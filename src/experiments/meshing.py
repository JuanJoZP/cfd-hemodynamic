import itertools
import os
import shutil
import traceback
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import yaml
except ImportError:
    yaml = None

from src.geom.tree.vascusynth_wrapper import generate_vascusynth_tree
from src.utils.hpc import load_config


def generate_experiment_matrix(config):
    """Genera todas las combinaciones de la matriz de experimentos."""
    matrix = config["matrix"]
    keys = matrix.keys()
    values = [v if isinstance(v, list) else [v] for v in matrix.values()]
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


def _rotate_tree_to_align(vtree, root_id, target_dir):
    """
    Rotate all tree nodes around the root so the tree's natural direction
    (root → first child) aligns with target_dir.
    """
    if root_id is None:
        return

    root_pos = np.array(vtree.nodes[root_id])

    # Find the tree's natural direction: root → first child
    first_child = None
    for edge in vtree.edges:
        if edge["from"] == root_id:
            first_child = edge["to"]
            break
    if first_child is None:
        return

    tree_dir = np.array(vtree.nodes[first_child]) - root_pos
    norm = np.linalg.norm(tree_dir)
    if norm < 1e-12:
        return
    tree_dir = tree_dir / norm

    # Rotation via Rodrigues' formula
    target_dir = np.array(target_dir, dtype=float)
    dot = np.clip(np.dot(tree_dir, target_dir), -1.0, 1.0)

    if np.abs(dot - 1.0) < 1e-10:
        return  # already aligned

    if np.abs(dot + 1.0) < 1e-10:
        perp = np.array([1, 0, 0]) if abs(tree_dir[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(tree_dir, perp)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = np.cross(tree_dir, target_dir)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)

    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    for nid in vtree.nodes:
        pos = np.array(vtree.nodes[nid])
        vtree.nodes[nid] = tuple(root_pos + R @ (pos - root_pos))


def _merge_multiple_msh_files(out_msh, *msh_files):
    """
    Merge multiple .msh files using gmsh Merge + Coherence.
    Physical groups from meshes are preserved.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    for msh in msh_files:
        gmsh.merge(msh)

    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.write(out_msh)
    gmsh.finalize()
    print(f"[OK] Combined mesh written to {out_msh}")


def run_meshing(config_path, output_base, job_idx=None, mode="all"):
    if mode in ["all", "geometry"] and np is None:
        raise ImportError(
            "Numpy is required for geometry generation but not installed."
        )

    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    config = load_config(config_path)

    base_params = config["base_params"]
    combinations = generate_experiment_matrix(config)

    print(f"[INFO] Total experimentos posibles: {len(combinations)}")
    print(f"[INFO] Modo de ejecucion: {mode}")

    if job_idx is not None:
        if 0 <= job_idx < len(combinations):
            print(f"[INFO] Ejecutando SOLAMENTE el experimento indice {job_idx}")
            combinations_with_idx = [(job_idx, combinations[job_idx])]
        else:
            print(f"[ERROR] job_idx {job_idx} fuera de rango (0-{len(combinations)-1})")
            return
    else:
        combinations_with_idx = list(enumerate(combinations))

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    for i, experiment in combinations_with_idx:
        exp_name = f"exp_{i:03d}"
        for k, v in experiment.items():
            val_str = str(v).replace(".", "p")
            exp_name += f"_{k}_{val_str}"

        exp_dir = output_base / exp_name
        exp_dir.mkdir(exist_ok=True)

        print(f"\n[STEP] --- Experimento: {exp_name} (ID: {i}) ---")

        r_in = base_params["radius_in"]
        r_out = base_params["radius_out"]
        length = base_params["length"]
        severity = experiment.get("reduccion_lumen", 0.0)

        r_base_mid = (r_in + r_out) / 2
        min_radius = (1 - severity) * r_base_mid

        tree_xml_path = exp_dir / "tree_structure.xml"

        try:
            # ---------------------------
            # STAGE 1: Tree Generation
            # ---------------------------
            if mode in ["all", "tree"]:
                print("[INFO] Generando arbol vascular con VascuSynth...")
                tree_params = {**base_params, **experiment}

                tmp_dir = str(f"tmp_vascusynth_{i}")
                os.makedirs(tmp_dir, exist_ok=True)

                try:
                    gxl_file = generate_vascusynth_tree(
                        tree_params, tmp_dir=tmp_dir, bind=True
                    )
                    # Copiar el resultado al directorio del experimento para persistencia
                    shutil.copy(gxl_file, tree_xml_path)
                    print(f"[INFO] Arbol guardado en {tree_xml_path}")
                finally:
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)

            # ---------------------------
            # STAGE 2: Geometry & Meshing
            # ---------------------------
            if mode in ["all", "geometry"]:
                if not tree_xml_path.exists():
                    print(
                        f"[ERROR] No se encontro {tree_xml_path}. Ejecute primero en modo 'tree'."
                    )
                    continue

                # Import libraries only needed for geometry generation
                import cadquery as cq

                from src.geom.coupling import generate_coupling_geometry, mesh_coupling
                from src.geom.stenosis.stenosis import (
                    generate_stenosis_geometry,
                    mesh_and_export,
                )
                from src.geom.tree.tree_model import VascularTree

                start_pt = np.array([0, 0, 0])
                end_pt = np.array([length, 0, 0])
                stenosis_dir = end_pt - start_pt
                stenosis_dir = stenosis_dir / np.linalg.norm(stenosis_dir)

                print("[INFO] Generando y malleando estenosis...")
                current_params = {**base_params, **experiment}

                solid_stenosis = generate_stenosis_geometry(
                    tuple(start_pt),
                    tuple(end_pt),
                    r_in,
                    r_out,
                    min_radius,
                    slope=current_params.get("slope", 0.5),
                    position=current_params.get("stenosis_position", 0.5),
                )

                # We do NOT mesh stenosis separately anymore.

                print(f"[INFO] Cargando arbol desde {tree_xml_path}...")
                tree_params = {**base_params, **experiment}

                # Recalcular voxel_width
                tree_volume = tree_params.get("tree_volume", 70.0)
                vol_mm3 = tree_volume * 1000.0
                cube_side = vol_mm3 ** (1.0 / 3.0)
                grid_size = 100
                voxel_width = cube_side / grid_size
                tree_params["voxel_width"] = voxel_width

                vtree = VascularTree.from_xml(str(tree_xml_path), tree_params)

                root_id = next(
                    (nid for nid, nt in vtree.node_types.items() if "root node" in nt),
                    None,
                )

                # --- Coupling Logic ---
                root_edge = next((e for e in vtree.edges if e["from"] == root_id), None)
                if not root_edge:
                    raise ValueError("Root node has no outgoing edge?")
                root_radius = root_edge["radius"]
                print(f"[INFO] Tree Root Radius: {root_radius:.4f} mm")

                # Align tree
                vtree.apply_modifications()
                root_radius = root_edge["radius"]  # update
                _rotate_tree_to_align(vtree, root_id, stenosis_dir)

                # Generate Coupling
                c_slope = tree_params.get("coupling_slope", 0.5)
                if c_slope <= 1e-4:
                    c_slope = 0.05
                coupling_factor = 1.0 / c_slope

                # FORCE OVERLAP for robust union
                # Move coupling start slightly inside the stenosis
                overlap_dist = 0.1
                coupling_start = end_pt - stenosis_dir * overlap_dist

                coupling_solid, coupling_len = generate_coupling_geometry(
                    tuple(coupling_start),
                    tuple(stenosis_dir),
                    r_out,
                    root_radius,
                    length_ratio=coupling_factor,
                )
                print(
                    f"[INFO] Coupling generated with length {coupling_len:.4f} mm (overlap {overlap_dist}mm)"
                )

                # Position Tree
                coupling_end_pt = end_pt + stenosis_dir * coupling_len
                current_root_pos = np.array(vtree.nodes[root_id])
                translation = coupling_end_pt - current_root_pos

                # Apply translation to all nodes
                for nid in vtree.nodes:
                    vtree.nodes[nid] = tuple(np.array(vtree.nodes[nid]) + translation)

                # Build Tree Solid
                solid_tree = vtree.build_solid()
                if not solid_tree:
                    raise RuntimeError(
                        "No se pudo generar el solido del arbol vascular"
                    )

                # Clean individual solids
                try:
                    solid_stenosis = solid_stenosis.clean()
                    coupling_solid = coupling_solid.clean()
                    if hasattr(solid_tree, "clean"):
                        solid_tree = solid_tree.clean()
                except Exception as e:
                    print(f"[WARN] Failed to clean individual solids: {e}")

                # DEBUG: Export components
                print("[DEBUG] Exporting individual components for inspection...")
                try:
                    cq.exporters.export(
                        solid_stenosis, str(exp_dir / "debug_stenosis.brep")
                    )
                    cq.exporters.export(
                        coupling_solid, str(exp_dir / "debug_coupling.brep")
                    )
                    cq.exporters.export(solid_tree, str(exp_dir / "debug_tree.brep"))
                except Exception as e:
                    print(f"[WARN] Failed to export debug components: {e}")

                # --- UNION EVERYTHING ---
                print("[INFO] Uniendo solidos (Estenosis + Coupling + Arbol)...")

                # Use a combined approach with clean
                combined_solid = solid_stenosis.union(coupling_solid).union(solid_tree)
                try:
                    combined_solid = combined_solid.clean()
                except Exception as e:
                    print(f"[WARN] Failed to clean combined solid: {e}")

                merged_brep = str(exp_dir / "merged_geometry.brep")
                merged_msh = str(exp_dir / "mesh.msh")

                print(f"[INFO] Exportando geometria unificada a {merged_brep}")
                cq.exporters.export(combined_solid, merged_brep)

                # Mesh the merged geometry
                print("[INFO] Mallando geometria unificada...")

                # Collect terminals for tagging
                terminals = []
                for nid, ntype in vtree.node_types.items():
                    if "terminal node" in ntype:
                        terminals.append(vtree.nodes[nid])

                mesh_merged_geometry(
                    merged_brep,
                    merged_msh,
                    tuple(start_pt),  # Inlet (start of stenosis)
                    terminals,  # Outlets (tree terminals)
                )

                print(f"[OK] Experimento {exp_name} completado.")

        except Exception as e:
            print(f"[ERROR] Meshing failed for {exp_name}: {e}")
            traceback.print_exc()


def mesh_merged_geometry(brep_path, out_msh_path, inlet_pt, outlet_pts):
    """
    Meshes the single merged BREP file.
    Tags:
      - Inlet: Surface closest to inlet_pt
      - Outlets: Surfaces closest to outlet_pts
      - Walls: All other surfaces
    """
    import gmsh

    from src.geom.stenosis.stenosis import FLUID_TAG, INLET_TAG, OUTLET_TAG, WALL_TAG

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge(brep_path)

    try:
        # Heal shapes to fix potential tolerance issues
        gmsh.model.occ.healShapes()
    except Exception as e:
        print(f"[WARN] Failed to heal shapes: {e}")

    try:
        gmsh.model.occ.synchronize()
    except:
        pass

    # Tag Volume
    vols = gmsh.model.getEntities(3)
    if vols:
        gmsh.model.addPhysicalGroup(3, [vols[0][1]], FLUID_TAG)
        gmsh.model.setPhysicalName(3, FLUID_TAG, "Fluid")

    surfaces = gmsh.model.getBoundary(vols)

    # Helper for distance
    def get_surf_center_dist(tag, pt):
        bb = gmsh.model.getBoundingBox(2, tag)
        c = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
        return np.linalg.norm(np.array(c) - np.array(pt))

    # Identify Inlet
    best_in_dist = float("inf")
    inlet_tag = None

    # Pre-calculate distances to avoid re-querying
    surf_centers = []
    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(2, tag)
        c = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
        surf_centers.append((tag, np.array(c)))

    # Find Inlet
    inlet_pt_arr = np.array(inlet_pt)
    for tag, c in surf_centers:
        d = np.linalg.norm(c - inlet_pt_arr)
        if d < best_in_dist:
            best_in_dist = d
            inlet_tag = tag

    # Find Outlets
    outlet_tags = []
    for out_pt in outlet_pts:
        best_out_dist = float("inf")
        best_out_tag = None
        out_pt_arr = np.array(out_pt)
        for tag, c in surf_centers:
            if tag == inlet_tag or tag in outlet_tags:
                continue
            d = np.linalg.norm(c - out_pt_arr)
            if d < best_out_dist:
                best_out_dist = d
                best_out_tag = tag

        if best_out_tag:
            outlet_tags.append(best_out_tag)

    # Tagging
    walls = []
    for dim, tag in surfaces:
        if tag == inlet_tag:
            gmsh.model.addPhysicalGroup(2, [tag], INLET_TAG)
            gmsh.model.setPhysicalName(2, INLET_TAG, "Inlet")
        elif tag in outlet_tags:
            # We can group all outlets or individual
            # Let's group all as 2 for now, or distinct?
            # Existing code used 2 for "outlets" group.
            pass
        else:
            walls.append(tag)

    if outlet_tags:
        gmsh.model.addPhysicalGroup(2, outlet_tags, OUTLET_TAG)
        gmsh.model.setPhysicalName(2, OUTLET_TAG, "Outlets")

    if walls:
        gmsh.model.addPhysicalGroup(2, walls, WALL_TAG)
        gmsh.model.setPhysicalName(2, WALL_TAG, "Walls")

    # Mesh settings
    gmsh.option.setNumber("Mesh.Smoothing", 30)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 11)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(out_msh_path)
    gmsh.finalize()
    print(f"[OK] Combined mesh generated at {out_msh_path}")
