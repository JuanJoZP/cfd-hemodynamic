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
                solid_stenosis = generate_stenosis_geometry(
                    tuple(start_pt),
                    tuple(end_pt),
                    r_in,
                    r_out,
                    min_radius,
                    slope=base_params.get("slope", 0.5),
                )

                stenosis_brep = str(exp_dir / "stenosis.brep")
                stenosis_msh = str(exp_dir / "stenosis.msh")
                mesh_and_export(
                    solid_stenosis,
                    stenosis_brep,
                    stenosis_msh,
                    tuple(start_pt),
                    tuple(end_pt),
                )

                print(f"[INFO] Cargando arbol desde {tree_xml_path}...")
                tree_params = {**base_params, **experiment}

                # Recalcular voxel_width para asegurar escala correcta (mismo logica que wrapper)
                tree_volume = tree_params.get("tree_volume", 70.0)
                vol_mm3 = tree_volume * 1000.0
                cube_side = vol_mm3 ** (1.0 / 3.0)
                grid_size = 100
                voxel_width = cube_side / grid_size
                tree_params["voxel_width"] = voxel_width
                print(
                    f"[INFO] Using voxel_width: {voxel_width:.4f} mm for tree volume {tree_volume} mL"
                )

                vtree = VascularTree.from_xml(str(tree_xml_path), tree_params)

                root_id = next(
                    (nid for nid, nt in vtree.node_types.items() if "root node" in nt),
                    None,
                )

                # --- Coupling Logic ---
                # 1. Get root radius from tree (do not overwrite it)
                root_edge = next((e for e in vtree.edges if e["from"] == root_id), None)
                if not root_edge:
                    raise ValueError("Root node has no outgoing edge?")
                root_radius = root_edge["radius"]
                print(f"[INFO] Tree Root Radius: {root_radius:.4f} mm")

                # 2. Align tree to stenosis direction
                vtree.apply_modifications()  # modifiers might change radii
                # re-read radius just in case modifiers changed it (though usually only distal)
                root_radius = root_edge["radius"]

                _rotate_tree_to_align(vtree, root_id, stenosis_dir)

                # 3. Generate Coupling
                # Coupling starts at end_pt (end of stenosis)
                # Length depends on radius difference.
                # We control this via 'coupling_slope' (default 0.5).
                # Slope = delta_radius / length  =>  length = delta_radius / Slope
                # In generate_coupling_geometry, length = delta_radius * length_ratio
                # So length_ratio = 1 / coupling_slope

                c_slope = tree_params.get("coupling_slope", 0.5)
                # Avoid division by zero or negative
                if c_slope <= 1e-4:
                    c_slope = 0.5

                coupling_factor = 1.0 / c_slope

                # direction is stenosis_dir

                # IMPORTANT: function signature update in my head:
                # generate_coupling_geometry(start_pt, direction, r_start, r_end, length_ratio)

                coupling_solid, coupling_len = generate_coupling_geometry(
                    tuple(end_pt),
                    tuple(stenosis_dir),
                    r_out,
                    root_radius,
                    length_ratio=coupling_factor,
                )
                print(f"[INFO] Coupling generated with length {coupling_len:.4f} mm")

                coupling_msh = str(exp_dir / "coupling.msh")
                # We need to implement mesh_coupling or similar.
                # I'll assume we added mesh_coupling to src.geom.coupling
                # (Wait, I need to check if I added it in the previous step... yes I did)

                # To define the mesh properly we need start/end of coupling?
                # mesh_coupling in the file I just wrote takes (solid, filename). Generates generic mesh.
                mesh_coupling(coupling_solid, coupling_msh)

                # 4. Position Tree
                # Tree starts at end_pt + coupling_len * stenosis_dir
                coupling_end_pt = end_pt + stenosis_dir * coupling_len

                current_root_pos = np.array(vtree.nodes[root_id])
                translation = coupling_end_pt - current_root_pos
                for nid in vtree.nodes:
                    vtree.nodes[nid] = tuple(np.array(vtree.nodes[nid]) + translation)

                solid_tree = vtree.build_solid()
                if not solid_tree:
                    raise RuntimeError(
                        "No se pudo generar el solido del arbol vascular"
                    )

                tree_brep = str(exp_dir / "tree.brep")
                tree_msh = str(exp_dir / "tree.msh")
                cq.exporters.export(solid_tree, tree_brep)
                vtree.mesh_and_tag(tree_brep, tree_msh)

                # Save parameters to YAML
                if yaml:
                    with open(exp_dir / "params.yaml", "w") as f:
                        yaml.dump({"experiment": experiment, "base": base_params}, f)
                else:
                    print(
                        f"[WARN] PyYAML not available, skipping params.yaml dump for {exp_name}"
                    )

                print("[INFO] Merging meshes...")
                combined_msh = str(exp_dir / "mesh.msh")
                _merge_multiple_msh_files(
                    combined_msh, stenosis_msh, coupling_msh, tree_msh
                )

                print(f"[OK] Experimento {exp_name} completado.")

        except Exception as e:
            print(f"[ERROR] Meshing failed for {exp_name}: {e}")
            traceback.print_exc()
