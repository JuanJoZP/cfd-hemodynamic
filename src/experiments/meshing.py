import itertools
import os
import shutil
import traceback
from pathlib import Path

import cadquery as cq
import gmsh
import numpy as np
import yaml

from src.geom.stenosis.stenosis import generate_stenosis_geometry, mesh_and_export
from src.geom.tree.tree_model import VascularTree
from src.geom.tree.vascusynth_wrapper import generate_vascusynth_tree


def generate_experiment_matrix(config):
    """Genera todas las combinaciones de la matriz de experimentos."""
    matrix = config["matrix"]
    keys = matrix.keys()
    values = matrix.values()
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


def _merge_msh_files(stenosis_msh, tree_msh, out_msh):
    """
    Merge two already-meshed .msh files using gmsh Merge + Coherence.
    Physical groups from both meshes are preserved.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.merge(stenosis_msh)
    gmsh.merge(tree_msh)

    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.write(out_msh)
    gmsh.finalize()
    print(f"[OK] Combined mesh written to {out_msh}")


def run_meshing(config_path, output_base, job_idx=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_params = config["base_params"]
    combinations = generate_experiment_matrix(config)

    print(f"[INFO] Total experimentos posibles: {len(combinations)}")

    if job_idx is not None:
        if 0 <= job_idx < len(combinations):
            print(f"[INFO] Ejecutando SOLAMENTE el experimento índice {job_idx}")
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

        start_pt = np.array([0, 0, 0])
        end_pt = np.array([length, 0, 0])
        r_base_mid = (r_in + r_out) / 2
        min_radius = (1 - severity) * r_base_mid

        try:
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

            print("[INFO] Generando árbol vascular con VascuSynth...")
            tree_params = {**base_params, **experiment}

            tmp_dir = str(f"tmp_vascusynth_{i}")
            os.makedirs(tmp_dir, exist_ok=True)

            try:
                gxl_file = generate_vascusynth_tree(
                    tree_params, tmp_dir=tmp_dir, bind=True
                )
                vtree = VascularTree.from_xml(gxl_file, tree_params)
            finally:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)

            root_id = next(
                (nid for nid, nt in vtree.node_types.items() if "root node" in nt),
                None,
            )
            if root_id:
                for edge in vtree.edges:
                    if edge["from"] == root_id:
                        edge["radius"] = r_out

            vtree.apply_modifications()

            stenosis_dir = end_pt - start_pt
            stenosis_dir = stenosis_dir / np.linalg.norm(stenosis_dir)
            _rotate_tree_to_align(vtree, root_id, stenosis_dir)

            # Translation
            current_root_pos = np.array(vtree.nodes[root_id])
            translation = end_pt - current_root_pos
            for nid in vtree.nodes:
                vtree.nodes[nid] = tuple(np.array(vtree.nodes[nid]) + translation)

            solid_tree = vtree.build_solid()
            if not solid_tree:
                raise RuntimeError("No se pudo generar el sólido del árbol vascular")

            tree_brep = str(exp_dir / "tree.brep")
            tree_msh = str(exp_dir / "tree.msh")
            cq.exporters.export(solid_tree, tree_brep)
            vtree.mesh_and_tag(tree_brep, tree_msh)

            print("[INFO] Merging meshes...")
            combined_msh = str(exp_dir / "mesh.msh")
            _merge_msh_files(stenosis_msh, tree_msh, combined_msh)

            # os.remove(stenosis_msh)
            # os.remove(tree_msh)

            with open(exp_dir / "params.yaml", "w") as f:
                yaml.dump({"experiment": experiment, "base": base_params}, f)

            print(f"[OK] Experimento {exp_name} completado.")

        except Exception as e:
            print(f"[ERROR] Meshing failed for {exp_name}: {e}")
            traceback.print_exc()
