import os
import shutil

import cadquery as cq
import yaml

from .tree_model import VascularTree
from .vascusynth_wrapper import generate_vascusynth_tree


def run(args, _unknown=None):
    """Generate vascular tree with VascuSynth from parsed CLI args."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    params = {}
    if "base_params" in cfg:
        params.update(cfg["base_params"])
    if "simulation_params" in cfg:
        params.update(cfg["simulation_params"])

    if args.perf_point:
        params["perf_point_mm"] = args.perf_point

    tmp_dir = "src/geom/tree/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        gxl_file = generate_vascusynth_tree(params, tmp_dir=tmp_dir, bind=args.bind)
        vtree = VascularTree.from_xml(gxl_file, params)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    vtree.apply_modifications()

    solid = vtree.build_solid()
    if solid:
        msh_output = (
            args.output if args.output.endswith(".msh") else args.output + ".msh"
        )

        brep_temp = ".vessels_temp.brep"
        cq.exporters.export(solid, brep_temp)

        print(f"[INFO] Generando malla en {msh_output}...")
        vtree.mesh_and_tag(brep_temp, msh_output)

        if os.path.exists(brep_temp):
            os.remove(brep_temp)

        print(f"[OK] Malla {msh_output} generada.")
    else:
        print("[ERROR] No se pudo construir la geometría sólida")
        return 1

    return 0
