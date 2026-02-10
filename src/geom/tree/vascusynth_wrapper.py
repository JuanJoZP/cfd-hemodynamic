import os
import subprocess


def generate_vascusynth_tree(params, tmp_dir="src/geom/tree/tmp", bind=False):
    """
    Genera los archivos de configuración para VascuSynth y ejecuta el wrapper.
    """

    # Volumen de perfusión en mL -> mm^3
    tree_volume = params.get("tree_volume", 70.0)
    vol_mm3 = tree_volume * 1000.0
    cube_side = vol_mm3 ** (1.0 / 3.0)
    grid_size = 100
    voxel_width = cube_side / grid_size
    params["voxel_width"] = voxel_width

    # Mapas de suministro y oxigenación (constantes en un cubo)
    oxy_file = os.path.join(tmp_dir, "oxygenation.txt")
    with open(oxy_file, "w") as f:
        f.write(f"{grid_size} {grid_size} {grid_size}\n")
        f.write(f"0 0 0 {grid_size} {grid_size} {grid_size}\n")
        f.write("1\n")

    supply_file = os.path.join(tmp_dir, "supply.txt")
    with open(supply_file, "w") as f:
        f.write(f"{grid_size} {grid_size} {grid_size} 1\n")
        f.write(f"0 0 0 {grid_size} {grid_size} {grid_size}\n")
        f.write("1\n")

    # Archivo de parámetros vasculares
    config1_file = os.path.join(tmp_dir, "config1.txt")
    # PERF_FLOW: Usamos el flujo de entrada q_in
    perf_flow = params.get("q_in", 70.0) / 60.0  # mL/min -> mL/s

    # Point of perfusion: Center of the face X=0
    px = 0
    py = grid_size // 2
    pz = grid_size // 2

    vs_params = {
        "SUPPLY_MAP": "supply.txt",
        "OXYGENATION_MAP": "oxygenation.txt",
        "PERF_POINT": f"{px} {py} {pz}",
        "PERF_PRESSURE": int(params.get("perf_pressure", 13332)),  # ~100 mmHg
        "TERM_PRESSURE": int(params.get("term_pressure", 1000)),
        "PERF_FLOW": perf_flow,
        "RHO": params.get("mu", 0.0035),  # En VascuSynth RHO es viscosidad
        "GAMMA": params.get("murray_exponent", 3.0),
        "LAMBDA": params.get("lambda", 2.0),
        "MU": params.get("mu_vs", 1.0),
        "MIN_DISTANCE": params.get("min_distance", 1),
        "NUM_NODES": int(params.get("n_terminal", 200)),
        "VOXEL_WIDTH": voxel_width,
        "CLOSEST_NEIGHBOURS": int(params.get("closest_neighbours", 5)),
    }

    with open(config1_file, "w") as f:
        for k, v in vs_params.items():
            f.write(f"{k}: {v}\n")

    # configs.txt (lista de archivos de parámetros)
    with open(os.path.join(tmp_dir, "configs.txt"), "w") as f:
        f.write("config1.txt")

    # output_dir.txt (lista de carpetas de salida)
    with open(os.path.join(tmp_dir, "output_dir.txt"), "w") as f:
        f.write("output")

    # Ejecutar VascuSynth a través del script shell
    script_path = "src/geom/tree/vascusynth.sh"
    # El script espera ser ejecutado desde el root del proyecto
    cmd = [
        "bash",
        script_path,
        "--voxel_width",
        str(voxel_width),
        "--temp_dir",
        tmp_dir,
    ]
    if bind:
        cmd.extend(["--bind", "."])

    print(f"[INFO] Running VascuSynth wrapper: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] VascuSynth failed with code {result.returncode}")
        raise RuntimeError("VascuSynth execution failed")

    # El GXL de salida estará en tmp/output/output.gxl
    gxl_path = os.path.join(tmp_dir, "output", "tree_structure.xml")
    return gxl_path
