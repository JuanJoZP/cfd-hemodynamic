import sys
import traceback
from pathlib import Path

import yaml

from src.simulation import Simulation

from .meshing import generate_experiment_matrix
from .scenario_factory import create_experiment_scenario_class

# Añadir root al path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))


def run_solving(config_path, output_base, job_idx=None, mesh_source_dir=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    base_params = config["base_params"]
    sim_params = config.get(
        "simulation_params",
        {"solver": "step_p2p1", "T": 1.0, "dt": 0.01, "mu": 3.5e-3, "rho": 1.06e-3},
    )

    combinations = generate_experiment_matrix(config)
    output_base = Path(output_base)

    if rank == 0:
        print(f"[INFO] Total experimentos posibles: {len(combinations)}")

    if job_idx is not None:
        if 0 <= job_idx < len(combinations):
            if rank == 0:
                print(f"[INFO] Ejecutando SOLAMENTE el experimento indice {job_idx}")
            combinations_with_idx = [(job_idx, combinations[job_idx])]
        else:
            if rank == 0:
                print(
                    f"[ERROR] job_idx {job_idx} fuera de rango (0-{len(combinations)-1})"
                )
            return
    else:
        combinations_with_idx = list(enumerate(combinations))

    if rank == 0:
        print(
            f"[INFO] Iniciando resoluciones para {len(combinations_with_idx)} experimentos..."
        )

    for i, experiment in combinations_with_idx:
        exp_name = f"exp_{i:03d}"
        for k, v in experiment.items():
            val_str = str(v).replace(".", "p")
            exp_name += f"_{k}_{val_str}"

        exp_dir = output_base / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Mesh location logic
        mesh_path = (
            exp_dir / "mesh.msh"
        )  # Default: Look in the experiment folder itself

        if mesh_source_dir:
            # If explicit mesh source override provided
            explicit_mesh_base = Path(mesh_source_dir)
            explicit_mesh_path = explicit_mesh_base / exp_name / "mesh.msh"
            if explicit_mesh_path.exists():
                mesh_path = explicit_mesh_path

        if not mesh_path.exists():
            # Fallback logic for legacy structures (results vs meshes)
            if "results" in str(output_base):
                alt_base = Path(str(output_base).replace("results", "meshes"))
                alt_mesh_path = alt_base / exp_name / "mesh.msh"
                if alt_mesh_path.exists():
                    mesh_path = alt_mesh_path

        if not mesh_path.exists():
            if rank == 0:
                print(
                    f"[WARN] No se encontró malla para {exp_name} en {mesh_path} (ni alternativos). Saltando..."
                )
            continue

        if rank == 0:
            print(f"[SOLVE] {exp_name}")

        try:
            # 1. Obtener la CLASE del escenario con params de experimento 'congelados'
            ScenarioClass = create_experiment_scenario_class(
                mesh_path, experiment, base_params
            )

            # 2. Utilizar la clase Simulation para orquestar la ejecución
            sim = Simulation(
                name=exp_name,
                simulation=ScenarioClass,
                solver=sim_params["solver"],
                T=sim_params["T"],
                dt=sim_params["dt"],
                output_dir=output_base,
                mu=sim_params["mu"],
                rho=sim_params["rho"],
                **{k: v for k, v in experiment.items()},
            )

            # 3. Ejecutar la simulación en el directorio correspondiente
            # Simulation.run ya se encarga de setup() y de guardar los resultados
            results_dir = exp_dir / "solution"
            sim.run(save_path=results_dir)

            if rank == 0:
                print(f"[DONE] {exp_name}")

        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Solver failed for {exp_name}: {e}")
                traceback.print_exc()
