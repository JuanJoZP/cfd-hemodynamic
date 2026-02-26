import contextlib
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


def run_solving(config_path, output_base, job_idx=None, early_stop_override=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    base_params = config.get("base_params", {})
    sim_params = config.get(
        "simulation_params",
        {"solver": "step_p2p1", "T": 1.0, "dt": 0.01, "mu": 3.5e-3, "rho": 1.06e-3},
    )

    # Build a unified param dict: sim_params values complement base_params.
    # This allows q_in, q_in_hyper, p_inlet, p_terminal, bc_type, hyperemia, etc.
    # to be declared in simulation_params and still be found by scenario_factory.
    effective_base_params = {**base_params, **sim_params}

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
        # Merge: matrix values override effective_base_params for this run
        run_params = {**effective_base_params, **experiment}

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
            print(f"[SOLVE] {exp_name}", flush=True)

        try:
            # 1. Obtener la CLASE del escenario con params de experimento 'congelados'
            if rank == 0:
                print(f"  [DEBUG] Creating scenario class...", flush=True)
            ScenarioClass = create_experiment_scenario_class(
                mesh_path, experiment, run_params
            )
            if rank == 0:
                print(f"  [DEBUG] Scenario class created.", flush=True)

            if early_stop_override is not None:
                run_params["early_stop_tolerance"] = early_stop_override

            # 2. Utilizar la clase Simulation para orquestar la ejecución
            solver_name = run_params.get("solver")
            if not solver_name:
                raise ValueError(
                    "Solver not specified in experiment matrix or simulation_params"
                )

            if rank == 0:
                print(
                    f"  [DEBUG] Creating Simulation object (solver={solver_name})...",
                    flush=True,
                )
            sim = Simulation(
                name=exp_name,
                simulation=ScenarioClass,
                solver=solver_name,
                T=run_params["T"],
                dt=run_params["dt"],
                output_dir=output_base,
                mu=run_params["mu"],
                rho=run_params["rho"],
                early_stop_tolerance=run_params["early_stop_tolerance"],
                **{k: v for k, v in experiment.items() if k != "solver"},
            )
            if rank == 0:
                print(f"  [DEBUG] Simulation object created.", flush=True)

            # 3. Ejecutar la simulación
            results_dir = exp_dir / "solution"

            if rank == 0:
                print(f"  [DEBUG] Starting sim.run()...", flush=True)

            try:
                sim.run(save_path=results_dir)
                if rank == 0:
                    print(f"[DONE] {exp_name}", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[ERROR] Simulation failed: {e}", flush=True)
                    traceback.print_exc()
                raise e

        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Solver failed for {exp_name}: {e}", flush=True)
                traceback.print_exc()
