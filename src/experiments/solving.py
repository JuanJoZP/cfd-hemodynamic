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


def run_solving(config_path, output_base):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_params = config["base_params"]
    # Parámetros de simulación por defecto (pueden venir del YAML o ser sobreescritos)
    sim_params = config.get(
        "simulation_params",
        {"solver": "step_p2p1", "T": 1.0, "dt": 0.01, "mu": 3.5e-3, "rho": 1.06e-3},
    )

    combinations = generate_experiment_matrix(config)
    output_base = Path(output_base)

    print(f"[INFO] Iniciando resoluciones para {len(combinations)} experimentos...")

    for i, experiment in enumerate(combinations):
        exp_name = f"exp_{i:03d}"
        for k, v in experiment.items():
            exp_name += f"_{k}_{v}"

        exp_dir = output_base / exp_name
        mesh_path = exp_dir / "mesh.msh"

        if not mesh_path.exists():
            print(f"[WARN] No se encontró malla para {exp_name}. Saltando...")
            continue

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

            print(f"[DONE] {exp_name}")

        except Exception as e:
            print(f"[ERROR] Solver failed for {exp_name}: {e}")

            traceback.print_exc()
