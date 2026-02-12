import argparse
import ast
import sys


def run_simulate(args, unknown):
    """Run a single CFD simulation."""
    from src.simulation import Simulation

    kwargs = {}

    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:]
            val = None
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                i += 1
            else:
                val = True

            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass

            kwargs[key] = val
        i += 1

    if args.mu is not None:
        kwargs["mu"] = args.mu
    if args.rho is not None:
        kwargs["rho"] = args.rho

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Running simulation with extra args: {kwargs}")

    try:
        sim = Simulation(
            name=args.name,
            simulation=args.simulation,
            solver=args.solver,
            T=args.T,
            dt=args.dt,
            output_dir=args.output_dir,
            **kwargs,
        )
    except ValueError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Invalid configuration: {e}")
        return 1
    except ImportError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Failed to load module: {e}")
        return 1
    except SyntaxError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Syntax error in module: {e}")
        return 1
    except RuntimeError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Initialization failed: {e}")
        return 1
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}")
        raise

    try:
        sim.run()
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Simulation failed: {type(e).__name__}: {e}")
        raise

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CFD Hemodynamic - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Argumentos globales/comunes
    hpc_parent = argparse.ArgumentParser(add_help=False)
    hpc_parent.add_argument(
        "--hpc", action="store_true", help="Ejecutar en HPC via Slurm/Singularity"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando disponible")

    # ── simulate ──────────────────────────────────────────────────────────
    sim_parser = subparsers.add_parser(
        "simulate", parents=[hpc_parent], help="Ejecutar una simulación CFD"
    )
    sim_parser.add_argument(
        "--simulation", required=True, help="Scenario name (e.g. dfg_1)"
    )
    sim_parser.add_argument(
        "--solver", required=True, help="Solver name (e.g. stabilized_schur)"
    )
    sim_parser.add_argument("--mu", type=float, default=None, help="Viscosity")
    sim_parser.add_argument("--rho", type=float, default=None, help="Density")
    sim_parser.add_argument("--T", type=float, required=True, help="Total time")
    sim_parser.add_argument("--dt", type=float, required=True, help="Time step")
    sim_parser.add_argument("--name", required=True, help="Name of the run")
    sim_parser.add_argument("--output_dir", default="results", help="Output directory")

    # ── experiment ────────────────────────────────────────────────────────
    # ── experiment ────────────────────────────────────────────────────────
    exp_parser = subparsers.add_parser(
        "experiment", help="Gestor de matriz de experimentos"
    )

    # Argumentos comunes para los subcomandos de experiment
    exp_common = argparse.ArgumentParser(add_help=False)
    exp_common.add_argument(
        "--config", type=str, required=True, help="Ruta al YAML de configuración"
    )
    exp_common.add_argument(
        "--output",
        type=str,
        default="results/experiments",
        dest="exp_output",
        help="Directorio base para resultados",
    )
    exp_common.add_argument(
        "--job_idx",
        type=int,
        default=None,
        help="Índice del experimento a ejecutar (para Job Arrays)",
    )

    exp_subparsers = exp_parser.add_subparsers(
        dest="exp_command", help="Subcomandos de experiment"
    )

    exp_mesh_parser = exp_subparsers.add_parser(
        "mesh",
        parents=[exp_common, hpc_parent],
        help="Generar mallas para la matriz de experimentos",
    )
    exp_mesh_parser.add_argument(
        "--mode",
        choices=["all", "tree", "geometry"],
        default="all",
        dest="meshing_mode",
        help="Modo de ejecución del mallado",
    )

    exp_solve_parser = exp_subparsers.add_parser(
        "solve",
        parents=[exp_common, hpc_parent],
        help="Resolver ecuaciones para la matriz de experimentos",
    )
    exp_solve_parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of MPI cores per job (for HPC solve)",
    )

    # ── tree ──────────────────────────────────────────────────────────────
    tree_parser = subparsers.add_parser(
        "tree", parents=[hpc_parent], help="Generar árbol vascular con VascuSynth"
    )
    tree_parser.add_argument(
        "--config", type=str, required=True, help="Path al YAML de configuración"
    )
    tree_parser.add_argument(
        "--output", type=str, required=True, help="Path de salida de la malla (.msh)"
    )
    tree_parser.add_argument(
        "--bind",
        action="store_true",
        help="Bind al directorio actual para Singularity",
    )
    tree_parser.add_argument(
        "--perf_point",
        type=float,
        nargs=3,
        help="Punto de perfusión en mm (x y z)",
    )

    # ── Parse & dispatch ──────────────────────────────────────────────────
    args, unknown = parser.parse_known_args()

    # --hpc flag logic
    if getattr(args, "hpc", False):
        from src.utils.hpc import dispatch_hpc

        # Remove --hpc from sys.argv so it doesn't get passed downstream
        sys.argv = [a for a in sys.argv if a != "--hpc"]
        dispatch_hpc(args, unknown)
        return 0

    if args.command == "simulate":
        return run_simulate(args, unknown)
    elif args.command == "experiment":
        from src.experiments.main import run

        return run(args)
    elif args.command == "tree":
        from src.geom.tree.main import run

        return run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
