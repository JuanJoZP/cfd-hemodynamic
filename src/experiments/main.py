import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent


def run(args, _unknown=None):
    """Run experiment matrix (mesh or solve) from parsed CLI args."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_PATH / config_path

    if not config_path.exists():
        print(f"[ERROR] Archivo de configuración no encontrado: {config_path}")
        return 1

    output_path = Path(args.exp_output)
    if not output_path.is_absolute():
        output_path = ROOT_PATH / output_path

    if args.exp_command == "mesh":
        from .meshing import run_meshing

        # Helper: detect if we are likely on login node but forgot --hpc
        if args.job_idx is not None:
            try:
                import cadquery
            except ImportError:
                print(
                    "\n[WARNING] You specified --job_idx but are running locally without CadQuery."
                )
                print(
                    "[HINT] Did you mean to dispatch this job to the HPC? If so, add the '--hpc' flag:"
                )
                print(f"       python main.py {' '.join(sys.argv[1:])} --hpc\n")

        run_meshing(
            config_path, output_path, job_idx=args.job_idx, mode=args.meshing_mode
        )
    elif args.exp_command == "solve":
        # Helper: detect if we are likely on login node but forgot --hpc
        if args.job_idx is not None:
            try:
                import dolfinx
            except ImportError:
                print(
                    "\n[WARNING] You specified --job_idx but are running locally without dolfinx."
                )
                print(
                    "[HINT] Did you mean to dispatch this job to the HPC? If so, add the '--hpc' flag:"
                )
                print(f"       python main.py {' '.join(sys.argv[1:])} --hpc\n")

        from .solving import run_solving

        # Extract mesh_source_dir if present
        mesh_source_dir = getattr(args, "mesh_source_dir", None)
        run_solving(
            config_path,
            output_path,
            job_idx=args.job_idx,
            mesh_source_dir=mesh_source_dir,
        )
    else:
        print("[ERROR] Subcomando de experiment no reconocido. Use 'mesh' o 'solve'.")
        return 1

    return 0
