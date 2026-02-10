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

        run_meshing(config_path, output_path)
    elif args.exp_command == "solve":
        from .solving import run_solving

        run_solving(config_path, output_path)
    else:
        print("[ERROR] Subcomando de experiment no reconocido. Use 'mesh' o 'solve'.")
        return 1

    return 0
