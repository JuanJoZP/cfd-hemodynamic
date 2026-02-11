import ast
import itertools
import subprocess
import sys
from pathlib import Path


def load_config(config_path):
    """
    Load the full configuration (including 'matrix' and 'base_params') from the YAML config file.
    Tries to use PyYAML if available.
    Falls back to a simple text parser if PyYAML is missing (e.g. on HPC login node or host).
    """
    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("[INFO] PyYAML not found. Using fallback parser for config.")
        config = {"matrix": {}, "base_params": {}}

        current_section = None
        # Simple indentation-based parser
        # Assumes structure:
        # base_params:
        #   key: value
        # matrix:
        #   key: [v1, v2]

        with open(config_path, "r") as f:
            for line in f:
                content = line.split("#")[0].rstrip()
                if not content:
                    continue

                stripped = content.strip()
                indent = len(content) - len(stripped)

                if indent == 0 and stripped.endswith(":"):
                    # Top-level section
                    key = stripped[:-1]
                    if key in ["matrix", "base_params"]:
                        current_section = key
                    else:
                        current_section = None
                    continue

                if current_section and ":" in stripped:
                    key, val_str = stripped.split(":", 1)
                    key = key.strip()
                    val_str = val_str.strip()

                    if not val_str:
                        continue

                    try:
                        # Try to evaluate safely (numbers, lists, bools)
                        val = ast.literal_eval(val_str)
                    except (ValueError, SyntaxError):
                        # Keep as string if eval fails
                        val = val_str

                    config[current_section][key] = val

        return config


def dispatch_hpc(args, unknown):
    """
    Submits a job to the HPC using the corresponding script.
    It reads the script template, replaces placeholders if needed, and submits via sbatch.
    """

    command = args.command

    if command == "experiment":
        if args.exp_command == "mesh":
            config_path = Path(args.config)
            full_config = load_config(config_path)
            matrix = full_config.get("matrix", {})

            if not matrix:
                print(
                    f"[WARN] usage of yaml is deprecated on HPC login node, and fallback parser found no 'matrix' in {config_path} or it was empty."
                )
                if not matrix:
                    print("[ERROR] Matrix configuration not found.")
                    return

            keys = matrix.keys()
            values = [v if isinstance(v, list) else [v] for v in matrix.values()]
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            num_experiments = len(combinations)

            print(f"[INFO] Dispatching meshing for {num_experiments} experiments.")

            array_range = f"0-{num_experiments-1}"

            # Determine steps to run
            steps = []
            if hasattr(args, "meshing_mode"):
                if args.meshing_mode == "all":
                    steps = ["tree", "geometry"]
                else:
                    steps = [args.meshing_mode]
            else:
                steps = ["geometry"]

            # Override output directory logic (common for both)
            config_name = config_path.stem
            override_output = str(Path.home() / "data/meshes" / config_name)
            print(f"[INFO] Overriding output directory to: {override_output}")

            last_job_id = None

            for step in steps:
                script_name = "hpc_tree.sh" if step == "tree" else "hpc_mesh.sh"
                script_path = Path("src/experiments") / script_name

                # Adjust output path for geometry step (running in container with /data bind)
                step_output = override_output
                if step == "geometry":
                    home_data = str(Path.home() / "data")
                    if override_output.startswith(home_data):
                        step_output = override_output.replace(home_data, "/data", 1)

                # Reconstruct arguments: pass specific mode to the script
                step_args = [args.exp_command, "--config", args.config, "--mode", step]
                step_args.extend(["--output", step_output])

                cmd = ["sbatch", f"--array={array_range}"]

                # Dependency chain
                if last_job_id:
                    cmd.append(f"--dependency=afterok:{last_job_id}")

                cmd.append(str(script_path))
                cmd.extend(step_args)

                print(f"[INFO] Submitting {step} job via {script_name}...")
                print(f"       Command: {' '.join(cmd)}")

                try:
                    res = subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    print(res.stdout)
                    # Extract Job ID"Submitted batch job 123456"
                    for line in res.stdout.splitlines():
                        if line.startswith("Submitted batch job"):
                            last_job_id = line.split()[-1]
                            break
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Submission failed: {e.stderr}")
                    return

        elif args.exp_command == "solve":
            print("[WARN] HPC dispatch for 'experiment solve' not implemented yet.")

    elif command == "simulate":
        script_path = Path("src/simulation_hpc.sh")
        print("[INFO] Dispatching simulation job.")

        filtered_args = []
        skip_next = False
        for arg in sys.argv[1:]:  # Skip script name
            if skip_next:
                skip_next = False
                continue
            if arg == "--hpc":
                continue
            if arg == "--output_dir":
                skip_next = True
                continue
            if arg.startswith("--output_dir="):
                continue
            filtered_args.append(arg)

        cmd = ["sbatch", str(script_path)] + filtered_args
        print(f"[INFO] Submitting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
