import ast
import itertools
import subprocess
import sys
from pathlib import Path


def load_config_matrix(config_path):
    """
    Load the 'matrix' section from the YAML config file.
    Tries to use PyYAML if available.
    Falls back to a simple text parser if PyYAML is missing (e.g. on HPC login node).
    """
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("matrix", {})
    except ImportError:
        print("[INFO] PyYAML not found. Using fallback parser for config matrix.")
        matrix = {}
        in_matrix = False
        indent_level = None
        parent_indent = 0

        with open(config_path, "r") as f:
            for line in f:
                # Remove comments
                line_content = line.split("#")[0].rstrip()
                if not line_content:
                    continue

                stripped = line_content.strip()

                # Detect start of matrix section
                if stripped.startswith("matrix:"):
                    in_matrix = True
                    # indentation of "matrix:" itself
                    parent_indent = len(line_content) - len(line_content.lstrip())
                    continue

                if in_matrix:
                    current_indent = len(line_content) - len(line_content.lstrip())

                    # If indentation drops back to parent level, we are out of matrix
                    if indent_level is None:
                        # Establish indentation level for matrix items
                        if current_indent > parent_indent:
                            indent_level = current_indent
                        else:
                            # Empty matrix or immediate end?
                            in_matrix = False
                            continue

                    if current_indent < indent_level:
                        in_matrix = False
                        continue

                    # Parse key: value
                    if ":" in stripped:
                        key, val_str = stripped.split(":", 1)
                        key = key.strip()
                        val_str = val_str.strip()
                        try:
                            # Evaluate list safely
                            val = ast.literal_eval(val_str)
                            if isinstance(val, list):
                                matrix[key] = val
                            else:
                                # Start of a list across multiple lines?
                                # This simple parser might not handle multi-line lists perfectly
                                # unless they are within [] brackets on new lines.
                                # For now assume standard format: key: [1, 2, 3]
                                print(
                                    f"[WARN] Fallback parser skipped non-list value for {key}: {val_str}"
                                )
                        except (ValueError, SyntaxError):
                            print(
                                f"[WARN] Fallback parser failed to parse value for {key}: {val_str}"
                            )

        return matrix


def dispatch_hpc(args, unknown):
    """
    Submits a job to the HPC using the corresponding script.
    It reads the script template, replaces placeholders if needed, and submits via sbatch.
    """

    command = args.command

    if command == "experiment":
        if args.exp_command == "mesh":
            script_path = Path("src/experiments/hpc_mesh.sh")
            config_path = Path(args.config)

            matrix = load_config_matrix(config_path)

            if not matrix:
                print(
                    f"[WARN] usage of yaml is deprecated on HPC login node, and fallback parser found no 'matrix' in {config_path} or it was empty."
                )
                # We do not crash, maybe user wants to run single job?
                # But for array logic we need matrix.
                # Let's assume 1 job if matrix is empty? Or error out?
                # The original code would crash on key error.
                if not matrix:
                    print("[ERROR] Matrix configuration not found.")
                    return

            keys = matrix.keys()
            values = matrix.values()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            num_experiments = len(combinations)

            print(f"[INFO] Dispatching meshing for {num_experiments} experiments.")

            array_range = f"0-{num_experiments-1}"

            filtered_args = []
            skip_next = False
            # Start from index 2 to skip 'main.py' and 'experiment'
            for arg in sys.argv[2:]:
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--hpc":
                    continue
                if arg == "--output":
                    skip_next = True  # Skip the value of --output
                    continue
                if arg.startswith("--output="):
                    continue
                filtered_args.append(arg)

            # Override output directory
            config_name = config_path.stem
            override_output = f"/data/meshes/{config_name}"
            print(f"[INFO] Overriding output directory to: {override_output}")
            filtered_args.extend(["--output", override_output])

            cmd = ["sbatch", f"--array={array_range}", str(script_path)] + filtered_args
            print(f"[INFO] Submitting: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

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
