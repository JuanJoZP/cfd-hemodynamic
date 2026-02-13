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
        buffer = ""
        in_multiline_value = False
        bracket_count = 0
        current_key = None
        expecting_value = False

        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            # Remove comments and whitespace
            line_content = line.split("#")[0]
            stripped = line_content.strip()

            if not stripped:
                continue

            # Determine indentation
            indent = len(line_content) - len(line_content.lstrip())

            # Check for top-level sections
            if indent == 0 and stripped.endswith(":"):
                key = stripped[:-1]
                if key in ["matrix", "base_params", "simulation_params"]:
                    current_section = key
                    if current_section not in config:
                        config[current_section] = {}
                    current_key = None
                    buffer = ""
                    in_multiline_value = False
                    bracket_count = 0
                    expecting_value = False
                else:
                    current_section = None
                continue

            if not current_section:
                continue

            # If we are parsing a multiline value
            if in_multiline_value:
                buffer += " " + stripped
                bracket_count += stripped.count("[") - stripped.count("]")
                if bracket_count == 0:
                    # Value finished
                    try:
                        val = ast.literal_eval(buffer)
                        config[current_section][current_key] = val
                    except (ValueError, SyntaxError):
                        # Fallback for weird strings, maybe simple list manual parse needed?
                        # For now assume AST works for lists
                        print(f"[WARN] Failed to ast.parse: {buffer}")
                        config[current_section][current_key] = buffer

                    in_multiline_value = False
                    buffer = ""
                    current_key = None
                continue

            if expecting_value:
                if stripped.startswith("["):
                    buffer = stripped
                    bracket_count = stripped.count("[") - stripped.count("]")
                    if bracket_count > 0:
                        in_multiline_value = True
                    else:
                        try:
                            val = ast.literal_eval(buffer)
                        except:
                            val = buffer
                        config[current_section][current_key] = val
                        current_key = None
                else:
                    try:
                        val = ast.literal_eval(stripped)
                    except:
                        val = stripped
                    config[current_section][current_key] = val
                    current_key = None
                expecting_value = False
                continue

            # Standard key parsing
            if ":" in stripped:
                parts = stripped.split(":", 1)
                key = parts[0].strip()
                val_str = parts[1].strip()

                # Check if value starts a list but doesn't end it
                if val_str.startswith("[") and not val_str.endswith("]"):
                    current_key = key
                    buffer = val_str
                    bracket_count = val_str.count("[") - val_str.count("]")
                    in_multiline_value = True
                    continue

                # Check empty value (maybe start of block list, but we only support inline lists properly or need logic)
                # But previous logic supported '- item', let's just stick to the requested inline fix + multiline robustness

                if not val_str:
                    current_key = key
                    expecting_value = True
                    continue

                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
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

            # Check if job_idx is specified
            target_idx = None
            for i, arg in enumerate(sys.argv):
                if arg == "--job_idx":
                    if i + 1 < len(sys.argv):
                        target_idx = sys.argv[i + 1]

            if target_idx:
                print(f"[INFO] Dispatching single job for index {target_idx}")
                array_range = target_idx
            else:
                array_range = f"0-{num_experiments-1}"

            # Determine steps to run
            steps = []
            if hasattr(args, "meshing_mode"):
                if args.meshing_mode == "all":
                    # Check if we assume tree is needed based on config
                    # If geometry_type is present and ONLY contains 'stenosis', skip tree step
                    geo_types = matrix.get("geometry_type", [])
                    if geo_types and all(g == "stenosis" for g in geo_types):
                        print(
                            "[INFO] Detected pure stenosis experiment. Skipping tree generation step."
                        )
                        steps = ["geometry"]
                    else:
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
                # Correct path relative to where dispatch is called (root)
                script_path = Path("src/experiments") / script_name

                # Adjust output path for geometry step (running in container with /data bind)
                step_output = override_output

                # Only remap path for the meshing step which runs inside Singularity
                if step == "geometry":
                    container_output = step_output
                    home_data = str(Path.home() / "data")
                    # On HPC, home is /home/juanjo.zuluaga.
                    # We want to replace /home/juanjo.zuluaga/data -> /data
                    # But ONLY if the path actually starts with that.

                    # Hardcode the HPC home path for robustness if Path.home() is local
                    hpc_home_data = "/home/juanjo.zuluaga/data"

                    if step_output.startswith(hpc_home_data):
                        container_output = step_output.replace(
                            hpc_home_data, "/data", 1
                        )
                    elif step_output.startswith(home_data):
                        container_output = step_output.replace(home_data, "/data", 1)
                else:
                    # Tree generation runs on bare metal python (no container bind), so use absolute path
                    container_output = step_output

                # Reconstruct arguments: pass specific mode to the script
                # We need to construct the ARGS passed TO THE SCRIPT (python main.py ...)
                # The script itself (hpc_mesh.sh) takes arguments as $*

                # NOTE: We DO NOT pass --job_idx here because hpc_mesh.sh uses $SLURM_ARRAY_TASK_ID
                # However, if we use --array=5, then TASK_ID will be 5.

                step_args = ["mesh", "--config", str(config_path), "--mode", step]
                step_args.extend(["--output", container_output])

                cmd = ["sbatch", f"--array={array_range}"]
                if last_job_id:
                    cmd.append(f"--dependency=afterok:{last_job_id}")

                cmd.append(str(script_path))
                # Append python args to the sbatch script arguments
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

            print(f"[INFO] Dispatching solving for {num_experiments} experiments.")

            # Check if job_idx is specified
            target_idx = None
            for i, arg in enumerate(sys.argv):
                if arg == "--job_idx":
                    if i + 1 < len(sys.argv):
                        target_idx = sys.argv[i + 1]

            if target_idx:
                print(f"[INFO] Dispatching single job for index {target_idx}")
                array_range = target_idx
            else:
                array_range = f"0-{num_experiments-1}"

            # Output directory logic (must match meshing output location)
            # User correction: should use 'data/results' not 'data/meshes'
            config_name = config_path.stem
            override_output = str(Path.home() / "data/results" / config_name)
            print(f"[INFO] Using output directory: {override_output}")

            script_name = "hpc_solve.sh"
            script_path = Path("src/experiments") / script_name

            # Adjust output path for container (running in container with /data bind)
            container_output = override_output
            home_data = str(Path.home() / "data")
            # Hardcode HPC home path for robustness
            hpc_home_data = "/home/juanjo.zuluaga/data"

            if override_output.startswith(hpc_home_data):
                container_output = override_output.replace(hpc_home_data, "/data", 1)
            elif override_output.startswith(home_data):
                container_output = override_output.replace(home_data, "/data", 1)

            # Reconstruct arguments
            # We pass args to main.py experiment solve ...
            # The script (hpc_solve.sh) binds /data so we use container_output

            step_args = ["solve", "--config", str(config_path)]
            step_args.extend(["--output", container_output])

            # Logic for custom cores (ntasks)
            num_cores = getattr(args, "cores", 1)

            # Verify array range is valid (single job or range)
            cmd = ["sbatch", f"--array={array_range}"]
            # Override ntasks from command line based on user request
            cmd.append(f"--ntasks={num_cores}")

            # Note: This command-line argument overrides the #SBATCH --array directive in the script.
            cmd.append(str(script_path))
            cmd.extend(step_args)

            print(f"[INFO] Submitting solve job via {script_name}...")
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
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Submission failed: {e.stderr}")
                return

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
