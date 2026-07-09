import ast
import itertools
import subprocess
import sys
from pathlib import Path


def load_config(config_path):
    """
    Load the full configuration from a YAML config file (strict mode).

    Supports both the legacy flat 'base_params' layout and the new structured
    layout with 'artery_params', 'tree_params', and 'fluid_params' sections.
    All named sections are merged into a single 'base_params' dict so the rest
    of the pipeline works without changes.

    Raises ValueError if any unknown top-level section or any unknown parameter
    within a known section is found.

    Tries to use PyYAML if available; falls back to a simple text parser.
    """
    # ------------------------------------------------------------------ #
    # Schema: which sections and which keys are valid in each section.    #
    # Add new params here when the pipeline supports them.                #
    # ------------------------------------------------------------------ #

    KNOWN_SECTIONS = {
        "matrix",
        "base_params",
        "simulation_params",
        "artery_params",
        "tree_params",
        "fluid_params",
    }
    MERGE_INTO_BASE = {"artery_params", "tree_params", "fluid_params"}

    # Parameters allowed inside base_params (or its structured equivalents).
    VALID_BASE_PARAMS = {
        # --- Artery geometry ---
        "radius_in",
        "radius_out",
        "length",
        "slope",
        "stenosis_position",
        "stenosis_severity",
        "stenosis_slope",
        "coupling_slope",
        # --- Boundary-condition flow values ---
        "q_in",
        "q_in_hyper",
        "p_terminal",
        "p_inlet",
        "p_outlet",
        # --- Vascular tree (VascuSynth) ---
        "tree_volume",
        "n_terminal",
        "perf_pressure",
        "term_pressure",
        "murray_exponent",
        "closest_neighbours",
        "random_seed",
        # --- DMV pathology modifiers ---
        "wall_thickening_severity",
        "thickening_level_threshold",
        "vessel_loss_factor",
        "hyperemia_dilation_factor",
        # --- Fluid properties ---
        "mu",
        "rho",
        "artery_mesh_size_from_curvature",
        # --- Legacy flat-layout solver params (test_simple / arteria_lad) ---
        "solver",
        "T",
        "dt",
        "early_stop_tolerance",
        "bc_type",
    }

    # Parameters allowed inside simulation_params.
    VALID_SIMULATION_PARAMS = {
        "solver",
        "T",
        "dt",
        "mu",
        "rho",
        "q_in",
        "q_in_hyper",
        "p_inlet",
        "p_outlet",
        "p_terminal",
        "bc_type",
        "geometry_type",
        "hyperemia",
        "early_stop_tolerance",
    }

    # Parameters allowed inside the matrix section.
    # These are the experiment axes that drive the combinatorial sweep.
    VALID_MATRIX_PARAMS = {
        "hyperemia",
        "vessel_loss_factor",
        "wall_thickening_severity",
        "thickening_level_threshold",
        "stenosis_severity",
        "stenosis_position",
        "lumen_thickening_factor",
        "hyperemia_dilation_factor",
        "bc_type",
        "geometry_type",
        "solver",
        "stenosis_slope",
        "p_inlet",
        "p_terminal",
        "q_in",
        "q_in_hyper",
        "p_outlet",
        "artery_mesh_size_from_curvature",
        "early_stop_tolerance",
    }

    # Map each logical section name (after merging) to its allowed-key set.
    SECTION_SCHEMA = {
        "base_params": VALID_BASE_PARAMS,
        "simulation_params": VALID_SIMULATION_PARAMS,
        "matrix": VALID_MATRIX_PARAMS,
    }

    # ------------------------------------------------------------------ #
    # Internal helper: validate the final dict (after merges).            #
    # ------------------------------------------------------------------ #
    def _validate(config: dict, source: str) -> None:
        """Raise ValueError for unknown sections or unknown parameters."""
        unknown_sections = set(config.keys()) - KNOWN_SECTIONS
        if unknown_sections:
            raise ValueError(
                f"[CONFIG ERROR] {source}: unknown top-level section(s): "
                f"{sorted(unknown_sections)}.\n"
                f"  Allowed sections: {sorted(KNOWN_SECTIONS)}"
            )

        for section, schema in SECTION_SCHEMA.items():
            if section not in config:
                continue
            section_data = config[section]
            if not isinstance(section_data, dict):
                continue
            unknown_keys = set(section_data.keys()) - schema
            if unknown_keys:
                raise ValueError(
                    f"[CONFIG ERROR] {source}: unknown parameter(s) in "
                    f"'{section}': {sorted(unknown_keys)}.\n"
                    f"  Allowed parameters: {sorted(schema)}"
                )

    try:
        import yaml

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(
                f"[CONFIG ERROR] {config_path}: file is empty or could not be parsed."
            )

        # Validate top-level sections *before* merging so structured names
        # like 'artery_params' are still visible.
        unknown_sections = set(raw.keys()) - KNOWN_SECTIONS
        if unknown_sections:
            raise ValueError(
                f"[CONFIG ERROR] {config_path}: unknown top-level section(s): "
                f"{sorted(unknown_sections)}.\n"
                f"  Allowed sections: {sorted(KNOWN_SECTIONS)}"
            )

        # Validate each structured section against VALID_BASE_PARAMS before
        # merging (they share the same allowed set).
        for section in MERGE_INTO_BASE:
            if section in raw and isinstance(raw[section], dict):
                unknown_keys = set(raw[section].keys()) - VALID_BASE_PARAMS
                if unknown_keys:
                    raise ValueError(
                        f"[CONFIG ERROR] {config_path}: unknown parameter(s) in "
                        f"'{section}': {sorted(unknown_keys)}.\n"
                        f"  Allowed parameters: {sorted(VALID_BASE_PARAMS)}"
                    )

        # Merge structured sections into base_params for pipeline compatibility.
        if any(k in raw for k in MERGE_INTO_BASE):
            merged = dict(raw.get("base_params", {}))
            for section in MERGE_INTO_BASE:
                merged.update(raw.get(section, {}))
            # Remove the now-merged structured keys so the final dict only
            # contains canonical sections.
            for section in MERGE_INTO_BASE:
                raw.pop(section, None)
            raw["base_params"] = merged

        # Final validation on the merged config.
        _validate(raw, str(config_path))

        return raw

    except ImportError:
        print("[INFO] PyYAML not found. Using fallback parser for config.")
        import re

        config = {s: {} for s in KNOWN_SECTIONS if s not in MERGE_INTO_BASE}
        config["base_params"] = {}

        current_dict = None
        # stack of (indent, dict_object)
        stack = []

        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, start=1):
            line_content = line.split("#")[0]
            stripped = line_content.strip()
            if not stripped:
                continue

            indent = len(line_content) - len(line_content.lstrip())

            # Match "key: value" or just "key:"
            m = re.match(r"^([^:]+):\s*(.*)$", stripped)
            if not m:
                continue

            key = m.group(1).strip()
            val_str = m.group(2).strip()

            if indent == 0:
                # Top level section
                sec_name = key
                if sec_name in MERGE_INTO_BASE:
                    current_dict = config["base_params"]
                elif sec_name in KNOWN_SECTIONS:
                    current_dict = config[sec_name]
                else:
                    raise ValueError(
                        f"[CONFIG ERROR] {config_path} line {line_no}: "
                        f"unknown section '{sec_name}'"
                    )
                stack = [(0, current_dict)]
                continue

            if current_dict is None:
                continue

            # Adjust stack based on indentation
            while stack and indent <= stack[-1][0] and stack[-1][0] != 0:
                stack.pop()

            if not stack:
                continue  # Should not happen if indent > 0

            parent = stack[-1][1]

            if not val_str:
                # Nested block start
                nested = {}
                parent[key] = nested
                stack.append((indent, nested))
            else:
                # Leaf key-value
                try:
                    # Handle basic types: [1,2], 0.1, true, false, "str"
                    # For YAML 'true'/'false', ast needs 'True'/'False'
                    p_val = val_str
                    if p_val.lower() == "true":
                        val = True
                    elif p_val.lower() == "false":
                        val = False
                    else:
                        try:
                            val = ast.literal_eval(p_val)
                        except:
                            val = p_val
                except:
                    val = val_str
                parent[key] = val

        # Final validation after full parse
        _validate(config, str(config_path))
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
                array_range = f"0-{num_experiments - 1}"

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
                array_range = f"0-{num_experiments - 1}"

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

            # Override time limit if specified
            time_limit = getattr(args, "time_limit", None)
            if time_limit:
                cmd.append(f"--time={time_limit}")

            if getattr(args, "monitor", False):
                step_args.extend(
                    [
                        "-snes_monitor",
                        "-snes_view",
                        "-ksp_monitor",
                    ]
                )

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

                # Extract job ID from sbatch output
                job_id = None
                for line in res.stdout.splitlines():
                    if line.startswith("Submitted batch job"):
                        job_id = line.split()[-1]
                        break

                # --watch: open tmux with sacct watcher + wjob alias
                if job_id and getattr(args, "watch", False):
                    log_dir = str(Path.home() / "data/logs")
                    tmux_session = f"watch_{job_id}"

                    # Write a temp bashrc with the wjob function
                    rc_path = Path.home() / ".wjob_rc"
                    rc_path.write_text(
                        f'wjob() {{ tail -f {log_dir}/solve_{job_id}_"$1".out; }}\n'
                        f'echo "wjob alias ready. Usage: wjob <idx>"\n'
                        f'echo "Example: wjob 0"\n'
                    )

                    subprocess.run(
                        f"tmux new-session -d -s {tmux_session} "
                        f"'watch -n 5 sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Elapsed'",
                        shell=True,
                    )
                    subprocess.run(
                        f"tmux split-window -t {tmux_session} -h 'bash --rcfile {rc_path}'",
                        shell=True,
                    )

                    print(f"[INFO] Launching tmux watch session '{tmux_session}'...")
                    print(f"[INFO] Use 'wjob <idx>' in the right pane to tail logs.")
                    subprocess.run(f"tmux attach -t {tmux_session}", shell=True)

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Submission failed: {e.stderr}")
                return

    elif command == "simulate":
        script_path = Path("src/simulation_hpc.sh")
        print("[INFO] Dispatching simulation job.")

        filtered_args = []
        skip_next = False
        num_cores = getattr(args, "cores", 1)
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
            if arg == "--cores":
                skip_next = True
                continue
            if arg.startswith("--cores="):
                continue
            filtered_args.append(arg)

        # Scenarios that need a bare-metal VascuSynth pre-job before fenicsx
        SCENARIOS_WITH_TREE = {"stenosis_with_tree"}
        simulation_name = None
        for i, a in enumerate(filtered_args):
            if a == "--simulation" and i + 1 < len(filtered_args):
                simulation_name = filtered_args[i + 1]

        dependency_flag = []
        if simulation_name in SCENARIOS_WITH_TREE:
            pretree_script = Path("src/geom/tree/vascusynth_pretree.sh")
            print(f"[INFO] Submitting VascuSynth pre-job: {pretree_script}")
            pretree_result = subprocess.run(
                ["sbatch", str(pretree_script)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            print(pretree_result.stdout.strip())
            pretree_job_id = None
            for line in pretree_result.stdout.splitlines():
                if line.startswith("Submitted batch job"):
                    pretree_job_id = line.split()[-1]
                    break
            if pretree_job_id:
                dependency_flag = [f"--dependency=afterok:{pretree_job_id}"]
                print(
                    f"[INFO] Simulation will start after pre-job {pretree_job_id} completes."
                )

        time_limit = getattr(args, "time_limit", None)
        time_flag = [f"--time={time_limit}"] if time_limit else []

        cmd = (
            ["sbatch", f"--ntasks={num_cores}"]
            + time_flag
            + dependency_flag
            + [str(script_path)]
            + filtered_args
        )
        print(f"[INFO] Submitting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
