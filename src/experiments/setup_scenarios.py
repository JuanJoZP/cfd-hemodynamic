import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Adjust path to import generate_experiment_matrix
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_PATH))

from src.experiments.meshing import generate_experiment_matrix
from src.utils.hpc import load_config


def run_command(cmd):
    print(f"[CMD] {cmd}")
    subprocess.check_call(cmd, shell=True)


def setup_scenarios(config_path, output_dir):
    config_path = Path(config_path)
    output_base = Path(output_dir)
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    config = load_config(config_path)

    combinations = generate_experiment_matrix(config)

    # Determine type based on matrix
    # I check the first combination
    first_exp = combinations[0]
    geo_type = "stenosis"
    if "tree" in config_path.name or "tree" in str(first_exp.get("geometry_type", "")):
        geo_type = "tree"

    print(f"[INFO] Setting up scenarios for {geo_type} from {config_path.name}")
    print(f"[INFO] Output Directory: {output_base}")

    # Generate Base Mesh in CWD/tmp_meshes (reused across calls if present)
    tmp_mesh_dir = Path("tmp_meshes")
    tmp_mesh_dir.mkdir(exist_ok=True)

    base_msh = tmp_mesh_dir / "base_{}.msh".format(geo_type)

    if not base_msh.exists():
        print("[INFO] Generating base mesh for {}...".format(geo_type))
        if geo_type == "stenosis":
            # Hardcoded stenosis params matching YAML base params roughly
            # radius_in: 3.0, length: 50.0, severity: 0.5
            # radius_in: 3.0, length: 50.0, severity: 0.5
            cmd = (
                "python src/geom/stenosis/stenosis.py "
                "--start 0 0 0 --end 50 0 0 "
                "--radius-in 3.0 --radius-out 3.0 "
                "--severity 0.5 --slope 0.5 "
                "--output_dir {}".format(tmp_mesh_dir)
            )
            # Ensure command is a single line for shell=True stability
            cmd = " ".join(cmd.split())

            try:
                run_command(cmd)
                # stenosis.py outputs 'stenosis.msh', rename to base_stenosis.msh
                # Wait, verify stenosis.py output name. It is joining output_dir + "stenosis.msh"
                (tmp_mesh_dir / "stenosis.msh").rename(base_msh)
            except subprocess.CalledProcessError:
                print(
                    "[WARN] Could not generate base mesh locally. Skipping. Expecting HPC generation."
                )
            except Exception as e:
                print(
                    "[WARN] Unexpected error generating mesh: {}. Skipping.".format(e)
                )

        elif geo_type == "tree":
            # Use existing tree structure
            tree_xml = Path("src/geom/vascular_tree/tree_structure.xml")
            if not tree_xml.exists():
                print("[ERROR] Tree XML not found for tree meshing.")
                return

            cmd = (
                "python src/geom/vascular_tree/treeToMesh.py "
                "--input {} --output {}".format(tree_xml, base_msh)
            )
            cmd = " ".join(cmd.split())

            try:
                run_command(cmd)
            except subprocess.CalledProcessError:
                print(
                    "[WARN] Could not generate base mesh locally. Skipping. Expecting HPC generation."
                )

    # Deploy to experiment folders
    output_base.mkdir(parents=True, exist_ok=True)

    for i, experiment in enumerate(combinations):
        exp_name = "exp_{:03d}".format(i)
        for k, v in experiment.items():
            val_str = str(v).replace(".", "p")
            exp_name += "_{}_{}".format(k, val_str)

        # We must match main.py naming exactly, so no custom prefix here.
        # User should use different output_dir.

        exp_dir = output_base / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Copy Config (manual dump since yaml might not be available)
        with open(exp_dir / "experiment_params.yaml", "w") as f:
            for k, val in experiment.items():
                f.write("{}: {}\n".format(k, val))
            # yaml.dump(experiment, f)

        # Copy Mesh
        dest_msh = exp_dir / "mesh.msh"
        if base_msh.exists():
            if not dest_msh.exists():
                shutil.copy(base_msh, dest_msh)
                print("[DEPLOY] Copied mesh to {}".format(exp_dir))
            else:
                print("[SKIP] Mesh already exists in {}".format(exp_dir))
        else:
            print("[INFO] No base mesh found. Skipping copy to {}".format(exp_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output_dir", required=True, help="Directory to create experiment folders in"
    )
    args = parser.parse_args()

    setup_scenarios(args.config, args.output_dir)
