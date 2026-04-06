#!/bin/bash

#SBATCH --job-name=vascusynth-pretree
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/juanjo.zuluaga/data/logs/pretree_%j.out
#SBATCH --error=/home/juanjo.zuluaga/data/logs/pretree_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=juanjo.zuluaga@urosario.edu.co

# Generates the VascuSynth GXL on bare metal (singularity available here).
# The GXL is cached at TREE_TMP_DIR/output/tree_structure.xml and reused
# by the fenicsx simulation job (no nested singularity needed).

cd /home/juanjo.zuluaga/simulatio.nova
export PYTHONPATH=$(pwd):$PYTHONPATH

python3 - <<'EOF'
import os
from src.scenarios.stenosis_with_tree_params import TREE_TMP_DIR
from src.scenarios.stenosis_with_tree_params import (
    TREE_N_TERMINAL, TREE_VOLUME_ML, TREE_Q_IN, TREE_PERF_PRESSURE,
    TREE_TERM_PRESSURE, TREE_MURRAY_EXPONENT, TREE_LAMBDA, TREE_MU_VS,
    TREE_MIN_DISTANCE, TREE_CLOSEST_NEIGHBOURS, TREE_RANDOM_SEED,
    TREE_VESSEL_LOSS_FACTOR,
)
from src.geom.tree.vascusynth_wrapper import generate_vascusynth_tree

os.makedirs(os.path.join(TREE_TMP_DIR, "output"), exist_ok=True)

vs_params = {
    "n_terminal":         TREE_N_TERMINAL,
    "tree_volume":        TREE_VOLUME_ML,
    "q_in":               TREE_Q_IN,
    "perf_pressure":      TREE_PERF_PRESSURE,
    "term_pressure":      TREE_TERM_PRESSURE,
    "murray_exponent":    TREE_MURRAY_EXPONENT,
    "lambda":             TREE_LAMBDA,
    "mu_vs":              TREE_MU_VS,
    "min_distance":       TREE_MIN_DISTANCE,
    "closest_neighbours": TREE_CLOSEST_NEIGHBOURS,
    "random_seed":        TREE_RANDOM_SEED,
    "vessel_loss_factor": TREE_VESSEL_LOSS_FACTOR,
    "mu":                 3.5e-3,
}

gxl = generate_vascusynth_tree(vs_params, TREE_TMP_DIR)
print(f"[OK] GXL ready at: {gxl}")
EOF
