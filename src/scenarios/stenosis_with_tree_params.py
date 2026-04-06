# ══════════════════════════════════════════════════════════════════════════════
# VascuSynth + tree geometry configuration for stenosis_with_tree scenario.
# Edit constants here — imported by both the scenario and vascusynth_pretree.sh
# (no heavy dependencies so it can be used outside the fenicsx container).
# ══════════════════════════════════════════════════════════════════════════════

# ── VascuSynth ────────────────────────────────────────────────────────────────
TREE_N_TERMINAL = 5  # number of terminal vessels / outlets
TREE_VOLUME_ML = 0.01  # perfusion volume (mL); drives voxel size
TREE_Q_IN = 1.0  # inlet flow rate (mL/min)
TREE_PERF_PRESSURE = 13332  # perfusion pressure (Pa-equiv, ~100 mmHg)
TREE_TERM_PRESSURE = 1000  # terminal pressure (Pa-equiv)
TREE_MURRAY_EXPONENT = 3.0  # Murray's law exponent γ
TREE_LAMBDA = 2.0  # optimality weight λ
TREE_MU_VS = 1.0  # viscosity weight μ (VascuSynth internal param)
TREE_MIN_DISTANCE = 2  # minimum inter-node distance (voxels)
TREE_CLOSEST_NEIGHBOURS = 5  # optimizer neighbour count
TREE_RANDOM_SEED = 42  # RNG seed (change for different tree topologies)
TREE_VESSEL_LOSS_FACTOR = 0.0  # fraction of vessels to prune (0 = keep all)
TREE_TMP_DIR = "src/geom/tree/tmp"  # working directory for VascuSynth I/O

# ── 2-D projection / scaling ──────────────────────────────────────────────────
# TREE_COORD_SCALE = None  →  auto-fit: scale tree Y-extent to artery height H
# TREE_COORD_SCALE = float →  explicit meters-per-mm conversion factor
TREE_COORD_SCALE = None


# Slope of the 2-D trapezoidal coupling between the stenosis outlet (width H)
# and the tree root channel (width 2*r_root).
# Defined as: slope = (H/2 - r_root) / coupling_length  →  same convention as
# the stenosis slope.  The coupling_length is computed automatically.
TREE_COUPLING_SLOPE = 0.1
