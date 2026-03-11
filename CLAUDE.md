# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CFD hemodynamic simulation framework using FEniCSx for Navier-Stokes equations. Supports multiple scenarios (benchmarks, stenosis, vascular trees) and solvers, running locally via Docker or on HPC clusters via SLURM/Singularity.

## Running Simulations

The main entry point is `main.py`. It requires FEniCSx which is not installed locally — run inside Docker:

```bash
# Build Docker image (first time)
docker build -t cfd-hemodynamic .

# Interactive Docker session
docker run -it --rm -v $(pwd):/app -w /app cfd-hemodynamic bash

# Single simulation (inside Docker or Singularity)
python main.py simulate \
  --simulation dfg_1 \
  --solver stabilized_schur \
  --T 1.0 --dt 0.01 \
  --name test_run

# MPI parallel
mpirun -n 4 python main.py simulate --simulation dfg_1 --solver stabilized_schur --T 1.0 --dt 0.01 --name run_mpi

# Experiment matrix (mesh generation)
python main.py experiment mesh --config src/experiments/config/example.yaml

# Experiment matrix (solve)
python main.py experiment solve --config src/experiments/config/example.yaml

# HPC dispatch (any command, adds --hpc flag)
python main.py simulate --simulation dfg_1 --solver stabilized_schur --T 1.0 --dt 0.01 --name hpc_run --hpc
```

Output is written to `results/<scenario>/<timestamp>_<name>/` as VTX files (view with ParaView). Each run also produces `simulation_params.txt` and `norms.txt`.

## Architecture

### Core Abstractions

**`src/scenario.py` — `Scenario` (ABC)**
Base class for all scenarios. Defines the simulation lifecycle: `setup()` → `solve()`. Subclasses must implement:
- `mesh` property (dolfinx `Mesh`)
- `bcu` / `bcp` properties (velocity/pressure boundary conditions)
- `initial_velocity(x)`
- Optionally `exact_velocity(t)` for error computation

`Scenario.__init__` dynamically imports the solver from `src/solvers/<solver_name>.py` and instantiates it.

**`src/solverBase.py` — `SolverBase` (ABC)**
Base class for all solvers. Manages FEniCSx function spaces (`V` for velocity, `Q` for pressure). Subclasses must implement `setup(bcu, bcp)` and `solveStep()`. Provides `initVelocitySpace()`, `initPressureSpace()`, `initStressForm()`, and `assemble_wss()`.

**`src/simulation.py` — `Simulation`**
Orchestrator that dynamically loads a scenario by name from `src/scenarios/<name>.py`, infers constructor parameters via inspection, then calls `scenario.setup()` and `scenario.solve()`.

**`src/boundaryCondition.py` — `BoundaryCondition`**
Wraps dolfinx Dirichlet BCs. Call `initTopological(fdim, entities)` after creation.

### Solvers (`src/solvers/`)

All solvers define a `Solver` class inheriting `SolverBase`. The main ones:
- `stabilized_schur` — SUPG/PSPG/LSIC stabilized Navier-Stokes, Newton linearization, block FGMRES + Schur field-split preconditioner
- `stabilized_lsc` — similar, with LSC Schur approximation
- `stabilized_pcd` — similar, with PCD preconditioner
- `*_bdf2` variants — BDF2 time integration instead of Crank-Nicolson

Solvers use P1/P1 elements with stabilization (not inf-sup stable by default).

### Scenarios (`src/scenarios/`)

- `dfg_1` — DFG 2D cylinder benchmark (generates mesh via gmsh if `meshes/pipe_cylinder.xdmf` absent)
- `unit_square`, `unit_cube_pipe` — simple geometries for testing
- `pipe_cylinder`, `stenosis`, `vascular_tree` — hemodynamic scenarios
- `taylor_green`, `lid_driven2D` — classical CFD benchmarks

### Experiments (`src/experiments/`)

Batch experiment framework driven by YAML config files in `src/experiments/config/`. A config has:
- `artery_params` / `tree_params` / `fluid_params` / `base_params` — base geometry & fluid properties
- `simulation_params` — solver, T, dt, boundary conditions
- `matrix` — cartesian product of parameter variations; one job per combination

`scenario_factory.py` dynamically creates scenario classes (`LADExperimentScenario`) from mesh paths and experiment parameters.

### Geometry (`src/geom/`)

- `src/geom/stenosis/stenosis.py` — CadQuery + gmsh stenosis mesh generation (tags: `INLET_TAG=1`, `OUTLET_TAG=2`, `WALL_TAG=3`, `FLUID_TAG=4`)
- `src/geom/tree/` — VascuSynth wrapper for vascular tree generation

### HPC (`src/utils/hpc.py`, `src/experiments/hpc_*.sh`)

`dispatch_hpc()` in `hpc.py` submits jobs to SLURM using Singularity (`fenicsx.sif`). The `--hpc` flag on any CLI command redirects execution to HPC. Shell scripts in `src/experiments/` are used as SLURM job templates.

## Adding a New Scenario

1. Create `src/scenarios/<name>.py`
2. Define a class inheriting `Scenario`
3. Implement `mesh`, `bcu`, `bcp`, `initial_velocity`
4. Run with `--simulation <name>`

## Adding a New Solver

1. Create `src/solvers/<name>.py`
2. Define a `Solver` class inheriting `SolverBase`
3. Call `initVelocitySpace()` and `initPressureSpace()` in `__init__`
4. Implement `setup(bcu, bcp, facet_tags, tags)` and `solveStep()`
5. Run with `--solver <name>`
