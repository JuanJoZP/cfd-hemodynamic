#!/bin/bash

#SBATCH --job-name=cfd-meshing
#SBATCH --output=/home/juanjo.zuluaga/data/logs/mesh_%A_%a.out
#SBATCH --error=/home/juanjo.zuluaga/data/logs/mesh_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-5

# Imagen de Singularity
image="/home/juanjo.zuluaga/simulatio.nova/fenicsx.sif"

singularity exec \
    --bind /home/juanjo.zuluaga/simulatio.nova:/work \
    --bind /home/juanjo.zuluaga/data:/data \
    --pwd /work \
    $image \
    bash -c "PYTHONPATH=/work:\$PYTHONPATH python3 main.py experiment $* --job_idx $SLURM_ARRAY_TASK_ID"
