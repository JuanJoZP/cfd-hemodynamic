#!/bin/bash

#SBATCH --job-name=cfd-hemodynamic
#SBATCH --output=$HOME/data/logs/output_%j.log
#SBATCH --error=$HOME/data/logs/error_%j.log
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=usuario@example.com

mpich="/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpirun"
image="$HOME/fenicsx.sif"

$mpich -n $SLURM_NTASKS singularity exec \
    --bind $(dirname $image):/work \
    --bind $HOME/data:/data \
    --pwd /work \
    $image \
    bash -c "PYTHONPATH=/work:\$PYTHONPATH python3 main.py simulate --output_dir=/data/results $*"
