#!/bin/bash

#SBATCH --job-name=cfd-hemodynamic
#SBATCH --output=/home/juanjo.zuluaga/data/logs/output_%j.log
#SBATCH --error=/home/juanjo.zuluaga/data/logs/error_%j.log
#SBATCH --ntasks=4
#SBATCH --time=12:00:00

mpich="/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpirun"
image="/home/juanjo.zuluaga/simulatio.nova/fenicsx.sif"

$mpich -n $SLURM_NTASKS singularity exec \
    --bind /home/juanjo.zuluaga/simulatio.nova:/work \
    --bind /home/juanjo.zuluaga/data:/data \
    --pwd /work \
    $image \
    bash -c "PYTHONPATH=/work:\$PYTHONPATH python3 main.py --output_dir=/data/results $*"
