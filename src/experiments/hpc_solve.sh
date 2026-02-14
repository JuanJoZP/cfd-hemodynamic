#!/bin/bash

#SBATCH --job-name=cfd-solve
#SBATCH --output=/home/juanjo.zuluaga/data/logs/solve_%A_%a.out
#SBATCH --error=/home/juanjo.zuluaga/data/logs/solve_%A_%a.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-1

# Definir variables de entorno
mpich="/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpirun"
image="/home/juanjo.zuluaga/simulatio.nova/fenicsx.sif"

# Importante: Para DolfinX con MPI, usamos mpirun fuera de Singularity
# y llamamos a singularity exec.
$mpich -n $SLURM_NTASKS singularity exec \
    --bind /home/juanjo.zuluaga/simulatio.nova:/work \
    --bind /home/juanjo.zuluaga/data:/data \
    --pwd /work \
    $image \
    bash -c "PYTHONPATH=/work:\$PYTHONPATH python3 main.py experiment solve $* --job_idx $SLURM_ARRAY_TASK_ID"
