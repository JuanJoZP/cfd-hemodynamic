#!/bin/bash

#SBATCH --job-name=cfd-tree-gen
#SBATCH --output=/home/juanjo.zuluaga/data/logs/tree_%A_%a.out
#SBATCH --error=/home/juanjo.zuluaga/data/logs/tree_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-5

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 main.py experiment $* --job_idx $SLURM_ARRAY_TASK_ID
