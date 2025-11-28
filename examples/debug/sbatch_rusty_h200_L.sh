#!/usr/bin/bash
#SBATCH --job-name=jaxionL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition eval
#SBATCH --constraint=h200
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=144G
#SBATCH --time=00-00:05

module purge
module load python/3.12.9

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/jaxion-venv/bin/activate

srun --gpu-bind=none --cpu-bind=cores python debug.py --res=64 --distributed
