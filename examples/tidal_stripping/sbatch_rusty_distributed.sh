#!/usr/bin/bash
#SBATCH --job-name=tidal_stripping
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00-00:20

module purge
module load python/3.12.9

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/jaxion-venv/bin/activate

srun --gpu-bind=none --cpu-bind=cores python tidal_stripping.py --res=8 --distributed
