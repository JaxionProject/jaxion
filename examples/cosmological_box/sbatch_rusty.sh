#!/usr/bin/bash
#SBATCH --job-name=cosmological_box
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00-02:00

module purge
module load python/3.12.9

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/jaxion-venv/bin/activate

srun python cosmological_box.py
