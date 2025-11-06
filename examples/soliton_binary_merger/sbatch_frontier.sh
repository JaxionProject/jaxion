#!/usr/bin/bash
#SBATCH -A AST231
#SBATCH --job-name=soliton_binary_merger
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00-00:10

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

export PYTHONUNBUFFERED=TRUE

conda activate jaxion-venv

srun python soliton_binary_merger.py --res=8
