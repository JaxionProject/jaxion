#!/usr/bin/bash
#SBATCH -A AST231
#SBATCH --job-name=jaxionS
#SBATCH --output=slurm-mi250xS-%j.out
#SBATCH --error=slurm-mi250xS-%j.err
#SBATCH --partition batch
#SBATCH --nodes=1
#SBATCH --time=00-00:05

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/7.0.2
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

export LD_LIBRARY_PATH=/lustre/orion/ast231/scratch/pmocz/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
# export FI_MR_CACHE_MONITOR=userfaultfd
export FI_MR_CACHE_MONITOR=kdreg2
export PYTHONUNBUFFERED=TRUE

conda activate /lustre/orion/scratch/pmocz/ast231/venvs/jaxion-venv

srun -N$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=7 --gpu-bind=closest --cpu-bind=cores python debug.py --res=16
