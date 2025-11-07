#!/bin/bash

# Create virtual environment on Frontier

# Set up JAX on Frontier. See: https://docs.olcf.ornl.gov/software/analytics/jax.html
module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4 # may work with ROCm 6.0.0 and 6.1.x
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda env remove --name jaxion-venv

conda create -n jaxion-venv python=3.11 numpy scipy -c conda-forge --yes
conda activate jaxion-venv

pip install jax-rocm60-pjrt==0.5.0 jax-rocm60-plugin==0.5.0 --no-cache-dir
pip install jax==0.5.0 --no-cache-dir

# HACK: inside requirements.txt, replace "jax==0.5.3" with "jax==0.5.0"
# since jax 0.5.3 is not available on Frontier
sed -i 's/jax==0.5.3/jax==0.5.0/g' requirements.txt

pip install .

# HACK: change back requirements.txt
sed -i 's/jax==0.5.0/jax==0.5.3/g' requirements.txt

conda deactivate

echo "Virtual environment 'jaxion-venv' created and jaxion installed."
