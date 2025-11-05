#!/bin/bash

# Create virtual environment on Frontier

conda env remove --name jaxion-venv

# Set up JAX on Frontier. See: https://docs.olcf.ornl.gov/software/analytics/jax.html
module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4 # may work with ROCm 6.0.0 and 6.1.x
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0ls

conda create -n jaxion-venv python=3.11 numpy scipy -c conda-forge
conda activate jaxion-venv

pip install jax-rocm60-pjrt==0.5.0 jax-rocm60-plugin==0.5.0 --no-cache-dir
pip install jax==0.5.0 --no-cache-dir

pip install .
