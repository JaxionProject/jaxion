#!/bin/bash

# Create virtual environment on Frontier

# Variables

rocm_version=7.0.2
jax_version=0.6.0

ENV_PATH='/lustre/orion/scratch/pmocz/ast231/venvs'
ENV_NAME='jaxion-venv'

# Set up JAX on Frontier. See: https://docs.olcf.ornl.gov/software/analytics/jax.html
module purge
module load PrgEnv-gnu/8.6.0
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda env remove --name $ENV_NAME

conda create -p $ENV_PATH/$ENV_NAME python=3.12.8 numpy scipy -c conda-forge --yes
conda activate $ENV_PATH/$ENV_NAME

pip install jax-rocm7-pjrt==$jax_version jax-rocm7-plugin==$jax_version --no-cache-dir
pip install jaxlib==$jax_version jax==$jax_version --no-cache-dir

pip install .

conda deactivate

echo "Virtual environment '$ENV_NAME' created and jaxion installed."
