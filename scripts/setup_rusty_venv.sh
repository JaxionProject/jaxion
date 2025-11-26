#!/bin/bash

# check that VENVDIR is set
if [ -z "$VENVDIR" ]; then
  echo "Error: VENVDIR environment variable is not set."
  exit 1
fi

# Create virtual environment on rusty
rm -fr $VENVDIR/jaxion-venv

module purge
module load python/3.12.9
python -m venv --system-site-packages $VENVDIR/jaxion-venv
source $VENVDIR/jaxion-venv/bin/activate
pip install --upgrade pip
pip install .[cuda12]