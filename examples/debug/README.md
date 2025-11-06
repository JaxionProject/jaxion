# Debug

Reproducer to setup/debug large runs on Frontier

Philip Mocz (2025)


## Install Jaxion on Frontier

1. Clone the repo:

```console
git clone git@github.com:JaxionProject/jaxion.git
```

2. Set up virtual environment (sets up conda-env jaxion-venv, installs jaxion -- see: https://docs.olcf.ornl.gov/software/analytics/jax.html):

```console
cd jaxion
./scripts/setup_venv_on_frontier.sh
```


## Submit jobs

Submit a Large (L) job (resolution 2046^3) on 16 nodes / 64 GPUs

```console
cd examples/debug
sbatch sbatch_frontier_L.sh
```

Submit an Extra Large (XL) job (resolution 4098^3) on 64 nodes / 512 GPUs

```console
cd examples/debug
sbatch sbatch_frontier_XL.sh
```
