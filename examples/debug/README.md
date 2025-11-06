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

Submit a Large (L) job (resolution 2046^3) on 16 nodes / 64 GPUs:

```console
cd examples/debug
sbatch sbatch_frontier_L.sh
```

(this test should run in about 23 minutes.)

Submit an Extra Large (XL) job (resolution 4098^3) on 64 nodes / 512 GPUs:

```console
cd examples/debug
sbatch sbatch_frontier_XL.sh
```

This test currently fails with:

```console
E1105 14:17:03.123606  763404 pjrt_stream_executor_client.cc:3045] Execution of replica 0 failed: INVALID_ARGUMENT: CliqueIds size must be 1 for NCCL communicator initialization
```


## Notes

* May want to check out the install script `scripts/setup_venv_on_frontier.sh` for improved/more up-to-date ways to install JAX on Frontier.

* May want to check out the submit scripts `examples/debug/sbatch_frontier_XL.sh` for missing flags/environment variables.
