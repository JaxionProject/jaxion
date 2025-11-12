import sys
import os
import importlib.util
from pathlib import Path
import importlib.resources
import json
from importlib.metadata import version
import jax
import jax.numpy as jnp


def print_parameters(params):
    print(json.dumps(params, indent=2))


def print_distributed_info():
    for env_var in [
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_NODELIST",
        "SLURM_STEP_NODELIST",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS",
    ]:
        print(f"{env_var}: {os.getenv(env_var, '')}")
    print("Total number of processes: ", jax.process_count())
    print("Total number of devices: ", jax.device_count())
    print("List of devices: ", jax.devices())
    print("Number of devices on this process: ", jax.local_device_count())


def set_up_parameters(user_overwrites):
    # first load the default params
    params_path = importlib.resources.files("jaxion") / "defaults.json"
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    # go down to lowest level in dict and eliminate meta-data
    params = _eliminate_metadata(params)

    # update default values with user-supplied overwrites
    params = _update_dicts(params, user_overwrites)

    # detect jaxion version
    params["version"] = version("jaxion")

    return params


def _eliminate_metadata(params):
    for key, value in params.items():
        if isinstance(value, dict):
            if "default" not in value:
                _eliminate_metadata(value)
            else:
                params[key] = value["default"]

    return params


def _update_dicts(orig_dict, new_dict):
    for key, value in new_dict.items():
        if (
            key in orig_dict
            and isinstance(orig_dict[key], dict)
            and isinstance(value, dict)
        ):
            _update_dicts(orig_dict[key], value)
        else:
            if key in orig_dict:
                if not isinstance(value, dict):
                    orig_dict[key] = value
                else:
                    raise ValueError(
                        f"Value: {value} for parameter key: {key} must be a value, not dict"
                    )
            else:
                raise KeyError(f"Unknown parameter key: {key}")
    return orig_dict


def run_example_main(example_path, argv=None):
    """
    Utility to run an example script's main() as if from its directory, with optional argv patching.
    Args:
        example_path (str or Path): Path to the example script (e.g., 'examples/foo/foo.py')
        argv (list, optional): List of arguments to patch sys.argv with. If None, uses [script_name].
    """
    example_path = Path(example_path)
    example_dir = example_path.parent
    script_name = example_path.name
    old_cwd = os.getcwd()
    old_argv = sys.argv.copy()
    try:
        spec = importlib.util.spec_from_file_location(
            script_name.rstrip(".py"), example_path
        )
        os.chdir(example_dir)
        sys.argv = [script_name] + (argv if argv is not None else [])
        module = importlib.util.module_from_spec(spec)
        sys.modules[script_name.rstrip(".py")] = module
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            result = module.main()
        else:
            raise AttributeError(f"No main() in {example_path}")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return result


# Make a distributed meshgrid function
def xmeshgrid(x_lin):
    xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    return xx, yy, zz


# NOTE: jaxdecomp (jd) has pfft3d transpose the axis (X, Y, Z) --> (Y, Z, X), and pifft3d undo it
# so the fourier space variables (e.g. kx, ky, kz) all need to be transposed


def xmeshgrid_transpose(x_lin):
    xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    xx = jnp.transpose(xx, (1, 2, 0))
    yy = jnp.transpose(yy, (1, 2, 0))
    zz = jnp.transpose(zz, (1, 2, 0))
    return xx, yy, zz


def xzeros(nx):
    return jnp.zeros((nx, nx, nx))


def xones(nx):
    return jnp.ones((nx, nx, nx))
