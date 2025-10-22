import jax
import jax.numpy as jnp
import jaxion
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import argparse
import os
import sys  # XXX

sys.path.append("../../")  # XXX

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Tidal Stripping of Fuzzy Dark Matter soliton in an external potential

Philip Mocz (2025)

Usage:
  python tidal_stripping.py --res <resolution_multiplier>

  python tidal_stripping.py --distributed --emulate
"""


def set_up_simulation(resolution_multiplier, sharding):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "external_potential": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params, sharding=sharding)

    # Set initial conditions (orbiting soliton in external potential)
    M_soliton = 1.0e9  # mass of soliton (M_sun)
    k_soliton = 4.0  # wave-number for orbital motion of soliton
    r_separation = 2.0  # separation of soliton from center (kpc)
    m_22 = sim.params["quantum"]["m_22"]
    m = sim.axion_mass
    hbar = jaxion.constants["reduced_planck_constant"]
    G = jaxion.constants["gravitational_constant"]
    r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc
    box_size = sim.params["domain"]["box_size"]

    xx, yy, zz = sim.grid
    r = jnp.sqrt(
        (xx - 0.5 * box_size) ** 2
        + (yy - 0.5 * box_size - r_separation) ** 2
        + (zz - 0.5 * box_size) ** 2
    )
    sim.state["psi"] = (
        jnp.sqrt(
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )
        + 0.0j
    )
    # add circular orbit velocity
    sim.state["psi"] *= jnp.exp(1.0j * k_soliton * xx)

    # add external potential (host halo)
    M_halo = 0.25 * box_size * k_soliton**2 * hbar**2 / (G * m**2)
    print(f"M_halo: {M_halo:.2e} M_sun")
    assert M_halo > M_soliton * 2.0  # halo should be much more massive than soliton

    r = jnp.sqrt(
        (xx - 0.5 * box_size) ** 2
        + (yy - 0.5 * box_size) ** 2
        + (zz - 0.5 * box_size) ** 2
    )
    sim.state["V_ext"] = -G * M_halo / (r + 0.5 * sim.dx)  # softening

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    parser.add_argument(
        "--distributed", action="store_true", help="run in distributed mode"
    )
    parser.add_argument(
        "--emulate", action="store_true", help="emulate distributed mode on CPU"
    )
    args = parser.parse_args()

    if args.distributed:
        if args.emulate:
            flags = os.environ.get("XLA_FLAGS", "")
            flags += " --xla_force_host_platform_device_count=8"  # change to, e.g., 8 for testing sharding virtually
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["XLA_FLAGS"] = flags
            print("Using emulated distributed CPU mode")
        else:
            jax.distributed.initialize()
            if jax.process_index() == 0:
                print("Using distributed GPU mode")
        # Create mesh and sharding for distributed computation
        n_devices = jax.device_count()
        devices = mesh_utils.create_device_mesh((1, n_devices))
        mesh = Mesh(devices, axis_names=("x", "y"))
        sharding = NamedSharding(mesh, PartitionSpec("x", "y"))
        # mesh = jax.make_mesh((1, n_devices, 1), ("x", "y", "z"),axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit))
        # jax.set_mesh(mesh)
        # sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

        if jax.process_index() == 0:
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
    else:
        sharding = None

    sim = set_up_simulation(args.res, sharding)
    sim.run()
    print("mean |psi| =", jnp.mean(jnp.abs(sim.state["psi"])))

    return sim


if __name__ == "__main__":
    main()
