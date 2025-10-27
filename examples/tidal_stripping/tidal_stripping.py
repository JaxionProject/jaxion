import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import argparse
import os
import sys

sys.path.append("../../")  # XXX
import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Tidal Stripping of Fuzzy Dark Matter soliton in an external potential

Philip Mocz (2025)

Usage:
  python tidal_stripping.py --res <resolution_multiplier>

  python tidal_stripping.py --distributed --emulate
"""


def set_up_simulation(resolution_multiplier, sharding, save):
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
            "save": save,
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
    if jax.process_index() == 0:
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
        "--save", type=bool, default=True, help="Save simulation output"
    )
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
            if jax.process_index() == 0:
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
    else:
        sharding = None

    sim = set_up_simulation(args.res, sharding, args.save)
    sim.run()
    mean_psi = jnp.mean(jnp.abs(sim.state["psi"]))
    if jax.process_index() == 0:
        print("mean |psi| =", mean_psi)

    return sim


if __name__ == "__main__":
    main()
