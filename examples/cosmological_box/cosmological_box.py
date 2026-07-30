import h5py
import jax.numpy as jnp

import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Fuzzy Dark Matter cosmological box

Philip Mocz (2025)

Usage:
  python cosmological_box.py
"""


def set_up_simulation():
    # Parameters added/changed from default values
    params = {
        "physics": {
            "cosmology": True,
        },
        "domain": {
            "box_size": 1000.0,  # in h^-1 kpc
            "resolution_base": 256,
        },
        "time": {
            "start": 127.0,  # start at z=127
            "end": 7.0,  # end at z=7
        },
        "output": {
            "path": "./checkpoints/",
        },
        "quantum": {
            "m_22": 2.5,  # m = 2.5e-22 eV
        },
        "cosmology": {
            "omega_matter": 0.3,
            "omega_lambda": 0.7,
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions from precomputed cosmological ICs
    # read in hdf5 file (which is in units of 1e10 Msun/kpc^2)
    with h5py.File("fdm_1mpc_256_m2.5e-22_z127_ic.hdf5", "r") as f:
        psi = jnp.array(f["psiRe"]) + 1.0j * jnp.array(f["psiIm"])
        psi *= 1e5

    sim.state["psi"] = psi

    return sim


def main():
    sim = set_up_simulation()
    sim.run()

    return sim


if __name__ == "__main__":
    main()
