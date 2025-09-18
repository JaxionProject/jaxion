import jax
import jax.numpy as jnp
import sys  # XXX TODO: remove

sys.path.append("../../")  # XXX TODO: remove
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Tidal Stripping of Fuzzy Dark Matter soliton

Philip Mocz (2025)

Usage:
  python tidal_stripping.py --res <resolution_multiplier>
"""


def setup_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "external_potential": True,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions
    M_soliton = 1.0e9  # mass of soliton (Msun)
    k_soliton = 4.0  # wave-number for orbital motion of soliton
    r_separation = 2.0  # separation of soliton from center (kpc)
    m22 = sim.params["quantum"]["m22"]

    # XXX

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    args = parser.parse_args()

    sim = setup_simulation(args.res)
    sim.run()


if __name__ == "__main__":
    main()
