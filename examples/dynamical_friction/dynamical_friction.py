import jax.numpy as jnp
import sys  # XXX

sys.path.append("../../")  # XXX
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Dynamical of Fuzzy Dark Matter soliton in an external potential

Philip Mocz (2025)

Usage:
  python dynamical_friction.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "external_potential": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
        },
        "time": {
            "end": 0.1,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
            "plot_dynamic_range": 5.0,
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions (uniform density w/ relative motion, in external potential)
    density = 1.0e6  # dm density M_sun/kpc^3
    k_rel = 5.0  # wavenumber of relative motion
    G = jaxion.constants["gravitational_constant"]
    box_size = sim.params["domain"]["box_size"]

    X, Y, Z = sim.grid
    sim.state["psi"] = jnp.sqrt(density) * jnp.exp(
        1.0j * (2.0 * jnp.pi / box_size) * k_rel * X
    )

    # add external potential (host halo)
    M_halo = 1e9  # M_sun
    print(f"M_halo: {M_halo:.2e} M_sun")

    r = jnp.sqrt(
        (X - 0.5 * box_size) ** 2
        + (Y - 0.5 * box_size) ** 2
        + (Z - 0.5 * box_size) ** 2
    )
    sim.state["V_ext"] = -G * M_halo / (r + 0.5 * sim.dx)  # softening

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    args = parser.parse_args()

    sim = set_up_simulation(args.res)
    sim.run()
    print("mean |psi| =", jnp.mean(jnp.abs(sim.state["psi"])))

    return sim


if __name__ == "__main__":
    main()
