import jax.numpy as jnp
import sys  # XXX

sys.path.append("../../")  # XXX
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Tidal Stripping of Fuzzy Dark Matter soliton in an external potential

Philip Mocz (2025)

Usage:
  python tidal_stripping.py --res <resolution_multiplier>
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
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

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

    X, Y, Z = sim.grid
    r = jnp.sqrt(
        (X - 0.5 * box_size) ** 2
        + (Y - 0.5 * box_size - r_separation) ** 2
        + (Z - 0.5 * box_size) ** 2
    )
    sim.state["psi"] = (
        jnp.sqrt(
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )
        + 0.0j
    )
    # add circular orbit velocity
    sim.state["psi"] *= jnp.exp(1.0j * k_soliton * X)

    # add external potential (host halo)
    M_halo = 0.25 * box_size * k_soliton**2 * hbar**2 / (G * m**2)
    print(f"M_halo: {M_halo:.2e} M_sun")
    assert M_halo > M_soliton * 2.0  # halo should be much more massive than soliton

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
