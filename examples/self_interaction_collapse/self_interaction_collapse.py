import jax
import jax.numpy as jnp
import argparse
import sys

sys.path.append("../../")  # XXX
import jaxion

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Collapse a soliton due to attractive self-interaction

Philip Mocz (2025)

Usage:
  python self_interaction_collapse.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "domain": {
            "resolution_multiplier": resolution_multiplier,
            "box_size": 4.0,
        },
        "time": {
            "end": 0.1,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
            "save": True,
        },
        "quantum": {
            "m_22": 1.0,  # axion mass in 10^-22 eV
            "f_15": -0.8,  # decay constant in 10^15 GeV
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions (randomly placed solitons)
    m_22 = sim.params["quantum"]["m_22"]
    hbar = jaxion.constants["reduced_planck_constant"]
    G = jaxion.constants["gravitational_constant"]
    m = sim.axion_mass
    a_s = sim.scattering_length
    box_size = sim.params["domain"]["box_size"]
    xx, yy, zz = sim.grid

    print("a_s =", a_s)
    M_soliton = 1.0e9
    r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc

    M_crit = 1.012 * hbar / jnp.sqrt(G * m * jnp.abs(a_s))
    print("M_soliton/M_crit =", M_soliton / M_crit)

    def rho_soliton(r, r_soliton, m_22):
        return (
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )

    r = jnp.sqrt(
        (xx - 0.5 * box_size) ** 2
        + (yy - 0.5 * box_size) ** 2
        + (zz - 0.5 * box_size) ** 2
    )
    rho = rho_soliton(r, r_soliton, m_22)

    sim.state["psi"] = jnp.array(jnp.sqrt(rho)) + 0j

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    args = parser.parse_args()

    sim = set_up_simulation(args.res)
    sim.run()

    return sim


if __name__ == "__main__":
    main()
