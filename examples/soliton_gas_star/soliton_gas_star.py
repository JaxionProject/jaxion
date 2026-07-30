import argparse

import jax.numpy as jnp

import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Explore the dynamics of a soliton, gas, and single star

Philip Mocz (2025)

Usage:
  python soliton_gas_star.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "hydro": True,
            "particles": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
        },
        "hydro": {
            "sound_speed": 40.0,  # km/s
        },
        "particles": {"num_particles": 1, "particle_mass": 1.0e7},
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions
    M_soliton = 1.0e9  # mass of soliton (M_sun)
    m_22 = sim.params["quantum"]["m_22"]
    r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc
    box_size = sim.params["domain"]["box_size"]
    nx = sim.resolution

    xx, yy, zz = sim.grid
    r = jnp.sqrt(
        (xx - 0.5 * box_size) ** 2
        + (yy - 0.5 * box_size) ** 2
        + (zz - 0.5 * box_size) ** 2
    )
    sim.state["psi"] = (
        jnp.sqrt(
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )
        + 0.0j
    )

    # add gas
    frac_gas = 0.1
    rho_gas = frac_gas * M_soliton / box_size**3
    sim.state["rho"] = jnp.ones((nx, nx, nx)) * rho_gas
    sim.state["vx"] = jnp.zeros((nx, nx, nx))
    sim.state["vy"] = jnp.zeros((nx, nx, nx))
    sim.state["vz"] = jnp.zeros((nx, nx, nx))

    # add star
    pos = jnp.array([[0.6 * box_size, 0.5 * box_size, 0.5 * box_size]])
    vel = jnp.array([[0.0, 40.0, 0.0]])

    sim.state["pos"] = pos
    sim.state["vel"] = vel

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
