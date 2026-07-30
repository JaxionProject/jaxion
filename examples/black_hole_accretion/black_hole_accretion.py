#!/usr/bin/env python3
import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Accrete gas onto a black hole particle

Philip Mocz (2025)

Usage:
  python black_hole_accretion.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    M_bh = 1.0e7
    params = {
        "physics": {
            "quantum": True,
            "gravity": True,
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
        "particles": {"num_particles": 1, "particle_mass": M_bh, "accrete_gas": True},
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # these are the fields we evolve:
    print(sim.state.keys())

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

    # add callback to record info about state
    n_buffer = sim.nt + 1
    sim.state["tt"] = jnp.full((n_buffer,), jnp.nan)
    sim.state["m_bh"] = jnp.full((n_buffer,), jnp.nan)
    sim.state["tt"] = sim.state["tt"].at[0].set(0.0)
    sim.state["m_bh"] = sim.state["m_bh"].at[0].set(M_bh)
    sim.callback = callback

    return sim


def callback(i, state):
    # record the black hole mass at end of timestep i
    state["tt"] = state["tt"].at[i + 1].set(state["t"])
    state["m_bh"] = state["m_bh"].at[i + 1].set(state["mass"][0])
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    args = parser.parse_args()

    sim = set_up_simulation(args.res)
    rho_gas0 = jnp.mean(sim.state["rho"])
    sound_speed = sim.params["hydro"]["sound_speed"]
    t_span = sim.params["time"]["end"] - sim.params["time"]["start"]

    sim.run()

    M_init = sim.params["particles"]["particle_mass"]
    G = jaxion.constants["gravitational_constant"]
    lam = jnp.exp(1.5) / 4.0
    dM_dt = 4.0 * jnp.pi * lam * (G * M_init) ** 2 / sound_speed**3 * rho_gas0
    M_estimate = M_init + t_span * dM_dt
    M_final = sim.state["mass"][0]
    print("BH mass baseline accretion:", M_estimate)
    print("Final BH mass:", M_final)

    # check mass accreted approximately matches gas lost:
    box_size = sim.params["domain"]["box_size"]
    dm_bh = M_final - M_init
    dm_gas = (jnp.mean(sim.state["rho"]) - rho_gas0) * box_size**3
    print("mass gained:", dm_bh)
    print("mass lost:", dm_gas)

    # Make a plot of black hole mass vs time:
    plt.plot(sim.state["tt"], sim.state["m_bh"])
    plt.xlabel("time [kpc/(km/s)]")
    plt.ylabel(r"mass [$M_\odot$]")
    plt.yscale("log")
    plt.savefig("callback.png")
    plt.close()

    return sim


if __name__ == "__main__":
    main()
