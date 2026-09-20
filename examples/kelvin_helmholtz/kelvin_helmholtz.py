#!/usr/bin/env python3
import argparse

import jax.numpy as jnp

import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Kelvin-Helmholtz instability of an ideal gas (Euler equations)

Philip Mocz (2026)

Usage:
  python kelvin_helmholtz.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier, save):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "quantum": False,  # pure hydrodynamics test
            "gravity": False,
            "hydro": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
            "box_size": 1.0,  # kpc
        },
        "time": {
            "end": 2.0,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
            "save": save,
            "plot_dynamic_range": 2.0,
        },
        "hydro": {
            "eos": "ideal",
            "gamma": 5.0 / 3.0,
            "sound_speed": 2.6,  # km/s (max signal speed |v| + c_s; sets dt)
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions
    # (opposite moving streams with a perturbation, uniform along z)
    box_size = sim.box_size
    gamma = sim.adiabatic_index
    xx, yy, _zz = sim.grid

    w0 = 0.1  # amplitude of the seed perturbation
    sigma = 0.05 / jnp.sqrt(2.0)  # width of the seed perturbation (kpc)
    rho_0 = 1.0  # density of the outer stream (M_sun / kpc^3)
    v_0 = 0.5  # speed of each stream (km/s)
    P_0 = 2.5  # (uniform) pressure (M_sun / kpc^3 * (km/s)^2)

    stream = jnp.abs(yy - 0.5 * box_size) < 0.25 * box_size

    rho = rho_0 * (1.0 + stream)
    vx = v_0 * (-1.0 + 2.0 * stream)
    vy = (
        w0
        * jnp.sin(4.0 * jnp.pi * xx / box_size)
        * (
            jnp.exp(-((yy - 0.25 * box_size) ** 2) / (2.0 * sigma**2))
            + jnp.exp(-((yy - 0.75 * box_size) ** 2) / (2.0 * sigma**2))
        )
    )
    vz = jnp.zeros(sim.shape)
    P = P_0 * jnp.ones(sim.shape)

    # check that the shear between the two streams is subsonic
    c_sound = jnp.sqrt(gamma * P_0 / rho_0)
    assert 2.0 * v_0 < c_sound

    sim.state["rho"] = rho
    sim.state["vx"] = vx
    sim.state["vy"] = vy
    sim.state["vz"] = vz
    sim.state["P"] = P

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    parser.add_argument(
        "--save", type=bool, default=True, help="Save simulation output"
    )
    args = parser.parse_args()

    sim = set_up_simulation(args.res, args.save)
    sim.run()
    print("min rho =", jnp.min(sim.state["rho"]))
    print("max rho =", jnp.max(sim.state["rho"]))
    print("mean |vy| =", jnp.mean(jnp.abs(sim.state["vy"])))
    print("mean P =", jnp.mean(sim.state["P"]))

    return sim


if __name__ == "__main__":
    main()
