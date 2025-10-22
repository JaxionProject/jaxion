import jax.numpy as jnp
import numpy as np
import jaxdecomp as jd
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Soliton formation in the kinetic regime

Philip Mocz (2025)

Usage:
  python kinetic_condensation.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "domain": {
            "resolution_multiplier": resolution_multiplier,
            "box_size": 6.0,  # kpc
        },
        "time": {
            "end": 10.0,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
            "plot_dynamic_range": 4.0,
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # average density of dark matter in the simulation (in units of Msun / kpc^3)
    rho_bar = 1.0e8

    # dark matter
    sigma = 100.0  # velocity dispersion of dm

    m = sim.axion_mass
    hbar = jaxion.constants["reduced_planck_constant"]
    m_per_hbar = m / hbar

    kx, ky, kz = sim.kgrid
    k_sq = kx**2 + ky**2 + kz**2

    nx = sim.resolution
    box_size = sim.box_size
    G = jaxion.constants["gravitational_constant"]

    # check that de broglie wavelength fits into box
    de_broglie_wavelength = hbar / (m * sigma)
    n_wavelengths = box_size / de_broglie_wavelength
    assert n_wavelengths > 1

    # check timescales
    lambda_fac = np.log(m * sigma * box_size / hbar)
    b = 1.0
    n = rho_bar / m
    # eqn 4 of Levkov
    tau_gr = (
        b
        * np.sqrt(2.0)
        / (12.0 * np.pi**3)
        * m
        * sigma**6
        / (G**2 * n**2 * lambda_fac)
        / hbar**3
    )
    # we want to see condensation happen in the simulation time
    assert 10.0 * tau_gr < sim.params["time"]["end"]

    kin1 = m * sigma * box_size / hbar  # box crossing time >> 1
    kin2 = m * sigma**2 * tau_gr / hbar  # condensation time  >> 1   eqn 1 of Levkov
    assert kin1 > 1
    assert kin2 > 1

    # check Jeans lengths
    jeans_length = sigma * np.sqrt(np.pi / (G * rho_bar))  # kinetic Jeans
    # quantum Jeans, eqn 40 of Levkov
    jeans_length_Q = np.pi / ((np.pi * G * rho_bar) ** 0.25 * m**0.5 / np.sqrt(hbar))

    assert jeans_length > box_size
    assert jeans_length_Q < box_size

    # dark matter
    # construct in fourier space according to Eq (27) of our paper [https://arxiv.org/abs/1801.03507]
    np.random.seed(17)
    # initialize random phases
    # (use order to set lowest k modes first)
    sid = np.argsort(k_sq.flatten(), stable=True)
    psi = np.zeros((nx**3,), dtype=complex)
    psi[sid] = np.exp(1.0j * 2.0 * np.pi * np.random.rand(nx**3))
    psi = psi.reshape(k_sq.shape)
    psi = jnp.array(psi)
    psi *= np.sqrt(np.exp(-k_sq / (2.0 * sigma**2 * m_per_hbar**2)))
    psi = jd.fft.pifft3d(psi)
    # re-normalize it
    psi *= jnp.sqrt(rho_bar / jnp.mean(jnp.abs(psi) ** 2))

    sim.state["psi"] = psi

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
