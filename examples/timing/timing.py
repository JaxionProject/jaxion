import jax.numpy as jnp
import numpy as np
import sys  # XXX

sys.path.append("../../")  # XXX
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Timing test based on heating_gas/

Philip Mocz (2025)

Usage:
  python timing.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "hydro": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
            "box_size": 4.0,  # kpc
        },
        "time": {
            "end": 0.01 * (1.0 / resolution_multiplier**2),
        },
        "output": {
            "num_checkpoints": 100,
            "save": False,
        },
        "hydro": {
            "sound_speed": 20.0,  # km/s
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # average density of all matter (dm+gas) in the simulation (in units of Msun / kpc^3)
    rho_bar = 1.0e7

    # gas
    frac_gas = 0.15  # fraction of total mass in gas
    rho_gas = frac_gas * rho_bar  # average density of gas
    c_sound = sim.sound_speed  # sound speed (km/s)

    # dark matter
    frac_dm = 1.0 - frac_gas  # fraction of total mass in dark matter
    sigma = 40.0  # velocity dispersion of dm

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

    # check the Jeans length
    jeans_length = c_sound * jnp.sqrt(jnp.pi / (G * rho_gas))
    n_jeans = box_size / jeans_length
    assert n_jeans < 0.5

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
    psi = np.fft.ifftn(psi)
    # re-normalize it
    psi *= jnp.sqrt(frac_dm * rho_bar / jnp.mean(jnp.abs(psi) ** 2))

    # gas is initially uniform
    rho = jnp.ones((nx, nx, nx)) * rho_gas
    vx = jnp.zeros((nx, nx, nx))
    vy = jnp.zeros((nx, nx, nx))
    vz = jnp.zeros((nx, nx, nx))

    sim.state["psi"] = psi
    sim.state["rho"] = rho
    sim.state["vx"] = vx
    sim.state["vy"] = vy
    sim.state["vz"] = vz

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
