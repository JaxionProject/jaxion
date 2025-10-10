import jax.numpy as jnp
import numpy as np
import sys  # XXX

sys.path.append("../../")  # XXX
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Heating Stars due to Fuzzy Dark Matter fluctuations

Philip Mocz (2025)

Usage:
  python heating_stars.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # average density of all matter (dm+stars) in the simulation (in units of Msun / kpc^3)
    rho_bar = 1.0e7

    # stars
    frac_stars = 0.15  # fraction of total mass in stars
    rho_stars = frac_stars * rho_bar  # average density of stars
    sigma_stars = 20.0  # velocity dispersion (1d) of stars (km/s)

    box_size = 4.0  # kpc

    n_stars = 400
    m_stars = rho_stars * box_size**3 / n_stars  # Msun

    # Parameters added/changed from default values
    params = {
        "physics": {
            "particles": True,
        },
        "domain": {
            "resolution_multiplier": resolution_multiplier,
            "box_size": box_size,
        },
        "time": {
            "end": 0.4,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
            "plot_dynamic_range": 2.0,
        },
        "particles": {"num_particles": n_stars, "particle_mass": m_stars},
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # dark matter
    frac_dm = 1.0 - frac_stars  # fraction of total mass in dark matter
    sigma = 40.0  # velocity dispersion of dm (km/s)

    m = sim.axion_mass
    hbar = jaxion.constants["reduced_planck_constant"]
    m_per_hbar = m / hbar

    kx, ky, kz = sim.kgrid
    k_sq = kx**2 + ky**2 + kz**2

    nx = sim.resolution
    G = jaxion.constants["gravitational_constant"]

    # check that de broglie wavelength fits into box
    de_broglie_wavelength = hbar / (m * sigma)
    n_wavelengths = box_size / de_broglie_wavelength
    assert n_wavelengths > 1

    # check the Jeans length
    jeans_length = sigma_stars * jnp.sqrt(jnp.pi / (G * rho_stars))
    n_jeans = box_size / jeans_length
    assert n_jeans < 0.5

    # dark matter
    # construct in fourier space according to Eq (27) of our paper [https://arxiv.org/abs/1801.03507]
    np.random.seed(17)
    # initialize random phases
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

    # stars are initially uniform
    np.random.seed(17)
    pos = np.random.rand(n_stars, 3) * box_size
    vel = np.random.randn(n_stars, 3) * sigma_stars
    pos = jnp.array(pos)
    vel = jnp.array(vel)

    sim.state["psi"] = psi
    sim.state["pos"] = pos
    sim.state["vel"] = vel

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1, help="Resolution multiplier")
    args = parser.parse_args()

    sim = set_up_simulation(args.res)
    sim.run()
    print("mean |psi| =", jnp.mean(jnp.abs(sim.state["psi"])))
    print("mean |vx| =", jnp.mean(jnp.abs(sim.state["vel"][:, 0])))
    print("mean |vy| =", jnp.mean(jnp.abs(sim.state["vel"][:, 1])))
    print("mean |vz| =", jnp.mean(jnp.abs(sim.state["vel"][:, 2])))
    print(
        "std vel =",
        jnp.sqrt(
            jnp.std(sim.state["vel"][:, 0]) ** 2
            + jnp.std(sim.state["vel"][:, 1]) ** 2
            + jnp.std(sim.state["vel"][:, 2]) ** 2
        ),
    )

    return sim


if __name__ == "__main__":
    main()
