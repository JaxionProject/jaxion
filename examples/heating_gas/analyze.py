import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Plot power spectrum of Heating Gas example
Philip Mocz (2025)

Usage:
  python analyze.py
"""


def main():
    # load the simulation
    sim = jaxion.Simulation("checkpoints2/")

    # plot power spectrum of dm and gas density at final time
    rho_dm = jnp.abs(sim.state["psi"]) ** 2
    rho_gas = sim.state["rho"]
    box_size = sim.box_size
    kx, ky, kz = sim.kgrid

    Pf_dm, k, _ = jaxion.radial_power_spectrum(rho_dm, kx, ky, kz, box_size)
    Pf_gas, k, _ = jaxion.radial_power_spectrum(rho_gas, kx, ky, kz, box_size)

    plt.figure()
    plt.loglog(k, Pf_dm, label="dm")
    plt.loglog(k, Pf_gas, label="gas")
    # k^-{5/3} reference line
    k_ref = jnp.array([k[1], k[-1]])
    P_ref = 1e14 * k_ref ** (-5.0 / 3.0)
    plt.loglog(k_ref, P_ref, "k--", label=r"$k^{-5/3}$")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.legend()
    plt.ylim(1e10, 2e15)
    plt.savefig("power_spectrum.png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
