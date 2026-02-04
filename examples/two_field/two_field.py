import jax
import jax.numpy as jnp
import numpy as np
import jaxdecomp as jd
import argparse
import jaxion

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Merge solitons to form an idealized Two-Field Fuzzy Dark Matter halo
Demonstrates adding custom fields

Philip Mocz (2025)

Usage:
  python two_field.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "quantum": False,  # use custom fields instead for psi1, psi2
            "gravity": True,
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

    # Set initial conditions (randomly placed solitons)
    m_22_1 = 1.0
    m_22_2 = 2.0
    hbar = jaxion.constants["reduced_planck_constant"]
    eV = jaxion.constants["electron_volt"]
    c = jaxion.constants["speed_of_light"]
    m1 = m_22_1 * 1.0e-22 * eV / c**2
    m2 = m_22_2 * 1.0e-22 * eV / c**2
    m1_per_hbar = m1 / hbar
    m2_per_hbar = m2 / hbar

    box_size = sim.params["domain"]["box_size"]
    nx = sim.resolution
    xx, yy, zz = sim.grid

    def rho_soliton(r, r_soliton, m_22):
        return (
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )

    np.random.seed(17)
    n_solitons = 8

    rho1 = np.zeros((nx, nx, nx), dtype=complex)
    rho2 = np.zeros((nx, nx, nx), dtype=complex)
    for _ in range(n_solitons):
        r_soliton = (0.05 + 0.03 * np.random.rand()) * box_size
        x_soliton = (0.25 + 0.5 * np.random.rand()) * box_size
        y_soliton = (0.25 + 0.5 * np.random.rand()) * box_size
        z_soliton = (0.25 + 0.5 * np.random.rand()) * box_size

        r = jnp.sqrt(
            (xx - x_soliton) ** 2 + (yy - y_soliton) ** 2 + (zz - z_soliton) ** 2
        )
        rho1 += rho_soliton(r, r_soliton, m_22_1)
        rho2 += rho_soliton(r, r_soliton, m_22_2)

    sim.state["psi1"] = jnp.array(jnp.sqrt(rho1))
    sim.state["psi2"] = jnp.array(jnp.sqrt(rho2))

    # Define custom functions
    def custom_density(state):
        return jnp.abs(state["psi1"]) ** 2 + jnp.abs(state["psi2"]) ** 2

    def custom_kick(state, V, dt):
        state["psi1"] = jnp.exp(-1j * m1_per_hbar * dt * V) * state["psi1"]
        state["psi2"] = jnp.exp(-1j * m2_per_hbar * dt * V) * state["psi2"]

        return state

    def custom_drift(state, k_sq, dt):
        psi1_hat = jd.fft.pfft3d(state["psi1"])
        psi1_hat = jnp.exp(dt * (-1.0j * k_sq / m1_per_hbar / 2.0)) * psi1_hat
        state["psi1"] = jd.fft.pifft3d(psi1_hat)

        psi2_hat = jd.fft.pfft3d(state["psi2"])
        psi2_hat = jnp.exp(dt * (-1.0j * k_sq / m2_per_hbar / 2.0)) * psi2_hat
        state["psi2"] = jd.fft.pifft3d(psi2_hat)

        return state

    def custom_plot(state, checkpoint_dir, i, params):
        import matplotlib.pyplot as plt
        import os

        dynamic_range = params["output"]["plot_dynamic_range"]

        # process distributed data
        nx = state["psi1"].shape[0]
        rho_bar_1 = jnp.mean(jnp.abs(state["psi1"]) ** 2)
        rho_proj_1 = jnp.log10(
            jax.experimental.multihost_utils.process_allgather(
                jnp.mean(jnp.abs(state["psi1"]) ** 2, axis=2), tiled=True
            )
        ).T
        rho_bar_2 = jnp.mean(jnp.abs(state["psi2"]) ** 2)
        rho_proj_2 = jnp.log10(
            jax.experimental.multihost_utils.process_allgather(
                jnp.mean(jnp.abs(state["psi2"]) ** 2, axis=2), tiled=True
            )
        ).T

        # create plot on process 0
        if jax.process_index() == 0:
            plt.clf()

            # Field 1 projection
            vmin1 = jnp.log10(rho_bar_1 / dynamic_range)
            vmax1 = jnp.log10(rho_bar_1 * dynamic_range)

            ax = plt.gca()
            ax.imshow(
                rho_proj_1,
                cmap="inferno",
                origin="lower",
                vmin=vmin1,
                vmax=vmax1,
                extent=[0, nx, 0, nx],
            )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"rho1_{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # Field 2 projection
            vmin2 = jnp.log10(rho_bar_2 / dynamic_range)
            vmax2 = jnp.log10(rho_bar_2 * dynamic_range)
            plt.clf()

            ax = plt.gca()
            ax.imshow(
                rho_proj_2,
                cmap="inferno",
                origin="lower",
                vmin=vmin2,
                vmax=vmax2,
                extent=[0, nx, 0, nx],
            )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"rho2_{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    sim.custom_density = custom_density
    sim.custom_plot = custom_plot
    sim.custom_kick = custom_kick
    sim.custom_drift = custom_drift

    print("rho_bar:", sim.rho_bar)

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
