import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os


def plot_sim(state, checkpoint_dir, i, params):
    """Plot the simulation state."""

    dynamic_range = params["output"]["plot_dynamic_range"]
    box_size = params["domain"]["box_size"]
    aspect_ratio = params["domain"].get("aspect_ratio", 1)
    lx = aspect_ratio * box_size
    ly = box_size
    extent = [0, lx, 0, ly]

    # process distributed data
    if params["physics"]["quantum"]:
        rho_bar_dm = jnp.mean(jnp.abs(state["psi"]) ** 2)
        rho_proj_dm = jnp.log10(
            jax.experimental.multihost_utils.process_allgather(
                jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2), tiled=True
            )
        ).T
    if params["physics"]["hydro"]:
        rho_bar_gas = jnp.mean(state["rho"])
        rho_proj_gas = jnp.log10(
            jax.experimental.multihost_utils.process_allgather(
                jnp.mean(state["rho"], axis=2), tiled=True
            )
        ).T

    # create plot on process 0
    if jax.process_index() == 0:
        if params["physics"]["quantum"]:
            plt.clf()

            # DM projection
            vmin = jnp.log10(rho_bar_dm / dynamic_range)
            vmax = jnp.log10(rho_bar_dm * dynamic_range)

            ax = plt.gca()
            ax.imshow(
                rho_proj_dm,
                cmap="inferno",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if params["physics"]["particles"]:
                # draw particles
                plt.plot(
                    state["pos"][:, 0],
                    state["pos"][:, 1],
                    color="cyan",
                    marker=".",
                    linestyle="None",
                    markersize=5,
                )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"dm{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        if params["physics"]["hydro"]:
            plt.clf()

            # gas projection
            vmin = jnp.log10(rho_bar_gas / dynamic_range)
            vmax = jnp.log10(rho_bar_gas * dynamic_range)

            ax = plt.gca()
            ax.imshow(
                rho_proj_gas,
                cmap="viridis",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if params["physics"]["particles"]:
                # draw particles
                plt.plot(
                    state["pos"][:, 0],
                    state["pos"][:, 1],
                    color="red",
                    marker=".",
                    linestyle="None",
                    markersize=5,
                )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"gas{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
