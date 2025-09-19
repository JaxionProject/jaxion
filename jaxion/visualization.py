import jax.numpy as jnp
import matplotlib.pyplot as plt
import os


def plot_sim(state, checkpoint_dir, i, vmin, vmax):
    """Plot the simulation state."""

    plt.clf()

    # DM projection
    rho_proj_dm = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2)).T
    ax = plt.gca()
    ax.imshow(
        rho_proj_dm,
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig(
        os.path.join(checkpoint_dir, f"snap{i:03d}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
