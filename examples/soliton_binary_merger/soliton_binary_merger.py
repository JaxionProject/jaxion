import jax.numpy as jnp
import numpy as np
import jaxion
import argparse

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Collide two solitons with opposite phases

Philip Mocz (2025)

Usage:
  python soliton_binary_merger.py --res <resolution_multiplier>
"""


def set_up_simulation(resolution_multiplier):
    # Parameters added/changed from default values
    params = {
        "domain": {
            "resolution_multiplier": resolution_multiplier,
        },
        "output": {
            "path": f"./checkpoints{resolution_multiplier}/",
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # Set initial conditions (randomly places solitons)
    m_22 = sim.params["quantum"]["m_22"]
    box_size = sim.params["domain"]["box_size"]
    nx = sim.resolution
    X, Y, Z = sim.grid

    def rho_soliton(r, r_soliton, m_22):
        return (
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )

    rho = np.zeros((nx, nx, nx), dtype=complex)
    r_soliton = 0.02 * box_size
    for i in range(2):
        x_soliton = (0.5 + 0.3 * (i - 0.5)) * box_size
        y_soliton = 0.5 * box_size
        z_soliton = 0.5 * box_size

        r = jnp.sqrt((X - x_soliton) ** 2 + (Y - y_soliton) ** 2 + (Z - z_soliton) ** 2)
        rho += rho_soliton(r, r_soliton, m_22)

    half = (X - 0.5 * box_size) > 0
    sim.state["psi"] = jnp.array(jnp.sqrt(rho)) * (1.0 * half + -1.0 * (1.0 - half))

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
