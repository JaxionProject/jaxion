Tutorial 3: Sponge Boundary Conditions
======================================

This tutorial describes how to add sponge boundary conditions.
By default, Jaxion uses periodic boundary conditions.
Sponge boundary conditions can be used to absorb outgoing waves at the domain boundaries.

An example of sponge boundary conditions is provided in `examples/tidal_stripping <https://github.com/JaxionProject/jaxion/tree/main/examples/tidal_stripping>`_

It is created by adding an external potential to the simulation.
We need to set ``params["physics"]["external_potential"] = True``.

.. code-block:: python

    # V_0 is the depth of the sponge potential ~ G * M_tot / box_size
    # r_N, r_p, and r_s define the sponge region
    r_N = 0.5 * box_size
    r_p = (7 / 8) * r_N
    r_s = 0.5 * (r_N + r_p)
    delta = r_N - r_p
    V_sponge = (
        -0.5j
        * V_0
        * (2 + jnp.tanh((R - r_s) / delta) - jnp.tanh(r_s / delta))
        * jnp.heaviside(R - r_p, 0.0)
    )
    sim.state["V_ext"] = V_sponge

Since the sponge potential is imaginary, it will absorb outgoing waves in the sponge region.
