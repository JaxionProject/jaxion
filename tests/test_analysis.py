import jax.numpy as jnp
import jaxion
from jaxion.quantum import quantum_velocity
from jaxion.analysis import radial_power_spectrum
import pytest


def test_quantum_velocity_and_radial_power_spectrum():
    sim = jaxion.Simulation({})
    box_size = sim.box_size
    m_per_hbar = sim.m_per_hbar
    xx, yy, _ = sim.grid
    kx, ky, kz = sim.kgrid

    psi = (
        jnp.cos(2.0 * jnp.pi * xx / box_size) ** 2
        + jnp.cos(2.0 * jnp.pi * yy / box_size) * 1j
    )

    vx, _, _ = quantum_velocity(psi, box_size, m_per_hbar)

    Pf, _, _ = radial_power_spectrum(vx, kx, ky, kz, box_size)

    assert jnp.max(Pf) == pytest.approx(8244.606, rel=1e-4)
