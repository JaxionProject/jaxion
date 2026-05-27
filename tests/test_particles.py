import jax.numpy as jnp
import pytest

from jaxion.particles import bin_particles, particles_drift


def test_bin_particles_rectangular_shape_conserves_mass():
    dx = 1.0
    shape = (8, 4, 4)
    pos = jnp.array([[1.5, 1.5, 1.5], [6.5, 3.5, 3.5]])
    masses = jnp.array([2.0, 3.0])

    rho = bin_particles(pos, masses, dx, shape, multiple_masses=True)

    assert rho.shape == shape
    assert float(jnp.sum(rho) * dx**3) == pytest.approx(float(jnp.sum(masses)))


def test_particles_drift_wraps_rectangular_domain():
    pos = jnp.array([[7.5, 3.5, 3.5], [0.25, 0.25, 0.25]])
    vel = jnp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    domain_size = jnp.array([8.0, 4.0, 4.0])

    pos = particles_drift(pos, vel, dt=1.0, domain_size=domain_size)

    expected = jnp.array([[0.5, 0.5, 0.5], [7.25, 3.25, 3.25]])
    assert jnp.allclose(pos, expected)
