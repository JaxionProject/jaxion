import jax
import jax.numpy as jnp
import pytest

from jaxion.hydro import get_conserved_ideal, get_primitive_ideal, hydro_fluxes_ideal


def test_ideal_eos_round_trips_primitive_variables():
    gamma = 5.0 / 3.0
    vol = 0.5**3
    rho = jnp.array([[[1.0, 2.0]]])
    vx = jnp.array([[[0.3, -0.7]]])
    vy = jnp.array([[[-0.2, 0.4]]])
    vz = jnp.array([[[0.1, 0.9]]])
    P = jnp.array([[[1.5, 2.5]]])

    conserved = get_conserved_ideal(rho, vx, vy, vz, P, gamma, vol)
    primitive = get_primitive_ideal(*conserved, gamma, vol)

    for got, expected in zip(primitive, (rho, vx, vy, vz, P)):
        assert jnp.allclose(got, expected)


def test_ideal_eos_fluxes_conserve_mass_momentum_and_energy():
    n = 8
    dx = 1.0 / n
    gamma = 5.0 / 3.0
    dt = 0.1 * dx

    keys = jax.random.split(jax.random.PRNGKey(17), 5)
    rho = 1.0 + 0.2 * jax.random.uniform(keys[0], (n, n, n))
    vx = 0.1 * jax.random.normal(keys[1], (n, n, n))
    vy = 0.1 * jax.random.normal(keys[2], (n, n, n))
    vz = 0.1 * jax.random.normal(keys[3], (n, n, n))
    P = 1.0 + 0.2 * jax.random.uniform(keys[4], (n, n, n))

    conserved_start = get_conserved_ideal(rho, vx, vy, vz, P, gamma, dx**3)

    for _ in range(10):
        rho, vx, vy, vz, P = hydro_fluxes_ideal(rho, vx, vy, vz, P, dt, dx, gamma)

    conserved_end = get_conserved_ideal(rho, vx, vy, vz, P, gamma, dx**3)

    # the periodic finite volume scheme is conservative by construction
    # (momenta sum to nearly zero, so measure the drift against |q| instead)
    for start, end in zip(conserved_start, conserved_end):
        scale = float(jnp.sum(jnp.abs(start)))
        assert float(jnp.sum(end)) == pytest.approx(
            float(jnp.sum(start)), abs=1.0e-5 * scale
        )


def test_ideal_eos_advects_a_sound_wave_at_the_sound_speed():
    n = 64
    dx = 1.0 / n
    gamma = 5.0 / 3.0
    rho_0 = 1.0
    P_0 = 1.0 / gamma  # so that c_sound == 1
    amplitude = 1.0e-3

    c_sound = jnp.sqrt(gamma * P_0 / rho_0)
    x = ((jnp.arange(n) + 0.5) * dx)[:, None, None] * jnp.ones((n, 4, 4))

    def wave(x):
        return jnp.sin(2.0 * jnp.pi * x)

    rho = rho_0 * (1.0 + amplitude * wave(x))
    vx = c_sound * amplitude * wave(x)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    P = P_0 * (1.0 + gamma * amplitude * wave(x))

    # advance the wave by exactly one period
    dt = 0.2 * dx / c_sound
    nt = round(float(1.0 / c_sound / dt))
    for _ in range(nt):
        rho, vx, vy, vz, P = hydro_fluxes_ideal(rho, vx, vy, vz, P, dt, dx, gamma)

    # the wave returns to its starting phase, damped by numerical diffusion
    error = jnp.max(jnp.abs(rho - rho_0 * (1.0 + amplitude * wave(x)))) / amplitude
    assert float(error) < 0.1
