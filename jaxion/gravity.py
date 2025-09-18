import jax.numpy as jnp

# Pure functions for gravity calculations


def calculate_gravitational_potential(rho, k_sq, G, rho_bar):
    Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(Vhat))
    return V
