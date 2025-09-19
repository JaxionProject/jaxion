import jax.numpy as jnp

# Pure functions for quantum simulation


def quantum_kick(psi, V, m_per_hbar, dt):
    psi = jnp.exp(-1.0j * m_per_hbar * dt * V) * psi
    return psi


def quantum_drift(psi, k_sq, m_per_hbar, dt):
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)
    return psi
