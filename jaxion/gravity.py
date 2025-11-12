import jax.numpy as jnp
import jaxdecomp as jd

# Pure functions for gravity calculations


def calculate_gravitational_potential(rho, k_sq, G, rho_bar):
    V_hat = -jd.fft.pfft3d(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jd.fft.pifft3d(V_hat))
    return V
