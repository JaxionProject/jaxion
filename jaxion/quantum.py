import jax.numpy as jnp
import jaxdecomp as jd

# Pure functions for quantum simulation


def quantum_kick(psi, V, m_per_hbar, dt):
    psi = jnp.exp(-1.0j * m_per_hbar * dt * V) * psi
    return psi


def quantum_drift(psi, k_sq, m_per_hbar, dt):
    psi_hat = jd.fft.pfft3d(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jd.fft.pifft3d(psi_hat)
    return psi


def get_gradient(psi, kx, ky, kz):
    """
    Computes gradient of wavefunction psi
    psi: jnp.ndarray (3D)
    Returns: grad_psi_x, grad_psi_y, grad_psi_z
    """
    psi_hat = jd.fft.pfft3d(psi)
    grad_psi_x_hat = 1.0j * kx * psi_hat
    grad_psi_y_hat = 1.0j * ky * psi_hat
    grad_psi_z_hat = 1.0j * kz * psi_hat

    grad_psi_x = jd.fft.pifft3d(grad_psi_x_hat)
    grad_psi_y = jd.fft.pifft3d(grad_psi_y_hat)
    grad_psi_z = jd.fft.pifft3d(grad_psi_z_hat)

    return grad_psi_x, grad_psi_y, grad_psi_z


def quantum_velocity(psi, box_size, m_per_hbar):
    """
    Compute the velocity from the wave-function
    v = nabla S / m
    psi = sqrt(rho) exp(i S / hbar)
    """
    N = psi.shape[0]
    dx = box_size / N

    S_per_hbar = jnp.angle(psi)

    # Central differences with phase unwrapping
    # vx = jnp.roll(S_per_hbar, -1, axis=0) - jnp.roll(S_per_hbar, 1, axis=0)
    # vy = jnp.roll(S_per_hbar, -1, axis=1) - jnp.roll(S_per_hbar, 1, axis=1)
    # vz = jnp.roll(S_per_hbar, -1, axis=2) - jnp.roll(S_per_hbar, 1, axis=2)
    # vx = jnp.where(vx > jnp.pi, vx - 2 * jnp.pi, vx)
    # vx = jnp.where(vx <= -jnp.pi, vx + 2 * jnp.pi, vx)
    # vy = jnp.where(vy > jnp.pi, vy - 2 * jnp.pi, vy)
    # vy = jnp.where(vy <= -jnp.pi, vy + 2 * jnp.pi, vy)
    # vz = jnp.where(vz > jnp.pi, vz - 2 * jnp.pi, vz)
    # vz = jnp.where(vz <= -jnp.pi, vz + 2 * jnp.pi, vz)
    # vx = vx / (2.0 * dx) / m_per_hbar
    # vy = vy / (2.0 * dx) / m_per_hbar
    # vz = vz / (2.0 * dx) / m_per_hbar

    # Forward differences
    vx1 = jnp.roll(S_per_hbar, -1, axis=0) - S_per_hbar
    vy1 = jnp.roll(S_per_hbar, -1, axis=1) - S_per_hbar
    vz1 = jnp.roll(S_per_hbar, -1, axis=2) - S_per_hbar
    vx1 = jnp.where(vx1 > jnp.pi, vx1 - 2 * jnp.pi, vx1)
    vx1 = jnp.where(vx1 <= -jnp.pi, vx1 + 2 * jnp.pi, vx1)
    vy1 = jnp.where(vy1 > jnp.pi, vy1 - 2 * jnp.pi, vy1)
    vy1 = jnp.where(vy1 <= -jnp.pi, vy1 + 2 * jnp.pi, vy1)
    vz1 = jnp.where(vz1 > jnp.pi, vz1 - 2 * jnp.pi, vz1)
    vz1 = jnp.where(vz1 <= -jnp.pi, vz1 + 2 * jnp.pi, vz1)
    vx1 = vx1 / dx / m_per_hbar
    vy1 = vy1 / dx / m_per_hbar
    vz1 = vz1 / dx / m_per_hbar

    # Backward differences
    vx2 = S_per_hbar - jnp.roll(S_per_hbar, 1, axis=0)
    vy2 = S_per_hbar - jnp.roll(S_per_hbar, 1, axis=1)
    vz2 = S_per_hbar - jnp.roll(S_per_hbar, 1, axis=2)
    vx2 = jnp.where(vx2 > jnp.pi, vx2 - 2 * jnp.pi, vx2)
    vx2 = jnp.where(vx2 <= -jnp.pi, vx2 + 2 * jnp.pi, vx2)
    vy2 = jnp.where(vy2 > jnp.pi, vy2 - 2 * jnp.pi, vy2)
    vy2 = jnp.where(vy2 <= -jnp.pi, vy2 + 2 * jnp.pi, vy2)
    vz2 = jnp.where(vz2 > jnp.pi, vz2 - 2 * jnp.pi, vz2)
    vz2 = jnp.where(vz2 <= -jnp.pi, vz2 + 2 * jnp.pi, vz2)
    vx2 = vx2 / dx / m_per_hbar
    vy2 = vy2 / dx / m_per_hbar
    vz2 = vz2 / dx / m_per_hbar

    # Average forward and backward
    vx = 0.5 * (vx1 + vx2)
    vy = 0.5 * (vy1 + vy2)
    vz = 0.5 * (vz1 + vz2)

    return vx, vy, vz
