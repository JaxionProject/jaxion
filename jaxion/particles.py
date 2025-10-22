import jax
import jax.numpy as jnp
import jaxdecomp as jd

# Pure functions for particle-mesh calculations (stars, BHs, ...)


def get_cic_indices_and_weights(pos, dx, resolution):
    """Compute the cloud-in-cell indices and weights for the particle positions."""
    nx = resolution
    dxs = jnp.array([dx, dx, dx])
    i = jnp.floor((pos - 0.5 * dxs) / dxs)
    ip1 = i + 1.0
    weight_i = ((ip1 + 0.5) * dxs - pos) / dxs
    weight_ip1 = (pos - (i + 0.5) * dxs) / dxs
    i = jnp.mod(i, jnp.array([nx, nx, nx])).astype(int)
    ip1 = jnp.mod(ip1, jnp.array([nx, nx, nx])).astype(int)
    return i, ip1, weight_i, weight_ip1


def bin_particles(pos, dx, resolution, m_particle):
    """Bin the particles into the grid using cloud-in-cell weights."""
    nx = resolution
    n_particle = pos.shape[0]
    rho = jnp.zeros((nx, nx, nx))
    i, ip1, w_i, w_ip1 = get_cic_indices_and_weights(pos, dx, resolution)

    def deposit_particle(s, rho):
        """Deposit the particle mass into the grid."""
        fac = m_particle / (dx * dx * dx)
        rho = rho.at[i[s, 0], i[s, 1], i[s, 2]].add(
            w_i[s, 0] * w_i[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], i[s, 2]].add(
            w_ip1[s, 0] * w_i[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], i[s, 2]].add(
            w_i[s, 0] * w_ip1[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], i[s, 1], ip1[s, 2]].add(
            w_i[s, 0] * w_i[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], i[s, 2]].add(
            w_ip1[s, 0] * w_ip1[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], ip1[s, 2]].add(
            w_ip1[s, 0] * w_i[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], ip1[s, 2]].add(
            w_i[s, 0] * w_ip1[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], ip1[s, 2]].add(
            w_ip1[s, 0] * w_ip1[s, 1] * w_ip1[s, 2] * fac
        )
        return rho

    rho = jax.lax.fori_loop(0, n_particle, deposit_particle, rho)
    return rho


def get_acceleration(pos, V, kx, ky, kz, dx):
    """Compute the acceleration of the particles."""
    n_particle = pos.shape[0]
    resolution = V.shape[0]
    i, ip1, w_i, w_ip1 = get_cic_indices_and_weights(pos, dx, resolution)

    # find accelerations on the grid
    V_hat = jd.fft.pfft3d(V)
    ax = -jnp.real(jd.fft.pifft3d(1.0j * kx * V_hat))
    ay = -jnp.real(jd.fft.pifft3d(1.0j * ky * V_hat))
    az = -jnp.real(jd.fft.pifft3d(1.0j * kz * V_hat))
    a_grid = jnp.stack((ax, ay, az), axis=-1)

    # interpolate the accelerations to the particle positions
    acc = jnp.zeros((n_particle, 3))
    acc += (w_i[:, 0] * w_i[:, 1] * w_i[:, 2])[:, None] * a_grid[
        i[:, 0], i[:, 1], i[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_i[:, 1] * w_i[:, 2])[:, None] * a_grid[
        ip1[:, 0], i[:, 1], i[:, 2]
    ]
    acc += (w_i[:, 0] * w_ip1[:, 1] * w_i[:, 2])[:, None] * a_grid[
        i[:, 0], ip1[:, 1], i[:, 2]
    ]
    acc += (w_i[:, 0] * w_i[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        i[:, 0], i[:, 1], ip1[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_ip1[:, 1] * w_i[:, 2])[:, None] * a_grid[
        ip1[:, 0], ip1[:, 1], i[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_i[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        ip1[:, 0], i[:, 1], ip1[:, 2]
    ]
    acc += (w_i[:, 0] * w_ip1[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        i[:, 0], ip1[:, 1], ip1[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_ip1[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        ip1[:, 0], ip1[:, 1], ip1[:, 2]
    ]
    return acc


def particles_accelerate(vel, pos, V, kx, ky, kz, dx, dt):
    """Apply the acceleration to the particles."""
    acc = get_acceleration(pos, V, kx, ky, kz, dx)
    vel += acc * dt
    return vel


def particles_drift(pos, vel, dt, box_size):
    pos += vel * dt
    pos = jnp.mod(pos, jnp.array([box_size, box_size, box_size]))

    return pos
