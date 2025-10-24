import jax.numpy as jnp
import jaxdecomp as jd


def radial_power_spectrum(data_cube, kx, ky, kz, box_size):
    """
    Computes the radially averaged power spectral density of a 3D datacube.

    Parameters
    ----------
        data_cube : jnp.ndarray
            3D data cube, must be cubic
        kx, ky, kz: jnp.ndarray
            wavenumber grids in each dimension
        box_size: float
            physical size of box

    Returns
    -------
        Pf: jnp.ndarray
            radial power spectrum
        k: jnp.ndarray
            wavenumbers
        total_power: float
            total power
    """
    dim = data_cube.ndim
    nx = data_cube.shape[0]
    dx = box_size / nx

    # Compute power spectrum
    data_cube_hat = jd.fft.pfft3d(data_cube)
    total_power = 0.5 * jnp.sum(jnp.abs(data_cube_hat) ** 2) / nx**dim * dx**dim
    phi_k = 0.5 * jnp.abs(data_cube_hat) ** 2 / nx**dim * dx**dim
    half_size = nx // 2 + 1

    # Compute radially-averaged power spectrum
    # if dim == 2:
    #    k_r = jnp.sqrt(kx**2 + ky**2)
    k_r = jnp.sqrt(kx**2 + ky**2 + kz**2)

    Pf, _ = jnp.histogram(
        k_r, range=(-0.5, half_size - 0.5), bins=half_size, weights=phi_k
    )
    norm, _ = jnp.histogram(k_r, range=(-0.5, half_size - 0.5), bins=half_size)
    Pf /= norm + (norm == 0)

    k = 2.0 * jnp.pi * jnp.arange(half_size) / box_size
    dk = 2.0 * jnp.pi / box_size

    Pf /= dk**dim

    # Add geometrical factor
    # if dim == 2:
    #     Pf = Pf * 2.0 * jnp.pi * k
    Pf *= 4.0 * jnp.pi * k**2

    return Pf, k, total_power
