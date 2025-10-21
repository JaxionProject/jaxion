import jax.numpy as jnp

# Pure functions for cosmology simulation


def get_physical_time_interval(z_start, z_end, omega_matter, omega_lambda, little_h):
    """Compute the total physical time between two redshifts."""
    #  da/dt = H0 * sqrt(omega_matter / a + omega_lambda * a^2)
    a_start = 1.0 / (1.0 + z_start)
    a_end = 1.0 / (1.0 + z_end)
    H0 = 0.1 * little_h  # Hubble constant in (km/s/kpc)
    n_quad = 10000
    a_lin = jnp.linspace(a_start, a_end, n_quad)
    a_dot = H0 * jnp.sqrt((omega_matter / a_lin) + (omega_lambda * a_lin**2))
    dt_hat = jnp.trapezoid(1.0 / a_dot, a_lin)
    return dt_hat


def get_supercomoving_time_interval(
    z_start, z_end, omega_matter, omega_lambda, little_h
):
    """Compute the total supercomoving time (dt_hat = a^-2 dt) between two redshifts."""
    #  da/dt = H0 * sqrt(omega_matter / a + omega_lambda * a^2)
    #  da/dt_hat = a^2 * da/dt
    #  dt_hat/da = a^-2 / (da/dt)
    a_start = 1.0 / (1.0 + z_start)
    a_end = 1.0 / (1.0 + z_end)
    H0 = 0.1 * little_h  # Hubble constant in (km/s/kpc)
    n_quad = 10000
    a_lin = jnp.linspace(a_start, a_end, n_quad)
    a_dot = H0 * jnp.sqrt((omega_matter / a_lin) + (omega_lambda * a_lin**2))
    dt_hat_da = a_lin**-2 / a_dot
    dt_hat = jnp.trapezoid(dt_hat_da, a_lin)
    return dt_hat


def get_scale_factor(z_start, dt_hat, omega_matter, omega_lambda, little_h):
    """Compute the scale factor corresponding to a given supercomoving time,
    by root finding."""
    a_start = 1.0 / (1.0 + z_start)
    a = a_start
    tolerance = 1e-6
    max_iterations = 100
    lower_bound = 0.0
    upper_bound = 2.0  # Set a reasonable upper bound for the scale factor
    for _ in range(max_iterations):
        dt_hat_guess = get_supercomoving_time_interval(
            z_start, 1.0 / a - 1.0, omega_matter, omega_lambda, little_h
        )
        error = dt_hat_guess - dt_hat
        if jnp.abs(error) < tolerance:
            break
        # Use bisection method
        if error > 0:
            upper_bound = a  # Move upper bound down
        else:
            lower_bound = a  # Move lower bound up
        a = (lower_bound + upper_bound) / 2  # Update scale factor to midpoint
    return a


def get_next_scale_factor(z_start, dt_hat, omega_matter, omega_lambda, little_h):
    """Advance scale factor by dt_hat using RK2 (midpoint) on da/dt_hat = a^2 * da/dt."""
    a_start = 1.0 / (1.0 + z_start)
    H0 = 0.1 * little_h

    def g(a):
        a_dot = H0 * jnp.sqrt((omega_matter / a) + (omega_lambda * a**2))
        return a**2 * a_dot  # da/dt_hat

    k1 = g(a_start)
    k2 = g(a_start + 0.5 * dt_hat * k1)
    a = a_start + dt_hat * k2
    return a
