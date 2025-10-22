import jax.numpy as jnp
import jaxdecomp as jd

# Pure functions for hydro simulation


def get_conserved(rho, vx, vy, vz, vol):
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Momz = rho * vz * vol

    return Mass, Momx, Momy, Momz


def get_primitive(Mass, Momx, Momy, Momz, vol):
    rho = Mass / vol
    vx = Momx / Mass
    vy = Momy / Mass
    vz = Momz / Mass

    return rho, vx, vy, vz


def get_gradient(f, dx):
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)
    f_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dx)

    return f_dx, f_dy, f_dz


def extrap_to_face(f, f_dx, f_dy, f_dz, dx):
    f_XL = f + 0.5 * f_dx * dx
    f_XR = f - 0.5 * f_dx * dx
    f_XR = jnp.roll(f_XR, -1, axis=0)

    f_YL = f + 0.5 * f_dy * dx
    f_YR = f - 0.5 * f_dy * dx
    f_YR = jnp.roll(f_YR, -1, axis=1)

    f_ZL = f + 0.5 * f_dz * dx
    f_ZR = f - 0.5 * f_dz * dx
    f_ZR = jnp.roll(f_ZR, -1, axis=2)

    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    fac = dt * dx * dx
    F += -fac * flux_F_X
    F += fac * jnp.roll(flux_F_X, 1, axis=0)

    F += -fac * flux_F_Y
    F += fac * jnp.roll(flux_F_Y, 1, axis=1)

    F += -fac * flux_F_Z
    F += fac * jnp.roll(flux_F_Z, 1, axis=2)

    return F


def get_flux(rho_L, vx_L, vy_L, vz_L, rho_R, vx_R, vy_R, vz_R, cs):
    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)

    P_star = rho_star * cs * cs

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Momz = momx_star * momz_star / rho_star

    # find wavespeeds
    C_L = cs + jnp.abs(vx_L)
    C_R = cs + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz


def hydro_accelerate(vx, vy, vz, V, kx, ky, kz, dt):
    V_hat = jd.fft.pfft3d(V)

    ax = -jnp.real(jd.fft.pifft3d(1.0j * kx * V_hat))
    ay = -jnp.real(jd.fft.pifft3d(1.0j * ky * V_hat))
    az = -jnp.real(jd.fft.pifft3d(1.0j * kz * V_hat))

    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    return vx, vy, vz


def hydro_fluxes(rho, vx, vy, vz, dt, dx, cs):
    # calculate gradients
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx, vx_dy, vx_dz = get_gradient(vx, dx)
    vy_dx, vy_dy, vy_dz = get_gradient(vy, dx)
    vz_dx, vz_dy, vz_dz = get_gradient(vz, dx)

    # slope limit gradients
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx, vx_dy, vx_dz = slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
    vy_dx, vy_dy, vy_dz = slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
    vz_dx, vz_dy, vz_dz = slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx
        + rho * vx_dx
        + vy * rho_dy
        + rho * vy_dy
        + vz * rho_dz
        + rho * vz_dz
    )
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx + vy * vx_dy + vz * vx_dz + (1.0 / rho) * rho_dx * cs * cs
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx + vy * vy_dy + vz * vy_dz + (1.0 / rho) * rho_dy * cs * cs
    )
    vz_prime = vz - 0.5 * dt * (
        vx * vz_dx + vy * vz_dy + vz * vz_dz + (1.0 / rho) * rho_dz * cs * cs
    )

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrap_to_face(
        rho_prime, rho_dx, rho_dy, rho_dz, dx
    )
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrap_to_face(
        vx_prime, vx_dx, vx_dy, vx_dz, dx
    )
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrap_to_face(
        vy_prime, vy_dx, vy_dy, vy_dz, dx
    )
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrap_to_face(
        vz_prime, vz_dx, vz_dy, vz_dz, dx
    )

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X = get_flux(
        rho_XL, vx_XL, vy_XL, vz_XL, rho_XR, vx_XR, vy_XR, vz_XR, cs
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momz_Y, flux_Momx_Y = get_flux(
        rho_YL, vy_YL, vz_YL, vx_YL, rho_YR, vy_YR, vz_YR, vx_YR, cs
    )
    flux_Mass_Z, flux_Momz_Z, flux_Momx_Z, flux_Momy_Z = get_flux(
        rho_ZL, vz_ZL, vx_ZL, vy_ZL, rho_ZR, vz_ZR, vx_ZR, vy_ZR, cs
    )

    Mass, Momx, Momy, Momz = get_conserved(rho, vx, vy, vz, dx**3)

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)

    # get Primitive variables
    rho, vx, vy, vz = get_primitive(Mass, Momx, Momy, Momz, dx**3)

    return rho, vx, vy, vz


def slope_limiter(f, dx, f_dx, f_dy, f_dz):
    """
    Apply slope limiter to slopes (minmod)
    """

    eps = 1.0e-12

    # Keep a copy of the original slopes
    orig_f_dx = f_dx
    orig_f_dy = f_dy
    orig_f_dz = f_dz

    # Function to adjust the denominator safely
    def adjust_denominator(denom):
        denom_safe = jnp.where(
            denom > 0, denom + eps, jnp.where(denom < 0, denom - eps, eps)
        )
        return denom_safe

    # For x-direction
    denom = adjust_denominator(orig_f_dx)
    num = (f - jnp.roll(f, 1, axis=0)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    ratio = num / denom  # Use the same adjusted denominator
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    # For y-direction
    denom = adjust_denominator(orig_f_dy)
    num = (f - jnp.roll(f, 1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    # For z-direction
    denom = adjust_denominator(orig_f_dz)
    num = (f - jnp.roll(f, 1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    return f_dx, f_dy, f_dz
