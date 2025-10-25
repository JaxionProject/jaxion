from jaxion.cosmology import (
    get_physical_time_interval,
    get_supercomoving_time_interval,
    get_scale_factor,
    get_next_scale_factor,
)
import pytest


def test_cosmology_functions():
    z_start = 127.0
    z_end = 0.0
    omega_matter = 0.3
    omega_lambda = 0.7
    little_h = 0.7
    dt_hat = 10.0

    assert get_physical_time_interval(
        z_start, z_end, omega_matter, omega_lambda, little_h
    ) == pytest.approx(13.76084)
    assert get_supercomoving_time_interval(
        z_start, z_end, omega_matter, omega_lambda, little_h
    ) == pytest.approx(530.44415)
    assert get_scale_factor(
        z_start, dt_hat, omega_matter, omega_lambda, little_h
    ) == pytest.approx(0.008084139320999384)
    assert get_next_scale_factor(
        z_start, dt_hat, omega_matter, omega_lambda, little_h
    ) == pytest.approx(0.00808401)
