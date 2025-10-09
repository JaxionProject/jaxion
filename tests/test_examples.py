import jax.numpy as jnp
from jaxion.utils import run_example_main
import pytest

rel_tol = 1e-5


def test_tidal_stripping():
    sim = run_example_main(
        "examples/tidal_stripping/tidal_stripping.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(639.0479, rel=rel_tol)


def test_heating_gas():
    sim = run_example_main("examples/heating_gas/heating_gas.py", argv=["--res", "1"])
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2581.588, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(3.535396, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(3.0527194, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vz"])) == pytest.approx(4.05517, rel=rel_tol)


def test_heating_stars():
    sim = run_example_main(
        "examples/heating_stars/heating_stars.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2586.3862, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 0])) == pytest.approx(
        16.76076, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 1])) == pytest.approx(
        17.233564, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 2])) == pytest.approx(
        16.62571, rel=rel_tol
    )
