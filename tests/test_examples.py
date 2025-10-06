import jax.numpy as jnp
from jaxion.utils import run_example_main
import pytest


def test_tidal_stripping():
    sim = run_example_main(
        "examples/tidal_stripping/tidal_stripping.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(639.0479)


def test_heating_gas():
    sim = run_example_main("examples/heating_gas/heating_gas.py", argv=["--res", "1"])
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2581.4824)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(3.5349982)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(3.0522914)
    assert jnp.mean(jnp.abs(sim.state["vz"])) == pytest.approx(4.054723)
