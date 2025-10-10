import jax.numpy as jnp
from jaxion.utils import run_example_main
import pytest

rel_tol = 1e-4


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
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2574.076, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(3.0708265, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(4.0393305, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vz"])) == pytest.approx(5.859339, rel=rel_tol)


def test_heating_stars():
    sim = run_example_main(
        "examples/heating_stars/heating_stars.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2574.089, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 0])) == pytest.approx(
        17.170446, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 1])) == pytest.approx(
        18.110512, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 2])) == pytest.approx(
        17.425604, rel=rel_tol
    )
