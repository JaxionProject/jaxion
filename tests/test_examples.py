import jax.numpy as jnp
import pytest

from jaxion.utils import run_example_main

rel_tol = 1e-3


def test_tidal_stripping():
    sim = run_example_main(
        "examples/tidal_stripping/tidal_stripping.py",
        argv=["--res", "1", "--save", "False"],
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(162.028, rel=rel_tol)


def test_tidal_stripping_distributed_emulate():
    sim = run_example_main(
        "examples/tidal_stripping/tidal_stripping.py",
        argv=["--res", "1", "--distributed", "--emulate"],
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(162.028, rel=rel_tol)


def test_heating_gas():
    sim = run_example_main("examples/heating_gas/heating_gas.py", argv=["--res", "1"])
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2573.253, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vx"])) == pytest.approx(4.6109166, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(2.9255207, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vz"])) == pytest.approx(4.1069093, rel=rel_tol)


def test_kelvin_helmholtz():
    sim = run_example_main(
        "examples/kelvin_helmholtz/kelvin_helmholtz.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    # mass is conserved exactly by the finite volume scheme
    assert jnp.mean(sim.state["rho"]) == pytest.approx(1.5, rel=rel_tol)
    assert jnp.max(sim.state["rho"]) == pytest.approx(1.9329057, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vy"])) == pytest.approx(0.00239378, rel=rel_tol)
    assert jnp.mean(sim.state["P"]) == pytest.approx(2.569811, rel=rel_tol)


def test_heating_stars():
    sim = run_example_main(
        "examples/heating_stars/heating_stars.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
    assert jnp.mean(jnp.abs(sim.state["psi"])) == pytest.approx(2574.395, rel=rel_tol)
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 0])) == pytest.approx(
        16.625353, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 1])) == pytest.approx(
        17.345486, rel=rel_tol
    )
    assert jnp.mean(jnp.abs(sim.state["vel"][:, 2])) == pytest.approx(
        18.218296, rel=rel_tol
    )
