from jaxion.utils import run_example_main


def test_tidal_stripping():
    sim = run_example_main(
        "examples/tidal_stripping/tidal_stripping.py", argv=["--res", "1"]
    )
    assert sim.resolution == 32
    assert sim.state["t"] > 0.0
