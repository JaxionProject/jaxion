from importlib.metadata import PackageNotFoundError, version

from .analysis import radial_power_spectrum as radial_power_spectrum
from .constants import constants as constants
from .simulation import Simulation as Simulation

"""
jaxion: A JAX library for simulations of fuzzy dark matter, stars, gas + more!
"""

try:
    __version__ = version("jaxion")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Simulation",
    "constants",
    "radial_power_spectrum",
]
