from importlib.metadata import version, PackageNotFoundError

from .simulation import Simulation as Simulation
from .constants import constants as constants
from .analysis import radial_power_spectrum as radial_power_spectrum

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
