# Copyright (c) 2025 The Jaxion Team.

"""
Physical constants in units of:
[L] = kpc,
[V] = km/s,
[M] = Msun

note: other units are derived from these base units, e.g., [T] = [L]/[V] = kpc / (km/s) ~= 0.978 Gyr
"""

constants = {
    "gravitational_constant": 4.30241002e-6,  # G / (kpc * (km/s)^2 / Msun)
    "reduced_planck_constant": 1.71818134e-87,  # hbar / ((km/s) * kpc * mass of sun)
    "electron_volt": 8.05478173e-56,  # eV / (mass of sun * (km/s)^2)
    "speed_of_light": 299792.458,  # c / (km/s)
}
