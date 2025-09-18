import jax
import jax.numpy as jnp
from functools import partial
import json

from .constants import constants
from .quantum import quantum_kick, quantum_drift
from .gravity import calculate_gravitational_potential


class Simulation:
    """
    Simulation: The base class for an astrophysics simulation.

    Parameters
    ----------
      params (dict): The Python dictionary that contains the simulation parameters.

    """

    def __init__(self, params):
        # start from default simulation parameters
        with open("params_default.json", "r") as f:
            self._params = json.load(f)
        # update with user params
        for key in params:
            if key in self._params:
                self._params[key].update(params[key])
            else:
                raise KeyError(f"Unknown parameter key: {key}")

        # simulation state
        self.state = {}
        self.state["t"] = 0.0
        if self.params["physics"]["quantum"]:
            self.state["psi"] = jnp.zeros((self.resolution, self.resolution)) * 1j

        # XXX TODO: finish

        # checks
        if self.resolution % 2 != 0:
            raise ValueError("Resolution must be divisible by 2.")

        # internal simulation state -- should not be touched by user!
        self._internal = {}
        if self.params["physics"]["gravity"]:
            self._internal["V"] = jnp.zeros((self._nx, self._ny))

    @property
    def resolution(self):
        return (
            self._params["domain"]["resolution_base"]
            * self._params["domain"]["resolution_multiplier"]
        )

    @property
    def box_size(self):
        return self._params["domain"]["box_size"]

    @property
    def dx(self):
        return self.box_size / self.resolution

    @property
    def params(self):
        return self._params

    @property
    def grid(self):
        dx = self.dx
        x_lin = jnp.linspace(0.5 * dx, self.box_size - 0.5 * dx, self.box_size)
        X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
        return X, Y, Z

    @property
    def fourier(self):
        nx = self.resolution
        k_lin = (2.0 * jnp.pi / self.box_size) * jnp.arange(-nx / 2, nx / 2)
        kx, ky, kz = jnp.meshgrid(k_lin, k_lin, k_lin, indexing="ij")
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        kz = jnp.fft.ifftshift(kz)
        return kx, ky, kz

    def _calc_rho_bar(self, state):
        rho_bar = 0.0
        if self.params["physics"]["quantum"]:
            rho_bar += jnp.mean(jnp.abs(state["psi"]) ** 2)
        return rho_bar

    def _calc_grav_potential(self, state, k_sq):
        G = constants["gravitational_constant"]
        rho_bar = self._calc_rho_bar(self.state)
        rho_tot = 0.0
        if self.params["physics"]["quantum"]:
            rho_tot += jnp.abs(state["psi"]) ** 2
        return calculate_gravitational_potential(rho_tot, k_sq, G, rho_bar)

    @property
    def potential(self):
        kx, ky, kz = self.fourier
        k_sq = kx**2 + ky**2 + kz**2
        return self._calc_grav_potential(self.state, k_sq)

    @partial(jax.jit, static_argnames=["self"])
    def _evolve(self, state):
        """
        This function evolves the simulation state according to the simulation parameters/physics.

        Parameters
        ----------
        state: jax.pytree
          The current state of the simulation.

        Returns
        -------
        state: jax.pytree
          The evolved state of the simulation.
        """

        # Simulation parameters
        dt = self._dt
        nt = self._nt
        dx = self._dx
        vol = self._vol

        # Fourier space variables
        if self.params["physics"]["gravity"] or self.params["physics"]["quantum"]:
            kx, ky, kz = self.fourier
            k_sq = kx**2 + ky**2 + kz**2

        # initialize potential
        if self.params["physics"]["gravity"]:
            self._internal["V"] = self._calc_grav_potential(state, k_sq)

        def update(i, state):
            # Update the simulation state by one timestep
            # according to a 2nd-order `kick-drift-kick` scheme

            # Kick (half-step)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(state["psi"], self._internal["V"], dt / 2.0)

            # Drift (full-step)
            if self.params["physics"]["quantum"]:
                state["psi"] = quantum_drift(state["psi"], k_sq, dt)

            # update potential
            if self.params["physics"]["gravity"]:
                self._internal["V"] = self._calc_grav_potential(state, k_sq)

            # Kick (half-step)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(state["psi"], self._internal["V"], dt / 2.0)

            # update time
            state["t"] += nt * dt

            return state

        # Simulation Main Loop
        state = jax.lax.fori_loop(0, nt, update, init_val=state)

        return state

    def run(self):
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
