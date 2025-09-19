import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import json
import time

from .constants import constants
from .quantum import quantum_kick, quantum_drift
from .gravity import calculate_gravitational_potential
from .utils import set_up_parameters, print_parameters
from .visualization import plot_sim


class Simulation:
    """
    Simulation: The base class for an astrophysics simulation.

    Parameters
    ----------
      params (dict): The Python dictionary that contains the simulation parameters.

    """

    def __init__(self, params):
        # start from default simulation parameters and update with user params
        self._params = set_up_parameters(params)

        # additional checks
        if self.resolution % 2 != 0:
            raise ValueError("Resolution must be divisible by 2.")

        print("Simulation initialized with parameters:")
        print_parameters(self.params)

        # simulation state
        self.state = {}
        self.state["t"] = 0.0
        if self.params["physics"]["quantum"]:
            self.state["psi"] = (
                jnp.zeros((self.resolution, self.resolution, self.resolution)) * 1j
            )
        if self.params["physics"]["external_potential"]:
            self.state["V_ext"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution)
            )

        # XXX TODO: finish

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
    def axion_mass(self):
        return (
            self.params["quantum"]["m_22"]
            * 1.0e-22
            * constants["electron_volt"]
            / constants["speed_of_light"] ** 2
        )

    @property
    def params(self):
        return self._params

    @property
    def grid(self):
        hx = 0.5 * self.dx
        x_lin = jnp.linspace(hx, self.box_size - hx, self.resolution)
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
        dx = self.dx
        m_per_hbar = self.axion_mass / constants["reduced_planck_constant"]

        dt_fac = 1.0
        dt_kin = dt_fac * (m_per_hbar / 6.0) * (dx * dx)
        t_end = self.params["time"]["end"]

        # round up to the nearest multiple of num_checkpoints
        num_checkpoints = self.params["output"]["num_checkpoints"]
        nt = int(jnp.ceil(jnp.ceil(t_end / dt_kin) / num_checkpoints) * num_checkpoints)
        nt_sub = int(jnp.round(nt / num_checkpoints))
        dt = t_end / nt

        # Fourier space variables
        if self.params["physics"]["gravity"] or self.params["physics"]["quantum"]:
            kx, ky, kz = self.fourier
            k_sq = kx**2 + ky**2 + kz**2

        # Checkpointer
        options = ocp.CheckpointManagerOptions()
        checkpoint_dir = checkpoint_dir = os.path.join(
            os.getcwd(), self.params["output"]["path"]
        )
        path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
        async_checkpoint_manager = ocp.CheckpointManager(path, options=options)

        @jax.jit
        def _update(_, state):
            # Update the simulation state by one timestep
            # according to a 2nd-order `kick-drift-kick` scheme

            # Kick (half-step)
            if self.params["physics"]["gravity"]:
                V = self._calc_grav_potential(state, k_sq)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, 0.5 * dt)
            if (
                self.params["physics"]["quantum"]
                and self.params["physics"]["external_potential"]
            ):
                state["psi"] = quantum_kick(
                    state["psi"], state["V_ext"], m_per_hbar, 0.5 * dt
                )

            # Drift (full-step)
            if self.params["physics"]["quantum"]:
                state["psi"] = quantum_drift(state["psi"], k_sq, m_per_hbar, dt)

            # Kick (half-step)
            if self.params["physics"]["gravity"]:
                V = self._calc_grav_potential(state, k_sq)
            if self.params["physics"]["quantum"] and self.params["physics"]["gravity"]:
                state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, 0.5 * dt)
            if (
                self.params["physics"]["quantum"]
                and self.params["physics"]["external_potential"]
            ):
                state["psi"] = quantum_kick(
                    state["psi"], state["V_ext"], m_per_hbar, 0.5 * dt
                )

            # update time
            state["t"] += dt

            return state

        # Simulation Main Loop
        print("Starting simulation ...")
        with open(os.path.join(checkpoint_dir, "params.json"), "w") as f:
            json.dump(self.params, f, indent=2)
        rho_bar = self._calc_rho_bar(state)
        vmin = jnp.log10(rho_bar / 100.0)
        vmax = jnp.log10(rho_bar * 100.0)
        # save initial state
        async_checkpoint_manager.save(0, args=ocp.args.StandardSave(state))
        plot_sim(state, checkpoint_dir, 0, vmin, vmax)
        async_checkpoint_manager.wait_until_finished()
        t_start_timer = time.time()
        for i in range(1, num_checkpoints + 1):
            state = jax.lax.fori_loop(0, nt_sub, _update, init_val=state)
            percent = int(100 * i / num_checkpoints)
            elapsed = time.time() - t_start_timer
            est_total = elapsed / i * num_checkpoints
            est_remaining = est_total - elapsed
            print(f"{percent:.1f}%: estimated time remaining (s): {est_remaining:.1f}")
            async_checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
            plot_sim(state, checkpoint_dir, i, vmin, vmax)
            async_checkpoint_manager.wait_until_finished()
        jax.block_until_ready(state)
        print("Simulation Run Time (s): ", time.time() - t_start_timer)

        return state

    def run(self):
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
