import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import json
import time

from .constants import constants
from .quantum import quantum_kick, quantum_drift
from .gravity import calculate_gravitational_potential
from .hydro import hydro_fluxes, hydro_accelerate
from .particles import particles_accelerate, particles_drift, bin_particles
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

        if self.params["output"]["save"]:
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
        if self.params["physics"]["hydro"]:
            self.state["rho"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution)
            )
            self.state["vx"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution)
            )
            self.state["vy"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution)
            )
            self.state["vz"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution)
            )
        if self.params["physics"]["particles"]:
            self.state["pos"] = jnp.zeros((self.num_particles, 3))
            self.state["vel"] = jnp.zeros((self.num_particles, 3))

    @property
    def resolution(self):
        return (
            self.params["domain"]["resolution_base"]
            * self.params["domain"]["resolution_multiplier"]
        )

    @property
    def num_particles(self):
        return self.params["particles"]["num_particles"]

    @property
    def box_size(self):
        return self.params["domain"]["box_size"]

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
    def sound_speed(self):
        return self.params["hydro"]["sound_speed"]

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
    def kgrid(self):
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
        if self.params["physics"]["hydro"]:
            rho_bar += jnp.mean(state["rho"])
        if self.params["physics"]["particles"]:
            m_particle = self.params["particles"]["particle_mass"]
            n_particles = self.num_particles
            box_size = self.box_size
            rho_bar += m_particle * n_particles / box_size
        return rho_bar

    def _calc_grav_potential(self, state, k_sq):
        G = constants["gravitational_constant"]
        m_particle = self.params["particles"]["particle_mass"]
        rho_bar = self._calc_rho_bar(self.state)
        rho_tot = 0.0
        if self.params["physics"]["quantum"]:
            rho_tot += jnp.abs(state["psi"]) ** 2
        if self.params["physics"]["hydro"]:
            rho_tot += state["rho"]
        if self.params["physics"]["particles"]:
            rho_tot += bin_particles(state["pos"], self.dx, self.resolution, m_particle)
        return calculate_gravitational_potential(rho_tot, k_sq, G, rho_bar)

    @property
    def potential(self):
        kx, ky, kz = self.kgrid
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

        c_sound = self.params["hydro"]["sound_speed"]
        box_size = self.box_size

        # round up to the nearest multiple of num_checkpoints
        num_checkpoints = self.params["output"]["num_checkpoints"]
        nt = int(round(round(t_end / dt_kin) / num_checkpoints) * num_checkpoints)
        nt_sub = int(round(nt / num_checkpoints))
        dt = t_end / nt

        # Fourier space variables
        if self.params["physics"]["gravity"] or self.params["physics"]["quantum"]:
            kx, ky, kz = self.kgrid
            k_sq = kx**2 + ky**2 + kz**2

        # Checkpointer
        if self.params["output"]["save"]:
            options = ocp.CheckpointManagerOptions()
            checkpoint_dir = checkpoint_dir = os.path.join(
                os.getcwd(), self.params["output"]["path"]
            )
            path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
            async_checkpoint_manager = ocp.CheckpointManager(path, options=options)

        def _kick(state, dt):
            # Kick (half-step)
            if (
                self.params["physics"]["gravity"]
                and self.params["physics"]["external_potential"]
            ):
                V = self._calc_grav_potential(state, k_sq) + state["V_ext"]
            elif self.params["physics"]["gravity"]:
                V = self._calc_grav_potential(state, k_sq)
            elif self.params["physics"]["external_potential"]:
                V = state["V_ext"]

            if (
                self.params["physics"]["gravity"]
                or self.params["physics"]["external_potential"]
            ):
                if self.params["physics"]["quantum"]:
                    state["psi"] = quantum_kick(state["psi"], V, m_per_hbar, dt)
                if self.params["physics"]["hydro"]:
                    state["vx"], state["vy"], state["vz"] = hydro_accelerate(
                        state["vx"], state["vy"], state["vz"], V, kx, ky, kz, dt
                    )
                if self.params["physics"]["particles"]:
                    state["vel"] = particles_accelerate(
                        state["vel"], state["pos"], V, kx, ky, kz, dx, dt
                    )

        def _drift(state, dt):
            # Drift (full-step)
            if self.params["physics"]["quantum"]:
                state["psi"] = quantum_drift(state["psi"], k_sq, m_per_hbar, dt)
            if self.params["physics"]["hydro"]:
                state["rho"], state["vx"], state["vy"], state["vz"] = hydro_fluxes(
                    state["rho"], state["vx"], state["vy"], state["vz"], dt, dx, c_sound
                )
            if self.params["physics"]["particles"]:
                state["pos"] = particles_drift(state["pos"], state["vel"], dt, box_size)

        @jax.jit
        def _update(_, state):
            # Update the simulation state by one timestep
            # according to a 2nd-order `kick-drift-kick` scheme
            _kick(state, 0.5 * dt)
            _drift(state, dt)
            _kick(state, 0.5 * dt)

            # update time
            state["t"] += dt

            return state

        # save initial state
        print("Starting simulation ...")
        if self.params["output"]["save"]:
            with open(os.path.join(checkpoint_dir, "params.json"), "w") as f:
                json.dump(self.params, f, indent=2)
            async_checkpoint_manager.save(0, args=ocp.args.StandardSave(state))
            plot_sim(state, checkpoint_dir, 0, self.params)
            async_checkpoint_manager.wait_until_finished()

        # Simulation Main Loop
        t_start_timer = time.time()
        if self.params["output"]["save"]:
            for i in range(1, num_checkpoints + 1):
                state = jax.lax.fori_loop(0, nt_sub, _update, init_val=state)
                jax.block_until_ready(state)
                # save state
                async_checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
                percent = int(100 * i / num_checkpoints)
                elapsed = time.time() - t_start_timer
                est_total = elapsed / i * num_checkpoints
                est_remaining = est_total - elapsed
                print(
                    f"{percent:.1f}%: estimated time remaining (s): {est_remaining:.1f}"
                )
                plot_sim(state, checkpoint_dir, i, self.params)
                async_checkpoint_manager.wait_until_finished()
        else:
            state = jax.lax.fori_loop(0, nt, _update, init_val=state)
        jax.block_until_ready(state)
        print("Simulation Run Time (s): ", time.time() - t_start_timer)

        return state

    def run(self):
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
