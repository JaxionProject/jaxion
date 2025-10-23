import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import json
import time

from .constants import constants
from .quantum import quantum_kick, quantum_drift, quantum_velocity
from .gravity import calculate_gravitational_potential
from .hydro import hydro_fluxes, hydro_accelerate
from .particles import particles_accelerate, particles_drift, bin_particles
from .cosmology import get_supercomoving_time_interval, get_next_scale_factor
from .utils import (
    set_up_parameters,
    print_parameters,
    xmeshgrid,
    xmeshgrid_transpose,
    xzeros,
    xones,
)
from .visualization import plot_sim


class Simulation:
    """
    Simulation: The base class for an astrophysics simulation.

    Parameters
    ----------
      params (dict): The Python dictionary that contains the simulation parameters.
                     Params can also be a string path to a checkpoint directory to load a saved simulation.

    """

    def __init__(self, params, sharding=None):
        # allow loading directly from a checkpoint path
        load_from_checkpoint = False
        checkpoint_dir = ""
        if isinstance(params, str):
            load_from_checkpoint = True
            checkpoint_dir = os.path.join(os.getcwd(), params)
            with open(os.path.join(checkpoint_dir, "params.json"), "r") as f:
                params = json.load(f)

        # start from default simulation parameters and update with user params
        self._params = set_up_parameters(params)

        # additional checks
        if self.resolution % 2 != 0:
            raise ValueError("Resolution must be divisible by 2.")

        if self.params["time"]["adaptive"]:
            raise NotImplementedError("Adaptive time stepping is not yet implemented.")

        if self.params["physics"]["cosmology"]:
            if (
                self.params["physics"]["hydro"]
                or self.params["physics"]["particles"]
                or self.params["physics"]["external_potential"]
            ):
                raise NotImplementedError(
                    "Cosmological hydro/particles/external_potential physics is not yet implemented."
                )

        if self.params["physics"]["hydro"] or self.params["physics"]["particles"]:
            if sharding is not None:
                raise NotImplementedError(
                    "hydro/particles sharding is not yet implemented."
                )

        if self.params["output"]["save"]:
            if jax.process_index() == 0:
                print("Simulation parameters:")
                print_parameters(self.params)

        # jitted functions
        self.xmeshgrid_jit = jax.jit(
            xmeshgrid, in_shardings=None, out_shardings=sharding
        )
        self.xmeshgrid_transpose_jit = jax.jit(
            xmeshgrid_transpose, in_shardings=None, out_shardings=sharding
        )
        self.xzeros_jit = jax.jit(
            xzeros, static_argnums=0, in_shardings=None, out_shardings=sharding
        )
        self.xones_jit = jax.jit(
            xones, static_argnums=0, in_shardings=None, out_shardings=sharding
        )

        # simulation state
        self.state = {}
        self.state["t"] = 0.0
        if self.params["physics"]["cosmology"]:
            self.state["redshift"] = 0.0
        if self.params["physics"]["quantum"]:
            self.state["psi"] = self.xzeros_jit(self.resolution) * 1j
        if self.params["physics"]["external_potential"]:
            self.state["V_ext"] = self.xzeros_jit(self.resolution)
        if self.params["physics"]["hydro"]:
            self.state["rho"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution),
            )
            self.state["vx"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution),
            )
            self.state["vy"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution),
            )
            self.state["vz"] = jnp.zeros(
                (self.resolution, self.resolution, self.resolution),
            )
        if self.params["physics"]["particles"]:
            self.state["pos"] = jnp.zeros((self.num_particles, 3))
            self.state["vel"] = jnp.zeros((self.num_particles, 3))

        if load_from_checkpoint:
            options = ocp.CheckpointManagerOptions()
            async_checkpoint_manager = ocp.CheckpointManager(
                checkpoint_dir, options=options
            )
            step = async_checkpoint_manager.latest_step()
            self.state = async_checkpoint_manager.restore(
                step, args=ocp.args.StandardRestore(self.state)
            )

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
        xx, yy, zz = self.xmeshgrid_jit(x_lin)
        return xx, yy, zz

    @property
    def kgrid(self):
        nx = self.resolution
        k_lin = (2.0 * jnp.pi / self.box_size) * jnp.arange(-nx / 2, nx / 2)
        kx, ky, kz = self.xmeshgrid_transpose_jit(k_lin)
        kx = jnp.fft.ifftshift(kx)
        ky = jnp.fft.ifftshift(ky)
        kz = jnp.fft.ifftshift(kz)
        return kx, ky, kz

    @property
    def quantum_velocity(self):
        m_per_hbar = self.axion_mass / constants["reduced_planck_constant"]
        return quantum_velocity(self.state["psi"], self.box_size, m_per_hbar)

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
        rho_bar = self._calc_rho_bar(state)
        rho_tot = 0.0
        if self.params["physics"]["quantum"]:
            rho_tot += jnp.abs(state["psi"]) ** 2
        if self.params["physics"]["hydro"]:
            rho_tot += state["rho"]
        if self.params["physics"]["particles"]:
            rho_tot += bin_particles(state["pos"], self.dx, self.resolution, m_particle)
        if self.params["physics"]["cosmology"]:
            scale_factor = 1.0 / (1.0 + state["redshift"])
            rho_bar *= scale_factor
            rho_tot *= scale_factor
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
        box_size = self.box_size

        dt_fac = 1.0
        dt_kin = dt_fac * (m_per_hbar / 6.0) * (dx * dx)
        t_start = self.params["time"]["start"]
        t_end = self.params["time"]["end"]
        t_span = t_end - t_start
        state["t"] = t_start

        # cosmology
        if self.params["physics"]["cosmology"]:
            z_start = self.params["time"]["start"]
            z_end = self.params["time"]["end"]
            omega_matter = self.params["cosmology"]["omega_matter"]
            omega_lambda = self.params["cosmology"]["omega_lambda"]
            little_h = self.params["cosmology"]["little_h"]
            t_span = get_supercomoving_time_interval(
                z_start, z_end, omega_matter, omega_lambda, little_h
            )
            state["t"] = 0.0
            state["redshift"] = z_start

        # hydro
        c_sound = self.params["hydro"]["sound_speed"]

        # round up to the nearest multiple of num_checkpoints
        num_checkpoints = self.params["output"]["num_checkpoints"]
        nt = int(round(round(t_span / dt_kin) / num_checkpoints) * num_checkpoints)
        nt_sub = int(round(nt / num_checkpoints))
        dt = t_span / nt

        # distributed arrays (fixed) needed for calculations
        kx, ky, kz = None, None, None
        k_sq = None

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
            path = os.path.join(os.getcwd(), checkpoint_dir)
            if jax.process_index() == 0:
                path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
            async_checkpoint_manager = ocp.CheckpointManager(path, options=options)

        carry = (state, kx, ky, kz, k_sq)

        def _kick(state, kx, ky, kz, k_sq, dt):
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

        def _drift(state, k_sq, dt):
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
        def _update(_, carry):
            # Update the simulation state by one timestep
            # according to a 2nd-order `kick-drift-kick` scheme
            state, kx, ky, kz, k_sq = carry
            _kick(state, kx, ky, kz, k_sq, 0.5 * dt)
            _drift(state, k_sq, dt)
            # update time & redshift
            state["t"] += dt
            if self.params["physics"]["cosmology"]:
                scale_factor = get_next_scale_factor(
                    state["redshift"],
                    dt,
                    self.params["cosmology"]["omega_matter"],
                    self.params["cosmology"]["omega_lambda"],
                    self.params["cosmology"]["little_h"],
                )
                state["redshift"] = 1.0 / scale_factor - 1.0
            _kick(state, kx, ky, kz, k_sq, 0.5 * dt)

            return state, kx, ky, kz, k_sq

        # save initial state
        if jax.process_index() == 0:
            print(f"Starting simulation (res={self.resolution}, nt={nt}) ...")
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
                carry = jax.lax.fori_loop(0, nt_sub, _update, init_val=carry)
                state, kx, ky, kz, k_sq = carry
                jax.block_until_ready(state)
                # save state
                async_checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
                percent = int(100 * i / num_checkpoints)
                elapsed = time.time() - t_start_timer
                est_total = elapsed / i * num_checkpoints
                est_remaining = est_total - elapsed
                if jax.process_index() == 0:
                    print(
                        f"{percent:.1f}%: estimated time remaining (s): {est_remaining:.1f}"
                    )
                plot_sim(state, checkpoint_dir, i, self.params)
                async_checkpoint_manager.wait_until_finished()
        else:
            state = jax.lax.fori_loop(0, nt, _update, init_val=state)
        jax.block_until_ready(state)
        if jax.process_index() == 0:
            print("Simulation Run Time (s): ", time.time() - t_start_timer)

        return state

    def run(self):
        self.state = self._evolve(self.state)
        jax.block_until_ready(self.state)
