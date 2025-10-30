import jax
import jax.numpy as jnp
import jaxion
import chex
from typing import NamedTuple
import optax
import time
import matplotlib.image as img

# switch on for double precision
# jax.config.update("jax_enable_x64", True)

"""
Inverse problem: Find initial conditions for velocity (dm+stars) to achieve target density at t=1

Philip Mocz (2025)

Usage:
  python logo_inverse_problem.py
"""


def set_up_simulation(save=False):
    # Parameters added/changed from default values
    params = {
        "physics": {
            "particles": True,
        },
        "domain": {
            "resolution_base": 32,
            "box_size": 20.0,  # kpc
        },
        "time": {
            "end": 1.0,
        },
        "output": {
            "path": "./checkpoints/",
            "num_checkpoints": 100,
            "save": save,
            "plot_dynamic_range": 10.0,
        },
        "particles": {
            "num_particles": 64,
            "particle_mass": 1.0e7,  # Msun
        },
    }

    # Initialize the simulation
    sim = jaxion.Simulation(params)

    # dark matter
    rho_dm = 1.0e5  # (Msun / kpc^3)

    box_size = sim.box_size
    nx = sim.resolution

    # dark matter
    psi = jnp.sqrt(rho_dm) * jnp.ones((nx, nx, nx)) + 0j

    # stars are initially uniformly distributed
    # Arrange positions in a uniform grid in x-y plane, z=box_size/2
    num_stars = params["particles"]["num_particles"]
    side = int(num_stars ** (1 / 2))
    xlin = jnp.linspace(0, box_size, side, endpoint=False) + box_size / (2 * side)
    ylin = jnp.linspace(0, box_size, side, endpoint=False) + box_size / (2 * side)
    zlin = jnp.array([box_size / 2.0])
    xx, yy, zz = jnp.meshgrid(xlin, ylin, zlin, indexing="ij")
    pos = jnp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    vel = jnp.zeros((num_stars, 3))

    sim.state["psi"] = psi
    sim.state["pos"] = pos
    sim.state["vel"] = vel

    return sim


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Loss: {v:.2e}, |grad|: {e:.2e}",
            i=state.iter_num,
            v=value,
            e=optax.tree_utils.tree_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def run_opt(init_params, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        err = optax.tree_utils.tree_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def solve_inverse_problem():
    """Optimize the initial star positions to recreate the logo"""
    # Load the target density field
    target_data = img.imread("target.png")[:, :, 0]
    target_data = target_data[::2, ::2]  # downsample
    target = jnp.flipud(jnp.array(target_data, dtype=float)).T
    target = 1.0 - 1.6 * (target - 0.5)
    target /= jnp.mean(target)

    @jax.jit
    def loss_function(theta):
        sim = set_up_simulation()
        sim.state["vel"] = theta

        sim.run()

        projected_density = jnp.mean(jnp.abs(sim.state["psi"]) ** 2, axis=2)
        norm = jnp.mean(projected_density)
        projected_density /= norm

        error_norm = jnp.mean((projected_density - target) ** 2)

        return error_norm

    # opt = optax.lbfgs()
    opt = optax.chain(print_info(), optax.lbfgs())

    sim = set_up_simulation()
    init_params = sim.state["vel"]

    print(
        f"Initial value: {loss_function(init_params):.2e} "
        f"Initial gradient norm: {optax.tree_utils.tree_norm(jax.grad(loss_function)(init_params)):.2e}"
    )
    t0 = time.time()
    final_params, _ = run_opt(init_params, loss_function, opt, max_iter=100, tol=1e-5)
    print("Inverse-problem solve time (s): ", time.time() - t0)
    print(
        f"Final value: {loss_function(final_params):.2e}, "
        f"Final gradient norm: {optax.tree_utils.tree_norm(jax.grad(loss_function)(final_params)):.2e}"
    )

    return final_params


def main():
    optimized_ics = solve_inverse_problem()

    sim = set_up_simulation(save=True)
    sim.state["vel"] = optimized_ics
    sim.run()

    return sim


if __name__ == "__main__":
    main()
