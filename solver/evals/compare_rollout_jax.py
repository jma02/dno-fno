from __future__ import annotations

import jax.numpy as jnp

from ..solvers.time_integrator import SolverParams, State, rollout


def relative_l2(prediction: jnp.ndarray, truth: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    numerator = jnp.linalg.norm(prediction - truth, axis=axis)
    denominator = jnp.linalg.norm(truth, axis=axis)
    return numerator / (denominator + eps)


def compare_rollout_to_truth(
    trajectory: dict,
    params: SolverParams,
) -> dict[str, jnp.ndarray]:
    times = jnp.asarray(trajectory["t"])
    truth_eta = jnp.asarray(trajectory["eta"])
    truth_xi = jnp.asarray(trajectory["xi"])
    truth_gxi = jnp.asarray(trajectory["gxi"])

    initial_state = State(eta=truth_eta[0], xi=truth_xi[0])
    prediction = rollout(initial_state, times, params, save_gxi=True)

    predicted_eta = prediction["eta"]
    predicted_xi = prediction["xi"]
    predicted_gxi = prediction["gxi"]

    eta_rel_l2 = relative_l2(predicted_eta, truth_eta)
    xi_rel_l2 = relative_l2(predicted_xi, truth_xi)
    gxi_rel_l2 = relative_l2(predicted_gxi, truth_gxi)

    return {
        "name": trajectory.get("name", ""),
        "times": times,
        "eta_rel_l2": eta_rel_l2,
        "xi_rel_l2": xi_rel_l2,
        "gxi_rel_l2": gxi_rel_l2,
        "eta_final_rel_l2": eta_rel_l2[-1],
        "xi_final_rel_l2": xi_rel_l2[-1],
        "gxi_final_rel_l2": gxi_rel_l2[-1],
        "eta_mean_rel_l2": jnp.mean(eta_rel_l2),
        "xi_mean_rel_l2": jnp.mean(xi_rel_l2),
        "gxi_mean_rel_l2": jnp.mean(gxi_rel_l2),
    }
