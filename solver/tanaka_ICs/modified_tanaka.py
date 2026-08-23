from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import numpy as np

TANAKA_DTYPE_NAME = os.environ.get("DNO_TANAKA_DTYPE", "float64").strip().lower()
if TANAKA_DTYPE_NAME not in {"float32", "float64"}:
    raise ValueError(f"Unsupported DNO_TANAKA_DTYPE={TANAKA_DTYPE_NAME!r}; expected 'float32' or 'float64'.")

jax.config.update("jax_enable_x64", TANAKA_DTYPE_NAME == "float64")
import jax.numpy as jnp  # noqa: E402

from ..solvers.time_integrator import (  # noqa: E402
    make_solver_params,
    myfft,
    myifft,
    spectral_dx,
)


REAL_DTYPE = jnp.float64 if TANAKA_DTYPE_NAME == "float64" else jnp.float32
DEFAULT_QC_UPPER = 1.0 - 1.0e-12
DEFAULT_OUTER_ITERATIONS = 48
AMPLITUDE_RELATIVE_TOLERANCE = 1.0e-6
AMPLITUDE_ABSOLUTE_TOLERANCE = 1.0e-14


def validate_solved_amplitudes(
    eta_profile: jax.Array,
    requested_amplitudes: jax.Array,
) -> None:
    """Fail if a profile solve did not realize every requested crest height."""

    eta = jnp.asarray(eta_profile)
    requested = jnp.asarray(requested_amplitudes, dtype=eta.dtype)
    if eta.ndim != 2 or requested.shape != (eta.shape[0],):
        raise ValueError(
            "Tanaka profiles and requested amplitudes must have matching batch size."
        )
    achieved = jnp.max(eta, axis=-1)
    absolute_error = jnp.abs(achieved - requested)
    allowed_error = jnp.maximum(
        AMPLITUDE_ABSOLUTE_TOLERANCE,
        AMPLITUDE_RELATIVE_TOLERANCE * requested,
    )
    valid = (
        jnp.isfinite(requested)
        & (requested > 0.0)
        & jnp.isfinite(achieved)
        & (absolute_error <= allowed_error)
    )
    if bool(jnp.all(valid)):
        return

    invalid = np.flatnonzero(~np.asarray(jax.device_get(valid), dtype=bool))
    requested_host = np.asarray(jax.device_get(requested), dtype=np.float64)
    achieved_host = np.asarray(jax.device_get(achieved), dtype=np.float64)
    details = ", ".join(
        (
            f"component {int(index)} requested={requested_host[index]:.17g} "
            f"achieved={achieved_host[index]:.17g}"
        )
        for index in invalid
    )
    raise ValueError(f"Tanaka profile amplitude solve failed: {details}")


@dataclass(frozen=True)
class ModifiedTanakaParams:
    amplitude: float
    depth: float = 1.0
    gravity: float = 1.0
    direction: int = 1
    nx: int = 1024
    length: float = 164.0
    center: float = 43.5625
    dno_order: int = 6
    pad_factor: int = 8
    grid_mode: str = "manual"
    collocation_points: int = 257
    quadrature_substeps: int = 4
    interpolation_degree: int = 3
    s_max: float = 2.5
    alpha: float = 0.01
    transform_power: int = 5
    qc_lower: float = 0.2
    qc_upper: float = DEFAULT_QC_UPPER
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS
    fixed_point_iterations: int = 80
    f2_tolerance: float = 1e-10
    cg_maxiter: int = 400
    cg_tol: float = 1e-10


@dataclass(frozen=True)
class ModifiedTanakaSolution:
    amplitude: float
    qc: float
    froude: float
    speed: float
    tau_profile: jnp.ndarray
    x_profile: jnp.ndarray
    eta_profile: jnp.ndarray
    phi_profile: jnp.ndarray
    q_profile: jnp.ndarray
    theta_profile: jnp.ndarray
    x_periodic: jnp.ndarray
    eta_periodic: jnp.ndarray
    xi_periodic: jnp.ndarray
    gxi_periodic: jnp.ndarray


@dataclass(frozen=True)
class ModifiedTanakaBatchSolution:
    amplitudes: jnp.ndarray
    qc: jnp.ndarray
    froude: jnp.ndarray
    speed: jnp.ndarray
    tau_profile: jnp.ndarray
    x_profile: jnp.ndarray
    eta_profile: jnp.ndarray
    phi_profile: jnp.ndarray
    q_profile: jnp.ndarray
    theta_profile: jnp.ndarray
    x_periodic: jnp.ndarray
    eta_periodic: jnp.ndarray
    xi_periodic: jnp.ndarray
    gxi_periodic: jnp.ndarray


@dataclass(frozen=True)
class ModifiedTanakaSeed:
    qc: float
    f2: float
    phi_profile: jnp.ndarray
    tau_profile: jnp.ndarray


def make_default_tanaka_template(
    *,
    depth: float = 1.0,
    gravity: float = 1.0,
    direction: int = 1,
    nx: int = 1024,
    length: float = 164.0,
    center: float = 0.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    grid_mode: str = "manual",
    collocation_points: int = 257,
    quadrature_substeps: int = 4,
    interpolation_degree: int = 3,
    s_max: float = 2.5,
    alpha: float = 0.01,
    transform_power: int = 5,
    qc_lower: float = 0.2,
    qc_upper: float = DEFAULT_QC_UPPER,
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS,
    fixed_point_iterations: int = 80,
    f2_tolerance: float = 1e-10,
    cg_maxiter: int = 400,
    cg_tol: float = 1e-10,
) -> ModifiedTanakaParams:
    return ModifiedTanakaParams(
        amplitude=0.0,
        depth=depth,
        gravity=gravity,
        direction=direction,
        nx=nx,
        length=length,
        center=center,
        dno_order=dno_order,
        pad_factor=pad_factor,
        grid_mode=grid_mode,
        collocation_points=collocation_points,
        quadrature_substeps=quadrature_substeps,
        interpolation_degree=interpolation_degree,
        s_max=s_max,
        alpha=alpha,
        transform_power=transform_power,
        qc_lower=qc_lower,
        qc_upper=qc_upper,
        outer_iterations=outer_iterations,
        fixed_point_iterations=fixed_point_iterations,
        f2_tolerance=f2_tolerance,
        cg_maxiter=cg_maxiter,
        cg_tol=cg_tol,
    )


def _trapz_weights(values: jnp.ndarray) -> jnp.ndarray:
    spacing = values[1:] - values[:-1]
    left = spacing[:1] / 2.0
    right = spacing[-1:] / 2.0
    middle = (spacing[:-1] + spacing[1:]) / 2.0
    return jnp.concatenate((left, middle, right))


def _cumulative_trapezoid(values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    increments = 0.5 * (values[..., 1:] + values[..., :-1]) * (x[1:] - x[:-1])
    zeros = jnp.zeros((*values.shape[:-1], 1), dtype=values.dtype)
    return jnp.concatenate((zeros, jnp.cumsum(increments, axis=-1)), axis=-1)


def _gradient_last_axis(values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    left = (values[..., 1] - values[..., 0]) / (x[1] - x[0])
    middle = (values[..., 2:] - values[..., :-2]) / (x[2:] - x[:-2])
    right = (values[..., -1] - values[..., -2]) / (x[-1] - x[-2])
    return jnp.concatenate((left[..., None], middle, right[..., None]), axis=-1)


def _effective_grid_params(params: ModifiedTanakaParams) -> tuple[float, int, float]:
    return params.alpha, params.transform_power, params.s_max


def _phi_from_gamma(gamma: jnp.ndarray, params: ModifiedTanakaParams) -> jnp.ndarray:
    alpha, transform_power, _ = _effective_grid_params(params)
    return alpha * gamma + gamma**transform_power


def _jacobian_from_gamma(gamma: jnp.ndarray, params: ModifiedTanakaParams) -> jnp.ndarray:
    alpha, transform_power, _ = _effective_grid_params(params)
    return alpha + transform_power * gamma ** (transform_power - 1)


def _build_phi_grid(params: ModifiedTanakaParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha, transform_power, s_max = _effective_grid_params(params)
    s_pos = jnp.linspace(0.0, s_max, params.collocation_points)
    s_full = jnp.concatenate((-s_pos[:0:-1], s_pos))
    phi = alpha * s_full + s_full ** transform_power
    jacobian = alpha + transform_power * s_full ** (transform_power - 1)
    gamma_weights = _trapz_weights(s_full)
    positive = jnp.arange(params.collocation_points - 1, phi.shape[0])
    return s_full, phi, jacobian, gamma_weights, positive


def _refine_uniform_grid(grid: jnp.ndarray, substeps: int) -> jnp.ndarray:
    if substeps <= 1:
        return grid
    start = grid[0]
    stop = grid[-1]
    n_refined = (grid.shape[0] - 1) * substeps + 1
    return jnp.linspace(start, stop, n_refined, dtype=grid.dtype)


def _lagrange_interp_uniform(
    source_grid: jnp.ndarray,
    source_values: jnp.ndarray,
    eval_grid: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    degree = int(max(1, degree))
    stencil_size = degree + 1
    h = source_grid[1] - source_grid[0]
    scaled = (eval_grid - source_grid[0]) / h
    base = jnp.floor(scaled).astype(jnp.int32) - degree // 2
    base = jnp.clip(base, 0, source_grid.shape[0] - stencil_size)

    offsets = jnp.arange(stencil_size, dtype=jnp.int32)
    indices = base[:, None] + offsets[None, :]
    x_nodes = source_grid[indices]
    y_nodes = jnp.take(source_values, indices, axis=-1)
    x_eval = eval_grid[:, None]

    weights = jnp.ones((eval_grid.shape[0], stencil_size), dtype=source_values.dtype)
    for k in range(stencil_size):
        weight_k = jnp.ones(eval_grid.shape[0], dtype=source_values.dtype)
        for m in range(stencil_size):
            if m == k:
                continue
            weight_k = weight_k * (x_eval[:, 0] - x_nodes[:, m]) / (x_nodes[:, k] - x_nodes[:, m])
        weights = weights.at[:, k].set(weight_k)
    return jnp.sum(weights * y_nodes, axis=-1)


def _strip_hilbert_transform(
    tau: jnp.ndarray,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    params: ModifiedTanakaParams,
) -> jnp.ndarray:
    gamma_refined = _refine_uniform_grid(gamma, params.quadrature_substeps)
    tau_refined = _lagrange_interp_uniform(gamma, tau, gamma_refined, params.interpolation_degree)
    phi_refined = _phi_from_gamma(gamma_refined, params)
    jacobian_refined = _jacobian_from_gamma(gamma_refined, params)
    phi_weights_refined = jacobian_refined * _trapz_weights(gamma_refined)

    delta = phi_refined[None, :] - phi[:, None]
    denom = 2.0 * jnp.sinh(0.5 * jnp.pi * delta)
    tau_phi_refined = _gradient_last_axis(tau_refined, phi_refined)
    tau_phi_at_nodes = _lagrange_interp_uniform(gamma_refined, tau_phi_refined, gamma, params.interpolation_degree)

    kernel = jnp.where(delta != 0.0, 1.0 / denom, 0.0)
    regularized = (tau_refined[..., None, :] - tau[..., :, None]) * kernel
    diagonal_limit = tau_phi_at_nodes / jnp.pi
    regularized = jnp.where(delta == 0.0, diagonal_limit[..., :, None], regularized)
    return -jnp.sum(regularized * phi_weights_refined, axis=-1)


def _integrate_sin_theta(
    theta_pos: jnp.ndarray,
    gamma_pos: jnp.ndarray,
    params: ModifiedTanakaParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma_refined = _refine_uniform_grid(gamma_pos, params.quadrature_substeps)
    theta_refined = _lagrange_interp_uniform(gamma_pos, theta_pos, gamma_refined, params.interpolation_degree)
    jacobian_refined = _jacobian_from_gamma(gamma_refined, params)
    cumulative = _cumulative_trapezoid(jnp.sin(theta_refined) * jacobian_refined, gamma_refined)
    sample_step = max(1, params.quadrature_substeps)
    cumulative_nodes = cumulative[..., ::sample_step]
    return cumulative[..., -1], cumulative_nodes


def _initial_tau(phi: jnp.ndarray, qc: float, tau_seed: jnp.ndarray | None = None) -> jnp.ndarray:
    qc = jnp.asarray(qc, dtype=phi.dtype)
    qc_column = qc[..., None]
    if tau_seed is not None:
        q_seed = jnp.exp(tau_seed)
        q_seed_center = q_seed[..., phi.shape[0] // 2]
        deficit_scale = (1.0 - qc) / jnp.maximum(1.0 - q_seed_center, 1e-12)
        q = 1.0 - deficit_scale[..., None] * (1.0 - q_seed)
        q = jnp.clip(q, 1e-12, None)
        return jnp.log(q)

    width = 2.0
    q = 1.0 - (1.0 - qc_column) * jnp.exp(-(phi / width) ** 2)
    return jnp.log(q)


def _reflect_even(values_pos: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate((values_pos[..., :0:-1], values_pos), axis=-1)


def _single_qc_iteration(
    qc: float,
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
    tau_seed: jnp.ndarray | None = None,
    f2_seed: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    qc = jnp.asarray(qc, dtype=phi.dtype)
    gamma_pos = gamma[positive]
    tau = _initial_tau(phi, qc, tau_seed=tau_seed)
    if f2_seed is None:
        f2 = jnp.ones_like(qc, dtype=phi.dtype)
    else:
        f2 = jnp.asarray(f2_seed, dtype=phi.dtype)

    for _ in range(params.fixed_point_iterations):
        theta = _strip_hilbert_transform(tau, gamma, phi, params)
        theta_pos = jnp.take(theta, positive, axis=-1)

        integral, cumulative_nodes = _integrate_sin_theta(theta_pos, gamma_pos, params)
        f2_new = -3.0 * integral / (1.0 - qc**3)

        q3_pos = qc[..., None] ** 3 - (3.0 / f2_new[..., None]) * cumulative_nodes
        q_pos = jnp.cbrt(jnp.maximum(q3_pos, 1e-12))
        tau = _reflect_even(jnp.log(q_pos))
        if qc.ndim == 0 and float(jnp.abs(f2_new - f2)) < params.f2_tolerance:
            f2 = f2_new
            break
        f2 = f2_new

    theta = _strip_hilbert_transform(tau, gamma, phi, params)
    theta_pos = jnp.take(theta, positive, axis=-1)
    integral, cumulative_nodes = _integrate_sin_theta(theta_pos, gamma_pos, params)
    f2 = -3.0 * integral / (1.0 - qc**3)
    q3_pos = qc[..., None] ** 3 - (3.0 / f2[..., None]) * cumulative_nodes
    q_pos = jnp.cbrt(jnp.maximum(q3_pos, 1e-12))
    q = _reflect_even(q_pos)
    return tau, theta, q, f2


def _single_qc_iteration_fixed(
    qc: jnp.ndarray,
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del jacobian
    qc = jnp.asarray(qc, dtype=phi.dtype)
    gamma_pos = gamma[positive]
    tau = _initial_tau(phi, qc)
    f2 = jnp.ones_like(qc, dtype=phi.dtype)

    def body_fn(_: int, state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        tau, f2 = state
        theta = _strip_hilbert_transform(tau, gamma, phi, params)
        theta_pos = jnp.take(theta, positive, axis=-1)
        integral, cumulative_nodes = _integrate_sin_theta(theta_pos, gamma_pos, params)
        f2_new = -3.0 * integral / (1.0 - qc**3)
        q3_pos = qc[..., None] ** 3 - (3.0 / f2_new[..., None]) * cumulative_nodes
        q_pos = jnp.cbrt(jnp.maximum(q3_pos, 1e-12))
        tau_new = _reflect_even(jnp.log(q_pos))
        return tau_new, f2_new

    tau, f2 = jax.lax.fori_loop(0, params.fixed_point_iterations, body_fn, (tau, f2))
    theta = _strip_hilbert_transform(tau, gamma, phi, params)
    theta_pos = jnp.take(theta, positive, axis=-1)
    integral, cumulative_nodes = _integrate_sin_theta(theta_pos, gamma_pos, params)
    f2 = -3.0 * integral / (1.0 - qc**3)
    q3_pos = qc[..., None] ** 3 - (3.0 / f2[..., None]) * cumulative_nodes
    q_pos = jnp.cbrt(jnp.maximum(q3_pos, 1e-12))
    q = _reflect_even(q_pos)
    return tau, theta, q, f2


def _reconstruct_profile(
    q: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    positive: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi_pos = phi[positive]
    q_pos = jnp.take(q, positive, axis=-1)
    theta_pos = jnp.take(theta, positive, axis=-1)

    dx_dphi = jnp.cos(theta_pos) / q_pos
    dy_dphi = jnp.sin(theta_pos) / q_pos

    x_pos = _cumulative_trapezoid(dx_dphi, phi_pos)
    y_antideriv = _cumulative_trapezoid(dy_dphi, phi_pos)
    eta_pos = y_antideriv - y_antideriv[..., -1][..., None]

    x_full = jnp.concatenate((-x_pos[..., :0:-1], x_pos), axis=-1)
    eta_full = jnp.concatenate((eta_pos[..., :0:-1], eta_pos), axis=-1)
    amplitude = eta_pos[..., 0]
    return x_full, eta_full, amplitude


def _amplitude_for_qc(
    qc: float,
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
    seed: ModifiedTanakaSeed | None = None,
) -> tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray]]:
    tau_seed = None
    f2_seed = None
    if seed is not None:
        tau_seed = jnp.interp(phi, seed.phi_profile, seed.tau_profile)
        f2_seed = seed.f2

    tau, theta, q, f2 = _single_qc_iteration(
        qc,
        params,
        gamma,
        phi,
        jacobian,
        positive,
        tau_seed=tau_seed,
        f2_seed=f2_seed,
    )
    x_profile, eta_profile, amplitude = _reconstruct_profile(q, theta, phi, positive)
    return amplitude, (tau, theta, q, f2, x_profile, eta_profile)


def _amplitude_for_qc_batched(
    qc: jnp.ndarray,
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    tau, theta, q, f2 = _single_qc_iteration_fixed(qc, params, gamma, phi, jacobian, positive)
    x_profile, eta_profile, amplitude = _reconstruct_profile(q, theta, phi, positive)
    return amplitude, (tau, theta, q, f2, x_profile, eta_profile)


def _solve_for_qc(
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
    seed: ModifiedTanakaSeed | None = None,
) -> tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray]]:
    target = params.amplitude / params.depth
    if seed is None:
        low = params.qc_lower
        high = params.qc_upper
    else:
        low = max(params.qc_lower, seed.qc - 0.08)
        high = min(params.qc_upper, seed.qc + 0.08)

    amp_low, payload_low = _amplitude_for_qc(low, params, gamma, phi, jacobian, positive, seed=seed)
    amp_high, payload_high = _amplitude_for_qc(high, params, gamma, phi, jacobian, positive, seed=seed)

    while amp_low < target and low > params.qc_lower:
        high = low
        amp_high, payload_high = amp_low, payload_low
        low = max(params.qc_lower, low - 0.08)
        amp_low, payload_low = _amplitude_for_qc(low, params, gamma, phi, jacobian, positive, seed=seed)

    while amp_high > target and high < params.qc_upper:
        low = high
        amp_low, payload_low = amp_high, payload_high
        high = min(params.qc_upper, high + 0.08)
        amp_high, payload_high = _amplitude_for_qc(high, params, gamma, phi, jacobian, positive, seed=seed)

    for _ in range(params.outer_iterations):
        mid = 0.5 * (low + high)
        amp_mid, payload_mid = _amplitude_for_qc(mid, params, gamma, phi, jacobian, positive, seed=seed)
        if amp_mid > target:
            low = mid
            amp_low, payload_low = amp_mid, payload_mid
        else:
            high = mid
            amp_high, payload_high = amp_mid, payload_mid

    if abs(amp_low - target) < abs(amp_high - target):
        return low, payload_low
    return high, payload_high


def _solve_for_qc_batched(
    amplitudes: jnp.ndarray,
    params: ModifiedTanakaParams,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    jacobian: jnp.ndarray,
    positive: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    targets = jnp.asarray(amplitudes, dtype=phi.dtype) / params.depth
    low = jnp.full_like(targets, params.qc_lower)
    high = jnp.full_like(targets, params.qc_upper)

    def body_fn(
        _: int,
        state: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        low, high = state
        mid = 0.5 * (low + high)
        amp_mid, _ = _amplitude_for_qc_batched(mid, params, gamma, phi, jacobian, positive)
        go_lower = amp_mid > targets
        low = jnp.where(go_lower, mid, low)
        high = jnp.where(go_lower, high, mid)
        return low, high

    low, high = jax.lax.fori_loop(0, params.outer_iterations, body_fn, (low, high))

    amp_low, payload_low = _amplitude_for_qc_batched(low, params, gamma, phi, jacobian, positive)
    amp_high, payload_high = _amplitude_for_qc_batched(high, params, gamma, phi, jacobian, positive)
    choose_low = jnp.abs(amp_low - targets) < jnp.abs(amp_high - targets)
    qc = jnp.where(choose_low, low, high)

    def select_payload_component(low_component: jnp.ndarray, high_component: jnp.ndarray) -> jnp.ndarray:
        if low_component.ndim == 1:
            return jnp.where(choose_low, low_component, high_component)
        return jnp.where(choose_low[:, None], low_component, high_component)

    payload = tuple(select_payload_component(low_component, high_component) for low_component, high_component in zip(payload_low, payload_high))
    return qc, payload


def _interpolate_to_periodic_grid(
    x_profile: jnp.ndarray,
    eta_profile: jnp.ndarray,
    params: ModifiedTanakaParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = params.length / params.nx
    x_periodic = dx * jnp.arange(params.nx)
    x_shifted = x_profile * params.depth + params.center
    eta_periodic = jnp.interp(x_periodic, x_shifted, eta_profile * params.depth, left=0.0, right=0.0)
    return x_periodic, eta_periodic


def _interpolate_to_periodic_grid_batched(
    x_profile: jnp.ndarray,
    eta_profile: jnp.ndarray,
    centers: jnp.ndarray,
    params: ModifiedTanakaParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = params.length / params.nx
    x_periodic = dx * jnp.arange(params.nx, dtype=x_profile.dtype)
    centers = jnp.asarray(centers, dtype=x_profile.dtype)

    def interp_one(x_profile_row: jnp.ndarray, eta_profile_row: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
        x_shifted = x_profile_row * params.depth + center
        return jnp.interp(x_periodic, x_shifted, eta_profile_row * params.depth, left=0.0, right=0.0)

    eta_periodic = jax.vmap(interp_one)(x_profile, eta_profile, centers)
    return x_periodic, eta_periodic


def _solve_surface_potential(
    eta_periodic: jnp.ndarray,
    speed: float,
    params: ModifiedTanakaParams,
    direction: int | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    solver_params = make_solver_params(
        nx=params.nx,
        length=params.length,
        depth=params.depth,
        gravity=params.gravity,
        dno_order=params.dno_order,
        pad_factor=params.pad_factor,
        filter_fraction=1.0,
    )
    direction_arr = jnp.asarray(params.direction if direction is None else direction, dtype=eta_periodic.dtype)
    speed_arr = jnp.asarray(speed, dtype=eta_periodic.dtype)
    signed_speed = direction_arr * speed_arr
    signed_speed_column = signed_speed[..., None]
    direction_column = direction_arr[..., None]
    eta_x = spectral_dx(eta_periodic, solver_params.k)
    rhs = -signed_speed_column * eta_x

    radical = (1.0 + eta_x**2) * (signed_speed_column**2 - 2.0 * params.gravity * eta_periodic)
    xi_x = signed_speed_column - direction_column * jnp.sqrt(jnp.maximum(radical, 0.0))

    inv_ik = jnp.where(solver_params.k != 0.0, 1.0 / (1j * solver_params.k), 0.0)
    xi = myifft(inv_ik * myfft(xi_x, xi_x.shape[-1]))
    xi = xi - jnp.mean(xi, axis=-1, keepdims=True)
    gxi = rhs
    return xi, gxi


@partial(jax.jit, static_argnames=("params",))
def _solve_modified_tanaka_batched_core(
    amplitudes: jnp.ndarray,
    centers: jnp.ndarray,
    directions: jnp.ndarray,
    params: ModifiedTanakaParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gamma, phi, jacobian, _, positive = _build_phi_grid(params)
    qc, payload = _solve_for_qc_batched(amplitudes, params, gamma, phi, jacobian, positive)
    tau, theta, q, f2, x_profile, eta_profile = payload

    x_periodic, eta_periodic = _interpolate_to_periodic_grid_batched(x_profile, eta_profile, centers, params)
    froude = jnp.sqrt(f2)
    speed = froude * jnp.sqrt(params.gravity * params.depth)
    xi_periodic, gxi_periodic = _solve_surface_potential(eta_periodic, speed, params, direction=directions)
    return qc, froude, speed, tau, x_profile, eta_profile, phi, q, theta, x_periodic, eta_periodic, xi_periodic, gxi_periodic


def solve_modified_tanaka(params: ModifiedTanakaParams, seed: ModifiedTanakaSeed | None = None) -> ModifiedTanakaSolution:
    if seed is None:
        batched = solve_modified_tanaka_batched(
            params,
            jnp.asarray([params.amplitude], dtype=REAL_DTYPE),
            centers=jnp.asarray([params.center], dtype=REAL_DTYPE),
            directions=jnp.asarray([params.direction], dtype=REAL_DTYPE),
        )
        return ModifiedTanakaSolution(
            amplitude=float(batched.amplitudes[0]),
            qc=float(batched.qc[0]),
            froude=float(batched.froude[0]),
            speed=float(batched.speed[0]),
            tau_profile=batched.tau_profile[0],
            x_profile=batched.x_profile[0],
            eta_profile=batched.eta_profile[0],
            phi_profile=batched.phi_profile,
            q_profile=batched.q_profile[0],
            theta_profile=batched.theta_profile[0],
            x_periodic=batched.x_periodic,
            eta_periodic=batched.eta_periodic[0],
            xi_periodic=batched.xi_periodic[0],
            gxi_periodic=batched.gxi_periodic[0],
        )

    gamma, phi, jacobian, gamma_weights, positive = _build_phi_grid(params)
    qc, payload = _solve_for_qc(params, gamma, phi, jacobian, positive, seed=seed)
    tau, theta, q, f2, x_profile, eta_profile = payload

    x_periodic, eta_periodic = _interpolate_to_periodic_grid(x_profile, eta_profile, params)
    speed = float(jnp.sqrt(f2) * jnp.sqrt(params.gravity * params.depth))
    xi_periodic, gxi_periodic = _solve_surface_potential(eta_periodic, speed, params, direction=params.direction)
    validate_solved_amplitudes(
        (eta_profile * params.depth)[None, :],
        jnp.asarray((params.amplitude,), dtype=eta_profile.dtype),
    )

    return ModifiedTanakaSolution(
        amplitude=params.amplitude,
        qc=float(qc),
        froude=float(jnp.sqrt(f2)),
        speed=speed,
        tau_profile=tau,
        x_profile=x_profile * params.depth,
        eta_profile=eta_profile * params.depth,
        phi_profile=phi,
        q_profile=q,
        theta_profile=theta,
        x_periodic=x_periodic,
        eta_periodic=eta_periodic,
        xi_periodic=xi_periodic,
        gxi_periodic=gxi_periodic,
    )


def solve_modified_tanaka_batched(
    template_params: ModifiedTanakaParams,
    amplitudes: jnp.ndarray,
    centers: jnp.ndarray | None = None,
    directions: jnp.ndarray | None = None,
) -> ModifiedTanakaBatchSolution:
    amplitudes = jnp.asarray(amplitudes, dtype=REAL_DTYPE)
    if amplitudes.ndim != 1:
        raise ValueError("amplitudes must have shape (batch,)")

    if centers is None:
        centers = jnp.full_like(amplitudes, template_params.center)
    else:
        centers = jnp.asarray(centers, dtype=amplitudes.dtype)

    if directions is None:
        directions = jnp.full(amplitudes.shape, template_params.direction, dtype=amplitudes.dtype)
    else:
        directions = jnp.asarray(directions, dtype=amplitudes.dtype)

    jit_params = replace(template_params, amplitude=0.0, center=0.0, direction=1)
    qc, froude, speed, tau, x_profile, eta_profile, phi, q, theta, x_periodic, eta_periodic, xi_periodic, gxi_periodic = _solve_modified_tanaka_batched_core(
        amplitudes,
        centers,
        directions,
        jit_params,
    )
    validate_solved_amplitudes(
        eta_profile * template_params.depth,
        amplitudes,
    )

    return ModifiedTanakaBatchSolution(
        amplitudes=amplitudes,
        qc=qc,
        froude=froude,
        speed=speed,
        tau_profile=tau,
        x_profile=x_profile * template_params.depth,
        eta_profile=eta_profile * template_params.depth,
        phi_profile=phi,
        q_profile=q,
        theta_profile=theta,
        x_periodic=x_periodic,
        eta_periodic=eta_periodic,
        xi_periodic=xi_periodic,
        gxi_periodic=gxi_periodic,
    )


def make_tanaka_seed(solution: ModifiedTanakaSolution) -> ModifiedTanakaSeed:
    return ModifiedTanakaSeed(
        qc=float(solution.qc),
        f2=float(solution.froude**2),
        phi_profile=solution.phi_profile,
        tau_profile=solution.tau_profile,
    )


def solve_tanaka_branch(amplitudes: list[float] | tuple[float, ...], template_params: ModifiedTanakaParams) -> list[ModifiedTanakaSolution]:
    seed: ModifiedTanakaSeed | None = None
    branch: list[ModifiedTanakaSolution] = []
    for amplitude in amplitudes:
        params = replace(template_params, amplitude=float(amplitude))
        solution = solve_modified_tanaka(params, seed=seed)
        branch.append(solution)
        seed = make_tanaka_seed(solution)
    return branch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a solitary-wave initial condition with the modified Tanaka iteration in JAX.")
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--center", type=float, default=43.5625)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--grid_mode", choices=("auto", "manual"), default="manual")
    parser.add_argument("--collocation_points", type=int, default=257)
    parser.add_argument("--quadrature_substeps", type=int, default=4)
    parser.add_argument("--interpolation_degree", type=int, default=3)
    parser.add_argument("--s_max", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--transform_power", type=int, default=5)
    parser.add_argument("--qc_lower", type=float, default=0.2)
    parser.add_argument("--qc_upper", type=float, default=DEFAULT_QC_UPPER)
    parser.add_argument(
        "--outer_iterations",
        type=int,
        default=DEFAULT_OUTER_ITERATIONS,
    )
    parser.add_argument("--fixed_point_iterations", type=int, default=80)
    parser.add_argument("--f2_tolerance", type=float, default=1e-10)
    parser.add_argument("--cg_maxiter", type=int, default=400)
    parser.add_argument("--cg_tol", type=float, default=1e-10)
    parser.add_argument("--output", default="outputs/tanaka_ic.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = replace(
        make_default_tanaka_template(
            depth=args.depth,
            gravity=args.gravity,
            direction=args.direction,
            nx=args.nx,
            length=args.length,
            center=args.center,
            dno_order=args.dno_order,
            pad_factor=args.pad_factor,
            grid_mode=args.grid_mode,
            collocation_points=args.collocation_points,
            quadrature_substeps=args.quadrature_substeps,
            interpolation_degree=args.interpolation_degree,
            s_max=args.s_max,
            alpha=args.alpha,
            transform_power=args.transform_power,
            qc_lower=args.qc_lower,
            qc_upper=args.qc_upper,
            outer_iterations=args.outer_iterations,
            fixed_point_iterations=args.fixed_point_iterations,
            f2_tolerance=args.f2_tolerance,
            cg_maxiter=args.cg_maxiter,
            cg_tol=args.cg_tol,
        ),
        amplitude=args.amplitude,
    )
    solution = solve_modified_tanaka(params)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(
        output_path,
        amplitude=solution.amplitude,
        qc=solution.qc,
        froude=solution.froude,
        speed=solution.speed,
        x_profile=solution.x_profile,
        eta_profile=solution.eta_profile,
        phi_profile=solution.phi_profile,
        q_profile=solution.q_profile,
        theta_profile=solution.theta_profile,
        x_periodic=solution.x_periodic,
        eta_periodic=solution.eta_periodic,
        xi_periodic=solution.xi_periodic,
        gxi_periodic=solution.gxi_periodic,
        params_json=json.dumps(asdict(params)),
    )
    print(output_path)


if __name__ == "__main__":
    main()
