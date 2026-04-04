from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from .dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol, myfft, myifft


class State(NamedTuple):
    eta: jnp.ndarray
    xi: jnp.ndarray


class SpectralState(NamedTuple):
    eta_hat: jnp.ndarray
    xi_hat: jnp.ndarray


class SolverParams(NamedTuple):
    nx: int
    length: float
    depth: float
    gravity: float
    dno_order: int
    pad_factor: int
    filter_fraction: float
    k: jnp.ndarray
    g0: jnp.ndarray


class RolloutSettings(NamedTuple):
    dno_order: int
    pad_factor: int
    filter_fraction: float
    method: str
    substeps_per_interval: int
    implicit_iterations: int
    implicit_relaxation: float
    zero_mean_xi: bool


def make_normalized_rollout_settings() -> RolloutSettings:
    return RolloutSettings(
        dno_order=6,
        pad_factor=8,
        filter_fraction=2.0 / 3.0,
        method="gl2_if",
        substeps_per_interval=8,
        implicit_iterations=4,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )


def make_solver_params(
    nx: int,
    length: float,
    depth: float,
    gravity: float = 1.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    filter_fraction: float = 1.0,
) -> SolverParams:
    _, k = build_grid(nx, length)
    g0 = make_linear_dno_symbol(k, depth)
    return SolverParams(
        nx=nx,
        length=length,
        depth=depth,
        gravity=gravity,
        dno_order=dno_order,
        pad_factor=pad_factor,
        filter_fraction=filter_fraction,
        k=k,
        g0=g0,
    )


def cast_state_dtype(state: State, dtype: jnp.dtype) -> State:
    return State(
        eta=jnp.asarray(state.eta, dtype=dtype),
        xi=jnp.asarray(state.xi, dtype=dtype),
    )


def cast_solver_params_dtype(params: SolverParams, dtype: jnp.dtype) -> SolverParams:
    return SolverParams(
        nx=params.nx,
        length=params.length,
        depth=params.depth,
        gravity=params.gravity,
        dno_order=params.dno_order,
        pad_factor=params.pad_factor,
        filter_fraction=params.filter_fraction,
        k=jnp.asarray(params.k, dtype=dtype),
        g0=jnp.asarray(params.g0, dtype=dtype),
    )


def spectral_dx(field: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    return myifft(1j * k * myfft(field, field.shape[-1]))


def linear_dno_action(xi: jnp.ndarray, g0: jnp.ndarray) -> jnp.ndarray:
    return myifft(g0 * myfft(xi, xi.shape[-1]))


def apply_lowpass(field: jnp.ndarray, k: jnp.ndarray, filter_fraction: float = 1.0) -> jnp.ndarray:
    if filter_fraction >= 1.0:
        return field

    k_max = jnp.max(jnp.abs(k))
    cutoff = filter_fraction * k_max
    mask = (jnp.abs(k) <= cutoff).astype(myfft(field, field.shape[-1]).dtype)
    return myifft(mask * myfft(field, field.shape[-1]))


def _state_to_hat(state: State, nx: int) -> SpectralState:
    return SpectralState(
        eta_hat=myfft(state.eta, nx),
        xi_hat=myfft(state.xi, nx),
    )


def _hat_to_state(state_hat: SpectralState) -> State:
    return State(
        eta=myifft(state_hat.eta_hat),
        xi=myifft(state_hat.xi_hat),
    )


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, scale: float):
    return jax.tree_util.tree_map(lambda x: scale * x, a)


def _tree_axpy(base, scale: float, delta):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, base, delta)


def project_zero_mean_xi(state: State) -> State:
    return State(eta=state.eta, xi=state.xi - jnp.mean(state.xi, axis=-1, keepdims=True))


def apply_linear_flow_hat(state_hat: SpectralState, tau: float | jnp.ndarray, params: SolverParams) -> SpectralState:
    omega = jnp.sqrt(params.gravity * params.g0)
    coswt = jnp.cos(omega * tau)
    sin_over_omega = jnp.where(omega > 0, jnp.sin(omega * tau) / omega, 0.0)

    eta_hat = coswt * state_hat.eta_hat + sin_over_omega * params.g0 * state_hat.xi_hat
    xi_hat = coswt * state_hat.xi_hat - sin_over_omega * params.gravity * state_hat.eta_hat
    return SpectralState(eta_hat=eta_hat, xi_hat=xi_hat)


def rhs_full(state: State, params: SolverParams) -> State:
    eta_x = spectral_dx(state.eta, params.k)
    xi_x = spectral_dx(state.xi, params.k)
    gxi = dno_series_eval(
        state.eta,
        state.xi,
        params.k,
        params.depth,
        params.dno_order,
        pad_factor=params.pad_factor,
    )

    eta_t = gxi
    numerator = gxi + eta_x * xi_x
    xi_t = -params.gravity * state.eta - 0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    return State(eta=eta_t, xi=xi_t)


def rhs_nonlinear(state: State, params: SolverParams) -> State:
    eta_x = spectral_dx(state.eta, params.k)
    xi_x = spectral_dx(state.xi, params.k)
    gxi = dno_series_eval(
        state.eta,
        state.xi,
        params.k,
        params.depth,
        params.dno_order,
        pad_factor=params.pad_factor,
    )
    linear_gxi = linear_dno_action(state.xi, params.g0)

    eta_t = gxi - linear_gxi
    numerator = gxi + eta_x * xi_x
    xi_t = -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    return State(eta=eta_t, xi=xi_t)


def rhs_nonlinear_if(v_hat: SpectralState, t: float | jnp.ndarray, params: SolverParams) -> SpectralState:
    physical_hat = apply_linear_flow_hat(v_hat, t, params)
    physical_state = _hat_to_state(physical_hat)
    nonlinear_state = rhs_nonlinear(physical_state, params)
    nonlinear_hat = _state_to_hat(nonlinear_state, params.nx)
    return apply_linear_flow_hat(nonlinear_hat, -t, params)


def rk4_if_step(state: State, t: float, dt: float, params: SolverParams) -> State:
    state_hat = _state_to_hat(state, params.nx)
    v0 = apply_linear_flow_hat(state_hat, -t, params)

    k1 = rhs_nonlinear_if(v0, t, params)
    k2 = rhs_nonlinear_if(_tree_axpy(v0, 0.5 * dt, k1), t + 0.5 * dt, params)
    k3 = rhs_nonlinear_if(_tree_axpy(v0, 0.5 * dt, k2), t + 0.5 * dt, params)
    k4 = rhs_nonlinear_if(_tree_axpy(v0, dt, k3), t + dt, params)

    v1 = _tree_add(
        v0,
        _tree_scale(
            _tree_add(
                _tree_add(k1, _tree_scale(k2, 2.0)),
                _tree_add(_tree_scale(k3, 2.0), k4),
            ),
            dt / 6.0,
        ),
    )

    next_state = _hat_to_state(apply_linear_flow_hat(v1, t + dt, params))
    if params.filter_fraction < 1.0:
        next_state = State(
            eta=apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def implicit_midpoint_if_step(
    state: State,
    t: float,
    dt: float,
    params: SolverParams,
    iterations: int = 8,
    relaxation: float = 1.0,
) -> State:
    state_hat = _state_to_hat(state, params.nx)
    v0 = apply_linear_flow_hat(state_hat, -t, params)
    midpoint_time = t + 0.5 * dt

    def body_fn(_: int, stage: SpectralState) -> SpectralState:
        rhs = rhs_nonlinear_if(stage, midpoint_time, params)
        candidate = _tree_axpy(v0, 0.5 * dt, rhs)
        if relaxation == 1.0:
            return candidate
        return jax.tree_util.tree_map(lambda s, c: (1.0 - relaxation) * s + relaxation * c, stage, candidate)

    stage = jax.lax.fori_loop(0, iterations, body_fn, v0)
    v1 = jax.tree_util.tree_map(lambda mid, start: 2.0 * mid - start, stage, v0)
    next_state = _hat_to_state(apply_linear_flow_hat(v1, t + dt, params))
    if params.filter_fraction < 1.0:
        next_state = State(
            eta=apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def gauss_legendre_2_if_step(
    state: State,
    t: float,
    dt: float,
    params: SolverParams,
    iterations: int = 4,
    relaxation: float = 1.0,
) -> State:
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = _state_to_hat(state, params.nx)
    v0 = apply_linear_flow_hat(state_hat, -t, params)

    def body_fn(_: int, stages: tuple[SpectralState, SpectralState]) -> tuple[SpectralState, SpectralState]:
        stage1, stage2 = stages
        f1 = rhs_nonlinear_if(stage1, t + c1 * dt, params)
        f2 = rhs_nonlinear_if(stage2, t + c2 * dt, params)
        candidate1 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, a11), _tree_scale(f2, a12)), dt))
        candidate2 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, a21), _tree_scale(f2, a22)), dt))
        if relaxation == 1.0:
            return candidate1, candidate2
        next_stage1 = jax.tree_util.tree_map(
            lambda s, c: (1.0 - relaxation) * s + relaxation * c,
            stage1,
            candidate1,
        )
        next_stage2 = jax.tree_util.tree_map(
            lambda s, c: (1.0 - relaxation) * s + relaxation * c,
            stage2,
            candidate2,
        )
        return next_stage1, next_stage2

    stage1, stage2 = jax.lax.fori_loop(0, iterations, body_fn, (v0, v0))
    f1 = rhs_nonlinear_if(stage1, t + c1 * dt, params)
    f2 = rhs_nonlinear_if(stage2, t + c2 * dt, params)
    v1 = _tree_add(v0, _tree_scale(_tree_add(f1, f2), 0.5 * dt))

    next_state = _hat_to_state(apply_linear_flow_hat(v1, t + dt, params))
    if params.filter_fraction < 1.0:
        next_state = State(
            eta=apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def take_step(
    state: State,
    t: float,
    dt: float,
    params: SolverParams,
    method: str = "gl2_if",
    implicit_iterations: int = 8,
    implicit_relaxation: float = 1.0,
) -> State:
    if method == "rk4_if":
        return rk4_if_step(state, t, dt, params)
    if method == "implicit_midpoint_if":
        return implicit_midpoint_if_step(
            state,
            t,
            dt,
            params,
            iterations=implicit_iterations,
            relaxation=implicit_relaxation,
        )
    if method == "gl2_if":
        return gauss_legendre_2_if_step(
            state,
            t,
            dt,
            params,
            iterations=implicit_iterations,
            relaxation=implicit_relaxation,
        )
    raise ValueError(f"Unknown time-stepping method {method!r}")


def rollout(
    initial_state: State,
    times: jnp.ndarray,
    params: SolverParams,
    save_gxi: bool = True,
    substeps_per_interval: int = 1,
    method: str = "gl2_if",
    implicit_iterations: int = 8,
    implicit_relaxation: float = 1.0,
    zero_mean_xi: bool = False,
) -> dict[str, jnp.ndarray]:
    times = jnp.asarray(times)

    initial_state = State(eta=jnp.asarray(initial_state.eta), xi=jnp.asarray(initial_state.xi))
    if zero_mean_xi:
        initial_state = project_zero_mean_xi(initial_state)

    if times.shape[0] == 1:
        eta = initial_state.eta[jnp.newaxis, :]
        xi = initial_state.xi[jnp.newaxis, :]
    else:
        dts = times[1:] - times[:-1]

        def step_fn(carry: State, data: tuple[jnp.ndarray, jnp.ndarray]):
            current_t, dt = data
            if substeps_per_interval == 1:
                next_state = take_step(
                    carry,
                    current_t,
                    dt,
                    params,
                    method=method,
                    implicit_iterations=implicit_iterations,
                    implicit_relaxation=implicit_relaxation,
                )
                if zero_mean_xi:
                    next_state = project_zero_mean_xi(next_state)
            else:
                dt_sub = dt / substeps_per_interval

                def body_fn(sub_idx: int, subcarry: State) -> State:
                    sub_t = current_t + dt_sub * sub_idx
                    next_state = take_step(
                        subcarry,
                        sub_t,
                        dt_sub,
                        params,
                        method=method,
                        implicit_iterations=implicit_iterations,
                        implicit_relaxation=implicit_relaxation,
                    )
                    if zero_mean_xi:
                        next_state = project_zero_mean_xi(next_state)
                    return next_state

                next_state = jax.lax.fori_loop(0, substeps_per_interval, body_fn, carry)
            return next_state, next_state

        _, saved_states = jax.lax.scan(step_fn, initial_state, (times[:-1], dts))
        eta = jnp.concatenate((initial_state.eta[jnp.newaxis, :], saved_states.eta), axis=0)
        xi = jnp.concatenate((initial_state.xi[jnp.newaxis, :], saved_states.xi), axis=0)

    payload: dict[str, jnp.ndarray] = {"times": times, "eta": eta, "xi": xi}
    if save_gxi:
        gxi = jax.vmap(
            lambda eta_row, xi_row: dno_series_eval(
                eta_row,
                xi_row,
                params.k,
                params.depth,
                params.dno_order,
                pad_factor=params.pad_factor,
            )
        )(eta, xi)
        payload["gxi"] = gxi
    return payload


def batched_rollout(
    initial_state: State,
    times: jnp.ndarray,
    params: SolverParams,
    save_gxi: bool = True,
    substeps_per_interval: int = 1,
    method: str = "gl2_if",
    implicit_iterations: int = 8,
    implicit_relaxation: float = 1.0,
    zero_mean_xi: bool = False,
) -> dict[str, jnp.ndarray]:
    """Batch rollout entry point for initial states of shape (batch, nx) and shared times."""
    return rollout(
        initial_state,
        times,
        params,
        save_gxi=save_gxi,
        substeps_per_interval=substeps_per_interval,
        method=method,
        implicit_iterations=implicit_iterations,
        implicit_relaxation=implicit_relaxation,
        zero_mean_xi=zero_mean_xi,
    )
