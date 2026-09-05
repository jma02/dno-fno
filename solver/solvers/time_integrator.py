from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from .dno_series_jax import (
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
    myfft,
    myifft,
)


class State(NamedTuple):
    eta: jnp.ndarray
    xi: jnp.ndarray


class SpectralState(NamedTuple):
    eta_hat: jnp.ndarray
    xi_hat: jnp.ndarray


class ImplicitStepTelemetry(NamedTuple):
    """Per-sample diagnostics for one implicit Runge--Kutta step."""

    residual: jnp.ndarray
    iterations: jnp.ndarray
    converged: jnp.ndarray
    stage_finite: jnp.ndarray
    state_finite: jnp.ndarray
    hit_iteration_cap: jnp.ndarray


class ImplicitStepResult(NamedTuple):
    state: State
    telemetry: ImplicitStepTelemetry


class _GL2IterationCarry(NamedTuple):
    stage1: SpectralState
    stage2: SpectralState
    candidate1: SpectralState
    candidate2: SpectralState
    f1: SpectralState
    f2: SpectralState
    residual: jnp.ndarray
    iteration_count: jnp.ndarray
    active: jnp.ndarray


class SolverParams(NamedTuple):
    nx: int
    length: float
    depth: float | jax.Array
    gravity: float
    dno_order: int
    pad_factor: int
    filter_fraction: float
    k: jnp.ndarray
    g0: jnp.ndarray
    nonlinear_ramp_time: float | jnp.ndarray | None = None
    nonlinear_ramp_order: int = 4


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
    depth: float | jax.Array,
    gravity: float = 1.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    filter_fraction: float = 1.0,
    nonlinear_ramp_time: float | jnp.ndarray | None = None,
    nonlinear_ramp_order: int = 4,
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
        nonlinear_ramp_time=nonlinear_ramp_time,
        nonlinear_ramp_order=nonlinear_ramp_order,
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
        nonlinear_ramp_time=(
            None
            if params.nonlinear_ramp_time is None
            else jnp.asarray(params.nonlinear_ramp_time, dtype=dtype)
        ),
        nonlinear_ramp_order=params.nonlinear_ramp_order,
    )


def spectral_dx(field: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    return myifft(1j * k * myfft(field, field.shape[-1]))


def _spectral_upsample_real(field: jnp.ndarray, output_nx: int) -> jnp.ndarray:
    """Interpolate a real periodic field onto a larger Fourier grid."""
    input_nx = field.shape[-1]
    spectrum = jnp.fft.rfft(field, axis=-1)
    spectrum = spectrum.at[..., input_nx // 2].set(0)
    return jnp.fft.irfft(spectrum, n=output_nx, axis=-1) * (output_nx / input_nx)


def _spectral_truncate_real(field: jnp.ndarray, output_nx: int) -> jnp.ndarray:
    """Project a real periodic field from a larger grid onto ``output_nx``."""
    input_nx = field.shape[-1]
    spectrum = jnp.fft.rfft(field, axis=-1)[..., : output_nx // 2 + 1]
    spectrum = spectrum.at[..., output_nx // 2].set(0)
    return jnp.fft.irfft(spectrum, n=output_nx, axis=-1) * (output_nx / input_nx)


def dealiased_zakharov_xi_rhs(
    eta_x: jnp.ndarray,
    xi_x: jnp.ndarray,
    gxi: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the nonlinear Zakharov ``xi_t`` on a two-times Fourier grid.

    JCP09 evaluates the nonlinear equations-of-motion terms on spectra extended
    by a factor of two.  The final truncation removes product modes that would
    otherwise alias into the retained base-grid spectrum.  Leading batch
    dimensions are preserved.
    """
    nx = eta_x.shape[-1]
    padded_nx = 2 * nx
    eta_x_padded = _spectral_upsample_real(eta_x, padded_nx)
    xi_x_padded = _spectral_upsample_real(xi_x, padded_nx)
    gxi_padded = _spectral_upsample_real(gxi, padded_nx)

    numerator = gxi_padded + eta_x_padded * xi_x_padded
    xi_t_padded = -0.5 * xi_x_padded**2 + 0.5 * numerator**2 / (1.0 + eta_x_padded**2)
    return _spectral_truncate_real(xi_t_padded, nx)


def linear_dno_action(xi: jnp.ndarray, g0: jnp.ndarray) -> jnp.ndarray:
    return myifft(g0 * myfft(xi, xi.shape[-1]))


def apply_lowpass(
    field: jnp.ndarray, k: jnp.ndarray, filter_fraction: float = 1.0
) -> jnp.ndarray:
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


def _tree_add(a: SpectralState, b: SpectralState) -> SpectralState:
    return SpectralState(
        eta_hat=a.eta_hat + b.eta_hat,
        xi_hat=a.xi_hat + b.xi_hat,
    )


def _tree_scale(state: SpectralState, scale: float | jax.Array) -> SpectralState:
    return SpectralState(
        eta_hat=scale * state.eta_hat,
        xi_hat=scale * state.xi_hat,
    )


def _tree_axpy(
    base: SpectralState,
    scale: float | jax.Array,
    delta: SpectralState,
) -> SpectralState:
    return SpectralState(
        eta_hat=base.eta_hat + scale * delta.eta_hat,
        xi_hat=base.xi_hat + scale * delta.xi_hat,
    )


def _sample_l2_norm(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.abs(field) ** 2, axis=-1))


def _relative_component_defect(
    current: jnp.ndarray,
    mapped: jnp.ndarray,
) -> jnp.ndarray:
    """Relative fixed-point defect for one physical component."""

    numerator = _sample_l2_norm(mapped - current)
    denominator = jnp.maximum(
        _sample_l2_norm(current),
        _sample_l2_norm(mapped),
    )
    return numerator / jnp.maximum(
        denominator,
        jnp.asarray(jnp.finfo(numerator.dtype).tiny, dtype=numerator.dtype),
    )


def relative_implicit_stage_residual(
    stage1: SpectralState,
    stage2: SpectralState,
    mapped1: SpectralState,
    mapped2: SpectralState,
) -> jnp.ndarray:
    """Return a dimensionless GL2 stage-equation residual per sample.

    Each of the four stage/component defects is divided by the larger of the
    current and mapped component norms before taking their maximum.  Separate
    normalization of ``eta`` and ``xi`` avoids adding quantities with
    different physical units.  Leading batch dimensions are preserved.
    """

    component_defects = jnp.stack(
        (
            _relative_component_defect(stage1.eta_hat, mapped1.eta_hat),
            _relative_component_defect(stage1.xi_hat, mapped1.xi_hat),
            _relative_component_defect(stage2.eta_hat, mapped2.eta_hat),
            _relative_component_defect(stage2.xi_hat, mapped2.xi_hat),
        ),
        axis=0,
    )
    return jnp.max(component_defects, axis=0)


def _spectral_states_finite(*states: SpectralState) -> jnp.ndarray:
    finite_components = tuple(
        jnp.all(jnp.isfinite(component), axis=-1)
        for state in states
        for component in state
    )
    return jnp.all(jnp.stack(finite_components, axis=0), axis=0)


def _state_finite(state: State) -> jnp.ndarray:
    return jnp.logical_and(
        jnp.all(jnp.isfinite(state.eta), axis=-1),
        jnp.all(jnp.isfinite(state.xi), axis=-1),
    )


def project_zero_mean_xi(state: State) -> State:
    return State(
        eta=state.eta, xi=state.xi - jnp.mean(state.xi, axis=-1, keepdims=True)
    )


def apply_linear_flow_hat(
    state_hat: SpectralState, tau: float | jnp.ndarray, params: SolverParams
) -> SpectralState:
    omega = jnp.sqrt(params.gravity * params.g0)
    coswt = jnp.cos(omega * tau)
    sin_over_omega = jnp.where(omega > 0, jnp.sin(omega * tau) / omega, 0.0)

    eta_hat = coswt * state_hat.eta_hat + sin_over_omega * params.g0 * state_hat.xi_hat
    xi_hat = (
        coswt * state_hat.xi_hat - sin_over_omega * params.gravity * state_hat.eta_hat
    )
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
    xi_t = (
        -params.gravity * state.eta
        - 0.5 * xi_x**2
        + 0.5 * numerator**2 / (1.0 + eta_x**2)
    )
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
    xi_t = dealiased_zakharov_xi_rhs(eta_x, xi_x, gxi)
    # Filter inside rhs: at k_max ≳ 100 the DNO series' k^M factor amplifies high-k roundoff faster than the implicit-iteration accumulates, so post-step filtering alone is too late.
    if params.filter_fraction < 1.0:
        eta_t = apply_lowpass(eta_t, params.k, params.filter_fraction)
        xi_t = apply_lowpass(xi_t, params.k, params.filter_fraction)
    return State(eta=eta_t, xi=xi_t)


def nonlinear_ramp_factor(
    time: float | jnp.ndarray,
    params: SolverParams,
) -> jnp.ndarray:
    """Return the Dommermuth factor multiplying the nonlinear residual."""

    if params.nonlinear_ramp_time is None:
        return jnp.asarray(1.0, dtype=params.k.dtype)
    ramp_time = jnp.asarray(params.nonlinear_ramp_time, dtype=params.k.dtype)
    if ramp_time.ndim > 0:
        ramp_time = ramp_time[..., jnp.newaxis]
    nonnegative_time = jnp.maximum(
        jnp.asarray(time, dtype=params.k.dtype),
        jnp.asarray(0.0, dtype=params.k.dtype),
    )
    if nonnegative_time.ndim > 0:
        nonnegative_time = nonnegative_time[..., jnp.newaxis]
    return 1.0 - jnp.exp(
        -((nonnegative_time / ramp_time) ** params.nonlinear_ramp_order)
    )


def rhs_nonlinear_if(
    v_hat: SpectralState, t: float | jnp.ndarray, params: SolverParams
) -> SpectralState:
    physical_hat = apply_linear_flow_hat(v_hat, t, params)
    physical_state = _hat_to_state(physical_hat)
    nonlinear_state = rhs_nonlinear(physical_state, params)
    nonlinear_hat = _state_to_hat(nonlinear_state, params.nx)
    nonlinear_hat = _tree_scale(
        nonlinear_hat,
        nonlinear_ramp_factor(t, params),
    )
    return apply_linear_flow_hat(nonlinear_hat, -t, params)


def rk4_if_step(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
    params: SolverParams,
) -> State:
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
    t: float | jax.Array,
    dt: float | jax.Array,
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
        return jax.tree_util.tree_map(
            lambda s, c: (1.0 - relaxation) * s + relaxation * c, stage, candidate
        )

    stage = jax.lax.fori_loop(0, iterations, body_fn, v0)
    v1 = jax.tree_util.tree_map(lambda mid, start: 2.0 * mid - start, stage, v0)
    next_state = _hat_to_state(apply_linear_flow_hat(v1, t + dt, params))
    if params.filter_fraction < 1.0:
        next_state = State(
            eta=apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def _gauss_legendre_2_stage_map(
    stage1: SpectralState,
    stage2: SpectralState,
    *,
    v0: SpectralState,
    stage_time1: float | jnp.ndarray,
    stage_time2: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    params: SolverParams,
    a11: float | jnp.ndarray,
    a12: float | jnp.ndarray,
    a21: float | jnp.ndarray,
    a22: float | jnp.ndarray,
) -> tuple[SpectralState, SpectralState, SpectralState, SpectralState]:
    f1 = rhs_nonlinear_if(stage1, stage_time1, params)
    f2 = rhs_nonlinear_if(stage2, stage_time2, params)
    candidate1 = _tree_add(
        v0,
        _tree_scale(
            _tree_add(_tree_scale(f1, a11), _tree_scale(f2, a12)),
            dt,
        ),
    )
    candidate2 = _tree_add(
        v0,
        _tree_scale(
            _tree_add(_tree_scale(f1, a21), _tree_scale(f2, a22)),
            dt,
        ),
    )
    return candidate1, candidate2, f1, f2


def _finish_gauss_legendre_2_step(
    v0: SpectralState,
    f1: SpectralState,
    f2: SpectralState,
    *,
    t: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    params: SolverParams,
) -> State:
    v1 = _tree_add(v0, _tree_scale(_tree_add(f1, f2), 0.5 * dt))
    next_state = _hat_to_state(apply_linear_flow_hat(v1, t + dt, params))
    if params.filter_fraction < 1.0:
        next_state = State(
            eta=apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def _gauss_legendre_2_if_step_result(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
    params: SolverParams,
    *,
    max_iterations: int,
    relaxation: float,
    residual_tolerance: float,
) -> ImplicitStepResult:
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = _state_to_hat(state, params.nx)
    v0 = apply_linear_flow_hat(state_hat, -t, params)

    tolerance = jnp.asarray(residual_tolerance, dtype=state.eta.dtype)
    iteration_count = jnp.zeros(state.eta.shape[:-1], dtype=jnp.int32)
    candidate1, candidate2, f1, f2 = _gauss_legendre_2_stage_map(
        v0,
        v0,
        v0=v0,
        stage_time1=t + c1 * dt,
        stage_time2=t + c2 * dt,
        dt=dt,
        params=params,
        a11=a11,
        a12=a12,
        a21=a21,
        a22=a22,
    )
    residual = relative_implicit_stage_residual(
        v0,
        v0,
        candidate1,
        candidate2,
    )
    active = jnp.logical_and(
        jnp.isfinite(residual),
        residual > tolerance,
    )

    def loop_body(
        _iteration: int,
        carry: _GL2IterationCarry,
    ) -> _GL2IterationCarry:
        if relaxation == 1.0:
            relaxed1, relaxed2 = carry.candidate1, carry.candidate2
        else:
            relaxed1 = jax.tree_util.tree_map(
                lambda s, c: (1.0 - relaxation) * s + relaxation * c,
                carry.stage1,
                carry.candidate1,
            )
            relaxed2 = jax.tree_util.tree_map(
                lambda s, c: (1.0 - relaxation) * s + relaxation * c,
                carry.stage2,
                carry.candidate2,
            )

        active_mask = carry.active[..., jnp.newaxis]
        stage1 = jax.tree_util.tree_map(
            lambda old, new: jnp.where(active_mask, new, old),
            carry.stage1,
            relaxed1,
        )
        stage2 = jax.tree_util.tree_map(
            lambda old, new: jnp.where(active_mask, new, old),
            carry.stage2,
            relaxed2,
        )
        counts = carry.iteration_count + carry.active.astype(jnp.int32)

        candidate1, candidate2, f1, f2 = _gauss_legendre_2_stage_map(
            stage1,
            stage2,
            v0=v0,
            stage_time1=t + c1 * dt,
            stage_time2=t + c2 * dt,
            dt=dt,
            params=params,
            a11=a11,
            a12=a12,
            a21=a21,
            a22=a22,
        )
        residual = relative_implicit_stage_residual(
            stage1,
            stage2,
            candidate1,
            candidate2,
        )
        active = jnp.logical_and(
            jnp.isfinite(residual),
            residual > tolerance,
        )
        return _GL2IterationCarry(
            stage1=stage1,
            stage2=stage2,
            candidate1=candidate1,
            candidate2=candidate2,
            f1=f1,
            f2=f2,
            residual=residual,
            iteration_count=counts,
            active=active,
        )

    iteration = jax.lax.fori_loop(
        0,
        max_iterations,
        loop_body,
        _GL2IterationCarry(
            stage1=v0,
            stage2=v0,
            candidate1=candidate1,
            candidate2=candidate2,
            f1=f1,
            f2=f2,
            residual=residual,
            iteration_count=iteration_count,
            active=active,
        ),
    )
    stage1 = iteration.stage1
    stage2 = iteration.stage2
    candidate1 = iteration.candidate1
    candidate2 = iteration.candidate2
    f1 = iteration.f1
    f2 = iteration.f2
    residual = iteration.residual
    iteration_count = iteration.iteration_count
    next_state = _finish_gauss_legendre_2_step(
        v0,
        f1,
        f2,
        t=t,
        dt=dt,
        params=params,
    )

    stage_finite = jnp.logical_and(
        jnp.isfinite(residual),
        _spectral_states_finite(
            stage1,
            stage2,
            candidate1,
            candidate2,
        ),
    )
    state_finite = _state_finite(next_state)
    numerically_finite = jnp.logical_and(stage_finite, state_finite)
    converged = jnp.logical_and(
        numerically_finite,
        residual <= tolerance,
    )
    hit_iteration_cap = jnp.logical_and(
        numerically_finite,
        jnp.logical_and(
            jnp.logical_not(converged),
            iteration_count >= max_iterations,
        ),
    )

    return ImplicitStepResult(
        state=next_state,
        telemetry=ImplicitStepTelemetry(
            residual=residual,
            iterations=iteration_count,
            converged=converged,
            stage_finite=stage_finite,
            state_finite=state_finite,
            hit_iteration_cap=hit_iteration_cap,
        ),
    )


def gauss_legendre_2_if_step(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
    params: SolverParams,
    iterations: int = 4,
    relaxation: float = 1.0,
) -> State:
    """Take the legacy fixed-count GL2 integrating-factor step."""

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = _state_to_hat(state, params.nx)
    v0 = apply_linear_flow_hat(state_hat, -t, params)

    def body_fn(
        _: int,
        stages: tuple[SpectralState, SpectralState],
    ) -> tuple[SpectralState, SpectralState]:
        stage1, stage2 = stages
        candidate1, candidate2, _f1, _f2 = _gauss_legendre_2_stage_map(
            stage1,
            stage2,
            v0=v0,
            stage_time1=t + c1 * dt,
            stage_time2=t + c2 * dt,
            dt=dt,
            params=params,
            a11=a11,
            a12=a12,
            a21=a21,
            a22=a22,
        )
        if relaxation == 1.0:
            return candidate1, candidate2
        return (
            jax.tree_util.tree_map(
                lambda s, c: (1.0 - relaxation) * s + relaxation * c,
                stage1,
                candidate1,
            ),
            jax.tree_util.tree_map(
                lambda s, c: (1.0 - relaxation) * s + relaxation * c,
                stage2,
                candidate2,
            ),
        )

    stage1, stage2 = jax.lax.fori_loop(
        0,
        iterations,
        body_fn,
        (v0, v0),
    )
    _, _, f1, f2 = _gauss_legendre_2_stage_map(
        stage1,
        stage2,
        v0=v0,
        stage_time1=t + c1 * dt,
        stage_time2=t + c2 * dt,
        dt=dt,
        params=params,
        a11=a11,
        a12=a12,
        a21=a21,
        a22=a22,
    )
    return _finish_gauss_legendre_2_step(
        v0,
        f1,
        f2,
        t=t,
        dt=dt,
        params=params,
    )


def gauss_legendre_2_if_step_with_telemetry(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
    params: SolverParams,
    *,
    max_iterations: int = 8,
    residual_tolerance: float = 1e-8,
    relaxation: float = 1.0,
) -> ImplicitStepResult:
    """Take a convergence-controlled GL2 step and return per-sample telemetry.

    Picard stages are updated until their dimensionless fixed-point residual
    does not exceed ``residual_tolerance`` or ``max_iterations`` updates have
    been applied.  Converged samples in a batch are frozen while unconverged
    samples continue.
    """

    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if not 0.0 < residual_tolerance < float("inf"):
        raise ValueError("residual_tolerance must be finite and positive")
    return _gauss_legendre_2_if_step_result(
        state,
        t,
        dt,
        params,
        max_iterations=max_iterations,
        relaxation=relaxation,
        residual_tolerance=residual_tolerance,
    )


def take_step(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
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


def take_step_with_telemetry(
    state: State,
    t: float | jax.Array,
    dt: float | jax.Array,
    params: SolverParams,
    *,
    method: str = "gl2_if",
    implicit_max_iterations: int = 8,
    implicit_residual_tolerance: float = 1e-8,
    implicit_relaxation: float = 1.0,
) -> ImplicitStepResult:
    """Take one convergence-controlled implicit step with diagnostics."""

    if method != "gl2_if":
        raise ValueError(
            "implicit-stage telemetry is currently available only for 'gl2_if'"
        )
    return gauss_legendre_2_if_step_with_telemetry(
        state,
        t,
        dt,
        params,
        max_iterations=implicit_max_iterations,
        residual_tolerance=implicit_residual_tolerance,
        relaxation=implicit_relaxation,
    )


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
    implicit_residual_tolerance: float | None = None,
) -> dict[str, jnp.ndarray]:
    """Integrate saved states, optionally recording convergence-controlled GL2.

    Existing calls with ``implicit_residual_tolerance=None`` retain the fixed
    ``implicit_iterations`` behavior and payload.  Supplying a tolerance
    interprets ``implicit_iterations`` as a per-step iteration cap and adds
    per-substep and trajectory-level ``gl2_*`` diagnostics to the payload.
    Per-step arrays have shape ``(number_of_actual_substeps, *batch_shape)``;
    ``gl2_all_stages_converged``, ``gl2_max_stage_residual``, and
    ``gl2_first_failed_step`` have shape ``batch_shape``.
    """

    times = jnp.asarray(times)
    if substeps_per_interval < 1:
        raise ValueError("substeps_per_interval must be positive")
    telemetry_enabled = implicit_residual_tolerance is not None
    if telemetry_enabled and method != "gl2_if":
        raise ValueError(
            "implicit_residual_tolerance is currently available only for 'gl2_if'"
        )
    if telemetry_enabled:
        assert implicit_residual_tolerance is not None
        if not 0.0 < implicit_residual_tolerance < float("inf"):
            raise ValueError("implicit_residual_tolerance must be finite and positive")

    initial_state = State(
        eta=jnp.asarray(initial_state.eta), xi=jnp.asarray(initial_state.xi)
    )
    if zero_mean_xi:
        initial_state = project_zero_mean_xi(initial_state)

    step_telemetry: ImplicitStepTelemetry | None = None
    if times.shape[0] == 1:
        eta = initial_state.eta[jnp.newaxis, :]
        xi = initial_state.xi[jnp.newaxis, :]
        if telemetry_enabled:
            empty_shape = (0, *initial_state.eta.shape[:-1])
            step_telemetry = ImplicitStepTelemetry(
                residual=jnp.empty(empty_shape, dtype=initial_state.eta.dtype),
                iterations=jnp.empty(empty_shape, dtype=jnp.int32),
                converged=jnp.empty(empty_shape, dtype=jnp.bool_),
                stage_finite=jnp.empty(empty_shape, dtype=jnp.bool_),
                state_finite=jnp.empty(empty_shape, dtype=jnp.bool_),
                hit_iteration_cap=jnp.empty(empty_shape, dtype=jnp.bool_),
            )
    else:
        dts = times[1:] - times[:-1]

        if telemetry_enabled:
            assert implicit_residual_tolerance is not None

            def step_fn_with_telemetry(
                carry: State,
                data: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[
                State,
                tuple[State, ImplicitStepTelemetry],
            ]:
                current_t, dt = data
                if substeps_per_interval == 1:
                    result = take_step_with_telemetry(
                        carry,
                        current_t,
                        dt,
                        params,
                        method=method,
                        implicit_max_iterations=implicit_iterations,
                        implicit_residual_tolerance=implicit_residual_tolerance,
                        implicit_relaxation=implicit_relaxation,
                    )
                    next_state = result.state
                    if zero_mean_xi:
                        next_state = project_zero_mean_xi(next_state)
                    interval_telemetry = result.telemetry
                else:
                    dt_sub = dt / substeps_per_interval

                    def substep_fn(
                        subcarry: State,
                        sub_idx: jnp.ndarray,
                    ) -> tuple[State, ImplicitStepTelemetry]:
                        sub_t = current_t + dt_sub * sub_idx
                        result = take_step_with_telemetry(
                            subcarry,
                            sub_t,
                            dt_sub,
                            params,
                            method=method,
                            implicit_max_iterations=implicit_iterations,
                            implicit_residual_tolerance=implicit_residual_tolerance,
                            implicit_relaxation=implicit_relaxation,
                        )
                        next_substate = result.state
                        if zero_mean_xi:
                            next_substate = project_zero_mean_xi(next_substate)
                        return next_substate, result.telemetry

                    next_state, interval_telemetry = jax.lax.scan(
                        substep_fn,
                        carry,
                        jnp.arange(substeps_per_interval),
                    )
                return next_state, (next_state, interval_telemetry)

            _, (saved_states, interval_telemetry) = jax.lax.scan(
                step_fn_with_telemetry,
                initial_state,
                (times[:-1], dts),
            )
            if substeps_per_interval == 1:
                step_telemetry = interval_telemetry
            else:
                step_telemetry = jax.tree_util.tree_map(
                    lambda field: field.reshape(
                        (
                            field.shape[0] * field.shape[1],
                            *field.shape[2:],
                        )
                    ),
                    interval_telemetry,
                )
        else:

            def step_fn(
                carry: State,
                data: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[State, State]:
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
                        next_substate = take_step(
                            subcarry,
                            sub_t,
                            dt_sub,
                            params,
                            method=method,
                            implicit_iterations=implicit_iterations,
                            implicit_relaxation=implicit_relaxation,
                        )
                        if zero_mean_xi:
                            next_substate = project_zero_mean_xi(next_substate)
                        return next_substate

                    next_state = jax.lax.fori_loop(
                        0,
                        substeps_per_interval,
                        body_fn,
                        carry,
                    )
                return next_state, next_state

            _, saved_states = jax.lax.scan(
                step_fn,
                initial_state,
                (times[:-1], dts),
            )
        eta = jnp.concatenate(
            (initial_state.eta[jnp.newaxis, :], saved_states.eta), axis=0
        )
        xi = jnp.concatenate(
            (initial_state.xi[jnp.newaxis, :], saved_states.xi), axis=0
        )

    payload: dict[str, jnp.ndarray] = {"times": times, "eta": eta, "xi": xi}
    if telemetry_enabled:
        assert step_telemetry is not None
        n_steps = step_telemetry.residual.shape[0]
        if n_steps == 0:
            sample_shape = initial_state.eta.shape[:-1]
            all_converged = jnp.ones(sample_shape, dtype=jnp.bool_)
            maximum_residual = jnp.zeros(
                sample_shape,
                dtype=initial_state.eta.dtype,
            )
            first_failed_step = jnp.full(
                sample_shape,
                -1,
                dtype=jnp.int32,
            )
            step_times = jnp.empty((0,), dtype=times.dtype)
            step_dts = jnp.empty((0,), dtype=times.dtype)
        else:
            failures = jnp.logical_not(step_telemetry.converged)
            any_failure = jnp.any(failures, axis=0)
            all_converged = jnp.logical_not(any_failure)
            maximum_residual = jnp.max(
                jnp.where(
                    step_telemetry.stage_finite,
                    step_telemetry.residual,
                    jnp.asarray(jnp.inf, dtype=step_telemetry.residual.dtype),
                ),
                axis=0,
            )
            first_failed_step = jnp.where(
                any_failure,
                jnp.argmax(failures, axis=0).astype(jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
            )
            interval_dts = times[1:] - times[:-1]
            offsets = (
                jnp.arange(substeps_per_interval, dtype=times.dtype)
                / substeps_per_interval
            )
            step_times = (
                times[:-1, jnp.newaxis]
                + interval_dts[:, jnp.newaxis] * offsets[jnp.newaxis, :]
            ).reshape(-1)
            step_dts = jnp.broadcast_to(
                interval_dts[:, jnp.newaxis] / substeps_per_interval,
                (interval_dts.shape[0], substeps_per_interval),
            ).reshape(-1)

        payload.update(
            {
                "gl2_stage_residual": step_telemetry.residual,
                "gl2_iterations": step_telemetry.iterations,
                "gl2_converged": step_telemetry.converged,
                "gl2_stage_finite": step_telemetry.stage_finite,
                "gl2_state_finite": step_telemetry.state_finite,
                "gl2_hit_iteration_cap": step_telemetry.hit_iteration_cap,
                "gl2_step_times": step_times,
                "gl2_step_dts": step_dts,
                "gl2_all_stages_converged": all_converged,
                "gl2_max_stage_residual": maximum_residual,
                "gl2_first_failed_step": first_failed_step,
                "gl2_residual_tolerance": jnp.asarray(
                    implicit_residual_tolerance,
                    dtype=initial_state.eta.dtype,
                ),
                "gl2_iteration_cap": jnp.asarray(
                    implicit_iterations,
                    dtype=jnp.int32,
                ),
            }
        )
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
        if params.filter_fraction < 1.0:
            # Match the resolved subspace used by the RHS and accepted states;
            # otherwise the order-M series amplifies discarded-mode roundoff.
            gxi = apply_lowpass(gxi, params.k, params.filter_fraction)
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
    implicit_residual_tolerance: float | None = None,
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
        implicit_residual_tolerance=implicit_residual_tolerance,
    )
