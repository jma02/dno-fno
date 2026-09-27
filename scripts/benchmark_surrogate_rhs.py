"""Benchmark-only spectral adapter and shared FFT-packing trials."""

from typing import Literal, cast

import jax
import jax.numpy as jnp

from solver.evals.model_rollout import Predictor
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import _dno_series_hat


def rotate_spectrum(
    state: ti.SpectralState, cosine: jax.Array, sine_over_omega: jax.Array, params: ti.SolverParams,
) -> ti.SpectralState:
    return ti.SpectralState(
        cosine * state.eta_hat + sine_over_omega * params.g0 * state.xi_hat,
        cosine * state.xi_hat - sine_over_omega * params.gravity * state.eta_hat,
    )


def shared_m1_hat(
    eta_hat: jax.Array, xi_hat: jax.Array, params: ti.SolverParams,
    predict_correction: Predictor,
) -> jax.Array:
    """Add the unchanged learned correction to the solver's physical-depth M1."""
    nx = params.nx
    modes = slice(0, nx // 2 + 1)
    baseline = _dno_series_hat(
        eta_hat, xi_hat, params.k[modes], params.g0[..., modes],
        nx=nx, order=1, pad_factor=params.pad_factor,
    )
    eta, xi = jnp.moveaxis(jnp.fft.irfft(
        jnp.stack((eta_hat, xi_hat), axis=-2), n=nx, axis=-1,
    ), -2, 0)
    correction = jnp.fft.rfft(predict_correction(eta, xi), axis=-1)
    return (baseline + correction).at[..., nx // 2].set(0)


def rhs_nonlinear_if_spectral(
    state_hat: ti.SpectralState,
    time_value: float | jax.Array,
    params: ti.SolverParams,
    predict_gxi: Predictor | None,
    *,
    pack_ffts: Literal["none", "inputs", "all"] = "none",
    predict_correction: Predictor | None = None,
    predict_hat: Predictor | None = None,
    rotation: tuple[jax.Array, jax.Array] | None = None,
    ramp_time: float | jax.Array | None = None,
    half_spectrum: bool = False,
) -> ti.SpectralState:
    physical_hat = (ti.apply_linear_flow_hat(state_hat, time_value, params) if rotation is None
                    else rotate_spectrum(state_hat, *rotation, params))
    nx = params.nx
    positive_modes = slice(0, nx // 2 + 1)
    eta_hat = physical_hat.eta_hat[..., positive_modes]
    xi_hat = physical_hat.xi_hat[..., positive_modes]
    k = params.k[positive_modes]
    if predict_hat is not None:
        gxi_hat = predict_hat(eta_hat, xi_hat).at[..., nx // 2].set(0)
    elif predict_correction is not None:
        gxi_hat = shared_m1_hat(eta_hat, xi_hat, params, predict_correction)
    elif predict_gxi is None:
        gxi_hat = _dno_series_hat(
            eta_hat, xi_hat, k, params.g0[..., positive_modes],
            nx=nx, order=params.dno_order, pad_factor=params.pad_factor,
        )
    else:
        if pack_ffts == "none":
            eta = jnp.fft.irfft(eta_hat, n=nx, axis=-1)
            xi = jnp.fft.irfft(xi_hat, n=nx, axis=-1)
        else:
            eta, xi = jnp.moveaxis(jnp.fft.irfft(
                jnp.stack((eta_hat, xi_hat), axis=-2), n=nx, axis=-1,
            ), -2, 0)
        gxi_hat = jnp.fft.rfft(predict_gxi(eta, xi), axis=-1)
        gxi_hat = gxi_hat.at[..., nx // 2].set(0)

    if pack_ffts == "all":
        eta_x, xi_x, gxi = jnp.moveaxis(2.0 * jnp.fft.irfft(
            jnp.stack((1j * k * eta_hat, 1j * k * xi_hat, gxi_hat), axis=-2),
            n=2 * nx, axis=-1,
        ), -2, 0)
    else:
        eta_x = 2.0 * jnp.fft.irfft(1j * k * eta_hat, n=2 * nx, axis=-1)
        xi_x = 2.0 * jnp.fft.irfft(1j * k * xi_hat, n=2 * nx, axis=-1)
        gxi = 2.0 * jnp.fft.irfft(gxi_hat, n=2 * nx, axis=-1)
    numerator = gxi + eta_x * xi_x
    xi_t = -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    xi_t_hat = 0.5 * jnp.fft.rfft(xi_t, axis=-1)[..., positive_modes]
    xi_t_hat = xi_t_hat.at[..., nx // 2].set(0)
    eta_t_hat = gxi_hat - params.g0[..., positive_modes] * xi_hat
    nonlinear_hat = ti.SpectralState(eta_t_hat, xi_t_hat) if half_spectrum else ti.SpectralState(
        jnp.concatenate((eta_t_hat, jnp.conj(eta_t_hat[..., 1:-1][..., ::-1])), axis=-1),
        jnp.concatenate((xi_t_hat, jnp.conj(xi_t_hat[..., 1:-1][..., ::-1])), axis=-1),
    )
    if params.filter_fraction < 1.0:
        nonlinear_hat = ti._lowpass_hat(nonlinear_hat, params)
    if predict_gxi is None or predict_correction is not None:
        nonlinear_hat = ti._tree_scale(nonlinear_hat, ti.nonlinear_ramp_factor(
            time_value if ramp_time is None else ramp_time, params))
    return (ti.apply_linear_flow_hat(nonlinear_hat, -time_value, params) if rotation is None
            else rotate_spectrum(nonlinear_hat, rotation[0], -rotation[1], params))


def rollout_if_trial(
    initial: ti.State, times: jax.Array, *, params: ti.SolverParams, substeps: int,
    carry_spectral: bool, local_if: bool, predict_gxi: Predictor | None = None,
    predict_hat: Predictor | None = None, half_spectrum: bool = False,
) -> dict[str, jax.Array]:
    """Paired GL2 with optional spectral carry and interval-cached local rotations."""
    initial = ti.project_zero_mean_xi(initial)
    flow_params = (params._replace(k=params.k[:params.nx // 2 + 1], g0=params.g0[..., :params.nx // 2 + 1])
                   if half_spectrum else params)
    root3 = jnp.sqrt(jnp.asarray(3.0, dtype=initial.eta.dtype))
    c = jnp.array([0.5 - root3 / 6, 0.5 + root3 / 6])
    weights = jnp.array([[0.25, 0.25 - root3 / 6], [0.25 + root3 / 6, 0.25]])

    def interval_step(
        carry: ti.State | ti.SpectralState, data: tuple[jax.Array, jax.Array],
    ) -> tuple[ti.State | ti.SpectralState, tuple[jax.Array, jax.Array, jax.Array | None]]:
        start, interval = data
        dt = interval / substeps
        tau = (dt * jnp.concatenate((c, jnp.ones(1)))).reshape((3,) + (1,) * flow_params.g0.ndim)
        omega = jnp.sqrt(flow_params.gravity * flow_params.g0)
        cosine = jnp.cos(omega * tau)
        sine = jnp.where(omega > 0, jnp.sin(omega * tau) / omega, 0.0)

        def step(index: int, state: ti.State | ti.SpectralState) -> ti.State | ti.SpectralState:
            time_value = start + dt * index
            state_hat = (cast(ti.SpectralState, state) if carry_spectral
                         else ti._state_to_hat(cast(ti.State, state), params.nx))
            v0 = state_hat if local_if else ti.apply_linear_flow_hat(state_hat, -time_value, flow_params)

            def stage_map(
                stages: tuple[ti.SpectralState, ti.SpectralState],
            ) -> tuple[tuple[ti.SpectralState, ti.SpectralState], ti.SpectralState, ti.SpectralState]:
                packed = jax.tree.map(lambda first, second: jnp.stack((first, second)), *stages)
                if local_if:
                    fields = jax.vmap(lambda s, t, co, si: rhs_nonlinear_if_spectral(
                        s, t, flow_params, predict_gxi, pack_ffts="all", predict_hat=predict_hat,
                        rotation=(co, si), ramp_time=time_value + t, half_spectrum=half_spectrum,
                    ))(packed, c * dt, cosine[:2], sine[:2])
                else:
                    fields = jax.vmap(lambda s, t: rhs_nonlinear_if_spectral(
                        s, t, flow_params, predict_gxi, pack_ffts="all", predict_hat=predict_hat,
                        half_spectrum=half_spectrum,
                    ))(packed, time_value + c * dt)
                f1 = ti.SpectralState(*(field[0] for field in fields))
                f2 = ti.SpectralState(*(field[1] for field in fields))
                candidates = tuple(ti._tree_add(v0, ti._tree_scale(ti._tree_add(
                    ti._tree_scale(f1, row[0]), ti._tree_scale(f2, row[1])), dt)) for row in weights)
                return (candidates[0], candidates[1]), f1, f2

            stages = jax.lax.fori_loop(0, 4, lambda _, pair: stage_map(pair)[0], (v0, v0))
            _, f1, f2 = stage_map(stages)
            v1 = ti._tree_add(v0, ti._tree_scale(ti._tree_add(f1, f2), 0.5 * dt))
            next_hat = (rotate_spectrum(v1, cosine[2], sine[2], flow_params) if local_if
                        else ti.apply_linear_flow_hat(v1, time_value + dt, flow_params))
            if params.filter_fraction < 1.0:
                next_hat = ti._lowpass_hat(next_hat, flow_params)
            if carry_spectral:
                fields = jnp.stack(next_hat)
                if half_spectrum:
                    fields = fields.at[..., 0].set(fields[..., 0].real)
                else:
                    reflected = jnp.concatenate((fields[..., :1], fields[..., :0:-1]), axis=-1)
                    fields = 0.5 * (fields + jnp.conj(reflected))
                return ti.SpectralState(*fields.at[..., params.nx // 2].set(0).at[1, ..., 0].set(0))
            return ti.project_zero_mean_xi(ti._hat_to_state(next_hat))

        next_state = jax.lax.fori_loop(0, substeps, step, carry)
        physical = (ti.project_zero_mean_xi(
            ti.State(*jnp.fft.irfft(jnp.stack(next_state), n=params.nx)) if half_spectrum
            else ti._hat_to_state(next_state)) if carry_spectral
                    else cast(ti.State, next_state))
        gxi = predict_gxi(*physical) if predict_gxi is not None else None
        return next_state, (physical.eta, physical.xi, gxi)

    carry = ti._state_to_hat(initial, params.nx) if carry_spectral else initial
    if half_spectrum:
        carry = ti.SpectralState(*(field[..., :params.nx // 2 + 1] for field in carry))
    _, (eta, xi, saved_gxi) = jax.lax.scan(interval_step, carry, (times[:-1], times[1:] - times[:-1]))
    eta = jnp.concatenate((initial.eta[None], eta))
    xi = jnp.concatenate((initial.xi[None], xi))
    if predict_gxi is None:
        gxi = jax.vmap(lambda e, x: ti.dno_series_eval(
            e, x, params.k, params.depth, params.dno_order, pad_factor=params.pad_factor,
        ))(eta, xi)
        if params.filter_fraction < 1.0:
            gxi = ti.apply_lowpass(gxi, params.k, params.filter_fraction)
    else:
        gxi = jnp.concatenate((predict_gxi(*initial)[None], cast(jax.Array, saved_gxi)))
    return {"times": times, "eta": eta, "xi": xi, "gxi": gxi}
