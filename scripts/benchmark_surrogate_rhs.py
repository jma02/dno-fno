"""Benchmark-only spectral adapter and shared FFT-packing trials."""

from typing import Literal

import jax
import jax.numpy as jnp

from solver.evals.model_rollout import Predictor
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import _dno_series_hat


def rhs_nonlinear_if_spectral(
    state_hat: ti.SpectralState,
    time_value: float | jax.Array,
    params: ti.SolverParams,
    predict_gxi: Predictor | None,
    *,
    pack_ffts: Literal["none", "inputs", "all"] = "none",
) -> ti.SpectralState:
    physical_hat = ti.apply_linear_flow_hat(state_hat, time_value, params)
    nx = params.nx
    positive_modes = slice(0, nx // 2 + 1)
    eta_hat = physical_hat.eta_hat[..., positive_modes]
    xi_hat = physical_hat.xi_hat[..., positive_modes]
    k = params.k[positive_modes]
    if predict_gxi is None:
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
    nonlinear_hat = ti.SpectralState(
        jnp.concatenate((eta_t_hat, jnp.conj(eta_t_hat[..., 1:-1][..., ::-1])), axis=-1),
        jnp.concatenate((xi_t_hat, jnp.conj(xi_t_hat[..., 1:-1][..., ::-1])), axis=-1),
    )
    if params.filter_fraction < 1.0:
        nonlinear_hat = ti._lowpass_hat(nonlinear_hat, params)
    if predict_gxi is None:
        nonlinear_hat = ti._tree_scale(nonlinear_hat, ti.nonlinear_ramp_factor(time_value, params))
    return ti.apply_linear_flow_hat(nonlinear_hat, -time_value, params)
