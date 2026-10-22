"""Active-mode phase-space-normalized complex loss for DNO data."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_mode_balanced_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k_rfft: jax.Array,
    *,
    k_max: float = 128.0,
    gravity: float = 1.0,
    activity_threshold: float = 1e-4,
    denominator_eps: float = 1e-6,
    absolute_floor: float = 1e-24,
    huber_delta: float = 1.0,
    ratio_cap: float = 100.0,
    dispersion_weighting: bool = False,
    sample_weight: jax.Array | None = None,
    subtract_before_fft: bool = False,
    gxi_error: jax.Array | None = None,
    return_irls_residuals: bool = False,
) -> jax.Array:
    """Compare complex DNO coefficients with soft balance over active modes.

    For each positive Fourier mode, the physical reference scale is

    ``|q_*|^2 + omega(k, h)^2 |eta|^2``,

    where ``q = G(eta) xi`` and
    ``omega^2 = g |k| tanh(|k| h)``.  A soft activity mask prevents empty or
    numerical-noise modes from receiving the same weight as physical modes,
    while normalization by the scale keeps a small sideband from being hidden
    by a large carrier.  The mask and all target-derived scales are detached;
    gradients act only through the complex prediction error.

    The zero mode and modes above ``k_max`` are excluded.  Fourier
    transforms use forward normalization, making the floors independent of
    grid resolution.  With ``dispersion_weighting``, the outer sample
    average is weighted by the elevation-energy-averaged linear frequency

    ``Omega_eff^2 = sum_k omega(k, h)^2 |eta_hat_k|^2 / sum_k |eta_hat_k|^2``,

    over the same positive modes scored by the loss.  Parseval multiplicities
    for omitted negative modes enter only this optional effective-frequency
    calculation, not the core per-mode loss.

    This changes only the relative weight of samples; the complex per-mode
    loss within each sample is unchanged.

    Optional sample weights multiply rows before the ordinary batch mean, so
    ablating one family does not increase the other families' coefficients.
    Subtracting before the FFT changes only error arithmetic, not target scales
    or activity weights. ``gxi_error`` supplies a directly computed physical
    residual for the numerator; target scales and activity weights stay unchanged.
    With ``return_irls_residuals``, return complex residuals whose locally frozen
    weights make ``0.5 * sum(abs(residuals)**2)`` have the same parameter gradient.
    This quadratic is for local linearization, not scalar loss evaluation.
    """
    real_dtype = eta.dtype
    eta_hat, prediction_hat, target_hat = (
        jnp.fft.rfft(field, axis=-1, norm="forward")
        for field in (eta, gxi_prediction, gxi_target)
    )

    k_abs = jnp.abs(k_rfft).astype(real_dtype)
    omega_squared = (
        jnp.asarray(gravity, dtype=real_dtype)
        * k_abs[None, :]
        * jnp.tanh(depth[:, None] * k_abs[None, :])
    )
    physical_scale = (
        jnp.abs(target_hat) ** 2 + omega_squared * jnp.abs(eta_hat) ** 2
    ).astype(real_dtype)
    physical_scale = jax.lax.stop_gradient(physical_scale)

    band = (k_abs > 0.0) & (k_abs <= jnp.asarray(k_max, dtype=real_dtype))
    band_scale = jnp.where(band[None, :], physical_scale, 0.0)
    sample_scale = jnp.max(band_scale, axis=-1, keepdims=True)
    floor = jnp.asarray(absolute_floor, dtype=real_dtype)
    denominator_floor, activity_floor = (
        jnp.asarray(coefficient, dtype=real_dtype) * sample_scale + floor
        for coefficient in (denominator_eps, activity_threshold)
    )
    soft_activity = physical_scale / (physical_scale + activity_floor)
    mode_weight = jax.lax.stop_gradient(jnp.where(band[None, :], soft_activity, 0.0))

    error_hat = (jnp.fft.rfft(gxi_prediction - gxi_target, axis=-1, norm="forward")
                 if subtract_before_fft else prediction_hat - target_hat)
    if gxi_error is not None:
        error_hat = jnp.fft.rfft(gxi_error, axis=-1, norm="forward")
    squared_error = jnp.abs(error_hat) ** 2
    squared_ratio = squared_error / (physical_scale + denominator_floor)
    clipped_ratio = jnp.minimum(squared_ratio, jnp.asarray(ratio_cap, dtype=real_dtype))
    scaled_ratio = clipped_ratio / jnp.asarray(huber_delta, dtype=real_dtype) ** 2
    penalty = 2.0 * clipped_ratio / (jnp.sqrt(1.0 + scaled_ratio) + 1.0)

    active_weight = jnp.sum(mode_weight, axis=-1)
    per_sample_loss = jnp.sum(mode_weight * penalty, axis=-1) / jnp.maximum(
        active_weight, floor
    )
    effective_frequency_squared = None
    if dispersion_weighting:
        parseval_weight = jnp.full_like(k_abs, 2.0)
        parseval_weight = parseval_weight.at[0].set(1.0)
        if eta.shape[-1] % 2 == 0:
            parseval_weight = parseval_weight.at[-1].set(1.0)
        eta_energy = parseval_weight[None, :] * jnp.abs(eta_hat) ** 2 * band[None, :]
        total_eta_energy = jnp.sum(eta_energy, axis=-1)
        safe_eta_energy = jnp.where(total_eta_energy > 0.0, total_eta_energy, 1.0)
        effective_frequency_squared = jax.lax.stop_gradient(
            jnp.sum(omega_squared * eta_energy, axis=-1) / safe_eta_energy
        )
        per_sample_loss = effective_frequency_squared * per_sample_loss
    if sample_weight is not None:
        per_sample_loss *= jax.lax.stop_gradient(sample_weight)
    if return_irls_residuals:
        cap = jnp.asarray(ratio_cap, dtype=real_dtype)
        cap_derivative = jnp.where(
            squared_ratio < cap, 1.0, jnp.where(squared_ratio == cap, 0.5, 0.0)
        )
        penalty_derivative = cap_derivative / jnp.sqrt(1.0 + scaled_ratio)
        weight = (
            (2.0 / eta.shape[0])
            * (mode_weight / jnp.maximum(active_weight[:, None], floor))
            * penalty_derivative / (physical_scale + denominator_floor)
        )
        if effective_frequency_squared is not None:
            weight *= effective_frequency_squared[:, None]
        if sample_weight is not None:
            weight *= sample_weight[:, None]
        return error_hat * jnp.sqrt(jax.lax.stop_gradient(weight))
    return jnp.mean(per_sample_loss)
