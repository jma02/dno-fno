"""Active-mode phase-space-normalized complex loss for DNO data."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Diagnostics = dict[str, jax.Array]


@dataclass(frozen=True)
class ModeBalancedConfig:
    """Configuration for balancing errors over reference-active modes."""

    k_max: float = 128.0
    gravity: float = 1.0
    active_scale_relative: float = 1e-4
    denominator_floor_relative: float = 1e-6
    absolute_floor: float = 1e-24
    huber_delta: float = 1.0
    ratio_cap: float = 100.0
    dispersion_weighting: bool = False


def _capped_pseudo_huber(
    squared_ratio: jax.Array,
    delta: jax.Array,
    cap: jax.Array,
) -> jax.Array:
    """Return a stable pseudo-Huber penalty after clipping extreme ratios."""
    clipped_ratio = jnp.minimum(squared_ratio, cap)
    scaled_ratio = clipped_ratio / delta**2
    return 2.0 * clipped_ratio / (jnp.sqrt(1.0 + scaled_ratio) + 1.0)


def compute_mode_balanced_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k_rfft: jax.Array,
    config: ModeBalancedConfig,
) -> tuple[jax.Array, Diagnostics]:
    """Compare complex DNO coefficients with soft balance over active modes.

    For each positive Fourier mode, the physical reference scale is

    ``|q_*|^2 + omega(k, h)^2 |eta|^2``,

    where ``q = G(eta) xi`` and
    ``omega^2 = g |k| tanh(|k| h)``.  A soft activity mask prevents empty or
    numerical-noise modes from receiving the same weight as physical modes,
    while normalization by the scale keeps a small sideband from being hidden
    by a large carrier.  The mask and all target-derived scales are detached;
    gradients act only through the complex prediction error.

    The zero mode and modes above ``config.k_max`` are excluded.  Fourier
    transforms use forward normalization, making the floors independent of
    grid resolution.  With ``config.dispersion_weighting``, the outer sample
    average is weighted by the elevation-energy-averaged linear frequency

    ``Omega_eff^2 = sum_k omega(k, h)^2 |eta_hat_k|^2 / sum_k |eta_hat_k|^2``,

    over the same positive modes scored by the loss.  Parseval multiplicities
    for omitted negative modes enter only this optional effective-frequency
    calculation, not the core per-mode loss.

    This changes only the relative weight of samples; the complex per-mode
    loss within each sample is unchanged.
    """
    real_dtype = eta.dtype
    eta_hat = jnp.fft.rfft(eta, axis=-1, norm="forward")
    prediction_hat = jnp.fft.rfft(
        gxi_prediction, axis=-1, norm="forward"
    )
    target_hat = jnp.fft.rfft(gxi_target, axis=-1, norm="forward")

    k_abs = jnp.abs(k_rfft).astype(real_dtype)
    omega_squared = (
        jnp.asarray(config.gravity, dtype=real_dtype)
        * k_abs[None, :]
        * jnp.tanh(depth[:, None] * k_abs[None, :])
    )
    physical_scale = (
        jnp.abs(target_hat) ** 2 + omega_squared * jnp.abs(eta_hat) ** 2
    ).astype(real_dtype)
    physical_scale = jax.lax.stop_gradient(physical_scale)

    band = jnp.logical_and(
        k_abs > jnp.asarray(0.0, dtype=real_dtype),
        k_abs <= jnp.asarray(config.k_max, dtype=real_dtype),
    )
    band_scale = jnp.where(band[None, :], physical_scale, 0.0)
    sample_scale = jnp.max(band_scale, axis=-1, keepdims=True)
    absolute_floor = jnp.asarray(config.absolute_floor, dtype=real_dtype)
    denominator_floor = (
        jnp.asarray(config.denominator_floor_relative, dtype=real_dtype)
        * sample_scale
        + absolute_floor
    )
    activity_floor = (
        jnp.asarray(config.active_scale_relative, dtype=real_dtype)
        * sample_scale
        + absolute_floor
    )
    soft_activity = physical_scale / (physical_scale + activity_floor)
    mode_weight = jax.lax.stop_gradient(
        jnp.where(band[None, :], soft_activity, 0.0)
    )

    squared_error = jnp.abs(prediction_hat - target_hat) ** 2
    squared_ratio = squared_error / (physical_scale + denominator_floor)
    ratio_cap = jnp.asarray(config.ratio_cap, dtype=real_dtype)
    penalty = _capped_pseudo_huber(
        squared_ratio,
        jnp.asarray(config.huber_delta, dtype=real_dtype),
        ratio_cap,
    )

    active_weight = jnp.sum(mode_weight, axis=-1)
    weight_floor = jnp.asarray(config.absolute_floor, dtype=real_dtype)
    per_sample_loss = jnp.sum(mode_weight * penalty, axis=-1) / jnp.maximum(
        active_weight, weight_floor
    )
    parseval_weight = jnp.full_like(k_abs, 2.0)
    parseval_weight = parseval_weight.at[0].set(1.0)
    if eta.shape[-1] % 2 == 0:
        parseval_weight = parseval_weight.at[-1].set(1.0)
    eta_energy = (
        parseval_weight[None, :]
        * jnp.abs(eta_hat) ** 2
        * band[None, :]
    )
    total_eta_energy = jnp.sum(eta_energy, axis=-1)
    safe_eta_energy = jnp.where(
        total_eta_energy > 0.0,
        total_eta_energy,
        jnp.ones_like(total_eta_energy),
    )
    effective_frequency_squared = jax.lax.stop_gradient(
        jnp.sum(omega_squared * eta_energy, axis=-1)
        / safe_eta_energy
    )
    sample_weight = (
        effective_frequency_squared
        if config.dispersion_weighting
        else jnp.ones_like(effective_frequency_squared)
    )
    loss = jnp.mean(sample_weight * per_sample_loss)

    total_weight = jnp.sum(mode_weight)
    total_weight_safe = jnp.maximum(total_weight, weight_floor)
    weighted_raw_ratio = jnp.sum(mode_weight * squared_ratio) / total_weight_safe
    clipped_fraction = jnp.sum(
        mode_weight * (squared_ratio >= ratio_cap).astype(real_dtype)
    ) / total_weight_safe
    diagnostics: Diagnostics = {
        "loss": loss,
        "unweighted_loss": jnp.mean(per_sample_loss),
        "effective_frequency_squared_mean": jnp.mean(
            effective_frequency_squared
        ),
        "relative_error_rms": jnp.sqrt(weighted_raw_ratio),
        "active_modes": jnp.mean(active_weight),
        "clipped_mode_fraction": clipped_fraction,
        "mean_denominator_floor": jnp.mean(denominator_floor),
        "max_raw_squared_ratio": jnp.max(
            jnp.where(mode_weight > 0.0, squared_ratio, 0.0)
        ),
    }
    return loss, diagnostics
