"""Centered, low-mode phase-rate supervision for DNO data.

For ``q = G(eta) xi``, the kinematic condition is ``eta_t = q``.  At a
nonzero Fourier coefficient of ``eta``, the instantaneous phase rate is

``phi_dot = Im(conj(eta_hat) * q_hat) / |eta_hat|^2``.

This module compares that quantity for a learned prediction and a reference
target.  It changes only the training objective: it neither evaluates nor
introduces analytic Craig--Sulem terms beyond the model's ``G0 + G1``
backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Diagnostics = dict[str, jax.Array]


@dataclass(frozen=True)
class ModalPhaseRateConfig:
    """Configuration for energy-weighted modal phase-rate matching."""

    k_min: float = 1.0
    k_max: float = 128.0
    active_scale_relative: float = 1e-4
    denominator_floor_relative: float = 1e-6
    absolute_floor: float = 1e-24
    source_ids: tuple[int, ...] = (5, 6, 14)


def _validate_config(config: ModalPhaseRateConfig) -> None:
    if config.k_min <= 0.0:
        raise ValueError(f"k_min must be positive, got {config.k_min}")
    if config.k_max < config.k_min:
        raise ValueError(
            f"k_max must be at least k_min, got {config.k_max} < {config.k_min}"
        )
    if config.active_scale_relative <= 0.0:
        raise ValueError(
            "active_scale_relative must be positive, got "
            f"{config.active_scale_relative}"
        )
    if config.denominator_floor_relative <= 0.0:
        raise ValueError(
            "denominator_floor_relative must be positive, got "
            f"{config.denominator_floor_relative}"
        )
    if config.denominator_floor_relative > config.active_scale_relative:
        raise ValueError(
            "denominator_floor_relative must not exceed active_scale_relative"
        )
    if config.absolute_floor <= 0.0:
        raise ValueError(
            f"absolute_floor must be positive, got {config.absolute_floor}"
        )
    if not config.source_ids:
        raise ValueError("source_ids must not be empty")


def _validate_shapes(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    source: jax.Array,
    k_rfft: jax.Array,
) -> None:
    if eta.ndim != 2:
        raise ValueError(f"eta must have shape (batch, grid), got {eta.shape}")
    if gxi_prediction.shape != eta.shape:
        raise ValueError(
            "gxi_prediction must have the same shape as eta, got "
            f"{gxi_prediction.shape} and {eta.shape}"
        )
    if gxi_target.shape != eta.shape:
        raise ValueError(
            "gxi_target must have the same shape as eta, got "
            f"{gxi_target.shape} and {eta.shape}"
        )
    if source.shape != (eta.shape[0],):
        raise ValueError(
            f"source must have shape ({eta.shape[0]},), got {source.shape}"
        )
    expected_modes = eta.shape[-1] // 2 + 1
    if k_rfft.shape != (expected_modes,):
        raise ValueError(
            f"k_rfft must have shape ({expected_modes},), got {k_rfft.shape}"
        )


def _selected_mean(
    values: jax.Array,
    selected: jax.Array,
    selected_count: jax.Array,
) -> jax.Array:
    """Average scalar-per-sample values without ``0 * nonfinite`` hazards."""
    zero = jnp.asarray(0.0, dtype=values.dtype)
    denominator = jnp.maximum(selected_count, jnp.asarray(1.0, values.dtype))
    return jnp.sum(jnp.where(selected, values, zero)) / denominator


def compute_modal_phase_rate_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    source: jax.Array,
    k_rfft: jax.Array,
    config: ModalPhaseRateConfig,
) -> tuple[jax.Array, Diagnostics]:
    """Return the selected-source modal phase-rate loss and diagnostics.

    The prediction and target are independently centered before the Fourier
    transform.  This implements the same quotient by spatial constants used
    in production rollout: adding any constant to either ``q`` field cannot
    change this objective.

    Only modes in ``[config.k_min, config.k_max]`` whose reference elevation
    energy exceeds a per-sample relative floor are active.  The mask, floors,
    and energy weights depend only on the reference elevation and are stopped
    from contributing gradients.  The optimized per-sample quantity is

    ``sum_k w_k |eta_hat_k|^2 |delta phi_dot_k|^2``
    ``/ sum_k w_k |eta_hat_k|^2``.

    Amplitude-rate and full complex kinematic-growth errors are diagnostics,
    not additional loss terms.  Together they satisfy
    ``kinematic_growth_loss = phase_rate_loss + amplitude_rate_loss`` up to
    floating-point roundoff on active modes.
    """
    _validate_config(config)
    _validate_shapes(eta, gxi_prediction, gxi_target, source, k_rfft)

    dtype = eta.dtype
    prediction_centered = gxi_prediction - jnp.mean(
        gxi_prediction, axis=-1, keepdims=True
    )
    target_centered = gxi_target - jnp.mean(gxi_target, axis=-1, keepdims=True)
    eta_hat = jnp.fft.rfft(eta, axis=-1, norm="forward")
    error_hat = jnp.fft.rfft(
        prediction_centered - target_centered,
        axis=-1,
        norm="forward",
    )

    eta_power = jax.lax.stop_gradient(jnp.abs(eta_hat).astype(dtype) ** 2)
    k_abs = jnp.abs(k_rfft).astype(dtype)
    band = jnp.logical_and(
        k_abs >= jnp.asarray(config.k_min, dtype=dtype),
        k_abs <= jnp.asarray(config.k_max, dtype=dtype),
    )
    band_power = jnp.where(band[None, :], eta_power, 0.0)
    sample_scale = jax.lax.stop_gradient(jnp.max(band_power, axis=-1, keepdims=True))
    absolute_floor = jnp.asarray(config.absolute_floor, dtype=dtype)
    activity_floor = jax.lax.stop_gradient(
        jnp.asarray(config.active_scale_relative, dtype=dtype) * sample_scale
        + absolute_floor
    )
    denominator_floor = jax.lax.stop_gradient(
        jnp.asarray(config.denominator_floor_relative, dtype=dtype) * sample_scale
        + absolute_floor
    )
    active = jax.lax.stop_gradient(
        jnp.logical_and(band[None, :], eta_power >= activity_floor)
    )
    active_weight = active.astype(dtype)
    safe_eta_power = jnp.maximum(eta_power, denominator_floor)

    eta_error_cross = jnp.conj(eta_hat) * error_hat
    amplitude_rate_error = jnp.real(eta_error_cross).astype(dtype) / safe_eta_power
    phase_rate_error = jnp.imag(eta_error_cross).astype(dtype) / safe_eta_power
    zero = jnp.asarray(0.0, dtype=dtype)
    active_energy = jnp.where(active, eta_power, zero)
    per_sample_energy = jnp.sum(active_energy, axis=-1)
    energy_safe = jnp.maximum(per_sample_energy, absolute_floor)
    per_sample_phase_loss = (
        jnp.sum(jnp.where(active, eta_power * phase_rate_error**2, zero), axis=-1)
        / energy_safe
    )
    per_sample_amplitude_loss = (
        jnp.sum(jnp.where(active, eta_power * amplitude_rate_error**2, zero), axis=-1)
        / energy_safe
    )
    per_sample_kinematic_loss = (
        jnp.sum(
            jnp.where(active, jnp.abs(error_hat).astype(dtype) ** 2, zero),
            axis=-1,
        )
        / energy_safe
    )
    per_sample_active_modes = jnp.sum(active_weight, axis=-1)

    source_selected = jnp.zeros_like(source, dtype=jnp.bool_)
    for source_id in config.source_ids:
        source_selected = jnp.logical_or(source_selected, source == source_id)
    selected = jnp.logical_and(source_selected, per_sample_active_modes > 0.0)
    selected_count = jnp.sum(selected.astype(dtype))

    loss = _selected_mean(per_sample_phase_loss, selected, selected_count)
    amplitude_loss = _selected_mean(per_sample_amplitude_loss, selected, selected_count)
    kinematic_loss = _selected_mean(per_sample_kinematic_loss, selected, selected_count)
    diagnostics: Diagnostics = {
        "loss": loss,
        "phase_rate_loss": loss,
        "phase_rate_rms": jnp.sqrt(loss),
        "amplitude_rate_loss": amplitude_loss,
        "amplitude_rate_rms": jnp.sqrt(amplitude_loss),
        "kinematic_growth_loss": kinematic_loss,
        "kinematic_growth_rms": jnp.sqrt(kinematic_loss),
        "active_modes": _selected_mean(
            per_sample_active_modes, selected, selected_count
        ),
        "active_elevation_energy": _selected_mean(
            per_sample_energy, selected, selected_count
        ),
        "selected_samples": selected_count,
    }
    return loss, diagnostics
