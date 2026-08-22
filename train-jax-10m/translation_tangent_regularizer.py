"""Localized translation-tangent matching for Dirichlet--Neumann data."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class TranslationTangentConfig:
    """Configuration for the local projection of DNO error onto ``eta_x``."""

    window_depths: float = 1.0
    energy_floor_relative: float = 1e-3
    gravity: float = 1.0


def _periodic_gaussian_smooth(
    fields: jax.Array,
    depth: jax.Array,
    k_rfft: jax.Array,
    window_depths: float,
) -> jax.Array:
    """Smooth ``(B, C, N)`` fields over a periodic window proportional to depth."""
    sigma = jnp.asarray(window_depths, dtype=fields.dtype) * depth
    multiplier = jnp.exp(
        -0.5 * (sigma[:, None] * k_rfft[None, :]) ** 2
    )
    fields_hat = jnp.fft.rfft(fields, axis=-1)
    return jnp.fft.irfft(
        fields_hat * multiplier[:, None, :],
        n=fields.shape[-1],
        axis=-1,
    ).astype(fields.dtype)


def compute_translation_tangent_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    config: TranslationTangentConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Match the local translation component of the DNO error.

    At each location, a periodic Gaussian window defines the least-squares
    coefficient of ``gxi_prediction - gxi_target`` along ``eta_x`` after each
    DNO output is projected to zero spatial mean, matching rollout semantics.
    Dividing that coefficient by ``sqrt(g h)`` gives a dimensionless error
    without the singular inverse fitted-speed weighting of a global
    traveling-wave fit.

    The primary return remains the dimensionless relative-speed loss.  The
    diagnostics additionally expose the phase-growth loss

    ``gamma^2 = ||eta_x||_2^2 / ||eta||_2^2 * g h * relative_speed_loss``,

    evaluated per sample before averaging.  For a locally rigid translation
    defect, ``gamma`` is the instantaneous growth rate of relative elevation
    error.  Keeping both quantities permits independent trainer weights while
    preserving the original objective exactly.
    """
    dtype = eta.dtype
    eta_hat = jnp.fft.fft(eta, axis=-1)
    eta_x = jnp.real(
        jnp.fft.ifft(1j * k[None, :] * eta_hat, axis=-1)
    ).astype(dtype)
    centered_prediction = gxi_prediction - jnp.mean(
        gxi_prediction, axis=-1, keepdims=True
    )
    centered_target = gxi_target - jnp.mean(
        gxi_target, axis=-1, keepdims=True
    )
    error = centered_prediction - centered_target
    fields = jnp.stack((eta_x * error, eta_x**2), axis=1)
    smoothed = _periodic_gaussian_smooth(
        fields,
        depth,
        jnp.abs(k[: eta.shape[-1] // 2 + 1]),
        config.window_depths,
    )
    local_cross = smoothed[:, 0]
    local_energy = jnp.maximum(smoothed[:, 1], jnp.asarray(0.0, dtype=dtype))
    energy_floor = (
        jnp.asarray(config.energy_floor_relative, dtype=dtype)
        * jnp.max(local_energy, axis=-1, keepdims=True)
    )
    local_speed_error = -local_cross / (
        local_energy + energy_floor + jnp.asarray(1e-30, dtype=dtype)
    )
    physical_speed = jnp.sqrt(
        jnp.asarray(config.gravity, dtype=dtype)
        * jnp.maximum(depth, jnp.asarray(1e-12, dtype=dtype))
    )
    local_relative_error = jnp.abs(local_speed_error) / physical_speed[:, None]
    local_weight = local_energy / (
        jnp.sum(local_energy, axis=-1, keepdims=True)
        + jnp.asarray(1e-30, dtype=dtype)
    )
    per_sample_loss = jnp.sum(
        local_weight * local_relative_error**2,
        axis=-1,
    )
    per_sample_abs_error = jnp.sum(
        local_weight * jnp.abs(local_speed_error),
        axis=-1,
    )
    per_sample_relative_error = jnp.sum(
        local_weight * local_relative_error,
        axis=-1,
    )
    signed_local_relative_error = local_speed_error / physical_speed[:, None]
    rigid_relative_speed = jnp.sum(
        local_weight * signed_local_relative_error,
        axis=-1,
    )
    per_sample_rigid_speed_loss = rigid_relative_speed**2
    per_sample_differential_speed_loss = jnp.maximum(
        per_sample_loss - per_sample_rigid_speed_loss,
        jnp.asarray(0.0, dtype=dtype),
    )

    eta_energy = jnp.sum(eta**2, axis=-1)
    slope_energy = jnp.sum(local_energy, axis=-1)
    rms_wave_number_sq = slope_energy / (
        eta_energy + jnp.asarray(1e-30, dtype=dtype)
    )
    phase_growth_multiplier = physical_speed**2 * rms_wave_number_sq
    per_sample_phase_growth_loss = phase_growth_multiplier * per_sample_loss
    per_sample_rigid_phase_growth_loss = (
        phase_growth_multiplier * per_sample_rigid_speed_loss
    )
    per_sample_differential_phase_growth_loss = (
        phase_growth_multiplier * per_sample_differential_speed_loss
    )

    selected = (
        jnp.max(local_energy, axis=-1) > jnp.asarray(1e-20, dtype=dtype)
    ).astype(dtype)
    selected_count = jnp.sum(selected)
    selected_denom = jnp.maximum(selected_count, jnp.asarray(1.0, dtype=dtype))

    def selected_mean(values: jax.Array) -> jax.Array:
        return jnp.sum(selected * values) / selected_denom

    loss = selected_mean(per_sample_loss)
    phase_growth_loss = selected_mean(per_sample_phase_growth_loss)
    diagnostics = {
        "loss": loss,
        "relative_speed_loss": loss,
        "phase_growth_loss": phase_growth_loss,
        "phase_growth_rate": selected_mean(
            jnp.sqrt(per_sample_phase_growth_loss)
        ),
        "rms_wave_number": selected_mean(jnp.sqrt(rms_wave_number_sq)),
        "phase_growth_multiplier": selected_mean(phase_growth_multiplier),
        "rigid_phase_growth_loss": selected_mean(
            per_sample_rigid_phase_growth_loss
        ),
        "differential_phase_growth_loss": selected_mean(
            per_sample_differential_phase_growth_loss
        ),
        "speed_abs_error": selected_mean(per_sample_abs_error),
        "speed_relative_error": selected_mean(per_sample_relative_error),
        "selected_samples": selected_count,
    }
    return loss, diagnostics
