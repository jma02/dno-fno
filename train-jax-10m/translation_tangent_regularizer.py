"""Localized translation-tangent matching for Dirichlet--Neumann data."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_translation_tangent_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    *,
    sample_mask: jax.Array | None = None,
    smoothing_scale: float = 1.0,
    denominator_eps: float = 1e-3,
    gravity: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Return the local translation loss and count of selected nonflat samples.

    An optional boolean ``sample_mask`` selects which batch rows contribute.

    At each location, a periodic Gaussian window defines the least-squares
    coefficient of ``gxi_prediction - gxi_target`` along ``eta_x`` after each
    DNO output is projected to zero spatial mean, matching rollout semantics.
    Dividing that coefficient by ``sqrt(g h)`` gives a dimensionless error
    without the singular inverse fitted-speed weighting of a global
    traveling-wave fit.
    """
    dtype = eta.dtype
    eta_hat = jnp.fft.fft(eta, axis=-1)
    eta_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * eta_hat, axis=-1)).astype(dtype)
    centered_prediction = gxi_prediction - jnp.mean(
        gxi_prediction, axis=-1, keepdims=True
    )
    centered_target = gxi_target - jnp.mean(gxi_target, axis=-1, keepdims=True)
    error = centered_prediction - centered_target
    fields = jnp.stack((eta_x * error, eta_x**2), axis=1)
    k_rfft = jnp.abs(k[: eta.shape[-1] // 2 + 1])
    sigma = jnp.asarray(smoothing_scale, dtype=fields.dtype) * depth
    multiplier = jnp.exp(-0.5 * (sigma[:, None] * k_rfft[None, :]) ** 2)
    fields_hat = jnp.fft.rfft(fields, axis=-1)
    smoothed = jnp.fft.irfft(
        fields_hat * multiplier[:, None, :], n=fields.shape[-1], axis=-1
    ).astype(fields.dtype)
    local_cross = smoothed[:, 0]
    local_energy = jnp.maximum(smoothed[:, 1], 0.0)
    peak_energy = jnp.max(local_energy, axis=-1)
    energy_floor = jnp.asarray(denominator_eps, dtype=dtype) * peak_energy[:, None]
    local_speed_error = -local_cross / (
        local_energy + energy_floor + jnp.asarray(1e-30, dtype=dtype)
    )
    physical_speed = jnp.sqrt(
        jnp.asarray(gravity, dtype=dtype)
        * jnp.maximum(depth, jnp.asarray(1e-12, dtype=dtype))
    )
    local_relative_error = jnp.abs(local_speed_error) / physical_speed[:, None]
    local_weight = local_energy / (
        jnp.sum(local_energy, axis=-1, keepdims=True) + jnp.asarray(1e-30, dtype=dtype)
    )
    per_sample_loss = jnp.sum(
        local_weight * local_relative_error**2,
        axis=-1,
    )
    selected = peak_energy > jnp.asarray(1e-20, dtype=dtype)
    if sample_mask is not None:
        selected = selected & sample_mask
    selected = selected.astype(dtype)
    selected_count = jnp.sum(selected)
    loss = jnp.sum(selected * per_sample_loss) / jnp.maximum(selected_count, 1.0)
    return loss, selected_count
