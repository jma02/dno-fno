"""Translation and coupled modal-phase matching for Dirichlet--Neumann data."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from solver.solvers.time_integrator import (
    _spectral_truncate_real,
    _spectral_upsample_real,
)


def compute_modal_phase_loss(
    eta: jax.Array,
    xi: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    *,
    sample_mask: jax.Array | None = None,
    k_max: float = 128.0,
    denominator_eps: float = 1e-3,
    gravity: float = 1.0,
    loss_norm: Literal["squared", "charbonnier"] = "squared",
    charbonnier_eps: float = 1e-7,
) -> tuple[jax.Array, jax.Array]:
    """Match the two traveling-mode phase rates of the coupled water-wave RHS.

    With forward-normalized Fourier coefficients, the normal coordinates are
    ``a_± = sqrt(g) eta_hat ± i sqrt(G0) xi_hat``. Their linear phase rates are
    ``∓omega``, where ``G0 = |k| tanh(|k| h)`` and ``omega = sqrt(g G0)``.
    The phase-rate defect is ``imag(conj(a_±) delta_a_t_±) / |a_±|^2``;
    division by omega makes it dimensionless. Squared defects are averaged
    with the reference normal-coordinate energies before the optional smooth
    RMS transform. The denominator floor is relative to the largest retained
    mode energy in each row, and zero-energy rows are excluded.

    Both equations use the same state, so their RHS difference needs only
    ``delta_q`` and the Bernoulli difference. The latter is evaluated in its
    cancellation-free factorized form on the solver's two-times Fourier grid.
    DNO outputs are mean-centered, but their high modes enter this product
    before the final mode restriction, matching surrogate rollout semantics.
    DC and Nyquist are excluded. Depth is physical and need not be clipped.
    """
    dtype = eta.dtype
    nx = eta.shape[-1]
    centered_prediction = gxi_prediction - jnp.mean(
        gxi_prediction, axis=-1, keepdims=True
    )
    centered_target = gxi_target - jnp.mean(gxi_target, axis=-1, keepdims=True)
    error = centered_prediction - centered_target
    eta_hat, xi_hat, error_hat = (
        jnp.fft.rfft(field, axis=-1, norm="forward").at[..., -1].set(0)
        for field in (eta, xi, error)
    )
    k_abs = jnp.abs(k[: nx // 2 + 1]).astype(dtype)
    eta_x, xi_x = (
        jnp.fft.irfft(1j * k_abs[None, :] * field_hat, n=nx, norm="forward")
        for field_hat in (eta_hat, xi_hat)
    )
    padded = _spectral_upsample_real(
        jnp.stack((eta_x, xi_x, centered_target, error), axis=1), 2 * nx
    )
    eta_x_padded, xi_x_padded, target_padded, error_padded = (
        padded[:, index] for index in range(4)
    )
    xi_rhs_error = _spectral_truncate_real(
        error_padded
        * (target_padded + eta_x_padded * xi_x_padded + 0.5 * error_padded)
        / (1.0 + eta_x_padded**2),
        nx,
    )
    xi_rhs_error_hat = jnp.fft.rfft(xi_rhs_error, axis=-1, norm="forward")

    g0 = k_abs[None, :] * jnp.tanh(depth[:, None] * k_abs[None, :])
    sqrt_g = jnp.sqrt(jnp.asarray(gravity, dtype=dtype))
    sqrt_g0 = jnp.sqrt(g0)
    signs = jnp.asarray([1.0, -1.0], dtype=dtype)[None, :, None]
    coordinates = sqrt_g * eta_hat[:, None, :] + (
        signs * 1j * sqrt_g0[:, None, :] * xi_hat[:, None, :]
    )
    rhs_error = sqrt_g * error_hat[:, None, :] + (
        signs * 1j * sqrt_g0[:, None, :] * xi_rhs_error_hat[:, None, :]
    )
    band = (k_abs > 0.0) & (k_abs <= k_max)
    band = band.at[-1].set(False)
    energy = jax.lax.stop_gradient(jnp.where(
        band[None, None, :], jnp.abs(coordinates) ** 2, 0.0
    ))
    peak_energy = jnp.max(energy, axis=(1, 2), keepdims=True)
    absolute_floor = jnp.asarray(1e-30, dtype=dtype)
    total_energy = jnp.sum(energy, axis=(1, 2))
    selected = total_energy > absolute_floor
    if sample_mask is not None:
        selected = selected & sample_mask
    denominator = jnp.where(
        selected[:, None, None],
        energy + jnp.asarray(denominator_eps, dtype=dtype) * peak_energy + absolute_floor,
        1.0,
    )
    omega = jnp.sqrt(jnp.where(band[None, :], gravity * g0, 1.0))
    phase_numerator = jnp.where(
        band[None, None, :], jnp.imag(jnp.conj(coordinates) * rhs_error), 0.0
    )
    phase_rate_error = phase_numerator / (
        omega[:, None, :] * denominator
    )
    mode_weight = energy / jnp.maximum(total_energy[:, None, None], absolute_floor)
    per_sample_loss = jnp.sum(mode_weight * phase_rate_error**2, axis=(1, 2))
    if loss_norm == "charbonnier":
        eps = jnp.asarray(charbonnier_eps, dtype=dtype)
        per_sample_loss = per_sample_loss / (
            jnp.sqrt(per_sample_loss + eps**2) + eps
        )
    selected_count = jnp.sum(selected.astype(dtype))
    loss = jnp.sum(jnp.where(selected, per_sample_loss, 0.0)) / jnp.maximum(
        selected_count, 1.0
    )
    return loss, selected_count


def compute_translation_tangent_loss(
    eta: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    *,
    sample_mask: jax.Array | None = None,
    global_relative_mask: jax.Array | None = None,
    smoothing_scale: float = 1.0,
    denominator_eps: float = 1e-3,
    gravity: float = 1.0,
    loss_norm: Literal["squared", "charbonnier"] = "squared",
    charbonnier_eps: float = 1e-7,
) -> tuple[jax.Array, jax.Array]:
    """Return the translation loss and count of selected nonflat samples.

    An optional boolean ``sample_mask`` selects which batch rows contribute.

    At each location, a periodic Gaussian window defines the least-squares
    coefficient of ``gxi_prediction - gxi_target`` along ``eta_x`` after each
    DNO output is projected to zero spatial mean, matching rollout semantics.
    Dividing that coefficient by ``sqrt(g h)`` gives a dimensionless error
    without the singular inverse fitted-speed weighting of a global
    traveling-wave fit.

    Rows in ``global_relative_mask`` instead use the squared global error
    projection divided by the squared target projection, with a scale-aware
    floor. Near-zero/cancelling target projections retain the local loss.

    ``charbonnier`` uses sqrt(row_loss + eps^2) - eps before averaging.
    This is a smooth absolute global or RMS local phase error, with finite
    zero-error gradients and nonvanishing slope outside the smoothing floor.
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
    if global_relative_mask is not None:
        error_projection = jnp.mean(error * eta_x, axis=-1)
        target_projection_squared = jnp.mean(centered_target * eta_x, axis=-1) ** 2
        norm_product = jnp.mean(centered_target**2, axis=-1) * jnp.mean(eta_x**2, axis=-1)
        projection_floor = jnp.asarray(denominator_eps, dtype=dtype) * norm_product
        use_global = global_relative_mask & (target_projection_squared > projection_floor)
        denominator = jnp.where(
            use_global, target_projection_squared + projection_floor + 1e-30, 1.0
        )
        per_sample_loss = jnp.where(
            use_global, error_projection**2 / denominator, per_sample_loss
        )
    if loss_norm == "charbonnier":
        eps = jnp.asarray(charbonnier_eps, dtype=dtype)
        # Rationalization avoids subtractive cancellation near zero.
        per_sample_loss = per_sample_loss / (
            jnp.sqrt(per_sample_loss + eps**2) + eps
        )
    selected = peak_energy > jnp.asarray(1e-20, dtype=dtype)
    if sample_mask is not None:
        selected = selected & sample_mask
    selected = selected.astype(dtype)
    selected_count = jnp.sum(selected)
    loss = jnp.sum(selected * per_sample_loss) / jnp.maximum(selected_count, 1.0)
    return loss, selected_count
