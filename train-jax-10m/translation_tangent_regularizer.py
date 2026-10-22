"""Translation and coupled modal-phase matching for Dirichlet--Neumann data."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from solver.solvers.time_integrator import (
    _spectral_truncate_real,
    _spectral_upsample_real,
    dealiased_zakharov_xi_rhs,
)


def _centered_error(
    prediction: jax.Array, target: jax.Array, subtract_before_centering: bool,
) -> tuple[jax.Array, jax.Array]:
    centered_target = target - jnp.mean(target, axis=-1, keepdims=True)
    if subtract_before_centering:
        error = prediction - target
        error = error - jnp.mean(error, axis=-1, keepdims=True)
    else:
        error = prediction - jnp.mean(prediction, axis=-1, keepdims=True) - centered_target
    return error, centered_target


def _bernoulli_rhs_defect(
    eta_x: jax.Array, xi_x: jax.Array, target: jax.Array, error: jax.Array,
) -> jax.Array:
    """Cancellation-free difference of the two dealiased xi equations."""
    nx = eta_x.shape[-1]
    padded = _spectral_upsample_real(jnp.stack((eta_x, xi_x, target, error), axis=1), 2 * nx)
    eta_x_padded, xi_x_padded, target_padded, error_padded = (
        padded[:, index] for index in range(4)
    )
    return _spectral_truncate_real(
        error_padded * (target_padded + eta_x_padded * xi_x_padded + 0.5 * error_padded)
        / (1.0 + eta_x_padded**2), nx,
    )


def compute_rhs_relative_error(
    eta: jax.Array,
    xi: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    *,
    gravity: float = 1.0,
    k_max: float = 128.0,
) -> jax.Array:
    """Per-row retained Zakharov RHS error in the quadratic wave-energy norm.

    Full mean-free q enters the doubled-grid Bernoulli term before projection.
    The cancellation-free xi defect includes both occurrences of predicted q.
    The reference xi equation includes gravity; the denominator is detached.
    The shared factor two for positive Fourier modes cancels in the ratio.
    """
    eta, xi, gxi_prediction, gxi_target, depth = map(
        jnp.float64, (eta, xi, gxi_prediction, gxi_target, depth),
    )
    nx = eta.shape[-1]
    error, target = _centered_error(gxi_prediction, gxi_target, True)
    k_abs = jnp.abs(k[:nx // 2 + 1]).astype(eta.dtype)
    eta_x, xi_x = (
        jnp.fft.irfft(
            1j * k_abs * jnp.fft.rfft(field, axis=-1, norm="forward").at[..., -1].set(0),
            n=nx, norm="forward",
        )
        for field in (eta, xi)
    )
    xi_error = _bernoulli_rhs_defect(eta_x, xi_x, target, error)
    xi_target = -gravity * eta + dealiased_zakharov_xi_rhs(eta_x, xi_x, target)
    error_hat, xi_error_hat, target_hat, xi_target_hat = (
        jnp.fft.rfft(field, axis=-1, norm="forward").astype(jnp.complex128)
        for field in (error, xi_error, target, xi_target)
    )
    g0 = (k_abs[None, :] * jnp.tanh(depth[:, None] * k_abs[None, :])).astype(jnp.float64)
    band = ((k_abs > 0.0) & (k_abs <= k_max)).at[-1].set(False)
    squared_error = jnp.sum(jnp.where(
        band, gravity * jnp.abs(error_hat)**2 + g0 * jnp.abs(xi_error_hat)**2, 0.0,
    ), axis=-1)
    squared_target = jnp.sum(jnp.where(
        band, gravity * jnp.abs(target_hat)**2 + g0 * jnp.abs(xi_target_hat)**2, 0.0,
    ), axis=-1)
    zero_error = squared_error == 0.0
    error_norm = jnp.where(zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error)))
    denominator = jax.lax.stop_gradient(jnp.sqrt(jnp.maximum(squared_target, 1e-30)))
    return error_norm / denominator


def compute_coherent_rollout_loss(
    stage_eta: jax.Array,
    stage_xi: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    stage_times: jax.Array,
    stage_weights: jax.Array,
    initial_eta: jax.Array,
    initial_xi: jax.Array,
    valid_mask: jax.Array,
    *,
    gravity: float = 1.0,
    k_max: float = 128.0,
) -> jax.Array:
    """Relative norm of a detached path's signed, GL2-weighted RHS defect.

    Fields have shape (steps, 2, batch, nx); times and weights are (steps, 2).
    Times are relative to the initial state, and weights include dt/2. Only
    predictions receive gradients. Invalid rows must be sanitized upstream.
    Full mean-free q enters Bernoulli before retaining modes and transporting
    both equations backwards by the flat-water linear flow. The denominator
    is the initial quadratic energy norm, not a per-stage RHS norm.
    """
    stage_eta, stage_xi, gxi_target, depth, k, stage_times, stage_weights, initial_eta, initial_xi = map(
        lambda value: jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float64)),
        (stage_eta, stage_xi, gxi_target, depth, k, stage_times, stage_weights, initial_eta, initial_xi),
    )
    valid_mask = jax.lax.stop_gradient(valid_mask)
    g = jax.lax.stop_gradient(jnp.asarray(gravity, dtype=jnp.float64))
    nx = stage_eta.shape[-1]
    k_abs = jnp.abs(k[:nx // 2 + 1])
    error, target = _centered_error(
        jnp.asarray(gxi_prediction, dtype=jnp.float64).reshape(-1, nx),
        gxi_target.reshape(-1, nx), True,
    )
    eta_x, xi_x = (
        jnp.fft.irfft(
            1j * k_abs * jnp.fft.rfft(field.reshape(-1, nx), norm="forward").at[..., -1].set(0),
            n=nx, norm="forward",
        )
        for field in (stage_eta, stage_xi)
    )
    eta_error_hat, xi_error_hat = (
        jnp.fft.rfft(field, norm="forward").reshape(*stage_eta.shape[:-1], nx // 2 + 1)
        for field in (error, _bernoulli_rhs_defect(eta_x, xi_x, target, error))
    )
    g0 = k_abs * jnp.tanh(depth[:, None] * k_abs)
    eta_error_hat, xi_error_hat = jnp.sqrt(g) * eta_error_hat, jnp.sqrt(g0) * xi_error_hat
    angle = stage_times[..., None, None] * jnp.sqrt(g * g0)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    accumulated_eta, accumulated_xi = (
        jnp.sum(stage_weights[..., None, None] * field, axis=(0, 1))
        for field in (cosine * eta_error_hat - sine * xi_error_hat,
                      sine * eta_error_hat + cosine * xi_error_hat)
    )
    band = ((k_abs > 0.0) & (k_abs <= k_max)).at[-1].set(False)
    squared_error = jnp.sum(jnp.where(
        band, jnp.abs(accumulated_eta)**2 + jnp.abs(accumulated_xi)**2, 0.0,
    ), axis=-1)
    initial_eta_hat, initial_xi_hat = (
        jnp.fft.rfft(field, norm="forward") for field in (initial_eta, initial_xi)
    )
    squared_reference = jnp.sum(jnp.where(
        band, g * jnp.abs(initial_eta_hat)**2 + g0 * jnp.abs(initial_xi_hat)**2, 0.0,
    ), axis=-1)
    zero_error = squared_error == 0.0
    error_norm = jnp.where(zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error)))
    relative = error_norm / jnp.sqrt(jnp.maximum(squared_reference, 1e-30))
    return jnp.sum(jnp.where(valid_mask, relative, 0.0)) / jnp.maximum(jnp.sum(valid_mask), 1)


def compute_phase_mechanism_loss(
    eta: jax.Array,
    xi: jax.Array,
    gxi_prediction: jax.Array,
    gxi_target: jax.Array,
    depth: jax.Array,
    k: jax.Array,
    *,
    mechanism: Literal["carrier", "sideband", "momentum"],
    sample_mask: jax.Array | None = None,
    carrier_mode: jax.Array | None = None,
    sideband_offset: jax.Array | None = None,
    horizon: float = 1.0,
    activity_floor: float = 1e-6,
    charbonnier_eps: float = 1e-7,
    gravity: float = 1.0,
    k_max: float = 128.0,
    subtract_before_centering: bool = False,
    return_irls_residuals: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Match a signed RHS defect, with smooth absolute loss.

    Carrier: T times the phase-rate defect of the dominant traveling coordinate
    at the prescribed carrier (or largest-energy mode when omitted). Sideband:
    T times delta_omega_lower + delta_omega_upper - 2 delta_omega_carrier, using
    the same traveling direction and fixed carrier/offset metadata, not a peak
    that can change during modulation. Normal coordinates are sqrt(g)*eta_hat
    +/- i*sqrt(G0)*xi_hat. Inactive modes are excluded, not denominator-floored
    or energy-weighted; this keeps active weak sidebands influential.

    Momentum: T*mean(xi_x*delta_q - eta_x*delta_xi_t)/(RMS(eta)*RMS(xi_x)).
    This matches momentum production of the supplied target RHS, not zero.
    Training labels are P128 M6 outputs; their missing high modes make this an
    approximation to the full-q M6 momentum target in the Bernoulli equation.
    All three residuals are dimensionless. T is a normalization horizon, not a
    trajectory prediction. Only predictions receive parameter gradients; no
    solver evaluation, trajectory differentiation, or new network call occurs.
    ``return_irls_residuals`` returns the signed, locally reweighted residual and
    selected count instead of the scalar mean, using the ``0.5 * ||r||^2`` convention.
    """
    nx, dtype = eta.shape[-1], eta.dtype
    error, target = _centered_error(gxi_prediction, gxi_target, subtract_before_centering)
    eta_hat, xi_hat, error_hat = (
        jnp.fft.rfft(field, axis=-1, norm="forward").at[..., -1].set(0)
        for field in (eta, xi, error)
    )
    k_abs = jnp.abs(k[:nx // 2 + 1]).astype(dtype)
    eta_x, xi_x = (
        jnp.fft.irfft(1j * k_abs * spectrum, n=nx, norm="forward")
        for spectrum in (eta_hat, xi_hat)
    )
    rhs_error = _bernoulli_rhs_defect(eta_x, xi_x, target, error)
    if mechanism == "momentum":
        scale = jnp.sqrt(jnp.mean(eta**2, axis=-1) * jnp.mean(xi_x**2, axis=-1))
        selected = scale > 1e-30
        residual = jnp.mean(xi_x * error - eta_x * rhs_error, axis=-1) / jnp.where(selected, scale, 1.0)
    else:
        signs = jnp.asarray([1.0, -1.0], dtype=dtype)[None, :, None]
        sqrt_g0 = jnp.sqrt(k_abs * jnp.tanh(depth[:, None] * k_abs))
        sqrt_g = jnp.sqrt(jnp.asarray(gravity, dtype=dtype))
        coordinates = sqrt_g * eta_hat[:, None, :] + signs * 1j * sqrt_g0[:, None, :] * xi_hat[:, None, :]
        band = (k_abs > 0) & (k_abs <= k_max)
        band = band.at[-1].set(False)
        energy = jnp.where(band, jnp.abs(coordinates)**2, 0.0)
        if carrier_mode is None:
            if mechanism == "sideband":
                raise ValueError("Sideband phase loss requires fixed carrier-mode metadata")
            carrier_mode = jnp.argmax(jnp.sum(energy, axis=1), axis=-1)
        if mechanism == "sideband":
            if sideband_offset is None:
                raise ValueError("Sideband phase loss requires fixed sideband-offset metadata")
            modes = jnp.stack((carrier_mode - sideband_offset, carrier_mode,
                               carrier_mode + sideband_offset), axis=-1)
        else:
            modes = carrier_mode[:, None]
        rows = jnp.arange(eta.shape[0])
        direction = jnp.argmax(energy[rows, :, jnp.clip(carrier_mode, 0, nx // 2)], axis=-1)
        selected_modes = jnp.clip(modes, 0, nx // 2)
        mode_energy = energy[rows[:, None], direction[:, None], selected_modes]
        peak = jnp.max(energy, axis=(1, 2))
        active = ((modes > 0) & (modes < nx // 2)
                  & (mode_energy > activity_floor * peak[:, None]) & (mode_energy > 1e-30))
        selected = jnp.all(active, axis=-1)
        coordinate_defect = sqrt_g * error_hat[:, None, :] + signs * 1j * sqrt_g0[:, None, :] * jnp.fft.rfft(
            rhs_error, axis=-1, norm="forward",
        )[:, None, :]
        numerator = jnp.imag(jnp.conj(coordinates) * coordinate_defect)
        rates = numerator[rows[:, None], direction[:, None], selected_modes] / jnp.where(active, mode_energy, 1.0)
        residual = rates[:, 0] + rates[:, 2] - 2.0 * rates[:, 1] if mechanism == "sideband" else rates[:, 0]
    if sample_mask is not None:
        selected = selected & sample_mask
    residual = jnp.where(selected, horizon * residual, 0.0)
    squared = residual**2
    eps = jnp.asarray(charbonnier_eps, dtype=dtype)
    per_sample = squared / (jnp.sqrt(squared + eps**2) + eps)
    count = jnp.sum(selected.astype(dtype))
    if return_irls_residuals:
        weight = selected.astype(dtype) / (
            jnp.maximum(count, 1.0) * jnp.sqrt(squared + eps**2)
        )
        return residual * jnp.sqrt(jax.lax.stop_gradient(weight)), count
    return jnp.sum(per_sample) / jnp.maximum(count, 1.0), count


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
    subtract_before_centering: bool = False,
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
    error, centered_target = _centered_error(
        gxi_prediction, gxi_target, subtract_before_centering,
    )
    eta_hat, xi_hat, error_hat = (
        jnp.fft.rfft(field, axis=-1, norm="forward").at[..., -1].set(0)
        for field in (eta, xi, error)
    )
    k_abs = jnp.abs(k[: nx // 2 + 1]).astype(dtype)
    eta_x, xi_x = (
        jnp.fft.irfft(1j * k_abs[None, :] * field_hat, n=nx, norm="forward")
        for field_hat in (eta_hat, xi_hat)
    )
    xi_rhs_error = _bernoulli_rhs_defect(eta_x, xi_x, centered_target, error)
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
    subtract_before_centering: bool = False,
    return_irls_residuals: bool = False,
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
    ``return_irls_residuals`` returns signed spatial residuals and the selected
    count, with locally frozen Charbonnier weights if requested. Their
    ``0.5 * ||r||^2`` gradient agrees with the scalar loss at the linearization point.
    """
    dtype = eta.dtype
    eta_hat = jnp.fft.fft(eta, axis=-1)
    eta_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * eta_hat, axis=-1)).astype(dtype)
    error, centered_target = _centered_error(
        gxi_prediction, gxi_target, subtract_before_centering,
    )
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
    residuals = None
    if return_irls_residuals:
        residuals = jnp.sqrt(local_weight) * local_speed_error / physical_speed[:, None]
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
        if residuals is not None:
            residuals = jnp.where(
                use_global[:, None],
                (error_projection / jnp.sqrt(denominator * eta.shape[-1]))[:, None],
                residuals,
            )
    row_squared_loss = per_sample_loss
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
    if residuals is not None:
        weight = 2.0 * selected / jnp.maximum(selected_count, 1.0)
        if loss_norm == "charbonnier":
            eps = jnp.asarray(charbonnier_eps, dtype=dtype)
            weight /= 2.0 * jnp.sqrt(row_squared_loss + eps**2)
        return residuals * jnp.sqrt(jax.lax.stop_gradient(weight[:, None])), selected_count
    loss = jnp.sum(selected * per_sample_loss) / jnp.maximum(selected_count, 1.0)
    return loss, selected_count
