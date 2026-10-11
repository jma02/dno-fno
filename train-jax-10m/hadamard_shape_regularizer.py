"""Hadamard consistency and teacher-labelled local responses for a learned DNO.

For a Dirichlet--Neumann operator ``G(eta)`` and arbitrary surface direction
``zeta``, the Hadamard identity is

    D_eta G(eta)[zeta] xi = -G(eta)(zeta * B) - d_x(zeta * V),

where ``B = (Gxi + eta_x * xi_x) / (1 + eta_x**2)`` and
``V = xi_x - B * eta_x``. ``compute_hadamard_loss`` measures this identity using
only the learned operator. ``compute_local_response_loss`` instead anchors
joint surface/potential responses to a frozen order-six Craig--Sulem reference.

The finite secant avoids differentiating a JVP through the parameter loss.  A
small, relative perturbation is used so the same configuration applies across
the range of surface amplitudes in the clean training data.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeAlias

import jax
import jax.numpy as jnp

from solver.gen_data.pipeline.dno_target import project_fixed_band
from solver.solvers.dno_series_jax import build_grid, dno_series_eval
from solver.solvers.time_integrator import dealiased_zakharov_xi_rhs


Array: TypeAlias = jax.Array
ApplyFn: TypeAlias = Callable[[dict[str, Any], Array, Array], Array]
NormalizeInputsFn: TypeAlias = Callable[[Array, Array], Array]
DenormalizeTargetsFn: TypeAlias = Callable[[Array], Array]


def projected_sobolev_energy(
    field: Array,
    k: Array,
    k_max: float,
    sobolev_order: Literal[0, 1],
) -> Array:
    """Per-sample L2 or H1 energy after ``|k| <= k_max`` projection.

    ``field`` has shape ``(..., nx)`` and ``k`` has shape ``(nx,)``.  The
    unnormalised FFT is divided by ``nx**2``, so order zero agrees with the
    physical-space mean square by Parseval.
    """
    nx = field.shape[-1]
    field_hat = jnp.fft.fft(field, axis=-1)
    projector = (jnp.abs(k) <= jnp.asarray(k_max, dtype=k.dtype)).astype(
        field_hat.real.dtype
    )
    weight_sq = jnp.ones_like(k) if sobolev_order == 0 else 1 + k**2
    weighted_energy = (
        projector * weight_sq.astype(field_hat.real.dtype) * jnp.abs(field_hat) ** 2
    )
    return jnp.sum(weighted_energy, axis=-1) / jnp.asarray(
        nx * nx, dtype=field_hat.real.dtype
    )


def spectral_dx(field: Array, k: Array) -> Array:
    """Periodic spectral derivative along the last axis."""
    field_hat = jnp.fft.fft(field, axis=-1)
    return jnp.real(jnp.fft.ifft(1j * k * field_hat, axis=-1))


def compute_hadamard_loss(
    rng: Array,
    apply_fn: ApplyFn,
    model_params: Any,
    eta_phys: Array,
    xi_phys: Array,
    batch_depth_local: Array,
    norm_inputs_fn: NormalizeInputsFn,
    denorm_targets_fn: DenormalizeTargetsFn,
    k: Array,
    dtype: jnp.dtype,
    *,
    k_max: float = 128.0,
    sobolev_order: Literal[0, 1] = 1,
    fd_step_min: float = 1e-3,
    fd_step_max: float = 3e-3,
    min_surface_rms: float = 1e-3,
    denominator_eps: float = 1e-12,
    flow_target: Array | None = None,
) -> Array:
    """Return the normalized finite-secant Hadamard loss.

    Inputs are clean physical training states of shape ``(batch, nx)``;
    ``batch_depth_local`` is the model's usual depth input for those samples.
    The loss is

    ``mean(||W P S||^2 / (||W P R||^2 + denominator_eps))``,

    where ``S`` is the Hadamard residual and
    ``R = G(eta)(zeta B) + d_x(zeta V)`` is the identity's right-hand-side
    magnitude (up to sign).

    ``fd_step_*`` are dimensionless: the direction is scaled to each surface's
    RMS amplitude, with ``min_surface_rms`` as the physical fallback near flatness.
    With physical ``flow_target = G(eta)xi``, half the rows in expectation use
    its retained, mean-zero direction instead of a random probe. Zero-flow rows
    retain the random probe; validation can omit this target to stay unchanged.
    """
    eta = eta_phys.astype(dtype)
    xi = xi_phys.astype(dtype)
    k_typed = k.astype(dtype)
    depth = batch_depth_local.astype(dtype)
    batch_size = eta.shape[0]
    key_probe, key_eps = jax.random.split(rng)

    # Keep nonzero modes up to k_max, dividing by the Sobolev weight so
    # the largest wavenumbers do not dominate the weighted probe energy.
    raw = jax.random.normal(key_probe, eta.shape, dtype=dtype)
    raw_hat = jnp.fft.fft(raw, axis=-1)
    k_abs = jnp.abs(k_typed)
    projector = (k_abs > 0.0) & (k_abs <= jnp.asarray(k_max, dtype=dtype))
    weight_sq = jnp.ones_like(k_typed) if sobolev_order == 0 else 1 + k_typed**2
    probe_hat = raw_hat * projector[None, :] / jnp.sqrt(weight_sq)[None, :]
    probe = jnp.real(jnp.fft.ifft(probe_hat, axis=-1))
    probe_rms = jnp.sqrt(jnp.mean(probe * probe, axis=-1))
    unit_probe = probe / jnp.maximum(
        probe_rms[:, None], jnp.asarray(1e-30, dtype=dtype)
    )
    if flow_target is not None:
        flow = jnp.fft.ifft(
            jnp.fft.fft(flow_target.astype(dtype), axis=-1) * projector, axis=-1,
        ).real
        flow_rms = jnp.sqrt(jnp.mean(flow**2, axis=-1, keepdims=True))
        use_flow = jax.random.bernoulli(
            jax.random.fold_in(rng, 104729), shape=(batch_size, 1),
        ) & (flow_rms > 1e-30)
        unit_probe = jnp.where(
            use_flow, flow / jnp.maximum(flow_rms, 1e-30), unit_probe,
        )

    # Match each direction's RMS to its mean-subtracted surface amplitude.
    eta_centered = eta - jnp.mean(eta, axis=-1, keepdims=True)
    eta_rms = jnp.sqrt(jnp.mean(eta_centered * eta_centered, axis=-1))
    eta_scale = jnp.maximum(eta_rms, jnp.asarray(min_surface_rms, dtype=dtype))
    zeta = unit_probe * eta_scale[:, None]

    step_min = jnp.asarray(fd_step_min, dtype=dtype)
    step_max = jnp.asarray(fd_step_max, dtype=dtype)
    if fd_step_min == fd_step_max:
        fd_step = jnp.full((batch_size,), step_min, dtype=dtype)
    else:
        log_step = jax.random.uniform(
            key_eps,
            (batch_size,),
            dtype=dtype,
            minval=jnp.log(step_min),
            maxval=jnp.log(step_max),
        )
        fd_step = jnp.clip(jnp.exp(log_step), step_min, step_max)

    def operator(eta: Array, xi: Array) -> Array:
        """Evaluate in physical units and subtract the output's spatial mean."""
        inputs = norm_inputs_fn(eta, xi)
        predictions = apply_fn({"params": model_params}, inputs, depth)
        gxi = denorm_targets_fn(predictions)[..., 0].astype(dtype)
        return gxi - jnp.mean(gxi, axis=-1, keepdims=True)

    gxi = operator(eta, xi)
    eta_x = spectral_dx(eta, k_typed)
    xi_x = spectral_dx(xi, k_typed)
    b_velocity = (gxi + eta_x * xi_x) / (1.0 + eta_x * eta_x)
    v_velocity = xi_x - b_velocity * eta_x

    fd_step_broadcast = fd_step[:, None]
    gxi_perturbed = operator(eta + fd_step_broadcast * zeta, xi)
    secant = (gxi_perturbed - gxi) / fd_step_broadcast

    forcing = operator(eta, zeta * b_velocity) + spectral_dx(zeta * v_velocity, k_typed)
    residual = secant + forcing

    residual_energy = projected_sobolev_energy(residual, k_typed, k_max, sobolev_order)
    forcing_energy = projected_sobolev_energy(forcing, k_typed, k_max, sobolev_order)
    denominator = forcing_energy + jnp.asarray(denominator_eps, dtype=dtype)
    return jnp.mean(residual_energy / denominator)


def compute_local_response_loss(
    rng: Array,
    apply_fn: ApplyFn,
    model_params: Any,
    eta_phys: Array,
    xi_phys: Array,
    batch_depth_local: Array,
    norm_inputs_fn: NormalizeInputsFn,
    denorm_targets_fn: DenormalizeTargetsFn,
    k: Array,
    dtype: jnp.dtype,
    *,
    mode: Literal["secant", "teacher", "targeted"],
    length: float,
    k_max: float = 128.0,
    sobolev_order: Literal[0, 1] = 1,
    fd_step_min: float = 1e-3,
    fd_step_max: float = 3e-3,
    min_surface_rms: float = 1e-3,
    denominator_eps: float = 1e-12,
) -> Array:
    """Match a joint eta/xi response to a frozen FP64, order-six DNO.

    ``secant`` compares central finite differences. ``teacher`` compares the
    student's central difference with the teacher's joint Hadamard derivative.
    ``targeted`` uses central differences with an equal per-row mixture of smooth
    random directions, independent Fourier-phase changes, and instantaneous
    model/reference flow defects. No rollout or gradient through that defect is
    used. The two defect components share one normalization factor: their mean
    squared amplitudes relative to the respective field RMS average to one.
    This preserves their relative physical amplitudes.

    All teacher evaluations use the actual depth ``exp(batch_depth_local)`` and
    dataset input projection P128. The Hadamard construction retains the full
    base DNO and product spectra until its final output projection. The loss
    projects to ``k_max`` and divides the
    response-error energy by the detached teacher-response energy. Directions,
    step sizes, and normalization are also frozen during parameter gradients.
    """
    del k  # Rebuild physical modes in FP64, independently of the training dtype.
    _, k64 = build_grid(eta_phys.shape[-1], length)
    k64 = k64.astype(jnp.float64)
    eta, xi = map(
        lambda field: jax.lax.stop_gradient(project_fixed_band(
            field.astype(jnp.float64), k64, maximum_wavenumber=128.0,
        )),
        (eta_phys, xi_phys),
    )
    log_depth = jax.lax.stop_gradient(batch_depth_local.astype(jnp.float64))
    physical_depth = jnp.exp(log_depth)
    batch_size, nx = eta.shape
    key_probe, key_eps, key_phase, key_choice = jax.random.split(rng, 4)
    cutoff = min(k_max, 128.0)
    direction_mask = (jnp.abs(k64) > 0.0) & (jnp.abs(k64) <= cutoff)
    fields = jnp.stack((eta, xi))
    centered = fields - jnp.mean(fields, axis=-1, keepdims=True)
    scales = jnp.maximum(
        jnp.sqrt(jnp.mean(centered**2, axis=-1)),
        jnp.asarray([min_surface_rms, 1e-12], dtype=jnp.float64)[:, None],
    )
    weight_sq = jnp.ones_like(k64) if sobolev_order == 0 else 1 + k64**2
    raw = jax.random.normal(key_probe, fields.shape, dtype=jnp.float64)
    directions = jnp.fft.ifft(
        jnp.fft.fft(raw, axis=-1) * direction_mask / jnp.sqrt(weight_sq), axis=-1,
    ).real
    directions *= scales[..., None] / jnp.maximum(
        jnp.sqrt(jnp.mean(directions**2, axis=-1, keepdims=True)), 1e-30,
    )

    def student(eta_input: Array, xi_input: Array) -> Array:
        inputs = norm_inputs_fn(eta_input.astype(dtype), xi_input.astype(dtype))
        predictions = apply_fn({"params": model_params}, inputs, log_depth.astype(dtype))
        values = denorm_targets_fn(predictions)[..., 0].astype(jnp.float64)
        return values - jnp.mean(values, axis=-1, keepdims=True)

    def teacher(eta_input: Array, xi_input: Array) -> Array:
        # Do not prefilter xi_input here: the Hadamard combined potential contains
        # product modes above P128 that must participate in the DNO recurrence.
        values = dno_series_eval(
            eta_input, xi_input, k64, physical_depth, order=6, pad_factor=8,
        )
        return jax.lax.stop_gradient(values - jnp.mean(values, axis=-1, keepdims=True))

    if mode == "targeted":
        # Independent rates per mode and field are phase changes, not a common
        # spatial translation (which would assign rates proportional to k).
        phase_rates = jax.random.normal(
            key_phase, (2, batch_size, nx // 2 + 1), dtype=jnp.float64,
        )
        phase_hat = 1j * phase_rates * jnp.fft.rfft(fields, axis=-1)
        phase_hat *= direction_mask[: nx // 2 + 1]
        phase_hat = phase_hat.at[..., nx // 2].set(0)
        phase_directions = jnp.fft.irfft(phase_hat, n=nx, axis=-1)
        phase_directions *= scales[..., None] / jnp.maximum(
            jnp.sqrt(jnp.mean(phase_directions**2, axis=-1, keepdims=True)), 1e-30,
        )

        q6 = teacher(eta, xi)
        q_model = jax.lax.stop_gradient(student(eta, xi))
        eta_x = spectral_dx(eta, k64)
        xi_x = spectral_dx(xi, k64)
        defect = jnp.stack((
            q_model - q6,
            dealiased_zakharov_xi_rhs(eta_x, xi_x, q_model)
            - dealiased_zakharov_xi_rhs(eta_x, xi_x, q6),
        ))
        defect = jnp.fft.ifft(
            jnp.fft.fft(defect, axis=-1) * direction_mask, axis=-1,
        ).real
        joint_rms = jnp.sqrt(jnp.mean(
            jnp.mean(defect**2, axis=-1) / scales**2, axis=0,
        ))
        defect_directions = jnp.where(
            joint_rms[None, :, None] > 1e-30,
            defect / jnp.maximum(joint_rms[None, :, None], 1e-30),
            directions,
        )
        choice = jax.random.randint(key_choice, (batch_size,), 0, 3)
        directions = jnp.where(
            (choice == 1)[None, :, None], phase_directions,
            jnp.where((choice == 2)[None, :, None], defect_directions, directions),
        )

    zeta, chi = jax.lax.stop_gradient(directions)
    if fd_step_min == fd_step_max:
        fd_step = jnp.full((batch_size,), fd_step_min, dtype=jnp.float64)
    else:
        fd_step = jnp.exp(jax.random.uniform(
            key_eps, (batch_size,), dtype=jnp.float64,
            minval=jnp.log(fd_step_min), maxval=jnp.log(fd_step_max),
        ))
    # Both perturbed surfaces keep at least half the base water-column height.
    positive_step = jnp.min(
        (physical_depth + eta) / jnp.maximum(jnp.abs(zeta), 1e-30), axis=-1,
    ) / 2.0
    fd_step = jax.lax.stop_gradient(jnp.minimum(fd_step, positive_step))[:, None]
    eta_plus, eta_minus, xi_plus, xi_minus = map(
        lambda field: jax.lax.stop_gradient(project_fixed_band(
            field, k64, maximum_wavenumber=128.0,
        )),
        (eta + fd_step * zeta, eta - fd_step * zeta,
         xi + fd_step * chi, xi - fd_step * chi),
    )
    response = (student(eta_plus, xi_plus) - student(eta_minus, xi_minus)) / (2 * fd_step)
    if mode == "teacher":
        eta_x = spectral_dx(eta, k64)
        xi_x = spectral_dx(xi, k64)
        b_velocity = (teacher(eta, xi) + eta_x * xi_x) / (1.0 + eta_x**2)
        v_velocity = xi_x - b_velocity * eta_x
        reference = teacher(eta, chi - zeta * b_velocity) - spectral_dx(
            zeta * v_velocity, k64,
        )
    else:
        reference = (teacher(eta_plus, xi_plus) - teacher(eta_minus, xi_minus)) / (2 * fd_step)
    reference = jax.lax.stop_gradient(project_fixed_band(
        reference, k64, maximum_wavenumber=128.0,
    ))
    residual_energy = projected_sobolev_energy(
        response - reference, k64, cutoff, sobolev_order,
    )
    reference_energy = jax.lax.stop_gradient(projected_sobolev_energy(
        reference, k64, cutoff, sobolev_order,
    ))
    return jnp.mean(residual_energy / (reference_energy + denominator_eps))
