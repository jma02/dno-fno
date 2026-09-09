"""Finite-secant Hadamard shape-consistency regularizer for a learned DNO.

For a Dirichlet--Neumann operator ``G(eta)`` and arbitrary surface direction
``zeta``, the Hadamard identity is

    D_eta G(eta)[zeta] xi = -G(eta)(zeta * B) - d_x(zeta * V),

where ``B = (Gxi + eta_x * xi_x) / (1 + eta_x**2)`` and
``V = xi_x - B * eta_x``.  This module measures the identity using only the
learned operator: it does not require an order-N Craig--Sulem reference.

The finite secant avoids differentiating a JVP through the parameter loss.  A
small, relative perturbation is used so the same configuration applies across
the range of surface amplitudes in the clean training data.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeAlias

import jax
import jax.numpy as jnp


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


def compute_hadamard_reg(
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
