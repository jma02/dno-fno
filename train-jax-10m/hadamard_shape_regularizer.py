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

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import jax
import jax.numpy as jnp


Array: TypeAlias = jax.Array
ApplyFn: TypeAlias = Callable[[dict[str, Any], Array, Array], Array]
NormalizeInputsFn: TypeAlias = Callable[[Array, Array], Array]
DenormalizeTargetsFn: TypeAlias = Callable[[Array], Array]
Diagnostics: TypeAlias = dict[str, Array]


@dataclass(frozen=True)
class HadamardRegConfig:
    """Static controls for :func:`compute_hadamard_reg`.

    ``relative_eps_*`` are dimensionless because the sampled direction is
    scaled to the RMS amplitude of each input surface.  ``eta_scale_floor`` is
    the physical-amplitude fallback for flat or nearly flat surfaces.
    """

    k_max: float = 128.0
    sobolev_order: int = 1
    relative_eps_min: float = 1e-3
    relative_eps_max: float = 3e-3
    eta_scale_floor: float = 1e-3
    denominator_floor: float = 1e-12


def sample_microbatch(
    rng: Array,
    eta: Array,
    xi: Array,
    depth_per_sample: Array,
    local_size: int,
) -> tuple[Array, Array, Array]:
    """Sample a device-local microbatch without replacement."""
    indices = jax.random.permutation(rng, eta.shape[0])[:local_size]
    return eta[indices], xi[indices], depth_per_sample[indices]


def _validate_config(cfg: HadamardRegConfig) -> None:
    if cfg.k_max <= 0.0:
        raise ValueError(f"k_max must be positive, got {cfg.k_max}")
    if cfg.sobolev_order < 0:
        raise ValueError(
            f"sobolev_order must be nonnegative, got {cfg.sobolev_order}"
        )
    if not 0.0 < cfg.relative_eps_min <= cfg.relative_eps_max:
        raise ValueError(
            "relative eps bounds must satisfy 0 < min <= max; got "
            f"{cfg.relative_eps_min}, {cfg.relative_eps_max}"
        )
    if cfg.eta_scale_floor <= 0.0:
        raise ValueError(
            f"eta_scale_floor must be positive, got {cfg.eta_scale_floor}"
        )
    if cfg.denominator_floor <= 0.0:
        raise ValueError(
            f"denominator_floor must be positive, got {cfg.denominator_floor}"
        )


def sobolev_weight_sq(k: Array, order: int) -> Array:
    """Return the squared spectral weight used by the supervised loss.

    The trainer's order-``s`` convention is
    ``W(k)^2 = 1 + |k|^2 + ... + |k|^(2s)`` rather than ``(1+k^2)^s``.
    """
    if order < 0:
        raise ValueError(f"order must be nonnegative, got {order}")
    k_abs = jnp.abs(k)
    weight_sq = jnp.ones_like(k_abs)
    for derivative_order in range(1, order + 1):
        weight_sq = weight_sq + k_abs ** (2 * derivative_order)
    return weight_sq


def projected_sobolev_energy(
    field: Array,
    k: Array,
    k_max: float,
    sobolev_order: int,
) -> Array:
    """Per-sample mean-square ``H^s`` energy after ``|k| <= k_max`` projection.

    ``field`` has shape ``(..., nx)`` and ``k`` has shape ``(nx,)``.  The
    unnormalised FFT is divided by ``nx**2``, so order zero agrees with the
    physical-space mean square by Parseval.
    """
    nx = field.shape[-1]
    field_hat = jnp.fft.fft(field, axis=-1)
    projector = (jnp.abs(k) <= jnp.asarray(k_max, dtype=k.dtype)).astype(
        field_hat.real.dtype
    )
    weight_sq = sobolev_weight_sq(k, sobolev_order)
    weighted_energy = (
        projector.astype(field_hat.real.dtype)
        * weight_sq.astype(field_hat.real.dtype)
        * jnp.abs(field_hat) ** 2
    )
    return jnp.sum(weighted_energy, axis=-1) / jnp.asarray(
        nx * nx, dtype=field_hat.real.dtype
    )


def construct_relative_eta_probe(
    rng: Array,
    eta_phys: Array,
    k: Array,
    cfg: HadamardRegConfig,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Sample ``(zeta, relative_eps, eta_scale)`` for a finite secant.

    The zero-mean random direction is supported on ``0 < |k| <= k_max``.  Its
    spectrum is divided by the Sobolev weight before RMS normalisation; hence
    weighted probe energy is spread across the retained modes instead of being
    dominated by the largest wavenumbers.  Finally, its physical RMS is set to
    ``max(rms(eta - mean(eta)), eta_scale_floor)`` independently per sample.
    """
    _validate_config(cfg)
    eta = eta_phys.astype(dtype)
    k_typed = k.astype(dtype)
    batch_size = eta.shape[0]
    key_probe, key_eps = jax.random.split(rng)

    raw = jax.random.normal(key_probe, eta.shape, dtype=dtype)
    raw_hat = jnp.fft.fft(raw, axis=-1)
    k_abs = jnp.abs(k_typed)
    projector = (k_abs > 0.0) & (
        k_abs <= jnp.asarray(cfg.k_max, dtype=dtype)
    )
    weight = jnp.sqrt(sobolev_weight_sq(k_typed, cfg.sobolev_order))
    probe_hat = raw_hat * projector[None, :] / weight[None, :]
    probe = jnp.real(jnp.fft.ifft(probe_hat, axis=-1))
    probe_rms = jnp.sqrt(jnp.mean(probe * probe, axis=-1))
    unit_probe = probe / jnp.maximum(
        probe_rms[:, None], jnp.asarray(1e-30, dtype=dtype)
    )

    eta_centered = eta - jnp.mean(eta, axis=-1, keepdims=True)
    eta_rms = jnp.sqrt(jnp.mean(eta_centered * eta_centered, axis=-1))
    eta_scale = jnp.maximum(
        eta_rms, jnp.asarray(cfg.eta_scale_floor, dtype=dtype)
    )
    zeta = unit_probe * eta_scale[:, None]

    eps_min = jnp.asarray(cfg.relative_eps_min, dtype=dtype)
    eps_max = jnp.asarray(cfg.relative_eps_max, dtype=dtype)
    if cfg.relative_eps_min == cfg.relative_eps_max:
        relative_eps = jnp.full((batch_size,), eps_min, dtype=dtype)
    else:
        log_eps = jax.random.uniform(
            key_eps,
            (batch_size,),
            dtype=dtype,
            minval=jnp.log(eps_min),
            maxval=jnp.log(eps_max),
        )
        relative_eps = jnp.clip(jnp.exp(log_eps), eps_min, eps_max)
    return zeta, relative_eps, eta_scale


def evaluate_operator(
    apply_fn: ApplyFn,
    model_params: Any,
    eta_phys: Array,
    xi_phys: Array,
    batch_depth_local: Array,
    norm_inputs_fn: NormalizeInputsFn,
    denorm_targets_fn: DenormalizeTargetsFn,
    dtype: jnp.dtype,
) -> Array:
    """Evaluate the learned DNO in physical units with production zero-mean output."""
    inputs = norm_inputs_fn(eta_phys, xi_phys)
    predictions = apply_fn({"params": model_params}, inputs, batch_depth_local)
    gxi = denorm_targets_fn(predictions)[..., 0].astype(dtype)
    return gxi - jnp.mean(gxi, axis=-1, keepdims=True)


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
    cfg: HadamardRegConfig,
    dtype: jnp.dtype,
) -> tuple[Array, Diagnostics]:
    """Return the normalized finite-secant Hadamard defect and diagnostics.

    Inputs are clean physical training states of shape ``(batch, nx)``;
    ``batch_depth_local`` is the model's usual depth input for those samples.
    The loss is

    ``mean(||W P S||^2 / (||W P R||^2 + denominator_floor))``,

    where ``S`` is the Hadamard residual and
    ``R = G(eta)(zeta B) + d_x(zeta V)`` is the identity's right-hand-side
    magnitude (up to sign).  All reported diagnostics are scalar arrays, so
    they can be stacked and averaged directly by the trainer.
    """
    _validate_config(cfg)
    eta = eta_phys.astype(dtype)
    xi = xi_phys.astype(dtype)
    k_typed = k.astype(dtype)
    depth = batch_depth_local.astype(dtype)
    zeta, relative_eps, eta_scale = construct_relative_eta_probe(
        rng, eta, k_typed, cfg, dtype
    )

    gxi = evaluate_operator(
        apply_fn,
        model_params,
        eta,
        xi,
        depth,
        norm_inputs_fn,
        denorm_targets_fn,
        dtype,
    )
    eta_x = spectral_dx(eta, k_typed)
    xi_x = spectral_dx(xi, k_typed)
    b_velocity = (gxi + eta_x * xi_x) / (1.0 + eta_x * eta_x)
    v_velocity = xi_x - b_velocity * eta_x

    eps_broadcast = relative_eps[:, None]
    gxi_perturbed = evaluate_operator(
        apply_fn,
        model_params,
        eta + eps_broadcast * zeta,
        xi,
        depth,
        norm_inputs_fn,
        denorm_targets_fn,
        dtype,
    )
    secant = (gxi_perturbed - gxi) / eps_broadcast

    zeta_b = zeta * b_velocity
    g_zeta_b = evaluate_operator(
        apply_fn,
        model_params,
        eta,
        zeta_b,
        depth,
        norm_inputs_fn,
        denorm_targets_fn,
        dtype,
    )
    product_dx = spectral_dx(zeta * v_velocity, k_typed)
    forcing = g_zeta_b + product_dx
    residual = secant + forcing

    residual_energy = projected_sobolev_energy(
        residual, k_typed, cfg.k_max, cfg.sobolev_order
    )
    forcing_energy = projected_sobolev_energy(
        forcing, k_typed, cfg.k_max, cfg.sobolev_order
    )
    secant_energy = projected_sobolev_energy(
        secant, k_typed, cfg.k_max, cfg.sobolev_order
    )
    denominator = forcing_energy + jnp.asarray(
        cfg.denominator_floor, dtype=dtype
    )
    per_sample_loss = residual_energy / denominator
    loss = jnp.mean(per_sample_loss)

    diagnostics: Diagnostics = {
        "hadamard_loss": loss,
        "hadamard_defect_rms": jnp.mean(jnp.sqrt(per_sample_loss)),
        "hadamard_residual_hs_rms": jnp.mean(jnp.sqrt(residual_energy)),
        "hadamard_forcing_hs_rms": jnp.mean(jnp.sqrt(forcing_energy)),
        "hadamard_secant_hs_rms": jnp.mean(jnp.sqrt(secant_energy)),
        "hadamard_relative_eps": jnp.mean(relative_eps),
        "hadamard_eta_scale": jnp.mean(eta_scale),
    }
    return loss, diagnostics
