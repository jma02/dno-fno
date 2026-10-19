"""Benchmark-only inference kernels and explicitly named low-rank model trials."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import ctypes
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import linen as nn  # noqa: E402
from flax.core import Scope  # noqa: E402
from jax.experimental.compute_on import compute_on  # noqa: E402
from jax.stages import Wrapped  # noqa: E402

from scripts.benchmark_cufftdx import fft, field_irfft, sandwich  # noqa: E402
from scripts.benchmark_classical import classical_hat  # noqa: E402
from scripts.benchmark_dno_dense import dense  # noqa: E402
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched, load_run  # noqa: E402
from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, CraigSulemTermsResidual, DepthAwareMultiplier, phase_rank_one_hat  # noqa: E402

VARIANTS = ("original", "g1", "packed", "cached", "all", "batched", "surface", "front")
TUNED_TRIALS = tuple("fourier_tuned_" + "_".join(map(str, epts)) + suffix
                     for epts in product((4, 8, 16), repeat=4) for suffix in ("", "_mask"))
G1_FP32_BASE = "fourier_fused_features_product_overlap_g1fp32"
G1_FP32_TRIALS = tuple(G1_FP32_BASE + suffix + tuning
    for suffix in ("", "_reuse", "_inputfp32", "_reuse_inputfp32")
    for tuning in ("", *(f"_g1e{a}x{b}" for a, b in product((4, 8, 16), repeat=2))))
TRIALS = ("terms_interleaved", "terms_interleaved_features", "fourier", "fourier_round", "fourier_reuse", "fourier_input",
          "fourier_g1fft", "fourier_features", "fourier_product", "fourier_allfft",
          "fourier_fields", "fourier_kernels", "fourier_kernels_e4", "fourier_kernels_e16",
          "fourier_transforms", "fourier_transforms_e4", "fourier_transforms_e16",
          "fourier_fused_fields", "fourier_fused_inputs", "fourier_fused_both", "fourier_fused_g1",
          "fourier_fused_fields_overlap", "fourier_fused_g1_overlap",
          "fourier_fused_features", "fourier_fused_features_overlap",
          "fourier_balanced_features_overlap", "fourier_fused_features_product_overlap",
          "fourier_fused_features_product_overlap_nestedfused",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch",
          "fourier_fused_features_product_overlap_grid512",
          "fourier_balanced_features_product_overlap",
          "cufftdx", "cufftdx_joint", "cufftdx_product", "cufftdx_both", "cufftdx_dense", "cufftdx_head",
          "cufftdx_lowrank16", "cufftdx_lowrank32", "cufftdx_e4", "cufftdx_e16",
          "cufftdx_product_e4", "cufftdx_product_e16", *TUNED_TRIALS, *G1_FP32_TRIALS)


def resample_rfft(spectrum: jax.Array, n: int) -> jax.Array:
    """Resample an even-grid, unnormalized real FFT, merging/splitting Nyquist."""
    source_n = 2 * (spectrum.shape[-1] - 1)
    if n == source_n:
        return spectrum
    scaled = spectrum * (n / source_n)
    if n < source_n:
        result = scaled[..., :n // 2 + 1]
        return result.at[..., -1].set(2 * result[..., -1].real)
    result = jnp.pad(scaled, ((0, 0),) * (scaled.ndim - 1) + ((0, (n - source_n) // 2),))
    return result.at[..., source_n // 2].set(0.5 * result[..., source_n // 2].real)


def build_variant(
    loaded: LoadedRun, depth: jax.Array, nx: int, variant: str, *, include_baselines: bool = True,
    include_correction: bool = True, fft_ept: int = 8, solver_g0: jax.Array | None = None,
) -> Wrapped:
    """Intercept only inference arithmetic; leave the production module untouched."""
    model = loaded.model
    if isinstance(model, CraigSulemTermsResidual):
        if not include_baselines or not include_correction or loaded.norm_mode != "scale":
            raise ValueError("CS-field inference requires the complete scale-normalized decoder.")
        if variant == "original":
            predict = build_predict_gxi_batched(loaded)
            return jax.jit(lambda eta, xi: predict(eta, xi, depth))
        if variant not in ("terms_interleaved", "terms_interleaved_features"):
            raise ValueError("CS-field inference requires an original or terms_interleaved variant.")
        batch_features = variant.endswith("_features")
        if batch_features and (model.use_learned_eta_filters or model.surface_feature_transform is not None):
            raise ValueError("Batched decoder features require fixed, unwhitened surface filters.")
        log_depth = depth.astype(jnp.float64).reshape(-1, 1)
        scales = jnp.asarray((model.eta_scale, model.xi_scale), dtype=jnp.float64)
        k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float64)
        g0 = k * jnp.tanh(jnp.exp(log_depth) * k)
        feature_k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float32)
        curvature = -(feature_k ** 2)
        if model.curvature_cutoff > 0:
            curvature /= jnp.sqrt(1.0 + (feature_k / model.curvature_cutoff) ** 2)
        feature_ops = jnp.stack([symbol for enabled, symbol in (
            (model.use_first_deriv, 1j * feature_k), (model.use_second_deriv, curvature),
            (model.use_half_deriv, jnp.sqrt(feature_k)), (model.use_hilbert, -1j * jnp.sign(feature_k)),
        ) if enabled])

        @jax.jit
        def forward_terms(eta: jax.Array, xi: jax.Array) -> jax.Array:
            inputs = jnp.stack((eta.astype(jnp.float64), xi.astype(jnp.float64)), axis=-1) / scales
            if batch_features:
                eta_norm = inputs[..., 0].astype(jnp.float32)
                derived = jnp.moveaxis(jnp.fft.irfft(
                    feature_ops[None, :, :] * jnp.fft.rfft(eta_norm)[:, None, :],
                    n=nx,
                ), 1, -1)
                features = jnp.concatenate((jnp.stack(
                    [eta_norm ** p for p in range(1, model.n_polys + 1)], axis=-1,
                ), derived), axis=-1)
                if model.use_relative_elevation:
                    features = jnp.concatenate((features, (eta_norm * model.eta_scale
                        / jnp.exp(log_depth.astype(jnp.float32)))[..., None]), axis=-1)
                if model.surface_feature_scales is not None:
                    features /= jnp.asarray(model.surface_feature_scales, dtype=jnp.float32)

                def intercept_features(next_fun: Callable, call_args: tuple, call_kwargs: dict, context: Any) -> Any:
                    if isinstance(context.module, nn.Dense) and context.module.name == "eta_feat_proj":
                        call_args = (features,)
                    return next_fun(*call_args, **call_kwargs)

                with nn.intercept_methods(intercept_features):
                    coefficients = model.apply({"params": loaded.params}, inputs, log_depth,
                                               method=model.encode_geometry)
            else:
                coefficients = model.apply({"params": loaded.params}, inputs, log_depth,
                                           method=model.encode_geometry)
            # Preserve the native decoder's FP64 normalization round trip and pad2 projections.
            physical = jnp.moveaxis(inputs * scales, -1, -2)
            eta_hat, xi_hat = jnp.moveaxis(jnp.fft.rfft(physical).at[..., -1].set(0), -2, 0)
            terms_hat = classical_hat(
                eta_hat, xi_hat, k, g0, nx=nx, order=3, pad_factor=2,
                backend="interleaved_cufftdx", precision="fp64", ept=fft_ept, return_terms=True,
            )
            terms = jnp.fft.irfft(terms_hat, n=nx)
            prediction = cast(jax.Array, model.apply({}, coefficients, terms, method=model.decode_terms))
            output = (prediction[..., 0] * model.target_scale).astype(eta.dtype)
            return output - output.mean(axis=-1, keepdims=True)

        return forward_terms
    if variant.startswith("terms_"):
        raise ValueError("CS-field inference variants require a CS-field decoder.")
    context_batch = variant.endswith("_contextbatch")
    variant = variant.removesuffix("_contextbatch")
    nested_fused = variant.endswith("_nestedfused")
    variant = variant.removesuffix("_nestedfused")
    exact_g0 = variant.endswith("_exactg0")
    variant = variant.removesuffix("_exactg0")
    coarse_correction = variant.endswith("_grid512")
    variant = variant.removesuffix("_grid512")
    correction_nx = 512 if coarse_correction else nx
    variant, _, g1_tuning = variant.partition("_g1e")
    input_fp32 = variant.endswith("_inputfp32")
    variant = variant.removesuffix("_inputfp32")
    reuse_g1 = variant.endswith("_reuse") and "_g1fp32" in variant
    if reuse_g1:
        variant = variant.removesuffix("_reuse")
    g1_fp32 = variant.endswith("_g1fp32")
    variant = variant.removesuffix("_g1fp32")
    g1_dtype = jnp.float32 if g1_fp32 else jnp.float64
    overlap_g1 = variant.endswith("_overlap")
    variant = variant.removesuffix("_overlap")
    if variant.endswith(("_e4", "_e16")):
        variant, _, suffix = variant.rpartition("_e")
        fft_ept = int(suffix)
    mask_output = variant.endswith("_mask")
    variant = variant.removesuffix("_mask")
    # Input inverse, G1 input forward, packed-field inverse, G1 sandwich.
    input_ept, g1_input_ept, field_ept, sandwich_ept = (fft_ept,) * 4
    if g1_tuning:
        g1_input_ept, sandwich_ept = map(int, g1_tuning.split("x"))
    if variant.startswith("fourier_tuned_"):
        input_ept, g1_input_ept, field_ept, sandwich_ept = map(int, variant.removeprefix("fourier_tuned_").split("_"))
        variant = "fourier_transforms"
    fused_g1 = variant == "fourier_fused_g1"
    balanced_features = variant.startswith("fourier_balanced_features")
    fused_product = variant.endswith("features_product")
    fused_features = balanced_features or variant in ("fourier_fused_features", "fourier_fused_features_product")
    if coarse_correction and not (overlap_g1 and fused_features and fused_product):
        raise ValueError("The 512-point correction requires fused features/product with G1 overlap.")
    fused_fields = fused_features or variant in ("fourier_fused_fields", "fourier_fused_both", "fourier_fused_g1")
    fused_inputs = variant in ("fourier_fused_inputs", "fourier_fused_both")
    if fused_fields or fused_inputs:
        variant = "fourier_transforms"
    trial = variant
    spectral_io = variant.startswith("fourier")
    cufftdx_g1 = variant.startswith("cufftdx")
    if cufftdx_g1 or spectral_io:
        variant = "front"
    predict = build_predict_gxi_batched(loaded)
    model = loaded.model
    if model.phase_rank_one and (trial == "fourier" or (not spectral_io and variant != "original")):
        raise ValueError("Phase rank-one inference requires original or physically rounded Fourier I/O.")
    if nested_fused and (nx != 1024 or coarse_correction or not model.nested_blocks or not spectral_io):
        raise ValueError("Nested fusion requires a 1024-point nested model with Fourier I/O.")
    if context_batch and not model.surface_context_width:
        raise ValueError("Context/nested FFT batching requires surface context channels.")
    if exact_g0 and (solver_g0 is None or not spectral_io or not include_baselines or model.baseline_order != 1):
        raise ValueError("Exact solver G0 requires its symbol and Fourier M1 inference with baselines.")
    if model.nested_blocks:
        if not include_baselines or not include_correction:
            raise ValueError("Nested inference requires the full baseline and correction.")
        if variant == "original":
            return jax.jit(lambda eta, xi: predict(eta, xi, depth))
        if not spectral_io:
            raise ValueError("Nested inference supports original or Fourier-I/O variants only.")
    ordinary_blocks = model.n_blocks - model.nested_blocks
    ordinary_branches = ordinary_blocks * model.latent
    factors = None
    if "lowrank" in trial:
        rank = int(trial.split("lowrank")[1])
        with np.load(ROOT / "outputs/cufftdx_20260925/lowrank_factors.npz") as saved:
            factors = (jnp.asarray(saved[f"A{rank}"]), jnp.asarray(saved[f"B{rank}"]))
    head_kernel = (jnp.concatenate(tuple(loaded.params[f"cs_block_{i}"]["phi_proj"]["kernel"]
                                        for i in range(ordinary_blocks)), axis=-1)
                   if ordinary_blocks else None)
    cached = None
    k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float32)
    second_symbol = -(k ** 2)
    if model.curvature_cutoff > 0:
        second_symbol = second_symbol / jnp.sqrt(1.0 + (k / model.curvature_cutoff) ** 2)
    surface_ops = jnp.stack([symbol for enabled, symbol in (
        (model.use_first_deriv, 1j * k),
        (model.use_second_deriv, second_symbol),
        (model.use_half_deriv, jnp.sqrt(k)),
        (model.use_hilbert, -1j * jnp.sign(k)),
    ) if enabled])
    if include_correction and variant in ("cached", "all", "batched", "surface", "front"):
        multiplier = DepthAwareMultiplier(model.latent, model.domain_length, model.h_clip_max,
                                          model.mult_hidden, model.multiplier_skip, model.log_multiplier_inputs)
        # This is a per-checkpoint/grid/depth inference constant, not a training cache.
        cached = jax.jit(lambda d: tuple(multiplier.apply(
            {"params": loaded.params[f"cs_block_{i}"]["m_shared"]}, d, nx // 2 + 1,
        ) for i in range(model.n_blocks)))(
            jnp.minimum(depth.astype(jnp.float32).reshape(-1, 1), jnp.log(model.h_clip_max))
        )
        jax.block_until_ready(cached)

    if spectral_io:
        if model.surface_depth_gate and head_kernel is not None:
            # Fold the fixed-depth gain into the head once, outside the rollout.
            log_h = jnp.minimum(depth.astype(jnp.float32).reshape(-1, 1), jnp.log(model.h_clip_max)) / 4.0
            gains = 1.0 + 0.5 * jnp.tanh(
                jnp.concatenate((log_h, log_h**2), axis=-1)
                @ loaded.params["eta_depth_gate"]["kernel"]
            )
            head_kernel = gains[..., None] * head_kernel[None, ...]
            if head_kernel.shape[0] == 1:
                head_kernel = head_kernel[0]
        first_projection_kernel = loaded.params["eta_feat_proj"]["kernel"]
        if model.surface_feature_transform is not None:
            first_projection_kernel = jnp.asarray(
                model.surface_feature_transform, dtype=first_projection_kernel.dtype,
            ) @ first_projection_kernel
        filters = jnp.moveaxis(jnp.concatenate(cast(tuple[jax.Array, ...], cached), axis=-1), -1, 1)
        inner_filters = gamma = nested_symbols = surface_mix = None
        if model.nested_blocks:
            inner = DepthAwareMultiplier(model.latent, model.domain_length, model.h_clip_max,
                                         model.mult_hidden, model.multiplier_skip, model.log_multiplier_inputs)
            inner_filters = jnp.moveaxis(jax.jit(lambda d: jnp.concatenate(tuple(cast(jax.Array, inner.apply(
                {"params": loaded.params[f"cs_block_{i}"]["m_inner"]}, d, nx // 2 + 1,
            )) for i in range(ordinary_blocks, model.n_blocks)), axis=-1))(
                jnp.minimum(depth.astype(jnp.float32).reshape(-1, 1), jnp.log(model.h_clip_max))
            ), -1, 1)
            gamma = jnp.concatenate(tuple(loaded.params[f"cs_block_{i}"]["gamma"]
                                         for i in range(ordinary_blocks, model.n_blocks)))
            if model.nested_surface_features:
                surface_mix = jnp.concatenate(tuple(loaded.params[f"cs_block_{i}"]["surface_mix"]
                                                    for i in range(ordinary_blocks, model.n_blocks)), axis=-1)
            jax.block_until_ready((inner_filters, gamma))
            if nested_fused:
                nested_symbols = jnp.ones(
                    (inner_filters.shape[0] * inner_filters.shape[1], 1, nx // 2 + 1), dtype=jnp.complex64,
                )
                jax.block_until_ready(nested_symbols)
        scales = jnp.asarray(loaded.stats["feature_absmax"], dtype=jnp.float32)
        output_scale = float(loaded.stats["target_absmax"]) / model.target_scale
        log_depth = depth.astype(jnp.float32).reshape(-1, 1)
        nested_modulation = ((jnp.tanh(loaded.params["nested_surface_gain"]), jnp.exp(log_depth))
                             if model.nested_surface_modulation else None)
        h = jnp.exp(log_depth if model.baseline_true_depth else jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
        g0 = k[None, :] * jnp.tanh(h * k[None, :])
        k64 = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float64)
        g0_64 = k64[None, :] * jnp.tanh(h.astype(jnp.float64) * k64[None, :])
        operators = jnp.stack((g0_64, jnp.broadcast_to(1j * k64, g0_64.shape)), axis=1)
        if g1_fp32:
            operators = operators.astype(jnp.complex64)
        context_multiplier = None
        if model.surface_context_width:
            context_k, context_log_h = jnp.broadcast_arrays(k[None, :], jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
            conditioning = jnp.stack((
                jnp.log1p(context_k), context_log_h, jnp.tanh(jnp.exp(context_log_h) * context_k),
            ), axis=-1)
            hidden = loaded.params["eta_context_hidden"]
            context_hidden = nn.tanh(conditioning @ hidden["kernel"] + hidden["bias"])
            output = loaded.params["eta_context_out"]
            context_multiplier = 0.1 * nn.tanh(context_hidden @ output["kernel"] + output["bias"])
            jax.block_until_ready(context_multiplier)

        if coarse_correction:
            surface_ops = surface_ops[..., :correction_nx // 2 + 1]
            filters = filters[..., :correction_nx // 2 + 1]
            if inner_filters is not None:
                inner_filters = inner_filters[..., :correction_nx // 2 + 1]
            if context_multiplier is not None:
                context_multiplier = context_multiplier[:, :correction_nx // 2 + 1]

        @jax.jit
        def m2_hat(inputs: jax.Array) -> jax.Array:
            physical = jnp.stack((inputs[:, 0] * model.eta_scale, inputs[:, 1]), axis=1)
            eta64_hat, xi64_hat = jnp.moveaxis(
                fft(physical.astype(jnp.float64), nx, ept=16).at[..., -1].set(0), 1, 0,
            )
            return classical_hat(
                eta64_hat, xi64_hat, k64, g0_64, nx=nx, order=2,
                pad_factor=2, backend="specialized_cufftdx", precision="fp64", ept=16,
            )

        @jax.jit
        def forward_hat(eta_hat: jax.Array, xi_hat: jax.Array) -> jax.Array:
            analytic_hat = None
            # Preserve the trained surface features' physical-space FP32 rounding.
            eta_norm = jnp.fft.irfft(eta_hat, n=nx).astype(jnp.float32) / scales[0]
            eta_norm_hat = jnp.fft.rfft(eta_norm)
            xi_phys_hat = (xi_hat.astype(jnp.complex64) / scales[1]) * model.xi_scale
            g1_xi_hat = xi_hat * (model.xi_scale / scales[1]).astype(jnp.float64)
            normalized_hat = normalized_inputs = None
            if trial == "fourier" and model.baseline_order == 2:
                xi_phys = (jnp.fft.irfft(xi_hat, n=nx).astype(jnp.float32) / scales[1]) * model.xi_scale
                normalized_inputs = jnp.stack((eta_norm, xi_phys), axis=1)
            if trial != "fourier":
                inputs = jnp.stack((eta_hat, xi_hat), axis=1)
                if input_fp32:
                    inputs = inputs.astype(jnp.complex64)
                if fused_inputs:
                    normalized_inputs = fft(inputs, nx, inverse=True, ept=input_ept,
                                            normalization=jnp.array([1 / scales[0], 1 / scales[1], model.xi_scale], dtype=jnp.float32))
                    eta_norm, xi_phys = normalized_inputs[:, 0], normalized_inputs[:, 1]
                else:
                    physical = (fft(inputs, nx, inverse=True, ept=input_ept) if trial not in ("fourier_round", "fourier_reuse")
                                else jnp.fft.irfft(inputs, n=nx)).astype(jnp.float32)
                    normalized = physical / scales[None, :, None]
                    eta_norm, xi_phys = normalized[:, 0], normalized[:, 1] * model.xi_scale
                    normalized_inputs = jnp.stack((eta_norm, xi_phys), axis=1)
                normalized_hat = (fft(normalized_inputs, nx, ept=fft_ept)
                                  if trial in ("fourier_features", "fourier_allfft") else jnp.fft.rfft(normalized_inputs))
                eta_norm_hat, xi_phys_hat = jnp.moveaxis(normalized_hat, 1, 0)
                g1_xi_hat = (xi_phys_hat.astype(jnp.complex128) if trial == "fourier_reuse"
                             else jnp.fft.rfft(xi_phys.astype(jnp.float64)))
                if trial in ("fourier_g1fft", "fourier_allfft", "fourier_kernels", "fourier_transforms"):
                    g1_xi_hat = fft(xi_phys if fused_g1 else xi_phys.astype(jnp.float64), nx,
                                    ept=g1_input_ept, promote=fused_g1)
                if overlap_g1:
                    with compute_on("gpu_stream:1"):
                        if model.baseline_order == 2:
                            analytic_hat = m2_hat(normalized_inputs)
                        else:
                            analytic_hat = jax.jit(lambda values: -sandwich(
                                values[0][:, 0] if fused_g1 else (values[0][:, 0] * model.eta_scale).astype(g1_dtype),
                                values[1] if reuse_g1 else fft(
                                    values[0][:, 1] if fused_g1 else values[0][:, 1].astype(g1_dtype), nx,
                                    ept=g1_input_ept, promote=fused_g1), operators,
                                spectral=True, ept=sandwich_ept,
                                eta_scale=model.eta_scale if fused_g1 else None,
                            ).sum(axis=1))((normalized_inputs, xi_phys_hat) if reuse_g1 else (normalized_inputs,))
            baseline_xi_hat = xi_phys_hat
            phase_hat = (phase_rank_one_hat(
                eta_norm_hat * model.eta_scale, xi_phys_hat, log_depth,
                loaded.params["phase_rank_one_coefficients"], model.domain_length,
            ) if model.phase_rank_one else None)
            if coarse_correction:
                normalized_hat = resample_rfft(cast(jax.Array, normalized_hat), correction_nx)
                normalized_inputs = fft(normalized_hat, correction_nx, inverse=True, ept=input_ept)
                eta_norm, xi_phys = jnp.moveaxis(normalized_inputs, 1, 0)
                eta_norm_hat, xi_phys_hat = jnp.moveaxis(normalized_hat, 1, 0)
            if fused_features:
                features, branch_fields = field_irfft(cast(jax.Array, normalized_hat), surface_ops, filters,
                                                       n=correction_nx, ept=field_ept, normalized_inputs=normalized_inputs,
                                                       balanced=balanced_features)
                features = jnp.moveaxis(features, 1, -1)
            else:
                if fused_fields:
                    fields = cast(jax.Array, field_irfft(cast(jax.Array, normalized_hat), surface_ops, filters, n=correction_nx, ept=field_ept))
                else:
                    field_spectra = jnp.concatenate((
                        surface_ops[None, :, :] * eta_norm_hat[:, None, :],
                        filters * xi_phys_hat[:, None, :],
                    ), axis=1)
                    fields = (fft(field_spectra, correction_nx, inverse=True, ept=field_ept)
                              if trial in ("fourier_features", "fourier_allfft", "fourier_fields", "fourier_kernels", "fourier_transforms")
                              else jnp.fft.irfft(field_spectra, n=correction_nx))
                features = jnp.concatenate((
                    jnp.stack([eta_norm ** p for p in range(1, model.n_polys + 1)], axis=-1),
                    jnp.moveaxis(fields[:, :surface_ops.shape[0], :], 1, -1),
                ), axis=-1)
                branch_fields = fields[:, surface_ops.shape[0]:, :]
            nested_eta = eta_norm
            if model.nested_blocks:
                nested_eta = eta_norm if model.surface_feature_scales is None else eta_norm / model.surface_feature_scales[0]
                if nested_modulation is not None:
                    gain, actual_depth = nested_modulation
                    nested_eta = nested_eta * (
                        1.0 + gain * jnp.tanh(eta_norm * model.eta_scale / actual_depth)
                    )
            if model.surface_feature_scales is not None:
                features = features / jnp.asarray(model.surface_feature_scales, dtype=features.dtype)
            nested_eta = nested_eta[:, None, :]
            if model.nested_surface_features:
                second_index = model.n_polys + int(model.use_first_deriv)
                half_index = second_index + int(model.use_second_deriv)
                nested_eta = nested_eta + jnp.moveaxis(
                    features[..., jnp.asarray([half_index, second_index])] @ cast(jax.Array, surface_mix), -1, 1,
                )
            polynomial_features = features[..., 1:3] if model.surface_polynomial_skip else None
            context = None
            nested_inner_hat = None
            for name in ("eta_feat_proj", "eta_feat_mix"):
                kernel = first_projection_kernel if name == "eta_feat_proj" else loaded.params[name]["kernel"]
                features = nn.gelu(features @ kernel)
                if name == "eta_feat_proj" and context_multiplier is not None:
                    context = features @ loaded.params["eta_context_in"]["kernel"]
                    if context_batch:
                        # Independent fields share one FFT batch, not Fourier modes.
                        combined_hat = jnp.fft.rfft(jnp.concatenate((
                            jnp.moveaxis(context, -1, 1),
                            nested_eta * branch_fields[:, ordinary_branches:],
                        ), axis=1))
                        context_hat = jnp.moveaxis(combined_hat[:, :model.surface_context_width], 1, -1)
                        nested_inner_hat = combined_hat[:, model.surface_context_width:]
                    else:
                        context_hat = jnp.fft.rfft(context, axis=1)
                    context = jnp.fft.irfft(
                        context_multiplier * context_hat, n=correction_nx, axis=1,
                    )
                    context = context @ loaded.params["eta_context_expand"]["kernel"]
                    features = features + context
            features = features * jnp.tanh(features) ** (2 if model.baseline_order == 2 else 1)
            if model.surface_context_gate and context is not None:
                features = features * (1.0 + jnp.tanh(context))
            if polynomial_features is not None:
                features = jnp.concatenate((features, polynomial_features), axis=-1)
            correction_hat = jnp.zeros_like(xi_phys_hat)
            if ordinary_blocks:
                weights = features @ cast(jax.Array, head_kernel)
                ordinary_fields = branch_fields[:, :ordinary_branches]
                ordinary_filters = filters[:, :ordinary_branches]
                if fused_product:
                    weighted_hat = fft(weights, correction_nx, b=ordinary_fields, outer=ordinary_filters,
                                       ept=fft_ept, channels_last=True)
                elif trial in ("fourier_product", "fourier_allfft", "fourier_kernels"):
                    weights = jnp.moveaxis(weights, -1, 1)
                    weighted_hat = correction_nx * fft(weights, correction_nx, b=ordinary_fields,
                                           outer=ordinary_filters.astype(jnp.complex64), ept=fft_ept)
                else:
                    weighted_hat = ordinary_filters * jnp.fft.rfft(jnp.moveaxis(weights, -1, 1) * ordinary_fields)
                correction_hat = jnp.sum(weighted_hat, axis=1)
            if model.nested_blocks:
                inner_hat = (cast(jax.Array, nested_inner_hat) if context_batch else
                             jnp.fft.rfft(nested_eta * branch_fields[:, ordinary_branches:]))
                if nested_fused:
                    inner_hat = cast(jax.Array, inner_filters) * inner_hat
                    # Preserve the first FFT; unit symbols fuse IFFT/product/FFT with unnormalized output.
                    surface = cast(jax.Array, gamma)[None, :, None] * nested_eta
                    nested_hat = filters[:, ordinary_branches:] * sandwich(
                        surface.reshape(-1, correction_nx), inner_hat.reshape(-1, inner_hat.shape[-1]),
                        cast(jax.Array, nested_symbols), spectral=True, ept=fft_ept,
                    ).reshape(inner_hat.shape)
                else:
                    middle = jnp.fft.irfft(cast(jax.Array, inner_filters) * inner_hat, n=correction_nx)
                    nested_hat = filters[:, ordinary_branches:] * jnp.fft.rfft(
                        cast(jax.Array, gamma)[None, :, None] * nested_eta * middle)
                correction_hat = correction_hat + jnp.sum(nested_hat, axis=1)
            correction_hat = correction_hat / (model.n_blocks * model.latent) ** 0.5
            if coarse_correction:
                correction_hat = resample_rfft(correction_hat, nx)
            if phase_hat is not None:
                correction_hat = correction_hat + phase_hat
            if not overlap_g1:
                if model.baseline_order == 2:
                    analytic_hat = m2_hat(cast(jax.Array, normalized_inputs))
                else:
                    analytic_hat = -sandwich(
                        eta_norm if fused_g1 else (eta_norm * model.eta_scale).astype(jnp.float64),
                        g1_xi_hat, operators,
                        spectral=True, ept=sandwich_ept, eta_scale=model.eta_scale if fused_g1 else None,
                    ).sum(axis=1)
            analytic_hat = cast(jax.Array, analytic_hat).astype(jnp.complex128)
            if model.baseline_order != 2 and not exact_g0:
                analytic_hat = g0 * baseline_xi_hat + analytic_hat
            result = output_scale * (analytic_hat + correction_hat)
            if exact_g0:
                result = result + cast(jax.Array, solver_g0) * xi_hat
            if mask_output:
                return cast(jax.Array, jnp.where(k64 == 0, 0, jnp.where(k64 == k64[-1], result.real, result)))
            return result.at[..., 0].set(0).at[..., -1].set(result[..., -1].real)

        return forward_hat

    @jax.jit
    def forward(eta: jax.Array, xi: jax.Array) -> jax.Array:
        spatial, multipliers = [], []
        front_fields = None
        dense_weights = None

        def intercept(
            next_fun: Callable[..., Any], call_args: tuple[Any, ...], call_kwargs: dict[str, Any],
            context: nn.module.InterceptorContext,
        ) -> Any:
            nonlocal front_fields, dense_weights
            module = context.module
            if not include_correction and isinstance(module, CraigSulemBlock) and context.method_name == "__call__":
                return jnp.zeros_like(call_args[1])
            if variant == "front" and isinstance(module, CraigSulemDNO) and context.method_name == "__call__":
                inputs, log_depth = call_args
                input_fields = (jnp.stack((inputs[..., 0], inputs[..., 1] * model.xi_scale), axis=1)
                                if include_correction else (inputs[..., 1] * model.xi_scale)[:, None, :])
                spectra = jnp.fft.rfft(input_fields)
                h = jnp.exp(log_depth if model.baseline_true_depth else jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
                base_k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1)
                symbol = base_k[None, :] * jnp.tanh(h * base_k[None, :])
                packed_hat = (symbol[:, None, :] * spectra[:, -1:, :])[:, :int(include_baselines), :]
                if include_correction:
                    filters = jnp.moveaxis(jnp.concatenate(cast(tuple[jax.Array, ...], cached), axis=-1), -1, 1)
                    packed_hat = jnp.concatenate((
                        surface_ops[None, :, :] * spectra[:, :1, :], packed_hat, filters * spectra[:, 1:, :],
                    ), axis=1)
                front_fields = jnp.fft.irfft(packed_hat, n=nx, axis=-1)
            if include_correction and variant in ("surface", "front") and isinstance(module, nn.Dense) and module.name == "eta_feat_proj":
                if model.surface_feature_transform is None:
                    features = call_args[0]
                    eta_norm = features[..., 0]
                    transformed = (front_fields[:, :surface_ops.shape[0], :] if front_fields is not None
                                   else jnp.fft.irfft(surface_ops[None, :, :] * jnp.fft.rfft(eta_norm)[:, None, :], n=nx))
                    call_args = (jnp.concatenate((features[..., :model.n_polys],
                                                 jnp.moveaxis(transformed, 1, -1)), axis=-1),)
                if trial == "cufftdx_dense":
                    hidden = dense(call_args[0], loaded.params["eta_feat_proj"]["kernel"], "gelu")
                    hidden = dense(hidden, loaded.params["eta_feat_mix"]["kernel"], "gelu_lift")
                    dense_weights = dense(hidden, cast(jax.Array, head_kernel))
                    return jnp.zeros_like(hidden)
            if isinstance(module, nn.Dense) and module.name == "eta_feat_mix":
                if trial == "cufftdx_dense":
                    return jnp.zeros_like(call_args[0])
                if factors is not None:
                    return (call_args[0] @ factors[0]) @ factors[1]
            if not include_baselines and context.method_name in ("_linear_baseline", "_g1_baseline"):
                return jnp.zeros_like(call_args[0])
            if front_fields is not None and context.method_name == "_linear_baseline":
                return (front_fields[:, surface_ops.shape[0] if include_correction else 0, :]
                        / model.target_scale).astype(call_args[0].dtype)
            if context.method_name == "_g1_baseline" and variant in ("g1", "all", "batched", "surface", "front"):
                eta_norm, xi_norm, log_depth = call_args
                eta_phys = (eta_norm * model.eta_scale).astype(jnp.float64)
                xi_phys = (xi_norm * model.xi_scale).astype(jnp.float64)
                h = jnp.exp(log_depth if model.baseline_true_depth else jnp.minimum(log_depth, jnp.log(model.h_clip_max))).astype(jnp.float64)
                k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float64)
                symbol = k[None, :] * jnp.tanh(h * k[None, :])
                if cufftdx_g1:
                    operators = jnp.stack((symbol, jnp.broadcast_to(1j * k, symbol.shape)), axis=1)
                    joint = trial in ("cufftdx_joint", "cufftdx_both")
                    result = sandwich(eta_phys, xi_phys, operators, joint=joint, ept=fft_ept)
                    return (-(result if joint else result.sum(axis=1)) / model.target_scale).astype(xi_norm.dtype)
                xi_hat = jnp.fft.rfft(xi_phys, axis=-1)
                if variant in ("batched", "surface", "front"):
                    operators = jnp.stack((symbol, jnp.broadcast_to(1j * k, symbol.shape)), axis=1)
                    fields = jnp.fft.irfft(operators * xi_hat[:, None, :], n=nx, axis=-1)
                    products_hat = jnp.fft.rfft(eta_phys[:, None, :] * fields, axis=-1)
                    spectrum = -jnp.sum(operators * products_hat, axis=1)
                    return (jnp.fft.irfft(spectrum, n=nx, axis=-1) / model.target_scale).astype(xi_norm.dtype)
                g0_xi = jnp.fft.irfft(symbol * xi_hat, n=nx, axis=-1)
                dx_xi = jnp.fft.irfft(1j * k[None, :] * xi_hat, n=nx, axis=-1)
                g0_hat = -symbol * jnp.fft.rfft(eta_phys * g0_xi, axis=-1)
                dx_hat = -1j * k[None, :] * jnp.fft.rfft(eta_phys * dx_xi, axis=-1)
                return (jnp.fft.irfft(g0_hat + dx_hat, n=nx, axis=-1) / model.target_scale).astype(xi_norm.dtype)

            is_multiplier = isinstance(module, DepthAwareMultiplier) and context.method_name == "__call__"
            if is_multiplier and cached is not None:
                block = int(cast(Scope, module.scope).path[-2].removeprefix("cs_block_"))
                result = cached[block]
            elif isinstance(module, nn.Dense) and module.name == "phi_proj" and trial in ("cufftdx_dense", "cufftdx_head"):
                if dense_weights is None:
                    dense_weights = dense(call_args[0], cast(jax.Array, head_kernel))
                block = int(cast(Scope, module.scope).path[-2].removeprefix("cs_block_"))
                result = dense_weights[..., block * model.latent:(block + 1) * model.latent]
            else:
                result = next_fun(*call_args, **call_kwargs)
            if variant in ("packed", "all", "batched", "surface", "front") and context.method_name == "__call__":
                if isinstance(module, nn.Dense) and module.name == "phi_proj":
                    spatial.append(result)
                elif is_multiplier:
                    multipliers.append(result)
                elif isinstance(module, CraigSulemBlock):
                    xi_phys = call_args[1]
                    if len(spatial) < model.n_blocks:
                        return jnp.zeros_like(xi_phys)
                    weights = jnp.concatenate(spatial, axis=-1)
                    filters = jnp.concatenate(multipliers, axis=-1)
                    if front_fields is not None:
                        filtered = jnp.moveaxis(front_fields[:, surface_ops.shape[0] + int(include_baselines):, :], 1, -1)
                    else:
                        xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")
                        filtered = jnp.fft.irfft(filters * xi_hat[..., None], n=nx, axis=1, norm="forward")
                    if trial in ("cufftdx_product", "cufftdx_both"):
                        spectra = fft(jnp.moveaxis(filtered, 1, -1), nx,
                                      b=jnp.moveaxis(weights, 1, -1),
                                      outer=jnp.moveaxis(filters, 1, -1).astype(jnp.complex64), ept=fft_ept)
                        spectrum = spectra.sum(axis=1)
                    else:
                        weighted_hat = jnp.fft.rfft(weights * filtered, axis=1, norm="forward")
                        spectrum = jnp.sum(filters * weighted_hat, axis=-1)
                    # Original per-group FFT outputs are unused and removed by XLA.
                    return jnp.fft.irfft(spectrum, n=nx, axis=-1, norm="forward")
            return result

        with nn.intercept_methods(intercept):
            return predict(eta, xi, depth)

    return forward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/c27_w320_b4_h80_tanaka_hard128_20260924")
    parser.add_argument("--checkpoint", choices=("best", "latest", "weight_ema"), default="best")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dno_fusion_20260924")
    parser.add_argument("--repeats", type=int, default=600)
    parser.add_argument("--variants", nargs="+", choices=(*VARIANTS, *TRIALS), default=VARIANTS)
    parser.add_argument("--reference", choices=(*VARIANTS, *TRIALS), default="original")
    parser.add_argument("--ept", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--capture", choices=(*VARIANTS, *TRIALS), help="Capture 100 warmed calls of one variant instead of timing/testing.")
    args = parser.parse_args()
    variants = args.variants
    if not args.capture and args.reference not in variants:
        parser.error("include the numerical reference in variants")
    loaded = load_run(args.run, checkpoint=args.checkpoint)
    source = next((ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918").glob(
        "eval_best_current_test_stratified_n32*/stokes_trajs.npz"
    ))
    with np.load(source) as saved:
        eta = jnp.asarray(saved["truth_eta"][0, :1], dtype=jnp.float64)
        xi = jnp.asarray(saved["truth_xi"][0, :1], dtype=jnp.float64)
        depth = jnp.log(jnp.asarray(saved["depths"][:1], dtype=jnp.float64))
    args.output.mkdir(parents=True, exist_ok=True)
    functions, records = {}, {}
    for variant in (args.capture,) if args.capture else variants:
        started = time.perf_counter()
        forward = build_variant(loaded, depth, eta.shape[-1], variant, fft_ept=args.ept)
        if variant.startswith("fourier"):
            forward = jax.jit(lambda e, x, fn=forward: jnp.fft.irfft(fn(jnp.fft.rfft(e), jnp.fft.rfft(x)), n=e.shape[-1]))
        executable = forward.lower(eta, xi).compile()
        records[variant] = {"setup_compile_s": time.perf_counter() - started, "seconds": [],
                            "xla_cost_estimate": executable.cost_analysis()}
        functions[variant] = executable
        (args.output / f"{variant}.hlo").write_text(executable.as_text())
        for _ in range(100):
            executable(eta, xi).block_until_ready()
        print(f"{variant}: compiled and warmed", flush=True)
    if args.capture:
        driver = ctypes.CDLL("libcuda.so.1")
        if driver.cuProfilerStart() != 0:
            raise RuntimeError("CUDA profiler start failed")
        for _ in range(100):
            functions[args.capture](eta, xi).block_until_ready()
        if driver.cuProfilerStop() != 0:
            raise RuntimeError("CUDA profiler stop failed")
        raise SystemExit(0)

    # Interleave variants to reduce ordering/clock bias; every timed call has batch one.
    for repeat in range(args.repeats):
        for variant in variants[::1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            functions[variant](eta, xi).block_until_ready()
            records[variant]["seconds"].append(time.perf_counter() - started)
    reference = np.asarray(functions[args.reference](eta, xi))
    for variant, record in records.items():
        actual = np.asarray(functions[variant](eta, xi))
        record.update(median_us=float(np.median(record["seconds"]) * 1e6),
                      p25_us=float(np.percentile(record["seconds"], 25) * 1e6),
                      p75_us=float(np.percentile(record["seconds"], 75) * 1e6),
                      relative_l2=float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)))
        print(f"{variant}: {record['median_us']:.2f} us; difference {record['relative_l2']:.3g}", flush=True)

    # Correctness uses 48 saved reference states, but is not part of the batch-one timing.
    validation_eta, validation_xi, validation_depth, families = [], [], [], []
    for family in ("stokes", "tanaka", "jonswap_tma", "benjamin_feir"):
        family_source = next(source.parent.parent.glob(f"eval_best_current_test_stratified_n32*/{family}_trajs.npz"))
        with np.load(family_source) as saved:
            indices = [0, len(saved["times"]) // 2, len(saved["times"]) - 1]
            validation_eta.append(saved["truth_eta"][indices, :4].reshape(-1, eta.shape[-1]))
            validation_xi.append(saved["truth_xi"][indices, :4].reshape(-1, xi.shape[-1]))
            validation_depth.append(np.tile(saved["depths"][:4], len(indices)))
            families.extend([family] * 12)
    check_eta, check_xi = map(jnp.asarray, (np.concatenate(validation_eta), np.concatenate(validation_xi)))
    check_depth = jnp.log(jnp.asarray(np.concatenate(validation_depth)))
    reference_predict = build_variant(loaded, check_depth, eta.shape[-1], args.reference, fft_ept=args.ept)
    if args.reference.startswith("fourier"):
        reference_predict = jax.jit(lambda e, x, fn=reference_predict: jnp.fft.irfft(
            fn(jnp.fft.rfft(e), jnp.fft.rfft(x)), n=e.shape[-1]))
    expected = np.asarray(reference_predict(check_eta, check_xi))
    for variant in (name for name in variants if name != args.reference):
        candidate = build_variant(loaded, check_depth, eta.shape[-1], variant, fft_ept=args.ept)
        if variant.startswith("fourier"):
            candidate = jax.jit(lambda e, x, fn=candidate: jnp.fft.irfft(fn(jnp.fft.rfft(e), jnp.fft.rfft(x)), n=e.shape[-1]))
        actual = np.asarray(candidate(check_eta, check_xi))
        relative = np.linalg.norm(actual - expected, axis=-1) / np.linalg.norm(expected, axis=-1)
        records[variant]["validation"] = {
            "samples": len(families), "max_relative_l2": float(relative.max()),
            "max_absolute": float(np.abs(actual - expected).max()),
            "family_max_relative_l2": {family: float(relative[np.asarray(families) == family].max()) for family in set(families)},
            "passed": (bool(np.isfinite(actual).all() and relative.max() <= 1e-5)
                       if "lowrank" not in variant else None),
        }
        if variant.startswith("fourier"):
            other_xi = jnp.roll(check_xi, 1, axis=0)
            other = np.asarray(candidate(check_eta, other_xi))
            mixed = np.asarray(candidate(check_eta, 0.3 * check_xi - 0.7 * other_xi))
            denominator = 0.3 * np.linalg.norm(actual) + 0.7 * np.linalg.norm(other)
            linear_error = float(np.linalg.norm(mixed - (0.3 * actual - 0.7 * other)) / denominator)
            other_reference = np.asarray(reference_predict(check_eta, other_xi))
            mixed_reference = np.asarray(reference_predict(check_eta, 0.3 * check_xi - 0.7 * other_xi))
            reference_linear_error = float(np.linalg.norm(mixed_reference - (0.3 * expected - 0.7 * other_reference)) /
                                          (0.3 * np.linalg.norm(expected) + 0.7 * np.linalg.norm(other_reference)))
            mixed_error = float(np.linalg.norm(mixed - mixed_reference) / np.linalg.norm(mixed_reference))
            flat = np.asarray(candidate(jnp.zeros_like(check_eta), check_xi))
            flat_reference = np.asarray(reference_predict(jnp.zeros_like(check_eta), check_xi))
            flat_error = float(np.linalg.norm(flat - flat_reference) / np.linalg.norm(flat_reference))
            zero_error = float(np.abs(np.asarray(candidate(check_eta, jnp.zeros_like(check_xi)))).max())
            records[variant]["validation"].update(linearity_relative_l2=linear_error,
                                                  reference_linearity_relative_l2=reference_linear_error,
                                                  mixed_relative_l2=mixed_error, flat_relative_l2=flat_error,
                                                  zero_xi_max_absolute=zero_error)
            records[variant]["validation"]["passed"] &= mixed_error <= 1e-5 and flat_error <= 1e-5 and zero_error == 0
        print(f"{variant}: 48-state max relative difference {relative.max():.3g}; "
              f"passed={records[variant]['validation']['passed']}", flush=True)
    report = {"run": str(args.run), "batch_size_timing": 1, "nx": eta.shape[-1],
              "parameters": sum(leaf.size for leaf in jax.tree.leaves(loaded.params)),
              "device": jax.devices()[0].device_kind, "reference": args.reference, "variants": records,
              "scope": "Model-only inference prototypes; production model and checkpoints unchanged."}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: dict[str, str] = dict(zip((*VARIANTS, "cufftdx"), (
        "Original", "Combine G₁ inverse FFTs", "Pack branch FFTs", "Cache depth multipliers",
        "Previous combined", "Batch G₁ FFTs", "Batch surface-feature FFTs", "Pack input-side FFTs", "cuFFTDx G₁",
    )))
    labels.update(terms_interleaved="CS-field decoder · batched recurrence",
                  terms_interleaved_features="CS-field decoder · batched recurrence + features",
                  fourier="Fourier input/output", fourier_round="Fourier I/O · physical input rounding",
                  fourier_reuse="Reuse FP32 input spectrum for G₁",
                  fourier_input="cuFFTDx FP64 input inverse FFT",
                  fourier_g1fft="cuFFTDx FP64 G₁ input FFT", fourier_features="cuFFTDx feature FFTs",
                  fourier_product="Fused learned product FFT", fourier_allfft="Combined FFT replacements",
                  fourier_fields="cuFFTDx packed field inverse FFT", fourier_kernels="G₁ FFT + field inverse + product FFT",
                  fourier_transforms="G₁ FFT + packed field inverse FFT",
                  fourier_fused_fields="Fuse field multipliers + inverse FFT",
                  fourier_balanced_features_overlap="Balance surface-feature stores + G₁ overlap",
                  fourier_fused_features_product_overlap="Fused head product FFT + G₁ overlap",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch="Batch context + nested first FFT",
                  fourier_fused_features_product_overlap_nestedfused="Fused nested FFT",
                  fourier_fused_features_product_overlap_g1fp32="Current network with FP32 G₁",
                  fourier_balanced_features_product_overlap="Balanced features + product FFT + G₁ overlap",
                  fourier_fused_inputs="Fuse input inverse FFT + normalization",
                  fourier_fused_both="Both inverse-FFT fusions",
                  fourier_fused_g1="Field fusion + G1 preparation fusion",
                  fourier_fused_fields_overlap="Overlap G1 and learned correction",
                  fourier_fused_g1_overlap="G1 preparation fusion + overlap",
                  fourier_fused_features="Field inverse FFT into surface-network features",
                  fourier_fused_features_overlap="Surface-feature fusion + G1 overlap",
                  cufftdx_joint="Joint G₁", cufftdx_product="Fused learned products",
                  cufftdx_both="Joint G₁ + learned products", cufftdx_dense="Fused dense activations",
                  cufftdx_head="Non-split projection", cufftdx_lowrank16="Rank-16 surface mixing",
                  cufftdx_lowrank32="Rank-32 surface mixing")
    labels.update({name: f"{labels[name.rpartition('_e')[0]]} · EPT {name.rpartition('_e')[2]}"
                   for name in TRIALS if name.endswith(("_e4", "_e16"))})
    labels.update({name: "FFT EPTs " + name.removeprefix("fourier_tuned_").replace("_", "/") for name in TUNED_TRIALS})
    labels.update({name: "FP32 G₁" + name.removeprefix(G1_FP32_BASE).replace("_", " · ") for name in G1_FP32_TRIALS})
    medians = [records[name]["median_us"] for name in variants]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh([labels[name] for name in variants], medians, color="#278475")
    ax.invert_yaxis()
    ax.set(xlabel="Warmed forward latency (µs); lower is better", xlim=(0, max(medians) * 1.25),
           title="Neural forward-pass candidates · one sample · no time integrator")
    for index, value in enumerate(medians):
        ax.text(value + 3, index, f"{value:.1f} µs", va="center")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"forward_fusion.{suffix}", dpi=180)
    if any(records[variant]["validation"]["passed"] is False for variant in variants if variant != args.reference):
        raise RuntimeError("Forward equivalence failed; see results.json for the rejected variants.")
