"""Benchmark-only inference kernels and explicitly named low-rank model trials."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from scripts.benchmark_cufft_library import FFI_LIBRARY, graph_fft  # noqa: E402
from scripts.benchmark_classical import classical_hat  # noqa: E402
from scripts.benchmark_dno_dense import (  # noqa: E402
    analytic_g1, context_filter, cuda_tail_weights, dense, exact_analytic_g1, pack_tail_weights, tail_weights,
    trunk_fields, trunk_head, trunk_inputs,
)
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched, load_run  # noqa: E402
from dno_net_v2 import (  # noqa: E402
    CraigSulemBlock, CraigSulemDNO, CraigSulemTermsResidual, DepthAwareMultiplier,
    PhaseWavenumberMultiplier, phase_rank_one_hat, phase_wavenumber_hat, surface_amplitude,
    mixed_surface_lift, parity_dense, parity_gelu, surface_parity_products,
)

VARIANTS = ("original", "g1", "packed", "cached", "all", "batched", "surface", "front")
TUNED_TRIALS = tuple("fourier_tuned_" + "_".join(map(str, epts)) + suffix
                     for epts in product((4, 8, 16), repeat=4) for suffix in ("", "_mask"))
G1_FP32_BASE = "fourier_fused_features_product_overlap_g1fp32"
G1_FP32_TRIALS = tuple(G1_FP32_BASE + suffix + tuning
    for suffix in ("", "_reuse", "_inputfp32", "_reuse_inputfp32")
    for tuning in ("", *(f"_g1e{a}x{b}" for a, b in product((4, 8, 16), repeat=2))))
TRIALS = ("terms_interleaved", "terms_interleaved_features", "terms_interleaved_features_cufftdx",
          "terms_interleaved_features_cufftdx_spectral",
          "fourier", "fourier_round", "fourier_reuse", "fourier_input",
          "fourier_g1fft", "fourier_features", "fourier_product", "fourier_allfft",
          "fourier_fields", "fourier_kernels", "fourier_kernels_e4", "fourier_kernels_e16",
          "fourier_transforms", "fourier_transforms_e4", "fourier_transforms_e16",
          "fourier_fused_fields", "fourier_fused_inputs", "fourier_fused_both", "fourier_fused_g1",
          "fourier_fused_fields_overlap", "fourier_fused_g1_overlap",
          "fourier_fused_features", "fourier_fused_features_overlap",
          "fourier_balanced_features_overlap", "fourier_fused_features_product_overlap",
          "fourier_fused_features_product_overlap_nestedfused",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_reflectionencoder",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_contextfold",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_contextinverse",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_headtranspose",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_contexttranspose",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_fieldstride",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_branchmajor",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_mixtranspose",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_nestedoverlap",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_reflectionencoder",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_headpad",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextalias",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextadd",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextadd_mixgelu",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_splitg1",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_splitg1_gelureuse_branchmajor",
          "fourier_fused_features_product_overlap_nestedfused_contextbatch_serial",
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


def transposed_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Expose the transposed projection to GPU layout selection."""
    return jnp.swapaxes(jnp.swapaxes(b, -1, -2) @ jnp.swapaxes(a, -1, -2), -1, -2)


def build_variant(
    loaded: LoadedRun, depth: jax.Array, nx: int, variant: str, *, include_baselines: bool = True,
    include_correction: bool = True, fft_ept: int = 8, solver_g0: jax.Array | None = None,
    return_components: bool = False, reflection_encoder: bool = False,
) -> Wrapped:
    """Intercept inference arithmetic; optionally return full and learned Fourier outputs."""
    reflection_encoder = reflection_encoder or variant.endswith("_reflectionencoder")
    variant = variant.removesuffix("_reflectionencoder")
    model = loaded.model
    if (model.surface_parity_products or model.surface_cubic_channels or model.pair_coupling) and not (
        variant == "original" or variant.startswith("fourier")
    ):
        raise ValueError("Experimental surface lifts require original or Fourier branch inference.")
    if reflection_encoder and (isinstance(model, CraigSulemTermsResidual) or not variant.startswith("fourier")):
        raise ValueError("Encoder reflection averaging requires Fourier branch inference.")
    if reflection_encoder and (model.use_learned_eta_filters or model.use_relative_elevation):
        raise ValueError("Encoder reflection averaging requires the fixed surface-feature path.")
    if isinstance(model, CraigSulemTermsResidual):
        if not include_baselines or not include_correction or loaded.norm_mode != "scale":
            raise ValueError("CS-field inference requires the complete scale-normalized decoder.")
        if variant == "original":
            predict = build_predict_gxi_batched(loaded)
            return jax.jit(lambda eta, xi: predict(eta, xi, depth))
        if variant not in ("terms_interleaved", "terms_interleaved_features", "terms_interleaved_features_cufftdx",
                           "terms_interleaved_features_cufftdx_spectral"):
            raise ValueError("CS-field inference requires an original or terms_interleaved variant.")
        batch_features = "_features" in variant
        boundary_cufftdx = "_cufftdx" in variant
        spectral_io = variant.endswith("_spectral")
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
            if spectral_io:
                eta_physical = jnp.fft.irfft(eta.astype(jnp.complex128), n=nx)
                inputs = jnp.stack((eta_physical / scales[0], jnp.zeros_like(eta_physical)), axis=-1)
            else:
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
            if spectral_io:
                # Keep original FP64 spectra, skipping the analytic input normalization/FFT round trip.
                physical_hat = jnp.stack((eta, xi), axis=-2).astype(jnp.complex128)
            else:
                physical = jnp.moveaxis(inputs * scales, -1, -2)
                physical_hat = fft(physical, nx, ept=fft_ept) if boundary_cufftdx else jnp.fft.rfft(physical)
            eta_hat, xi_hat = jnp.moveaxis(physical_hat.at[..., -1].set(0), -2, 0)
            terms_hat = classical_hat(
                eta_hat, xi_hat, k, g0, nx=nx, order=model.baseline_order, pad_factor=2,
                backend="interleaved_cufftdx", precision="fp64", ept=fft_ept, return_terms=True,
            )
            terms = (fft(terms_hat, nx, inverse=True, ept=fft_ept) if boundary_cufftdx
                     else jnp.fft.irfft(terms_hat, n=nx))
            if spectral_io:
                correction = cast(tuple[jax.Array, ...], model.apply(
                    {}, coefficients, terms, method=model.decode_terms, return_baseline=True,
                ))[3][..., 0] * model.target_scale
                output_hat = jnp.sum(terms_hat, axis=-2) + fft(correction, nx, ept=fft_ept)
                return output_hat.at[..., 0].set(0).at[..., -1].set(output_hat[..., -1].real)
            prediction = cast(jax.Array, model.apply({}, coefficients, terms, method=model.decode_terms))
            output = (prediction[..., 0] * model.target_scale).astype(eta.dtype)
            return output - output.mean(axis=-1, keepdims=True)

        return forward_terms
    if variant.startswith("terms_"):
        raise ValueError("CS-field inference variants require a CS-field decoder.")
    cuda_fields = variant.endswith("_cudafields")
    cuda_input = cuda_fields or variant.endswith("_cudainput")
    variant = variant.removesuffix("_cudainput").removesuffix("_cudafields")
    exact_g1 = variant.endswith("_cudaexactg1")
    cuda_g1 = exact_g1 or variant.endswith("_cudag1")
    variant = variant.removesuffix("_cudag1").removesuffix("_cudaexactg1")
    cuda_tail = variant.endswith("_cudatail")
    fuse_tail = cuda_tail or variant.endswith("_fusedtail")
    variant = variant.removesuffix("_cudatail").removesuffix("_fusedtail")
    cuda_head = variant.endswith("_cudahead")
    variant = variant.removesuffix("_cudahead")
    cuda_context = variant.endswith("_cudacontext")
    variant = variant.removesuffix("_cudacontext")
    fuse_second_gelu = variant.endswith("_mixgelu")
    variant = variant.removesuffix("_mixgelu")
    context_add = variant.endswith("_contextadd")
    variant = variant.removesuffix("_contextadd")
    context_alias = variant.endswith("_contextalias")
    variant = variant.removesuffix("_contextalias")
    pad_head = variant.endswith("_headpad")
    variant = variant.removesuffix("_headpad")
    context_real = variant.endswith("_contextreal")
    variant = variant.removesuffix("_contextreal")
    nested_overlap = variant.endswith("_nestedoverlap")
    variant = variant.removesuffix("_nestedoverlap")
    transpose_mix = variant.endswith("_mixtranspose")
    variant = variant.removesuffix("_mixtranspose")
    branch_major = variant.endswith("_branchmajor")
    variant = variant.removesuffix("_branchmajor")
    reuse_first_gelu = variant.endswith("_gelureuse")
    variant = variant.removesuffix("_gelureuse")
    split_g1 = variant.endswith("_splitg1")
    variant = variant.removesuffix("_splitg1")
    field_stride = variant.endswith("_fieldstride")
    variant = variant.removesuffix("_fieldstride")
    transpose_head = variant.endswith(("_headtranspose", "_projectiontranspose"))
    transpose_context = variant.endswith(("_contexttranspose", "_projectiontranspose"))
    variant = variant.removesuffix("_headtranspose").removesuffix("_contexttranspose").removesuffix("_projectiontranspose")
    serial_g1 = variant.endswith("_serial")
    variant = variant.removesuffix("_serial")
    context_inverse = variant.endswith("_contextinverse")
    variant = variant.removesuffix("_contextinverse")
    context_fold = variant.endswith("_contextfold")
    variant = variant.removesuffix("_contextfold")
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
    if model.pair_coupling and (pad_head or (fused_product and (correction_nx != 1024 or fft_ept != 8))):
        raise ValueError("Fused pair coupling requires the unpadded 1024-point EPT8 head.")
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
    if (model.phase_rank_one or model.phase_wavenumber_hidden) and (
        trial == "fourier" or (not spectral_io and variant != "original")
    ):
        raise ValueError("Phase correction inference requires original or physically rounded Fourier I/O.")
    if model.surface_amplitude_reference > 0 and not spectral_io and variant != "original":
        raise ValueError("Amplitude-conditioned inference requires original or Fourier I/O.")
    if nested_fused and (nx != 1024 or coarse_correction or not model.nested_blocks or not spectral_io):
        raise ValueError("Nested fusion requires a 1024-point nested model with Fourier I/O.")
    if context_batch and not model.surface_context_width:
        raise ValueError("Context/nested FFT batching requires surface context channels.")
    if context_add and (not context_real or model.surface_context_gate or context_fold):
        raise ValueError("The fused context residual requires an ungated real-context inverse.")
    if reflection_encoder and context_add:
        raise ValueError("Encoder reflection averaging does not support the fused context-add kernel.")
    if model.surface_parity_streams and (reflection_encoder or context_fold or context_add):
        raise ValueError("Parity streams require single-pass, unfused block-diagonal context projections.")
    if fuse_second_gelu and (not context_add or model.baseline_order != 1 or model.surface_cubic_channels):
        raise ValueError("The fused second projection requires contextadd and the quadratic lift.")
    if fuse_tail and (not context_real or context_add or fuse_second_gelu or reflection_encoder or pad_head
                      or model.baseline_order != 1 or model.surface_cubic_channels or model.surface_context_gate
                      or model.surface_polynomial_skip or model.surface_parity_products or model.pair_coupling
                      or model.surface_amplitude_reference > 0):
        raise ValueError("The fused tail implements only the hero's real-context M1 trunk.")
    if cuda_head and (not (fused_features and context_batch and nested_fused) or nested_overlap or reflection_encoder
                      or model.surface_feature_scales is None or model.nested_blocks != 1 or model.nested_surface_features
                      or model.nested_surface_modulation or model.surface_parity_products or model.surface_polynomial_skip
                      or model.use_relative_elevation or model.surface_amplitude_reference > 0):
        raise ValueError("The CUDA trunk head implements only the hero's fixed-feature, batched-context trunk.")
    if cuda_context and not (cuda_head and context_real and nested_fused) or cuda_context and context_add:
        raise ValueError("The CUDA context filter follows the CUDA head on the real-context fused nested path.")
    if cuda_input and (nx != 1024 or not spectral_io or trial == "fourier"):
        raise ValueError("The CUDA trunk-input transform supports 1024-point Fourier-I/O inference only.")
    if cuda_fields and not (fused_features and not balanced_features and not coarse_correction):
        raise ValueError("The CUDA trunk-fields kernel replaces the fixed fused surface features and branch fields.")
    if cuda_g1 and (not overlap_g1 or fused_g1 or split_g1 or g1_fp32 or model.baseline_order != 1
                    or g1_input_ept != 8 or sandwich_ept != 8):
        raise ValueError("The CUDA G1 kernel replaces the accepted EPT8 FP64 stream-1 M1 G1 region only.")
    if nested_overlap and not (nested_fused and context_batch):
        raise ValueError("Nested overlap requires the fused nested path and batched context FFT.")
    if context_fold and (not spectral_io or not model.surface_context_width
                         or model.surface_context_gate or model.surface_amplitude_reference > 0):
        raise ValueError("Context folding requires ungated Fourier-I/O surface context.")
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
    if model.pair_coupling and head_kernel is not None:
        head_kernel = jnp.concatenate((head_kernel, jnp.concatenate(tuple(
            loaded.params[f"cs_block_{i}"]["phi_cross"]["kernel"]
            for i in range(ordinary_blocks)), axis=-1)), axis=-1)
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
        if pad_head:
            head_kernel = jnp.pad(cast(jax.Array, head_kernel),
                                 ((0, 0),) * (cast(jax.Array, head_kernel).ndim - 1)
                                 + ((0, model.nested_blocks * model.latent),))
        tail_packed, cuda_tail_kernel = (), tail_weights
        if cuda_tail:
            *tail_packed, tail_scales = pack_tail_weights(loaded.params["eta_feat_mix"]["kernel"],
                                                          cast(jax.Array, head_kernel))
            tail_packed, cuda_tail_kernel = tuple(map(jnp.asarray, tail_packed)), cuda_tail_weights(tail_scales)
        first_projection_kernel = None
        if not model.surface_parity_streams:
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
        g1_factors = (model.eta_scale / float(scales[0]) / nx, model.xi_scale / float(scales[1]))
        input_multipliers = (1 / (nx * float(scales[0])), model.xi_scale / (nx * float(scales[1])))
        output_scale = float(loaded.stats["target_absmax"]) / model.target_scale
        log_depth = depth.astype(jnp.float32).reshape(-1, 1)
        phase_multiplier = None
        if model.phase_wavenumber_hidden:
            phase_multiplier = cast(jax.Array, PhaseWavenumberMultiplier(
                model.phase_wavenumber_hidden, model.domain_length,
            ).apply({"params": loaded.params["phase_wavenumber"]}, log_depth, nx // 2 + 1))
            jax.block_until_ready(phase_multiplier)
        nested_modulation = ((jnp.tanh(loaded.params["nested_surface_gain"]), jnp.exp(log_depth))
                             if model.nested_surface_modulation else None)
        h = jnp.exp(log_depth if model.baseline_true_depth else jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
        g0 = k[None, :] * jnp.tanh(h * k[None, :])
        k64 = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float64)
        g0_64 = k64[None, :] * jnp.tanh(h.astype(jnp.float64) * k64[None, :])
        operators = jnp.stack((g0_64, jnp.broadcast_to(1j * k64, g0_64.shape)), axis=1)
        if g1_fp32:
            operators = operators.astype(jnp.complex64)
        context_multiplier = context_joint_kernel = context_folded_kernel = None
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
            if context_fold:
                mixing = loaded.params["eta_feat_mix"]["kernel"]
                context_joint_kernel = jnp.concatenate((mixing, loaded.params["eta_context_in"]["kernel"]), axis=-1)
                context_folded_kernel = jnp.matmul(
                    loaded.params["eta_context_expand"]["kernel"], mixing, precision=jax.lax.Precision.HIGHEST,
                )
                jax.block_until_ready((context_joint_kernel, context_folded_kernel))

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
        def forward_hat(eta_hat: jax.Array, xi_hat: jax.Array) -> jax.Array | tuple[jax.Array, jax.Array]:
            analytic_hat = None
            # Preserve the trained surface features' physical-space FP32 rounding.
            eta_norm = jnp.fft.irfft(eta_hat, n=nx).astype(jnp.float32) / scales[0]
            eta_norm_hat = jnp.fft.rfft(eta_norm)
            xi_phys_hat = (xi_hat.astype(jnp.complex64) / scales[1]) * model.xi_scale
            g1_xi_hat = xi_hat * (model.xi_scale / scales[1]).astype(jnp.float64)
            normalized_hat = normalized_inputs = None
            field_outputs: list[jax.Array] | None = None
            if trial == "fourier" and model.baseline_order == 2:
                xi_phys = (jnp.fft.irfft(xi_hat, n=nx).astype(jnp.float32) / scales[1]) * model.xi_scale
                normalized_inputs = jnp.stack((eta_norm, xi_phys), axis=1)
            if trial != "fourier":
                if cuda_fields:
                    normalized_inputs, *field_outputs = trunk_fields(eta_hat, xi_hat, surface_ops, filters,
                                                                     *input_multipliers)
                    # Spectra of the normalized inputs, for consumers other than the fused fields.
                    normalized_hat = (jnp.stack((eta_hat, xi_hat), axis=1)
                                      * (nx * jnp.asarray(input_multipliers))[:, None]).astype(jnp.complex64)
                    eta_norm, xi_phys = normalized_inputs[:, 0], normalized_inputs[:, 1]
                elif cuda_input:
                    normalized_inputs, normalized_hat = trunk_inputs(eta_hat, xi_hat, *input_multipliers)
                    eta_norm, xi_phys = normalized_inputs[:, 0], normalized_inputs[:, 1]
                else:
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
                    with (nullcontext() if serial_g1 else compute_on("gpu_stream:1")):
                        if model.baseline_order == 2:
                            analytic_hat = m2_hat(normalized_inputs)
                        elif exact_g1:
                            analytic_hat = jax.jit(lambda eta, xi: exact_analytic_g1(eta, xi, operators, *g1_factors))(
                                eta_hat, xi_hat)
                        elif cuda_g1:
                            analytic_hat = jax.jit(lambda values: analytic_g1(values, operators, model.eta_scale))(
                                normalized_inputs)
                        elif split_g1:
                            analytic_hat = jax.jit(lambda eta, xi: -sandwich(
                                (eta * model.eta_scale).astype(g1_dtype),
                                fft(xi.astype(g1_dtype), nx, ept=g1_input_ept), operators,
                                spectral=True, ept=sandwich_ept,
                            ).sum(axis=1))(eta_norm, xi_phys)
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
            if phase_multiplier is not None:
                phase_hat = phase_wavenumber_hat(
                    eta_norm_hat * model.eta_scale, xi_phys_hat, phase_multiplier, model.domain_length,
                )
            if coarse_correction:
                normalized_hat = resample_rfft(cast(jax.Array, normalized_hat), correction_nx)
                normalized_inputs = fft(normalized_hat, correction_nx, inverse=True, ept=input_ept)
                eta_norm, xi_phys = jnp.moveaxis(normalized_inputs, 1, 0)
                eta_norm_hat, xi_phys_hat = jnp.moveaxis(normalized_hat, 1, 0)
            if fused_features:
                features, branch_fields = (field_outputs if field_outputs is not None else field_irfft(
                    cast(jax.Array, normalized_hat), surface_ops, filters, n=correction_nx, ept=field_ept,
                    normalized_inputs=normalized_inputs, balanced=balanced_features))
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
            if model.surface_parity_products:
                features = surface_parity_products(features, include_odd=model.surface_parity_streams)
            nested_eta = eta_norm
            if model.nested_blocks:
                nested_eta = eta_norm if model.surface_feature_scales is None else eta_norm / model.surface_feature_scales[0]
                if model.surface_amplitude_reference > 0:
                    nested_eta = eta_norm / model.surface_amplitude_reference
                if nested_modulation is not None:
                    gain, actual_depth = nested_modulation
                    nested_eta = nested_eta * (
                        1.0 + gain * jnp.tanh(eta_norm * model.eta_scale / actual_depth)
                    )
            amplitude = None
            if model.surface_amplitude_reference > 0:
                amplitude = surface_amplitude(eta_norm, model.surface_amplitude_reference)
                degrees = jnp.asarray(
                    [*range(1, model.n_polys + 1), *([1] * surface_ops.shape[0])], dtype=features.dtype,
                )
                features = features / amplitude[..., None] ** degrees
            if model.surface_feature_scales is not None and not cuda_head:
                features = features / jnp.asarray(model.surface_feature_scales, dtype=features.dtype)
            nested_eta = nested_eta[:, None, :]
            if model.nested_surface_features:
                second_index = model.n_polys + int(model.use_first_deriv)
                half_index = second_index + int(model.use_second_deriv)
                nested_eta = nested_eta + jnp.moveaxis(
                    features[..., jnp.asarray([half_index, second_index])] @ cast(jax.Array, surface_mix), -1, 1,
                )
            polynomial_features = features[..., 1:3] if model.surface_polynomial_skip else None
            parity_features = features[..., jnp.asarray([3, 6, 7])] if model.surface_parity_products else None
            if reflection_encoder:
                parity = jnp.asarray(
                    [1] * features.shape[-1] if model.surface_parity_products else [1] * model.n_polys + [
                        sign for enabled, sign in (
                            (model.use_first_deriv, -1), (model.use_second_deriv, 1),
                            (model.use_half_deriv, 1), (model.use_hilbert, -1),
                        ) if enabled
                    ], dtype=features.dtype,
                )
                # Two geometry views; the xi fields, nested branches and baseline stay single-pass.
                features = jnp.stack((features, features * parity), axis=-2)
            context_channels = model.surface_context_width * (2 if reflection_encoder else 1)
            context = None
            tail = head_input = cuda_context_field = nested_pre = None
            nested_inner_hat = None
            nested_hat = None
            for name in ("eta_feat_proj", "eta_feat_mix"):
                if fuse_tail and name == "eta_feat_mix":
                    continue
                if context_fold and name == "eta_feat_mix":
                    features = nn.gelu(features)
                    continue
                if name == "eta_feat_mix" and amplitude is not None:
                    log_amplitude = jnp.log(amplitude)[..., None]
                    if reflection_encoder:
                        log_amplitude = log_amplitude[..., None]
                    features = features * (1.0 + log_amplitude
                                           * loaded.params["eta_amplitude_gain"])
                if cuda_head and name == "eta_feat_proj":
                    features, head_input = trunk_head(
                        jnp.moveaxis(features, -1, 1), cast(jax.Array, normalized_inputs), branch_fields,
                        jnp.asarray(model.surface_feature_scales, dtype=jnp.float32), cast(jax.Array, first_projection_kernel),
                        loaded.params["eta_context_in"]["kernel"], cast(tuple, model.surface_feature_scales)[0], ordinary_branches)
                elif model.surface_parity_streams:
                    layer = loaded.params[name]
                    features = parity_gelu(parity_dense(features, layer["even_kernel"], layer["odd_kernel"]))
                else:
                    kernel = (cast(jax.Array, first_projection_kernel) if name == "eta_feat_proj"
                              else loaded.params[name]["kernel"])
                    if fuse_second_gelu and name == "eta_feat_mix":
                        features = dense(features, kernel, "gelu_lift",
                                         precision=jax.lax.Precision.DEFAULT)
                    else:
                        features = nn.gelu(transposed_matmul(features, kernel)
                                           if transpose_mix and name == "eta_feat_mix" else features @ kernel)
                if reuse_first_gelu and name == "eta_feat_proj" and not cuda_head:
                    features = jax.lax.optimization_barrier(features)
                if name == "eta_feat_proj" and context_multiplier is not None:
                    if cuda_head:
                        context = cast(jax.Array, head_input)
                    elif context_fold:
                        joint = features @ cast(jax.Array, context_joint_kernel)
                        features, context = joint[..., :-model.surface_context_width], joint[..., -model.surface_context_width:]
                    elif model.surface_parity_streams:
                        layer = loaded.params["eta_context_in"]
                        context = parity_dense(features, layer["even_kernel"], layer["odd_kernel"])
                    else:
                        context = (transposed_matmul(features, loaded.params["eta_context_in"]["kernel"])
                                   if transpose_context else features @ loaded.params["eta_context_in"]["kernel"])
                    if reflection_encoder:
                        context = context.reshape(context.shape[0], correction_nx, context_channels)
                    if cuda_context:
                        cuda_context_field, nested_pre = context_filter(
                            cast(jax.Array, head_input), cast(jax.Array, context_multiplier), cast(jax.Array, inner_filters),
                            cast(jax.Array, gamma), cast(jax.Array, normalized_inputs),
                            cast(tuple, model.surface_feature_scales)[0], ept=fft_ept)
                        context_hat = cast(jax.Array, None)  # the filter consumes the spectra internally
                    elif context_batch:
                        # Independent fields share one FFT batch, not Fourier modes.
                        combined_hat = jnp.fft.rfft(cast(jax.Array, head_input) if cuda_head else jnp.concatenate((
                            jnp.moveaxis(context, -1, 1),
                            nested_eta * branch_fields[:, ordinary_branches:],
                        ), axis=1))
                        context_hat = jnp.moveaxis(combined_hat[:, :context_channels], 1, -1)
                        nested_inner_hat = combined_hat[:, context_channels:]
                        if nested_overlap:
                            with compute_on("gpu_stream:2"):
                                nested_hat = jax.jit(lambda values: filters[:, ordinary_branches:] * sandwich(
                                    (cast(jax.Array, gamma)[None, :, None] * values[1]).reshape(-1, correction_nx),
                                    (cast(jax.Array, inner_filters) * values[0][:, context_channels:]).reshape(
                                        -1, values[0].shape[-1]),
                                    cast(jax.Array, nested_symbols), spectral=True, ept=fft_ept,
                                ).reshape(cast(jax.Array, inner_filters).shape))((combined_hat, nested_eta))
                    else:
                        context_hat = jnp.fft.rfft(context, axis=1)
                    context_symbol = (jnp.tile(context_multiplier, (1, 1, 2))
                                      if reflection_encoder else context_multiplier)
                    if cuda_context:
                        context = jnp.moveaxis(cast(jax.Array, cuda_context_field), -2, -1)
                    elif context_real:
                        # A real multiplier preserves the RFFT's real DC/Nyquist coefficients.
                        context = graph_fft(
                            context_symbol * context_hat, n=correction_nx, axis=1,
                            inverse=True, hermitian_input=True,
                            input_alias=context_alias,
                            norm="forward" if context_add or fuse_tail else None,
                            library=(ROOT / "outputs/cufft_graph_ffi_20261019/cufft_graph_ffi_alias.so"
                                     if context_alias else FFI_LIBRARY),
                        )
                    elif context_inverse:
                        context = jnp.moveaxis(fft(
                            jnp.moveaxis(context_symbol * context_hat, 1, -1),
                            correction_nx, inverse=True, ept=fft_ept,
                        ), -1, 1)
                    else:
                        context = jnp.fft.irfft(
                            context_symbol * context_hat, n=correction_nx, axis=1,
                        )
                    if reflection_encoder:
                        context = context.reshape(context.shape[0], correction_nx, 2, model.surface_context_width)
                    if fuse_tail:
                        tail = (cuda_tail_kernel(context, features, loaded.params["eta_context_expand"]["kernel"],
                                                 *tail_packed) if cuda_tail
                                else tail_weights(context, features, loaded.params["eta_context_expand"]["kernel"],
                                                  loaded.params["eta_feat_mix"]["kernel"], cast(jax.Array, head_kernel)))
                    elif context_add:
                        features = dense(context, loaded.params["eta_context_expand"]["kernel"],
                                         residual=features, input_scale=1 / correction_nx,
                                         tile=(32, 32, 16), num_warps=2, input_transposed=True,
                                         precision=jax.lax.Precision.DEFAULT)
                    elif model.surface_parity_streams:
                        layer = loaded.params["eta_context_expand"]
                        context = parity_dense(context, layer["even_kernel"], layer["odd_kernel"])
                        features = features + context
                    else:
                        context = context @ (cast(jax.Array, context_folded_kernel) if context_fold
                                             else loaded.params["eta_context_expand"]["kernel"])
                        features = features + context
            if not (fuse_second_gelu or fuse_tail):
                features = (mixed_surface_lift(features, model.surface_cubic_channels)
                            if model.surface_cubic_channels else
                            features * jnp.tanh(features) ** (2 if model.baseline_order == 2 else 1))
            if model.surface_context_gate and context is not None:
                features = features * (1.0 + jnp.tanh(context))
            if reflection_encoder:
                # The head is linear; average after every nonlinear geometry operation.
                features = jnp.mean(features, axis=-2)
            if polynomial_features is not None:
                features = jnp.concatenate((features, polynomial_features), axis=-1)
            if parity_features is not None:
                features = jnp.concatenate((features, parity_features), axis=-1)
            if amplitude is not None:
                features = features * amplitude[..., None] ** 2
            correction_hat = jnp.zeros_like(xi_phys_hat)
            if ordinary_blocks:
                weights = (cast(jax.Array, tail) if fuse_tail else transposed_matmul(features, cast(jax.Array, head_kernel))
                           if transpose_head else features @ cast(jax.Array, head_kernel))
                ordinary_fields = branch_fields if field_stride else branch_fields[:, :ordinary_branches]
                ordinary_filters = filters[:, :ordinary_branches]
                if pad_head:
                    ordinary_filters = jnp.pad(ordinary_filters,
                                              ((0, 0), (0, model.nested_blocks * model.latent), (0, 0)))
                if fused_product:
                    weighted_hat = fft(weights, correction_nx, b=ordinary_fields, outer=ordinary_filters,
                                       ept=fft_ept, channels_last=True,
                                       field_stride=field_stride or model.pair_coupling,
                                       branch_major=branch_major, pair_coupling=model.pair_coupling)
                elif model.pair_coupling:
                    weights = jnp.moveaxis(weights, -1, 1)
                    fields = ordinary_fields[:, :ordinary_branches]
                    weighted = (weights[:, :ordinary_branches] * fields
                                + jnp.repeat(weights[:, ordinary_branches:], 2, axis=1)
                                * fields[:, jnp.arange(ordinary_branches) ^ 1])
                    weighted_hat = ordinary_filters * jnp.fft.rfft(weighted)
                elif trial in ("fourier_product", "fourier_allfft", "fourier_kernels"):
                    weights = jnp.moveaxis(weights, -1, 1)
                    weighted_hat = correction_nx * fft(weights, correction_nx, b=ordinary_fields,
                                           outer=ordinary_filters.astype(jnp.complex64), ept=fft_ept)
                else:
                    weighted_hat = ordinary_filters * jnp.fft.rfft(jnp.moveaxis(weights, -1, 1) * ordinary_fields)
                correction_hat = jnp.sum(weighted_hat[:, :ordinary_branches], axis=1)
            if model.nested_blocks:
                inner_hat = (cast(jax.Array, nested_inner_hat) if context_batch else
                             jnp.fft.rfft(nested_eta * branch_fields[:, ordinary_branches:]))
                if cuda_context:
                    nested_hat = filters[:, ordinary_branches:] * cast(jax.Array, nested_pre)
                elif nested_fused and not nested_overlap:
                    inner_hat = cast(jax.Array, inner_filters) * inner_hat
                    # Preserve the first FFT; unit symbols fuse IFFT/product/FFT with unnormalized output.
                    surface = cast(jax.Array, gamma)[None, :, None] * nested_eta
                    nested_hat = filters[:, ordinary_branches:] * sandwich(
                        surface.reshape(-1, correction_nx), inner_hat.reshape(-1, inner_hat.shape[-1]),
                        cast(jax.Array, nested_symbols), spectral=True, ept=fft_ept,
                    ).reshape(inner_hat.shape)
                elif not nested_overlap:
                    middle = jnp.fft.irfft(cast(jax.Array, inner_filters) * inner_hat, n=correction_nx)
                    nested_hat = filters[:, ordinary_branches:] * jnp.fft.rfft(
                        cast(jax.Array, gamma)[None, :, None] * nested_eta * middle)
                correction_hat = correction_hat + jnp.sum(cast(jax.Array, nested_hat), axis=1)
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
                result = cast(jax.Array, jnp.where(k64 == 0, 0, jnp.where(k64 == k64[-1], result.real, result)))
            else:
                result = result.at[..., 0].set(0).at[..., -1].set(result[..., -1].real)
            if return_components:
                correction = output_scale * correction_hat
                correction = correction.at[..., 0].set(0).at[..., -1].set(correction[..., -1].real)
                return result, correction
            return result

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
    parser.add_argument("--checkpoint", choices=("best", "best_weight_ema", "latest", "weight_ema"), default="best")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dno_fusion_20260924")
    parser.add_argument("--repeats", type=int, default=600)
    parser.add_argument("--variants", nargs="+", choices=(*VARIANTS, *TRIALS), default=VARIANTS)
    parser.add_argument("--reference", choices=(*VARIANTS, *TRIALS), default="original")
    parser.add_argument("--ept", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--reference-ept", type=int, choices=(4, 8, 16), help="Hold the reference FFT execution layout fixed while tuning candidates.")
    parser.add_argument("--capture", choices=(*VARIANTS, *TRIALS), help="Capture 100 warmed calls of one variant instead of timing/testing.")
    args = parser.parse_args()
    reference_ept = args.reference_ept or args.ept
    variants = args.variants
    if not args.capture and args.reference not in variants:
        parser.error("include the numerical reference in variants")
    if any(name.endswith("_contextfold") for name in variants) and not args.reference.startswith("fourier"):
        parser.error("context folding requires a Fourier-I/O reference for direct-correction checks")
    loaded = load_run(args.run, checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir)
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
        forward = build_variant(loaded, depth, eta.shape[-1], variant,
                                fft_ept=reference_ept if variant == args.reference else args.ept)
        if variant.startswith("fourier") or variant == "terms_interleaved_features_cufftdx_spectral":
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
    reference_predict = build_variant(loaded, check_depth, eta.shape[-1], args.reference, fft_ept=reference_ept)
    if args.reference.startswith("fourier") or args.reference == "terms_interleaved_features_cufftdx_spectral":
        reference_predict = jax.jit(lambda e, x, fn=reference_predict: jnp.fft.irfft(
            fn(jnp.fft.rfft(e), jnp.fft.rfft(x)), n=e.shape[-1]))
    expected = np.asarray(reference_predict(check_eta, check_xi))
    for variant in (name for name in variants if name != args.reference):
        candidate = build_variant(loaded, check_depth, eta.shape[-1], variant, fft_ept=args.ept)
        if variant.startswith("fourier") or variant == "terms_interleaved_features_cufftdx_spectral":
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
        if variant.endswith(("_contextfold", "_contextinverse", "_headtranspose",
                             "_contexttranspose", "_projectiontranspose", "_fieldstride", "_splitg1", "_gelureuse", "_branchmajor", "_mixtranspose", "_nestedoverlap", "_contextreal", "_headpad", "_contextalias", "_contextadd", "_mixgelu")):
            correction_outputs = []
            instrumented_exact = True
            for name, original in ((args.reference, expected), (variant, actual)):
                components = build_variant(loaded, check_depth, eta.shape[-1], name,
                                           fft_ept=reference_ept if name == args.reference else args.ept,
                                           return_components=True)
                full_hat, correction_hat = components(jnp.fft.rfft(check_eta), jnp.fft.rfft(check_xi))
                instrumented_full = np.asarray(jnp.fft.irfft(full_hat, n=eta.shape[-1]))
                instrumented_exact &= bool(np.array_equal(instrumented_full, original))
                correction_outputs.append(np.asarray(jnp.fft.irfft(correction_hat, n=eta.shape[-1])))
            correction_expected, correction_actual = correction_outputs
            correction_relative = np.linalg.norm(correction_actual - correction_expected, axis=-1) / np.maximum(
                np.linalg.norm(correction_expected, axis=-1), 1e-30,
            )
            records[variant]["validation"]["correction_max_relative_l2"] = float(correction_relative.max())
            records[variant]["validation"]["instrumented_full_exact"] = instrumented_exact
            records[variant]["validation"]["passed"] &= bool(
                instrumented_exact and np.isfinite(correction_actual).all() and correction_relative.max() <= 3e-6,
            )
        if variant.startswith("fourier") or variant == "terms_interleaved_features_cufftdx_spectral":
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
    report = {"run": str(args.run), "checkpoint": args.checkpoint, "epoch": loaded.epoch,
              "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
              "batch_size_timing": 1, "nx": eta.shape[-1],
              "parameters": sum(leaf.size for leaf in jax.tree.leaves(loaded.params)),
              "device": jax.devices()[0].device_kind, "reference": args.reference, "variants": records,
              "fft_ept": args.ept, "reference_fft_ept": reference_ept,
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
                  terms_interleaved_features_cufftdx="CS-field decoder · cuFFTDx boundaries",
                  terms_interleaved_features_cufftdx_spectral="CS-field decoder · Fourier input/output",
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
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_serial="Serial G₁",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_contextfold="Fold context projections",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_contextinverse="cuFFTDx context inverse FFT",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_headtranspose="Transposed head GEMM",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_contexttranspose="Transposed context GEMM",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose="Transposed head + context GEMMs",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_fieldstride="Direct strided branch fields",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride="Transposed projections + strided fields",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse="Reuse first surface GELU",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_branchmajor="Branch-major product FFT",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor="Branch-major FFT and reused GELU",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_mixtranspose="Transposed second surface projection",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_nestedoverlap="Overlap nested branch with surface MLP",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal="Skip real-context endpoint clears",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_headpad="Zero-pad head to complete branch width",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextalias="Reuse dead context inverse input",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextadd="Fuse context projection and residual addition",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal_contextadd_mixgelu="Fuse second projection, GELU and lift",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_splitg1="Separate G₁ input arguments",
                  fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_splitg1_gelureuse_branchmajor="Separate G₁ inputs with branch-major FFT",
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
