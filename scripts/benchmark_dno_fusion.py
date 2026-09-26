"""Benchmark-only inference kernels and explicitly named low-rank model trials."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import ctypes
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
from jax.stages import Wrapped  # noqa: E402

from scripts.benchmark_cufftdx import fft, sandwich  # noqa: E402
from scripts.benchmark_dno_dense import dense  # noqa: E402
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched, load_run  # noqa: E402
from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, DepthAwareMultiplier  # noqa: E402

VARIANTS = ("original", "g1", "packed", "cached", "all", "batched", "surface", "front")
TRIALS = ("cufftdx", "cufftdx_joint", "cufftdx_product", "cufftdx_both", "cufftdx_dense", "cufftdx_head",
          "cufftdx_lowrank16", "cufftdx_lowrank32", "cufftdx_e4", "cufftdx_e16",
          "cufftdx_product_e4", "cufftdx_product_e16")


def build_variant(
    loaded: LoadedRun, depth: jax.Array, nx: int, variant: str, *, include_baselines: bool = True,
    include_correction: bool = True, fft_ept: int = 8,
) -> Wrapped:
    """Intercept only inference arithmetic; leave the production module untouched."""
    if variant.endswith(("_e4", "_e16")):
        variant, _, suffix = variant.rpartition("_e")
        fft_ept = int(suffix)
    trial = variant
    cufftdx_g1 = variant.startswith("cufftdx")
    if cufftdx_g1:
        variant = "front"
    predict = build_predict_gxi_batched(loaded)
    model = loaded.model
    factors = None
    if "lowrank" in trial:
        rank = int(trial.split("lowrank")[1])
        with np.load(ROOT / "outputs/cufftdx_20260925/lowrank_factors.npz") as saved:
            factors = (jnp.asarray(saved[f"A{rank}"]), jnp.asarray(saved[f"B{rank}"]))
    head_kernel = jnp.concatenate(tuple(loaded.params[f"cs_block_{i}"]["phi_proj"]["kernel"]
                                        for i in range(model.n_blocks)), axis=-1)
    cached = None
    k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float32)
    surface_ops = jnp.stack([symbol for enabled, symbol in (
        (model.use_first_deriv, 1j * k),
        (model.use_second_deriv, -(k ** 2)),
        (model.use_half_deriv, jnp.sqrt(k)),
        (model.use_hilbert, -1j * jnp.sign(k)),
    ) if enabled])
    if include_correction and variant in ("cached", "all", "batched", "surface", "front"):
        multiplier = DepthAwareMultiplier(model.latent, model.domain_length, model.h_clip_max, model.mult_hidden)
        # This is a per-checkpoint/grid/depth inference constant, not a training cache.
        cached = jax.jit(lambda d: tuple(multiplier.apply(
            {"params": loaded.params[f"cs_block_{i}"]["m_shared"]}, d, nx // 2 + 1,
        ) for i in range(model.n_blocks)))(
            jnp.minimum(depth.astype(jnp.float32).reshape(-1, 1), jnp.log(model.h_clip_max))
        )
        jax.block_until_ready(cached)

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
                h = jnp.exp(jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
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
                features = call_args[0]
                eta_norm = features[..., 0]
                transformed = (front_fields[:, :surface_ops.shape[0], :] if front_fields is not None
                               else jnp.fft.irfft(surface_ops[None, :, :] * jnp.fft.rfft(eta_norm)[:, None, :], n=nx))
                call_args = (jnp.concatenate((features[..., :model.n_polys],
                                             jnp.moveaxis(transformed, 1, -1)), axis=-1),)
                if trial == "cufftdx_dense":
                    hidden = dense(call_args[0], loaded.params["eta_feat_proj"]["kernel"], "gelu")
                    hidden = dense(hidden, loaded.params["eta_feat_mix"]["kernel"], "gelu_lift")
                    dense_weights = dense(hidden, head_kernel)
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
                h = jnp.exp(jnp.minimum(log_depth, jnp.log(model.h_clip_max))).astype(jnp.float64)
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
                    dense_weights = dense(call_args[0], head_kernel)
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dno_fusion_20260924")
    parser.add_argument("--repeats", type=int, default=600)
    parser.add_argument("--variants", nargs="+", choices=(*VARIANTS, *TRIALS), default=VARIANTS)
    parser.add_argument("--ept", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--capture", choices=(*VARIANTS, *TRIALS), help="Capture 100 warmed calls of one variant instead of timing/testing.")
    args = parser.parse_args()
    variants = args.variants
    if not args.capture and "original" not in variants:
        parser.error("include original as the numerical reference")
    loaded = load_run(args.run)
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
    reference = np.asarray(functions["original"](eta, xi))
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
    original = build_predict_gxi_batched(loaded)
    expected = np.asarray(original(check_eta, check_xi, check_depth))
    for variant in (name for name in variants if name != "original"):
        candidate = build_variant(loaded, check_depth, eta.shape[-1], variant, fft_ept=args.ept)
        actual = np.asarray(candidate(check_eta, check_xi))
        relative = np.linalg.norm(actual - expected, axis=-1) / np.linalg.norm(expected, axis=-1)
        records[variant]["validation"] = {
            "samples": len(families), "max_relative_l2": float(relative.max()),
            "max_absolute": float(np.abs(actual - expected).max()),
            "family_max_relative_l2": {family: float(relative[np.asarray(families) == family].max()) for family in set(families)},
            "passed": (bool(np.isfinite(actual).all() and relative.max() <= 1e-5)
                       if "lowrank" not in variant else None),
        }
        print(f"{variant}: 48-state max relative difference {relative.max():.3g}; "
              f"passed={records[variant]['validation']['passed']}", flush=True)
    report = {"run": str(args.run), "batch_size_timing": 1, "nx": eta.shape[-1],
              "parameters": sum(leaf.size for leaf in jax.tree.leaves(loaded.params)),
              "device": jax.devices()[0].device_kind, "variants": records,
              "scope": "Model-only inference prototypes; production model and checkpoints unchanged."}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: dict[str, str] = dict(zip((*VARIANTS, "cufftdx"), (
        "Original", "Combine G₁ inverse FFTs", "Pack branch FFTs", "Cache depth multipliers",
        "Previous combined", "Batch G₁ FFTs", "Batch surface-feature FFTs", "Pack input-side FFTs", "cuFFTDx G₁",
    )))
    labels.update(cufftdx_joint="Joint G₁", cufftdx_product="Fused learned products",
                  cufftdx_both="Joint G₁ + learned products", cufftdx_dense="Fused dense activations",
                  cufftdx_head="Non-split projection", cufftdx_lowrank16="Rank-16 surface mixing",
                  cufftdx_lowrank32="Rank-32 surface mixing")
    labels.update({name: f"{labels[name.rpartition('_e')[0]]} · EPT {name.rpartition('_e')[2]}"
                   for name in TRIALS if name.endswith(("_e4", "_e16"))})
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
    if any(records[variant]["validation"]["passed"] is False for variant in variants if variant != "original"):
        raise RuntimeError("Forward equivalence failed; see results.json for the rejected variants.")
