"""Evaluate checkpoint accuracy with the timed Fourier-I/O implementation.

Reuses saved M6 references from this run, --reference-run, or --reference-dir. Initial conditions
come from the candidate's dataset, so native candidate rollouts are unnecessary.
Reported wall times are for 32 trajectories together, excluding compilation.
Set XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true for G1 overlap.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# Import the evaluation module first; the benchmark modules also depend on it.
from solver.evals.eval_suite import (  # noqa: E402
    FAMILY_CONFIGS, RolloutPayload, _load_paper_dataset_ics, _write_json,
    compute_macro_summary, run_family,
)
from solver.evals.model_rollout import build_predict_gxi_batched, load_run  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from scripts.benchmark_dno_fusion import build_variant  # noqa: E402
from scripts.benchmark_cufft_library import graph_fft  # noqa: E402
from scripts.benchmark_surrogate_rhs import physical_from_spectral, rollout_if_trial  # noqa: E402
from scripts.time_single_rollouts import packed_boundary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument("--matmul-precision", choices=("default", "highest"), default="default",
                        help="Precision for both cached multipliers and runtime matrix products.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("weight_ema", "weight_ema_secondary", "best_weight_ema", "latest"), default="weight_ema")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory containing metadata.json and ckpt_<epoch>; use --checkpoint best_weight_ema for EMA from a full state.")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-run", type=Path, help="Run with saved epoch-one M6 reference trajectories")
    reference.add_argument("--reference-dir", type=Path, help="Directory containing cached M6 references for each family")
    parser.add_argument("--ic-quantile-offset", type=float, default=0.5, help="Within-group stratified case quantile offset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--center-xi", action="store_true",
                        help="Remove the potential's mean at every DNO call, including internal stages.")
    parser.add_argument("--project-momentum", action="store_true", help="Experimental post-step momentum correction.")
    parser.add_argument("--reflection-average", action="store_true",
                        help="Diagnostic: average spectral predictions with reflected inputs/outputs; doubles model calls.")
    parser.add_argument("--reflection-encoder", action="store_true",
                        help="Average reflected geometry encodings; share the baseline and branch FFTs.")
    parser.add_argument("--exact-solver-g0", action="store_true", help="Use the solver's FP64 linear symbol and original potential spectrum.")
    parser.add_argument("--fuse-nested-fft", action="store_true", help="Fuse the final nested correction FFT pair.")
    parser.add_argument("--batch-context-fft", action="store_true", help="Batch context and nested first FFTs; requires --fuse-nested-fft.")
    parser.add_argument("--fold-context", action="store_true", help="Reassociate and batch the fixed context projections.")
    parser.add_argument("--graph-fft", action="store_true", help="Trial graph-compatible cuFFT calls throughout the rollout.")
    parser.add_argument("--graph-fft-mask", action="store_true", help="Fuse inverse endpoint preparation; requires --graph-fft.")
    parser.add_argument("--unroll-stages", action="store_true", help="Unroll the same four fixed-point iterations.")
    parser.add_argument("--materialize-spectra", action="store_true", help="Compute shared rotated spectra once.")
    parser.add_argument("--materialize-updated-state", action="store_true", help="Materialize the completed step before Hermitian projection.")
    parser.add_argument("--cache-rotation-products", action="store_true", help="Cache shared linear-flow coefficients once per saved interval.")
    parser.add_argument("--transpose-projections", action="store_true", help="Use context/head operand layouts from the timing trial.")
    parser.add_argument("--field-stride", action="store_true", help="Read ordinary branch fields without an intermediate copy.")
    parser.add_argument("--reuse-first-gelu", action="store_true", help="Reuse the first surface activation across its two consumers.")
    parser.add_argument("--branch-major", action="store_true", help="Consume the head projection without a paired-stage layout copy.")
    parser.add_argument("--real-context-endpoints", action="store_true", help="Skip clearing the already-real context FFT endpoints.")
    parser.add_argument("--fuse-context-add", action="store_true", help="Fuse the context projection and residual addition; requires --real-context-endpoints.")
    parser.add_argument("--cuda-head", action="store_true", help="CUDA kernel for the first projection, GELU, context projection and nested FFT inputs.")
    parser.add_argument("--cuda-context", action="store_true", help="cuFFTDx kernel for the context filter and nested sandwich; requires --cuda-head.")
    parser.add_argument("--fuse-tail", choices=("pallas", "cuda"), help="One Pallas pair or CUDA kernel for context expansion, mix MLP, lift and head; requires --real-context-endpoints.")
    parser.add_argument("--cuda-input", action="store_true", help="One FP32 cuFFTDx kernel for the normalized trunk inputs and their spectra.")
    parser.add_argument("--cuda-g1", choices=("rounded", "exact"), help="One FP64 cuFFTDx kernel for the stream-1 M1 G1 region, bit-identical (rounded) or from the exact FP64 state.")
    parser.add_argument("--bernoulli-fft", choices=("cufft", "cufftdx_forward", "cufftdx_both", "cufftdx_fused"), default="cufft")
    parser.add_argument("--batch-saved-dno", action="store_true", help="Compute all saved-frame DNO diagnostics after integration.")
    parser.add_argument("--pack-saved-inputs", action="store_true", help="Pack eta/xi input FFTs for saved DNO diagnostics.")
    parser.add_argument("--spectral-saved-dno", action="store_true", help="Evaluate saved DNO diagnostics directly from the spectral carry.")
    parser.add_argument("--cs-terms-variant", choices=("terms_interleaved", "terms_interleaved_features",
                                                    "terms_interleaved_features_cufftdx",
                                                    "terms_interleaved_features_cufftdx_spectral"),
                        help="Use optimized FP64 pad2 CS-field inference; decoder only. Default: native.")
    parser.add_argument("--correction-grid", type=int, choices=(512, 1024), default=1024,
                        help="Learned correction grid; the analytic baseline and solver stay at 1024.")
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    if args.reflection_average and args.reflection_encoder:
        parser.error("Choose either full-predictor or encoder reflection averaging")
    if args.batch_context_fft and not args.fuse_nested_fft:
        parser.error("--batch-context-fft requires --fuse-nested-fft")
    if args.graph_fft_mask and not args.graph_fft:
        parser.error("--graph-fft-mask requires --graph-fft")
    if args.spectral_saved_dno and args.batch_saved_dno:
        parser.error("--spectral-saved-dno cannot be combined with saved-DNO batching")
    if not 0.0 < args.ic_quantile_offset < 1.0:
        parser.error("ic-quantile-offset must be between 0 and 1")
    run_dir = args.run_dir.resolve()
    reference_dir = (args.reference_dir or args.reference_run or run_dir).resolve()
    output_dir = args.output_dir or run_dir / (
        f"eval_fast_{args.checkpoint}"
        + ("_matmulhighest" if args.matmul_precision == "highest" else "")
        + ("_centerxi" if args.center_xi else "")
        + (f"_q{args.ic_quantile_offset}" if args.ic_quantile_offset != 0.5 else "")
        + ("_momentum" if args.project_momentum else "")
        + ("_reflection" if args.reflection_average else "")
        + ("_reflectionencoder" if args.reflection_encoder else "")
        + ("_grid512" if args.correction_grid == 512 else "")
        + ("_exactg0" if args.exact_solver_g0 else "")
        + ("_nestedfused" if args.fuse_nested_fft else "")
        + ("_contextbatch" if args.batch_context_fft else "")
        + ("_contextfold" if args.fold_context else "")
        + ("_graphfft" if args.graph_fft else "")
        + ("_fftmask" if args.graph_fft_mask else "")
        + ("_unrolled" if args.unroll_stages else "")
        + ("_spectra" if args.materialize_spectra else "")
        + ("_statebarrier" if args.materialize_updated_state else "")
        + ("_rotationproducts" if args.cache_rotation_products else "")
        + ("_projections" if args.transpose_projections else "")
        + ("_fieldstride" if args.field_stride else "")
        + ("_gelureuse" if args.reuse_first_gelu else "")
        + ("_branchmajor" if args.branch_major else "")
        + ("_contextreal" if args.real_context_endpoints else "")
        + ("_contextadd" if args.fuse_context_add else "")
        + ("_cudacontext" if args.cuda_context else "")
        + ("_cudahead" if args.cuda_head else "")
        + {None: "", "pallas": "_fusedtail", "cuda": "_cudatail"}[args.fuse_tail]
        + {None: "", "rounded": "_cudag1", "exact": "_cudaexactg1"}[args.cuda_g1]
        + ("_cudainput" if args.cuda_input else "")
        + ("_bernoullifused" if args.bernoulli_fft == "cufftdx_fused" else "_bernoullidxboth" if args.bernoulli_fft == "cufftdx_both" else "_bernoullidx" if args.bernoulli_fft == "cufftdx_forward" else "")
        + ("_savedbatch" if args.batch_saved_dno else "")
        + ("_savedpacked" if args.pack_saved_inputs else "")
        + ("_savedspectral" if args.spectral_saved_dno else "")
        + (f"_{args.cs_terms_variant}" if args.cs_terms_variant else ""))
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    loaded = load_run(run_dir, checkpoint=args.checkpoint, checkpoint_dir=checkpoint_dir)
    dataset = Path(loaded.config["dataset"])
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    variant = ("fourier_fused_features_product_overlap"
               + ("_grid512" if args.correction_grid == 512 else "")
               + ("_exactg0" if args.exact_solver_g0 else "")
               + ("_nestedfused" if args.fuse_nested_fft else "")
               + ("_contextbatch" if args.batch_context_fft else "")
               + ("_contextfold" if args.fold_context else "")
               + ("_projectiontranspose" if args.transpose_projections else "")
               + ("_fieldstride" if args.field_stride else "")
               + ("_gelureuse" if args.reuse_first_gelu else "")
               + ("_branchmajor" if args.branch_major else "")
               + ("_contextreal" if args.real_context_endpoints else "")
               + ("_contextadd" if args.fuse_context_add else "")
               + ("_cudacontext" if args.cuda_context else "")
               + ("_cudahead" if args.cuda_head else "")
               + {None: "", "pallas": "_fusedtail", "cuda": "_cudatail"}[args.fuse_tail]
               + {None: "", "rounded": "_cudag1", "exact": "_cudaexactg1"}[args.cuda_g1]
               + ("_cudainput" if args.cuda_input else ""))
    terms_decoder = bool(loaded.config.get("cs_terms_residual", False))
    if args.cs_terms_variant and not terms_decoder:
        parser.error("--cs-terms-variant requires a CS-field decoder")
    if terms_decoder:
        if args.reflection_encoder:
            parser.error("--reflection-encoder requires a branch model, not a CS-field decoder")
        if args.reflection_average and not (args.cs_terms_variant or "").endswith("_spectral"):
            parser.error("--reflection-average requires a spectral predictor")
        if (args.fold_context or args.fuse_nested_fft or args.batch_context_fft or args.exact_solver_g0
                or args.correction_grid != 1024 or args.transpose_projections or args.field_stride
                or args.reuse_first_gelu or args.branch_major or args.real_context_endpoints or args.fuse_context_add):
            parser.error("CS-field decoder evaluation does not support branch-specific FFT variants")
        variant = args.cs_terms_variant or f"terms_m{loaded.config['cs_baseline_order']}_jax"
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation = (f"cuFFTDx FP64 CS fields, M{loaded.config['cs_baseline_order']}, pad2, ept8" if args.cs_terms_variant
                      else "native CS fields" if terms_decoder else "ept8")
    checkpoint_source = {
        "run_dir": str(run_dir), "selection": args.checkpoint,
        "path": str(checkpoint_dir or run_dir / ("best_val_ckpt" if args.checkpoint == "best_weight_ema"
                               else f"{args.checkpoint}_ckpt")), "epoch": loaded.epoch,
        "implementation": f"{variant}; {implementation}; spectral carry; local IF; packed boundaries",
        "matmul_precision": args.matmul_precision,
        "project_momentum": args.project_momentum,
        "center_xi": args.center_xi,
        "reflection_average": args.reflection_average,
        "reflection_encoder": args.reflection_encoder,
        "graph_fft": args.graph_fft,
        "graph_fft_mask": args.graph_fft_mask,
        "unroll_stages": args.unroll_stages,
        "materialize_spectra": args.materialize_spectra,
        "materialize_updated_state": args.materialize_updated_state,
        "cache_rotation_products": args.cache_rotation_products,
        "fuse_context_add": args.fuse_context_add,
        "bernoulli_fft": args.bernoulli_fft,
        "batch_saved_dno": args.batch_saved_dno,
        "pack_saved_inputs": args.pack_saved_inputs,
        "spectral_saved_dno": args.spectral_saved_dno,
    }
    summaries = {}
    for family in args.families:
        cfg = FAMILY_CONFIGS[family]
        source = (reference_dir if args.reference_dir else next(reference_dir.glob(
            f"eval_epoch1_weight_ema_test_stratified_n32_gpu*/{family}_trajs.npz")).parent)
        ics, _, nx, length = _load_paper_dataset_ics(
            dataset, family, 32, "stratified", ic_quantile_offset=args.ic_quantile_offset,
        )
        times = jnp.asarray(np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64))
        depths = jnp.asarray([ic.depth for ic in ics], dtype=jnp.float64)
        initial = ti.State(*(
            jnp.asarray(np.stack([getattr(ic, field) for ic in ics]), dtype=jnp.float64)
            for field in ("eta", "xi")
        ))
        params = ti.make_solver_params(
            nx, length, depths[:, None], dno_order=6,
            pad_factor=8, filter_fraction=cfg.filter_fraction,
        )
        jax.block_until_ready((initial, times, params))
        started = perf_counter()
        if terms_decoder and not variant.endswith("_spectral"):
            if args.spectral_saved_dno:
                parser.error("--spectral-saved-dno requires a spectral predictor")
            spectral = None
            if args.cs_terms_variant:
                physical = build_variant(loaded, jnp.log(depths), nx, variant, fft_ept=8)
                if args.center_xi:
                    physical = partial(lambda eta, xi, fn: fn(
                        eta, xi - jnp.mean(xi, axis=-1, keepdims=True),
                    ), fn=physical)
            else:
                physical = partial(lambda eta, xi, fn, depth: fn(eta, xi, depth),
                                   fn=build_predict_gxi_batched(loaded, center_xi=args.center_xi),
                                   depth=jnp.log(depths))
        else:
            spectral = build_variant(loaded, jnp.log(depths), nx, variant, fft_ept=8,
                                     solver_g0=params.g0[..., :nx // 2 + 1] if args.exact_solver_g0 else None,
                                     reflection_encoder=args.reflection_encoder)
            if args.center_xi:
                spectral = partial(lambda eta_hat, xi_hat, fn: fn(
                    eta_hat, xi_hat.at[..., 0].set(0),
                ), fn=spectral)
            if args.reflection_average:
                spectral = partial(
                    lambda eta_hat, xi_hat, fn: 0.5 * (
                        fn(eta_hat, xi_hat)
                        + jnp.conj(fn(jnp.conj(eta_hat), jnp.conj(xi_hat)))
                    ), fn=spectral,
                )
            physical = jax.jit(partial(physical_from_spectral, predict_hat=spectral)
                               if args.pack_saved_inputs else partial(lambda eta, xi, fn, n: jnp.fft.irfft(
                                   fn(jnp.fft.rfft(eta), jnp.fft.rfft(xi)), n=n,
                               ), fn=spectral, n=nx))
        function = partial(
            rollout_if_trial, params=params, substeps=cfg.substeps,
            carry_spectral=True, local_if=True, predict_gxi=physical, predict_hat=spectral,
            project_momentum=args.project_momentum, unroll_stages=args.unroll_stages,
            materialize_spectra=args.materialize_spectra,
            materialize_updated_state=args.materialize_updated_state,
            cache_rotation_products=args.cache_rotation_products,
            bernoulli_fft=args.bernoulli_fft, batch_saved_dno=args.batch_saved_dno,
            spectral_saved_dno=args.spectral_saved_dno,
        )
        with ExitStack() as patches:
            if args.graph_fft:
                transform = partial(graph_fft, endpoint_mask=args.graph_fft_mask)
                patches.enter_context(patch.object(jnp.fft, "rfft", transform))
                patches.enter_context(patch.object(jnp.fft, "irfft", partial(transform, inverse=True)))
            patches.enter_context(patch.object(ti, "_state_to_hat", packed_boundary))
            patches.enter_context(patch.object(ti, "_hat_to_state", partial(packed_boundary, inverse=True)))
            executable = jax.jit(function).lower(initial, times).compile()
        compiled = perf_counter()
        result = executable(initial, times)
        jax.block_until_ready(result)
        pred: RolloutPayload = {name: np.asarray(result[name]) for name in ("eta", "xi", "gxi")}
        pred["wall_s"] = perf_counter() - compiled
        print(f"[{family}] setup+compile={compiled-started:.3f}s; "
              f"batch32 execution+D2H={pred['wall_s']:.3f}s", flush=True)
        summaries[family] = run_family(
            family, cfg, loaded, partial(lambda eta, xi, _depth, fn: fn(eta, xi), fn=physical),
            output_dir, dataset, 32, source, checkpoint_source, 32,
            truth_batch_size=8, ic_selection="stratified", ic_quantile_offset=args.ic_quantile_offset, pred=pred,
            center_xi=args.center_xi,
        )
        del pred, result, executable, function, physical, spectral
        jax.clear_caches()
    _write_json(output_dir / "all_summaries.json", summaries)
    _write_json(output_dir / "macro_summary.json", compute_macro_summary(summaries))
    print(f"Done. Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
