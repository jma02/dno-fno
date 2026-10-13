"""Evaluate weight-EMA accuracy with the timed Fourier-I/O implementation.

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
from solver.evals.model_rollout import load_run  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from scripts.benchmark_dno_fusion import build_variant  # noqa: E402
from scripts.benchmark_surrogate_rhs import rollout_if_trial  # noqa: E402
from scripts.time_single_rollouts import packed_boundary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("weight_ema", "weight_ema_secondary"), default="weight_ema")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-run", type=Path, help="Run with saved epoch-one M6 reference trajectories")
    reference.add_argument("--reference-dir", type=Path, help="Directory containing cached M6 references for each family")
    parser.add_argument("--ic-quantile-offset", type=float, default=0.5, help="Within-group stratified case quantile offset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--project-momentum", action="store_true", help="Experimental post-step momentum correction.")
    parser.add_argument("--exact-solver-g0", action="store_true", help="Use the solver's FP64 linear symbol and original potential spectrum.")
    parser.add_argument("--fuse-nested-fft", action="store_true", help="Fuse the final nested correction FFT pair.")
    parser.add_argument("--batch-context-fft", action="store_true", help="Batch context and nested first FFTs; requires --fuse-nested-fft.")
    parser.add_argument("--correction-grid", type=int, choices=(512, 1024), default=1024,
                        help="Learned correction grid; the analytic baseline and solver stay at 1024.")
    args = parser.parse_args()
    if args.batch_context_fft and not args.fuse_nested_fft:
        parser.error("--batch-context-fft requires --fuse-nested-fft")
    if not 0.0 < args.ic_quantile_offset < 1.0:
        parser.error("ic-quantile-offset must be between 0 and 1")
    run_dir = args.run_dir.resolve()
    reference_dir = (args.reference_dir or args.reference_run or run_dir).resolve()
    output_dir = args.output_dir or run_dir / (
        f"eval_fast_{args.checkpoint}"
        + (f"_q{args.ic_quantile_offset}" if args.ic_quantile_offset != 0.5 else "")
        + ("_momentum" if args.project_momentum else "")
        + ("_grid512" if args.correction_grid == 512 else "")
        + ("_exactg0" if args.exact_solver_g0 else "")
        + ("_nestedfused" if args.fuse_nested_fft else "")
        + ("_contextbatch" if args.batch_context_fft else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    dataset = Path(loaded.config["dataset"])
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    variant = ("fourier_fused_features_product_overlap"
               + ("_grid512" if args.correction_grid == 512 else "")
               + ("_exactg0" if args.exact_solver_g0 else "")
               + ("_nestedfused" if args.fuse_nested_fft else "")
               + ("_contextbatch" if args.batch_context_fft else ""))
    checkpoint_source = {
        "run_dir": str(run_dir), "selection": args.checkpoint,
        "path": str(run_dir / f"{args.checkpoint}_ckpt"), "epoch": loaded.epoch,
        "implementation": f"{variant}; ept8; spectral carry; local IF; packed boundaries",
        "project_momentum": args.project_momentum,
    }
    summaries = {}
    for family, cfg in FAMILY_CONFIGS.items():
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
        spectral = build_variant(loaded, jnp.log(depths), nx, variant, fft_ept=8,
                                 solver_g0=params.g0[..., :nx // 2 + 1] if args.exact_solver_g0 else None)
        physical = jax.jit(partial(lambda eta, xi, fn, n: jnp.fft.irfft(
            fn(jnp.fft.rfft(eta), jnp.fft.rfft(xi)), n=n,
        ), fn=spectral, n=nx))
        function = partial(
            rollout_if_trial, params=params, substeps=cfg.substeps,
            carry_spectral=True, local_if=True, predict_gxi=physical, predict_hat=spectral,
            project_momentum=args.project_momentum,
        )
        with ExitStack() as patches:
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
        )
        del pred, result, executable, function, physical, spectral
        jax.clear_caches()
    _write_json(output_dir / "all_summaries.json", summaries)
    _write_json(output_dir / "macro_summary.json", compute_macro_summary(summaries))
    print(f"Done. Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
