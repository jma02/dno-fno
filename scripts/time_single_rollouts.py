"""Time one trajectory at a time after compilation and an untimed warm-up.

Use --steps 320 --repeats 3 for a short per-step benchmark; the default
times one full T=20/200 trajectory for each method and family.
The fused methods apply inference FFT optimizations to --candidate-run.
Spectral variants reduce adapter round trips; every classical order shares padded FFT batching.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, cast
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.benchmark_dno_fusion import build_variant  # noqa: E402
from scripts.benchmark_surrogate_rhs import rhs_nonlinear_if_spectral, shared_m1_hat  # noqa: E402
from solver.evals.eval_suite import FAMILY_CONFIGS  # noqa: E402
from solver.evals.model_rollout import Predictor, build_predict_gxi_batched, load_run, rollout_surrogate  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402

RUNS = {
    "full": ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918",
    "small": ROOT / "outputs/c27_branches16_tanaka_hard128_20260923",
}
FUSION_VARIANTS = {
    "fused": "all", "fused_g1": "batched", "fused_surface": "surface", "fused_front": "front",
    "fused_cufftdx": "cufftdx", "fused_joint": "cufftdx_joint", "fused_product": "cufftdx_product",
    "fused_both": "cufftdx_both", "fused_dense": "cufftdx_dense", "fused_head": "cufftdx_head",
    "fused_rank16": "cufftdx_lowrank16", "fused_rank32": "cufftdx_lowrank32",
    "fused_e4": "cufftdx_e4", "fused_e16": "cufftdx_e16",
    "fused_product_e4": "cufftdx_product_e4", "fused_product_e16": "cufftdx_product_e16",
}
FUSION_VARIANTS.update({f"{name}_spectral": variant for name, variant in tuple(FUSION_VARIANTS.items())})
FUSION_VARIANTS.update({f"fused_cufftdx_spectral_{packing}": "cufftdx" for packing in ("inputs", "all")})
FUSION_VARIANTS["baseline-only"] = "cufftdx"
FUSION_VARIANTS["shared_m1_spectral_all"] = "front"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument("--steps", type=int, default=0, help="0 uses each family's full saved trajectory.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ept", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--pad-factor", type=int, default=8, help="Padding for classical DNO and shared-M1 baseline; model-local baselines are unchanged.")
    parser.add_argument("--candidate-run", type=Path, help="Include another checkpoint as the compact model.")
    parser.add_argument("--methods", nargs="+", choices=("M1", "M2", "M3", "M4", "M5", "M6", "small", "full", "compact", *FUSION_VARIANTS))
    parser.add_argument("--reference", default="M6", help="Method used for numerical comparisons; must be included in --methods.")
    parser.add_argument("--reference-once", action="store_true", help="Run the numerical reference once while repeating the timing candidates.")
    parser.add_argument("--without-baselines", action="store_true", help="Speed-only diagnostic: omit model G0+G1, leaving the integrator unchanged.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_rollout_timing_20260923.json")
    args = parser.parse_args()
    if args.steps < 0 or args.repeats < 1 or args.pad_factor < 1:
        parser.error("steps must be nonnegative; repeats and pad-factor must be positive")
    if args.candidate_run:
        RUNS["compact"] = args.candidate_run.resolve()
    methods = args.methods or ["M6", "M1", "M2", "M3", "M4", "M5", *reversed(RUNS)]
    if args.reference not in methods or (set(methods) & {"compact", *FUSION_VARIANTS} and not args.candidate_run):
        parser.error("methods must include the reference; candidate methods require --candidate-run")
    if args.without_baselines and any(method.startswith("M") for method in methods):
        parser.error("--without-baselines applies only to neural methods")
    if args.without_baselines and set(methods) & {"baseline-only", "shared_m1_spectral_all"}:
        parser.error("baseline-only and shared_m1_spectral_all require analytic baselines")
    loaded = {name: load_run(run) for name, run in RUNS.items()
              if name in methods or (name == "compact" and set(methods) & FUSION_VARIANTS.keys())}
    predictors = {name: build_predict_gxi_batched(run) for name, run in loaded.items()}
    report: dict[str, Any] = {
        "device": jax.devices()[0].device_kind, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": 1, "nx": 1024, "internal_dt": 0.01, "gl2_iterations": 4,
        "fft_ept": args.ept, "classical_pad_factor": args.pad_factor,
        "precision": "FP64 classical/integration; FP32 learned inference",
        "timing": (f"synchronized GPU execution after {'same-executable' if args.steps else 'separate five-frame'} "
                   "warm-up; compilation and host transfers excluded"),
        "neural_runs": {name: str(RUNS["compact" if name in FUSION_VARIANTS else name])
                        for name in methods if not name.startswith("M")},
        "fusion_variants": {name: FUSION_VARIANTS[name] for name in methods if name in FUSION_VARIANTS},
        "reference": args.reference,
        "include_model_baselines": not args.without_baselines,
        "families": {},
    }
    for family in args.families:
        cfg = FAMILY_CONFIGS[family]
        if args.steps and (args.steps % cfg.substeps or args.steps * 0.01 > cfg.tmax):
            parser.error("steps must fit the saved trajectory and be divisible by the family's substeps")
        source = next(RUNS["full"].glob(f"eval_best_current_test_stratified_n32*/{family}_trajs.npz"))
        with np.load(source) as saved:
            frames = args.steps // cfg.substeps + 1 if args.steps else len(saved["times"])
            times = jnp.asarray(saved["times"][:frames])
            initial = ti.State(jnp.asarray(saved["truth_eta"][0, :1]), jnp.asarray(saved["truth_xi"][0, :1]))
            depth = jnp.asarray(saved["depths"][:1])
            simulation_id = int(saved["simulation_ids"][0])
            saved_reference = {key: saved[f"truth_{key}"][:frames, :1] for key in ("eta", "xi", "gxi")}
        records: dict[str, dict[str, Any]] = {}
        report["families"][family] = {
            "simulation_id": simulation_id, "steps": (frames - 1) * cfg.substeps,
            "horizon": float(times[-1]), "saved_frames": frames, "methods": records,
        }
        compiled, outputs = {}, {}
        for method in methods:
            started = time.perf_counter()
            shared_baseline = method == "shared_m1_spectral_all"
            correction = None
            params = ti.make_solver_params(
                initial.eta.shape[-1], 2 * np.pi, depth[:, None],
                dno_order=int(method[1:]) if method.startswith("M") else 1 if shared_baseline else 6,
                pad_factor=args.pad_factor, filter_fraction=cfg.filter_fraction,
            )
            if method.startswith("M"):
                function = partial(ti.rollout, params=params, save_gxi=True,
                                   substeps_per_interval=cfg.substeps, method="gl2_if",
                                   implicit_iterations=4, zero_mean_xi=True)
            else:
                predict = (cast(Predictor, build_variant(
                    loaded["compact" if method in FUSION_VARIANTS else method], jnp.log(depth), initial.eta.shape[-1],
                    FUSION_VARIANTS.get(method, "original"), include_baselines=not (args.without_baselines or shared_baseline),
                    include_correction=method != "baseline-only", fft_ept=args.ept,
                )) if method in FUSION_VARIANTS or args.without_baselines
                    else lambda eta, xi: predictors[method](eta, xi, jnp.log(depth)))
                if shared_baseline:
                    correction = predict
                    predict = partial(
                        lambda eta, xi, p, r: jnp.fft.irfft(shared_m1_hat(
                            jnp.fft.rfft(eta).at[..., p.nx // 2].set(0),
                            jnp.fft.rfft(xi).at[..., p.nx // 2].set(0), p, r,
                        ), n=p.nx), p=params, r=correction,
                    )
                function = partial(rollout_surrogate, params=params, substeps=cfg.substeps,
                                   predict_gxi=predict)
            runner = jax.jit(function)
            jax.block_until_ready((initial, times, params))
            setup_seconds = time.perf_counter() - started
            started = time.perf_counter()
            rhs_patch = nullcontext()
            if method.startswith("M"):
                rhs_patch = patch.object(ti, "gauss_legendre_2_if_step", partial(
                    ti.gauss_legendre_2_if_step, rhs=partial(
                        rhs_nonlinear_if_spectral, predict_gxi=None, pack_ffts="all")))
            elif "_spectral" in method or method == "baseline-only":
                packing = "all" if method == "baseline-only" else method.rsplit("_", 1)[-1]
                rhs_patch = patch("solver.evals.model_rollout._rhs_nonlinear_if_surrogate", partial(
                    rhs_nonlinear_if_spectral, pack_ffts=packing if packing in ("inputs", "all") else "none",
                    predict_correction=correction if shared_baseline else None))
            with rhs_patch:
                compiled[method] = runner.lower(initial, times).compile()
                compile_seconds = time.perf_counter() - started
                warm = compiled[method] if args.steps else runner.lower(initial, times[:5]).compile()
            warm_times = times if args.steps else times[:5]
            started = time.perf_counter()
            jax.block_until_ready(warm(initial, warm_times))
            records[method] = {"setup_s": setup_seconds, "compile_s": compile_seconds,
                               "warmup_s": time.perf_counter() - started, "seconds": []}
            print(f"{family} {method}: ready; batch=1, {report['families'][family]['steps']} steps", flush=True)
        for repeat in range(args.repeats):
            for method in list(compiled)[::1 if repeat % 2 == 0 else -1]:
                if args.reference_once and repeat > 0 and method == args.reference:
                    continue
                started = time.perf_counter()
                outputs[method] = jax.block_until_ready(compiled[method](initial, times))
                duration = time.perf_counter() - started
                records[method]["seconds"].append(duration)
                print(f"{family} {method}: repeat {repeat + 1}: {duration:.4f}s", flush=True)
        for method, result in outputs.items():
            eta = np.asarray(result["eta"])
            records[method]["finite"] = all(np.isfinite(np.asarray(result[key])).all().item() for key in ("eta", "xi", "gxi"))
            records[method]["minimum_depth"] = (float((eta + np.asarray(depth)[:, None]).min())
                                                 if records[method]["finite"] else None)
            if args.without_baselines or not records[method]["finite"]:
                continue
            for key in ("eta", "xi", "gxi"):
                actual = np.asarray(result[key])
                reference = np.asarray(outputs[args.reference][key])
                error = np.linalg.norm((actual - reference).reshape(frames, -1), axis=-1)
                scale = np.linalg.norm(reference.reshape(frames, -1), axis=-1)
                records[method][f"terminal_{key}_error"] = float(error[-1] / scale[-1])
                records[method][f"max_{key}_error"] = float((error / scale).max())
                records[method][f"trajectory_{key}_error"] = float(np.linalg.norm(error) / np.linalg.norm(scale))
                cached = saved_reference[key].reshape(frames, -1)
                cached_error = np.linalg.norm(actual.reshape(frames, -1) - cached, axis=-1)
                records[method][f"cached_reference_max_{key}_error"] = float((cached_error / np.linalg.norm(cached, axis=-1)).max())
                records[method][f"cached_reference_terminal_{key}_error"] = float(cached_error[-1] / np.linalg.norm(cached[-1]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{family}: saved {args.output}", flush=True)
        jax.clear_caches()
