"""Time trajectories after compilation and an untimed warm-up.

Use --steps 320 --repeats 3 for a short per-step benchmark; the default
times one full T=20/200 trajectory for each method and family.
--batch-size 32 times the complete saved cohort together, not one trajectory.
The fused methods apply inference FFT optimizations to --candidate-run.
Spectral variants reduce adapter round trips; every classical order shares padded FFT batching.
The _stages suffix pairs GL2 stages within one trajectory, not separate rollouts.
The _momentum suffix restores initial momentum after each full spectral step.
Overlap trials require XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, cast
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.benchmark_dno_fusion import G1_FP32_TRIALS, TUNED_TRIALS, build_variant  # noqa: E402
from scripts.benchmark_classical import classical_hat  # noqa: E402
from scripts.benchmark_surrogate_rhs import rhs_nonlinear_if_spectral, rollout_if_trial, shared_m1_hat  # noqa: E402
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
FUSION_VARIANTS["fused_front_spectral_all"] = "front"
FUSION_VARIANTS["fourier_spectral_all"] = "fourier"
FUSION_VARIANTS["fourier_round_spectral_all"] = "fourier_round"
FUSION_VARIANTS["fourier_reuse_spectral_all"] = "fourier_reuse"
FUSION_VARIANTS["fourier_input_spectral_all"] = "fourier_input"
FUSION_VARIANTS.update({f"{variant}_spectral_all": variant for variant in (
    "fourier_g1fft", "fourier_fields", "fourier_product", "fourier_kernels", "fourier_kernels_e4", "fourier_kernels_e16",
    "fourier_transforms", "fourier_transforms_e4", "fourier_transforms_e16",
    "fourier_fused_fields", "fourier_fused_inputs", "fourier_fused_both", "fourier_fused_g1",
    "fourier_fused_fields_overlap", "fourier_fused_g1_overlap",
    "fourier_fused_features", "fourier_fused_features_overlap",
    "fourier_balanced_features_overlap", "fourier_fused_features_product_overlap",
    "fourier_fused_features_product_overlap_nestedfused",
    "fourier_fused_features_product_overlap_nestedfused_contextbatch",
    "fourier_fused_features_product_overlap_exactg0",
    "fourier_fused_features_product_overlap_grid512",
    "fourier_balanced_features_product_overlap",
    *TUNED_TRIALS, *G1_FP32_TRIALS,
)})
FUSION_VARIANTS["baseline-only"] = "cufftdx"
FUSION_VARIANTS["shared_m1_spectral_all"] = "front"
FUSION_VARIANTS["terms_interleaved"] = "terms_interleaved"
FUSION_VARIANTS["terms_interleaved_features"] = "terms_interleaved_features"
FUSION_VARIANTS["terms_interleaved_features_cufftdx"] = "terms_interleaved_features_cufftdx"
FUSION_VARIANTS["terms_interleaved_features_cufftdx_spectral"] = "terms_interleaved_features_cufftdx_spectral"
METHODS = ("M1", "M2", "M3", "M4", "M5", "M6", "small", "full", "compact", *FUSION_VARIANTS)


def packed_boundary(
    state: ti.State | ti.SpectralState, nx: int = 0, *, inverse: bool = False,
) -> ti.State | ti.SpectralState:
    fields = jnp.stack(state)
    return ti.State(*ti.myifft(fields)) if inverse else ti.SpectralState(*ti.myfft(fields, nx))


def method_base(name: str) -> str:
    for suffix in ("_momentum", "_half", "_local", "_carry", "_stages", "_boundaries"):
        name = name.removesuffix(suffix)
    return name


def check_exclusive_gpu(enabled: bool) -> None:
    if enabled:
        pids = subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid", "--format=csv,noheader",
        ], text=True, timeout=10).split()
        foreign = [pid for pid in pids if int(pid) != os.getpid()]
        if foreign:
            print(f"GPU busy with other processes {foreign}; deferring timing", flush=True)
            raise SystemExit(75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument("--steps", type=int, default=0, help="0 uses each family's full saved trajectory.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, choices=range(1, 33), default=1)
    parser.add_argument("--require-idle-gpu", action="store_true", help="Exit 75 if another process uses the selected GPU before or during timings.")
    parser.add_argument("--ept", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--pad-factor", type=int, default=8, help="Padding for classical DNO and shared-M1 baseline; model-local baselines are unchanged.")
    parser.add_argument("--classical-backend", choices=("cufft", "interleaved_cufftdx"), default="cufft")
    parser.add_argument("--candidate-run", type=Path, help="Include another checkpoint as the compact model.")
    parser.add_argument("--checkpoint", choices=("best", "latest", "final", "weight_ema", "weight_ema_secondary"), default="best")
    parser.add_argument("--methods", nargs="+", choices=tuple(
        f"{method}{boundary}{stages}" for method in METHODS
        for boundary in ("", "_boundaries") for stages in ("", "_stages")) + tuple(
        f"{method}_boundaries_stages{trial}" for method in METHODS if method.startswith(("M", "fourier_", "terms_")) or method == "compact"
        for trial in ("_carry", "_local", "_carry_local", "_carry_local_half", "_carry_momentum", "_carry_local_momentum")))
    parser.add_argument("--reference", default="M6", help="Method in --methods, or cached_M6 to use the saved reference trajectory.")
    parser.add_argument("--reference-once", action="store_true", help="Run the numerical reference once while repeating the timing candidates.")
    parser.add_argument("--without-baselines", action="store_true", help="Speed-only diagnostic: omit model G0+G1, leaving the integrator unchanged.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_rollout_timing_20260923.json")
    args = parser.parse_args()
    if args.steps < 0 or args.repeats < 1 or args.pad_factor < 1:
        parser.error("steps must be nonnegative; repeats and pad-factor must be positive")
    if args.require_idle_gpu and (not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
                                 or "," in os.environ["CUDA_VISIBLE_DEVICES"]):
        parser.error("--require-idle-gpu requires one GPU UUID in CUDA_VISIBLE_DEVICES")
    check_exclusive_gpu(args.require_idle_gpu)
    if args.candidate_run:
        RUNS["compact"] = args.candidate_run.resolve()
    methods = args.methods or ["M6", "M1", "M2", "M3", "M4", "M5", *reversed(RUNS)]
    if args.classical_backend != "cufft" and any(
        name.startswith("M") and "carry" not in name and "local" not in name for name in methods
    ):
        parser.error("optimized classical timing requires a carry or local method")
    base_methods = set(map(method_base, methods))
    if (args.reference != "cached_M6" and args.reference not in methods) or (base_methods & {"compact", *FUSION_VARIANTS} and not args.candidate_run):
        parser.error("methods must include the reference unless cached_M6; candidate methods require --candidate-run")
    if args.without_baselines and any(method.startswith("M") for method in methods):
        parser.error("--without-baselines applies only to neural methods")
    if args.without_baselines and (base_methods & {"baseline-only", "shared_m1_spectral_all"}
                                   or any(method.startswith("fourier_") for method in methods)):
        parser.error("baseline-only, shared_m1_spectral_all and Fourier-I/O methods require analytic baselines")
    loaded = {name: load_run(run, checkpoint=args.checkpoint) for name, run in RUNS.items()
              if name in base_methods or (name == "compact" and base_methods & FUSION_VARIANTS.keys())}
    predictors = {name: build_predict_gxi_batched(run) for name, run in loaded.items()}
    report: dict[str, Any] = {
        "device": jax.devices()[0].device_kind, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.batch_size, "nx": 1024, "internal_dt": 0.01, "gl2_iterations": 4,
        "fft_ept": args.ept, "classical_pad_factor": args.pad_factor,
        "classical_backend": args.classical_backend,
        "precision": "FP64 classical/integration; FP32 learned inference",
        "checkpoint": args.checkpoint, "neural_epochs": {name: run.epoch for name, run in loaded.items()},
        "timing": "synchronized GPU execution after same-executable warm-up; compilation and host transfers excluded",
        "neural_runs": {name: str(RUNS["compact" if base in FUSION_VARIANTS else base])
                        for name in methods if not name.startswith("M")
                        for base in (method_base(name),)},
        "fusion_variants": {name: FUSION_VARIANTS[base] for name in methods
                            for base in (method_base(name),) if base in FUSION_VARIANTS},
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
            if len(saved["simulation_ids"]) < args.batch_size:
                parser.error(f"{family} has fewer than {args.batch_size} saved cases")
            frames = args.steps // cfg.substeps + 1 if args.steps else len(saved["times"])
            times = jnp.asarray(saved["times"][:frames])
            initial = ti.State(jnp.asarray(saved["truth_eta"][0, :args.batch_size]),
                               jnp.asarray(saved["truth_xi"][0, :args.batch_size]))
            depth = jnp.asarray(saved["depths"][:args.batch_size])
            simulation_ids = saved["simulation_ids"][:args.batch_size].tolist()
            saved_reference = {key: saved[f"truth_{key}"][:frames, :args.batch_size]
                               for key in ("eta", "xi", "gxi")}
        records: dict[str, dict[str, Any]] = {}
        report["families"][family] = {
            "simulation_ids": simulation_ids, "steps": (frames - 1) * cfg.substeps,
            **({"simulation_id": simulation_ids[0]} if args.batch_size == 1 else {}),
            "horizon": float(times[-1]), "saved_frames": frames, "methods": records,
        }
        compiled, outputs = {}, {}
        for label in methods:
            check_exclusive_gpu(args.require_idle_gpu)
            method = method_base(label)
            options = label[len(method):].split("_")
            started = time.perf_counter()
            shared_baseline = method == "shared_m1_spectral_all"
            correction = None
            spectral_predict = None
            params = ti.make_solver_params(
                initial.eta.shape[-1], 2 * np.pi, depth[:, None],
                dno_order=int(method[1:]) if method.startswith("M") else 1 if shared_baseline else 6,
                pad_factor=args.pad_factor, filter_fraction=cfg.filter_fraction,
            )
            if method.startswith("M"):
                if args.classical_backend == "interleaved_cufftdx":
                    spectral_predict = partial(
                        classical_hat, k=params.k[:params.nx // 2 + 1],
                        g0=params.g0[..., :params.nx // 2 + 1], nx=params.nx,
                        order=params.dno_order, pad_factor=args.pad_factor,
                        backend="interleaved_cufftdx", precision="fp64", ept=args.ept,
                    )
                function = (partial(rollout_if_trial, params=params, substeps=cfg.substeps,
                                    carry_spectral="carry" in options, local_if="local" in options,
                                    half_spectrum="half" in options, project_momentum="momentum" in options,
                                    predict_hat=spectral_predict)
                            if "carry" in options or "local" in options else partial(ti.rollout, params=params, save_gxi=True,
                                   substeps_per_interval=cfg.substeps, method="gl2_if",
                                   implicit_iterations=4, zero_mean_xi=True))
            else:
                predict = (cast(Predictor, build_variant(
                    loaded["compact" if method in FUSION_VARIANTS else method], jnp.log(depth), initial.eta.shape[-1],
                    FUSION_VARIANTS.get(method, "original"), include_baselines=not (args.without_baselines or shared_baseline),
                    include_correction=method != "baseline-only", fft_ept=args.ept,
                    **({"solver_g0": params.g0[..., :params.nx // 2 + 1]} if "_exactg0" in method else {}),
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
                if method.startswith("fourier_") or method == "terms_interleaved_features_cufftdx_spectral":
                    spectral_predict = predict
                    predict = partial(lambda eta, xi, fn: jnp.fft.irfft(
                        fn(jnp.fft.rfft(eta), jnp.fft.rfft(xi)), n=eta.shape[-1]), fn=spectral_predict)
                function = (partial(rollout_if_trial, params=params, substeps=cfg.substeps,
                                    carry_spectral="carry" in options, local_if="local" in options,
                                    predict_gxi=predict, predict_hat=spectral_predict, half_spectrum="half" in options,
                                    project_momentum="momentum" in options)
                            if "carry" in options or "local" in options else partial(
                                rollout_surrogate, params=params, substeps=cfg.substeps, predict_gxi=predict))
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
                    predict_correction=correction if shared_baseline else None, predict_hat=spectral_predict))
            with ExitStack() as patches:
                patches.enter_context(rhs_patch)
                if "boundaries" in options:
                    patches.enter_context(patch.object(ti, "_state_to_hat", packed_boundary))
                    patches.enter_context(patch.object(ti, "_hat_to_state", partial(packed_boundary, inverse=True)))
                if "stages" in options:
                    def paired_stage_map(
                        stage1: ti.SpectralState, stage2: ti.SpectralState, *, v0: ti.SpectralState,
                        stage_time1: float | jax.Array, stage_time2: float | jax.Array,
                        dt: float | jax.Array, params: ti.SolverParams,
                        a11: float | jax.Array, a12: float | jax.Array,
                        a21: float | jax.Array, a22: float | jax.Array,
                        rhs: Callable[..., ti.SpectralState] = ti.rhs_nonlinear_if,
                    ) -> tuple[ti.SpectralState, ti.SpectralState, ti.SpectralState, ti.SpectralState]:
                        fields = jax.vmap(rhs, in_axes=(0, 0, None))(
                            jax.tree.map(lambda first, second: jnp.stack((first, second)), stage1, stage2),
                            jnp.stack((stage_time1, stage_time2)), params,
                        )
                        f1 = ti.SpectralState(*(field[0] for field in fields))
                        f2 = ti.SpectralState(*(field[1] for field in fields))
                        candidate1 = ti._tree_add(v0, ti._tree_scale(
                            ti._tree_add(ti._tree_scale(f1, a11), ti._tree_scale(f2, a12)), dt))
                        candidate2 = ti._tree_add(v0, ti._tree_scale(
                            ti._tree_add(ti._tree_scale(f1, a21), ti._tree_scale(f2, a22)), dt))
                        return candidate1, candidate2, f1, f2

                    patches.enter_context(patch.object(ti, "_gauss_legendre_2_stage_map", paired_stage_map))
                compiled[label] = runner.lower(initial, times).compile()
                compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            jax.block_until_ready(compiled[label](initial, times))
            warmup_seconds = time.perf_counter() - started
            check_exclusive_gpu(args.require_idle_gpu)
            records[label] = {"setup_s": setup_seconds, "compile_s": compile_seconds,
                               "warmup_s": warmup_seconds, "seconds": []}
            print(f"{family} {label}: ready; batch={args.batch_size}, {report['families'][family]['steps']} steps", flush=True)
        for repeat in range(args.repeats):
            for method in list(compiled)[::1 if repeat % 2 == 0 else -1]:
                if args.reference_once and repeat > 0 and method == args.reference:
                    continue
                check_exclusive_gpu(args.require_idle_gpu)
                started = time.perf_counter()
                outputs[method] = jax.block_until_ready(compiled[method](initial, times))
                duration = time.perf_counter() - started
                check_exclusive_gpu(args.require_idle_gpu)
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
                reference = (saved_reference[key] if args.reference == "cached_M6"
                             else np.asarray(outputs[args.reference][key]))
                error = np.linalg.norm((actual - reference).reshape(frames, -1), axis=-1)
                scale = np.linalg.norm(reference.reshape(frames, -1), axis=-1)
                records[method][f"terminal_{key}_error"] = float(error[-1] / scale[-1])
                records[method][f"max_{key}_error"] = float((error / scale).max())
                records[method][f"trajectory_{key}_error"] = float(np.linalg.norm(error) / np.linalg.norm(scale))
                if key == "eta" and args.batch_size > 1:
                    records[method]["terminal_eta_errors"] = (
                        np.linalg.norm(actual[-1] - reference[-1], axis=-1)
                        / np.linalg.norm(reference[-1], axis=-1)
                    ).tolist()
                cached = saved_reference[key].reshape(frames, -1)
                cached_error = np.linalg.norm(actual.reshape(frames, -1) - cached, axis=-1)
                records[method][f"cached_reference_max_{key}_error"] = float((cached_error / np.linalg.norm(cached, axis=-1)).max())
                records[method][f"cached_reference_terminal_{key}_error"] = float(cached_error[-1] / np.linalg.norm(cached[-1]))
                for suffix, metric in (("_boundaries", "boundary"), ("_stages", "stage"),
                                       ("_carry_local", "carry_local"), ("_carry", "carry"), ("_local", "local"),
                                       ("_half", "half")):
                    if method.endswith(suffix) and method.removesuffix(suffix) in outputs:
                        unpacked = np.asarray(outputs[method.removesuffix(suffix)][key]).reshape(frames, -1)
                        records[method][f"{metric}_max_{key}_difference"] = float((
                            np.linalg.norm(actual.reshape(frames, -1) - unpacked, axis=-1)
                            / np.linalg.norm(unpacked, axis=-1)
                        ).max())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{family}: saved {args.output}", flush=True)
        jax.clear_caches()
