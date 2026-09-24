"""Time one trajectory at a time after compilation and an untimed warm-up.

Use --steps 320 --repeats 3 for a short per-step benchmark; the default
times one full T=20/200 trajectory for each method and family.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.evals.eval_suite import FAMILY_CONFIGS  # noqa: E402
from solver.evals.model_rollout import build_predict_gxi_batched, load_run, rollout_surrogate  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402

RUNS = {
    "full": ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918",
    "small": ROOT / "outputs/c27_branches16_tanaka_hard128_20260923",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument("--steps", type=int, default=0, help="0 uses each family's full saved trajectory.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--candidate-run", type=Path, help="Include another checkpoint as the compact model.")
    parser.add_argument("--methods", nargs="+", choices=("M1", "M2", "M3", "M4", "M5", "M6", "small", "full", "compact"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_rollout_timing_20260923.json")
    args = parser.parse_args()
    if args.steps < 0 or args.repeats < 1:
        parser.error("steps must be nonnegative and repeats positive")
    if args.candidate_run:
        RUNS["compact"] = args.candidate_run.resolve()
    methods = args.methods or ["M6", "M1", "M2", "M3", "M4", "M5", *reversed(RUNS)]
    if "M6" not in methods or ("compact" in methods and not args.candidate_run):
        parser.error("methods must include M6 as reference; compact requires --candidate-run")
    predictors = {name: build_predict_gxi_batched(load_run(run)) for name, run in RUNS.items() if name in methods}
    report: dict[str, Any] = {
        "device": jax.devices()[0].device_kind, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": 1, "nx": 1024, "internal_dt": 0.01, "gl2_iterations": 4,
        "precision": "FP64 classical/integration; FP32 learned inference",
        "timing": "synchronized GPU execution; compilation, warm-up and host transfers excluded",
        "neural_runs": {name: str(run) for name, run in RUNS.items() if name in methods},
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
        records: dict[str, dict[str, Any]] = {}
        report["families"][family] = {
            "simulation_id": simulation_id, "steps": (frames - 1) * cfg.substeps,
            "horizon": float(times[-1]), "saved_frames": frames, "methods": records,
        }
        compiled, outputs = {}, {}
        for method in methods:
            params = ti.make_solver_params(
                initial.eta.shape[-1], 2 * np.pi, depth[:, None],
                dno_order=int(method[1:]) if method.startswith("M") else 6,
                pad_factor=8, filter_fraction=cfg.filter_fraction,
            )
            if method.startswith("M"):
                function = partial(ti.rollout, params=params, save_gxi=True,
                                   substeps_per_interval=cfg.substeps, method="gl2_if",
                                   implicit_iterations=4, zero_mean_xi=True)
            else:
                function = partial(rollout_surrogate, params=params, substeps=cfg.substeps,
                                   predict_gxi=lambda eta, xi: predictors[method](eta, xi, jnp.log(depth)))
            runner = jax.jit(function)
            jax.block_until_ready((initial, times, params))
            started = time.perf_counter()
            compiled[method] = runner.lower(initial, times).compile()
            compile_seconds = time.perf_counter() - started
            warm = compiled[method] if args.steps else runner.lower(initial, times[:5]).compile()
            warm_times = times if args.steps else times[:5]
            started = time.perf_counter()
            jax.block_until_ready(warm(initial, warm_times))
            records[method] = {"compile_s": compile_seconds, "warmup_s": time.perf_counter() - started, "seconds": []}
            print(f"{family} {method}: ready; batch=1, {report['families'][family]['steps']} steps", flush=True)
        for repeat in range(args.repeats):
            for method in list(compiled)[::1 if repeat % 2 == 0 else -1]:
                started = time.perf_counter()
                outputs[method] = jax.block_until_ready(compiled[method](initial, times))
                duration = time.perf_counter() - started
                records[method]["seconds"].append(duration)
                print(f"{family} {method}: repeat {repeat + 1}: {duration:.4f}s", flush=True)
        reference = np.asarray(outputs["M6"]["eta"][-1])
        for method, result in outputs.items():
            eta = np.asarray(result["eta"])
            records[method]["finite"] = all(np.isfinite(np.asarray(result[key])).all().item() for key in ("eta", "xi", "gxi"))
            records[method]["minimum_depth"] = float((eta + np.asarray(depth)[:, None]).min())
            records[method]["terminal_eta_error"] = float(np.linalg.norm(eta[-1] - reference) / np.linalg.norm(reference))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{family}: saved {args.output}", flush=True)
        jax.clear_caches()
