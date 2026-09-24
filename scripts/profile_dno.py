"""Profile batch-one DNO inference; CUDA capture excludes compilation and warm-up.

Run directly for clean latency/HLO estimates, or use --capture 20 under
nsys --capture-range=cudaProfilerApi (ncu: --profile-from-start off).
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import ctypes
import json
import os
from pathlib import Path
import sqlite3
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

from solver.evals.model_rollout import build_predict_gxi_batched, load_run  # noqa: E402


def summarize_trace(path: Path) -> dict[str, Any]:
    """Summarize single-stream GPU events, assigning calls by CUDA launch/NVTX correlation."""
    with sqlite3.connect(path) as connection:
        calls = connection.execute("""
            SELECT n.start,n.end,n.globalTid FROM NVTX_EVENTS n
            JOIN StringIds s ON n.textId=s.id WHERE s.value LIKE 'XlaModule:%'
            ORDER BY n.start
        """).fetchall()
        events = connection.execute("""
            SELECT k.start,k.end,s.value,r.start,r.globalTid,k.gridX*k.gridY*k.gridZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
            UNION ALL
            SELECT k.start,k.end,'Device copy',r.start,r.globalTid,0
            FROM CUPTI_ACTIVITY_KIND_MEMCPY k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
            ON k.correlationId=r.correlationId ORDER BY 1
        """).fetchall()
        gpu = connection.execute("""
            SELECT name,smCount,l2CacheSize,memoryBandwidth FROM TARGET_INFO_GPU WHERE id=0
        """).fetchone()
        streams = connection.execute("SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if streams != 1:
        raise ValueError("This diagnostic assumes one compute stream; summed times would double-count overlap.")
    starts = [call[0] for call in calls]
    by_call: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    categories: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, float("inf"), 0.0])
    for start, end, name, launch, thread, blocks in events:
        index = bisect_right(starts, launch) - 1
        if index < 0 or launch > calls[index][1] or thread != calls[index][2]:
            raise ValueError("GPU event has no enclosing profiled call")
        category = "Other kernels"
        if "fft" in name or "scal_kernel" in name:
            category = "FP64 FFTs + scaling" if "double" in name else "FP32 FFTs + scaling"
        elif "gemm" in name:
            category = "Dense matrix multiplies"
        elif name == "Device copy":
            category = "Device copies"
        by_call[index].append((start, end, category))
        categories[category][0] += 1
        categories[category][1] += (end - start) / 1000
        kernels[name][0] += 1
        kernels[name][1] += (end - start) / 1000
        kernels[name][2] = min(kernels[name][2], blocks)
        kernels[name][3] = max(kernels[name][3], blocks)
    spans, busy = [], []
    for intervals in by_call.values():
        spans.append((intervals[-1][1] - intervals[0][0]) / 1000)
        covered, previous_end = 0, intervals[0][0]
        for start, end, _ in intervals:
            covered += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        busy.append(covered / 1000)
    representative = int(np.argsort(spans)[len(spans) // 2])
    origin = by_call[representative][0][0]
    return {
        "calls": len(calls), "gpu": dict(zip(("name", "sms", "l2_bytes", "bandwidth_bytes_s"), gpu)),
        "trace_span_median_us": float(np.median(spans)),
        "trace_busy_median_us": float(np.median(busy)),
        "trace_gap_median_us": float(np.median(np.asarray(spans) - busy)),
        "categories": {name: {"count_per_call": count / len(calls), "us_per_call": duration / len(calls)}
                       for name, (count, duration) in categories.items()},
        "kernels": {name: {"count_per_call": count / len(calls), "us_per_call": duration / len(calls),
                            "mean_us": duration / count, "min_blocks": low, "max_blocks": high}
                    for name, (count, duration, low, high) in kernels.items()},
        "timeline": [[(start - origin) / 1000, (end - start) / 1000, category]
                     for start, end, category in by_call[representative]],
        "caveat": "Kernel/copy times are traced GPU execution; trace gaps include profiler overhead, not just application overhead.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/c27_w320_b4_h80_tanaka_hard128_20260924")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dno_profile_20260924")
    parser.add_argument("--depth-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--capture", type=int, default=0, help="Capture this many warmed calls instead of benchmarking.")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--analyze", action="store_true", help="Summarize existing SQLite traces and draw an estimated roofline (no GPU work).")
    args = parser.parse_args()
    if args.capture < 0 or args.repeats < 1:
        parser.error("capture must be nonnegative and repeats positive")
    if args.analyze:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        forward = summarize_trace(args.output / "forward.sqlite")
        comparison = summarize_trace(args.output / "branches16/forward.sqlite")
        benchmarks = [json.loads(path.read_text()) for path in
                      (args.output / "fixed.json", args.output / "branches16/fixed.json")]
        analysis = {"forward": forward, "branches16_forward": comparison}
        (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        colors = dict(zip(("FP32 FFTs + scaling", "FP64 FFTs + scaling", "Other kernels",
                           "Dense matrix multiplies", "Device copies"),
                          ("#277da8", "#bb5548", "#909ba3", "#589365", "#b98b36")))
        fig = plt.figure(figsize=(12, 7), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(3, 1))
        breakdown = fig.add_subplot(grid[0, 0])
        roofline = fig.add_subplot(grid[0, 1])
        timeline = fig.add_subplot(grid[1, :])
        groups = sorted(forward["categories"], key=lambda key: forward["categories"][key]["us_per_call"], reverse=True)
        values = [forward["categories"][key]["us_per_call"] for key in groups]
        breakdown.barh(groups, values, color=[colors[key] for key in groups])
        breakdown.invert_yaxis()
        breakdown.set(xlabel="GPU execution per forward pass (µs)", xlim=(0, max(values) * 1.35),
                      title="Measured: FFT work dominates GPU time")
        for index, value in enumerate(values):
            breakdown.text(value + 1, index, f"{value:.1f} µs", va="center", fontsize=10)
        intensity = np.logspace(-1, 3, 300)
        bandwidth = forward["gpu"]["bandwidth_bytes_s"]
        # NVIDIA's published CUDA-core FP32 peak; not a mixed-precision/tensor-core roof.
        roofline.loglog(intensity, np.minimum(91.1e12, bandwidth * intensity) / 1e12,
                       color="#444444", label="FP32 / DRAM reference ceiling")
        for benchmark, color, offset in zip(benchmarks, ("#277da8", "#589365"), ((8, -16), (8, 8))):
            cost = benchmark["xla_cost_estimate"]
            x = cost["flops"] / cost["bytes accessed"]
            y = cost["flops"] / benchmark["median_us"] / 1e6
            roofline.scatter(x, y, color=color, s=65, zorder=3)
            roofline.annotate(f"{benchmark['parameters']:,} parameters\n{benchmark['median_us']:.0f} µs",
                             (x, y), xytext=offset, textcoords="offset points", fontsize=9)
        roofline.set(xlabel="Estimated intensity (XLA FLOP / compiler-counted byte)",
                     ylabel="Estimated work / measured time (TFLOP/s)", ylim=(0.05, 200),
                     title="Estimated roofline — hardware counters unavailable")
        roofline.legend(loc="upper left", fontsize=8, frameon=False)
        roofline.grid(alpha=0.2, which="both")
        for start, duration, category in forward["timeline"]:
            timeline.broken_barh([(start, duration)], (0, 1), facecolors=colors[category])
        timeline.set(ylim=(0, 1), yticks=[], xlabel="Time from first GPU operation (µs)",
                     title="One profiled forward pass: white gaps have no captured GPU work")
        timeline.set_xlim(0, forward["timeline"][-1][0] + forward["timeline"][-1][1])
        fig.suptitle("DNO forward pass · one sample · 1,024 grid points · RTX 6000 Ada", fontsize=15)
        fig.supxlabel("Roofline uses static estimates, not measured DRAM traffic. Mixed precision; FP32 ceiling is a reference only.\n"
                      "Kernel traces include profiler overhead; clean forward latency is measured separately.", fontsize=9)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"roofline.{suffix}", dpi=180)
        print(json.dumps({key: {field: value for field, value in result.items() if field not in ("kernels", "timeline")}
                          for key, result in analysis.items()}, indent=2))
        raise SystemExit(0)
    loaded = load_run(args.run)
    source = next((ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918").glob(
        "eval_best_current_test_stratified_n32*/stokes_trajs.npz"
    ))
    with np.load(source) as saved:
        eta = jnp.asarray(saved["truth_eta"][0, :1], dtype=jnp.float64)
        xi = jnp.asarray(saved["truth_xi"][0, :1], dtype=jnp.float64)
        depth = jnp.log(jnp.asarray(saved["depths"][:1], dtype=jnp.float64))
    predict = build_predict_gxi_batched(loaded)
    # Rollouts hold depth fixed: its multiplier MLPs can be constant-folded.
    function = jax.jit((lambda eta, xi: predict(eta, xi, depth)) if args.depth_mode == "fixed" else predict)
    inputs = (eta, xi) if args.depth_mode == "fixed" else (eta, xi, depth)
    label = args.depth_mode
    jax.block_until_ready(inputs)
    started = time.perf_counter()
    executable = function.lower(*inputs).compile()
    compile_s = time.perf_counter() - started
    for _ in range(50):
        jax.block_until_ready(executable(*inputs))
    actual = jax.block_until_ready(executable(*inputs))
    reference = predict(eta, xi, depth).block_until_ready()
    relative_error = float(jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference))
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(actual)) or relative_error > 1e-5:
        raise RuntimeError(f"Profile wrapper changed the prediction: relative error {relative_error}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.capture:
        (args.output / f"{label}_capture.hlo").write_text(executable.as_text())
        driver = ctypes.CDLL("libcuda.so.1")
        if driver.cuProfilerStart() != 0:
            raise RuntimeError("CUDA profiler start failed")
        for _ in range(args.capture):
            jax.block_until_ready(executable(*inputs))
        if driver.cuProfilerStop() != 0:
            raise RuntimeError("CUDA profiler stop failed")
        print(f"Captured {args.capture} batch-one {label} calls", flush=True)
    else:
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            jax.block_until_ready(executable(*inputs))
            durations.append(time.perf_counter() - started)
        report = {
            "run": str(args.run), "checkpoint_epoch": loaded.epoch,
            "parameters": sum(leaf.size for leaf in jax.tree.leaves(loaded.params)),
            "device": jax.devices()[0].device_kind, "jax_version": jax.__version__,
            "batch_size": 1, "nx": eta.shape[-1], "depth_mode": args.depth_mode,
            "workload": "one DNO forward pass; no time integrator",
            "compile_s": compile_s, "seconds": durations,
            "median_us": float(np.median(durations) * 1e6),
            "p25_us": float(np.percentile(durations, 25) * 1e6),
            "p75_us": float(np.percentile(durations, 75) * 1e6),
            "wrapper_relative_error": relative_error,
            "xla_cost_estimate": executable.cost_analysis(),
            "memory_analysis": str(executable.memory_analysis()),
            "caveat": "XLA operation/traffic estimates are not measured instructions or DRAM traffic; transcendental and FFT accounting may differ from hardware.",
        }
        (args.output / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / f"{label}.hlo").write_text(executable.as_text())
        print(json.dumps({key: value for key, value in report.items() if key != "seconds"}, indent=2), flush=True)
