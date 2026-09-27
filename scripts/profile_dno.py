"""Profile batch-one DNO inference; CUDA capture excludes compilation and warm-up.

Run directly for clean latency/HLO estimates, or use --capture 20 under
nsys --capture-range=cudaProfilerApi (ncu: --profile-from-start off).
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from contextlib import suppress
import ctypes
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
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
    parser.add_argument("--hardware-diagnosis", action="store_true", help="Compare dispatch, dependent execution, and 16 real GL2 steps with an annotated GPU trace.")
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
    if args.hardware_diagnosis:
        from scripts.benchmark_dno_fusion import build_variant
        from solver.evals.model_rollout import rollout_surrogate
        from solver.solvers import time_integrator as ti

        hardware_forward = build_variant(loaded, depth, eta.shape[-1], "cufftdx")

        @jax.jit
        def dependent(e: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            def body(_: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                q = hardware_forward(*state)
                return e + 1e-5 * q, x + 1e-5 * q
            return jax.lax.fori_loop(0, 64, body, (e, x))

        params = ti.make_solver_params(1024, 2 * np.pi, jnp.exp(depth)[:, None],
                                      dno_order=2, pad_factor=8, filter_fraction=0.25)
        times = jnp.arange(3, dtype=jnp.float64) * 0.08
        functions = {
            "elementwise_single": (jax.jit(lambda e, x: e * 1.000001 + x * 1e-7), 1, "call"),
            "elementwise_loop64": (jax.jit(lambda e, x: jax.lax.fori_loop(
                0, 64, lambda _, s: s * 1.000001 + x * 1e-7, e)), 64, "iteration"),
            "forward_single": (hardware_forward, 1, "forward"),
            "forward_loop64": (dependent, 64, "forward"),
            "neural_gl2_16_steps": (jax.jit(lambda e, x: rollout_surrogate(
                ti.State(e, x), times, params, lambda a, b: hardware_forward(a, b), substeps=8)), 16, "step"),
            "classical_M2_gl2_16_steps": (jax.jit(lambda e, x: ti.rollout(
                ti.State(e, x), times, params, save_gxi=True, substeps_per_interval=8,
                method="gl2_if", implicit_iterations=4, zero_mean_xi=True)), 16, "step"),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        records, compiled = {}, {}
        for name, (function, units, unit) in functions.items():
            started = time.perf_counter()
            compiled[name] = function.lower(eta, xi).compile()
            records[name] = {"compile_s": time.perf_counter() - started, "units": units,
                             "unit": unit, "seconds": []}
            for _ in range(5):
                jax.block_until_ready(compiled[name](eta, xi))
            print(f"{name}: compiled and warmed", flush=True)
            (args.output / f"{name}.hlo").write_text(compiled[name].as_text())
        expected = build_predict_gxi_batched(loaded)(eta, xi, depth)
        actual = compiled["forward_single"](eta, xi)
        relative_error = float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))
        if relative_error > 1e-5 or not bool(jnp.isfinite(actual).all()):
            raise RuntimeError(f"Forward numerical difference {relative_error}")
        fields = "timestamp,index,name,pstate,clocks.current.sm,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,utilization.gpu,utilization.memory,temperature.gpu"
        monitor = subprocess.Popen(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                                    f"--query-gpu={fields}", "--format=csv,noheader,nounits", "--loop-ms=100"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for repeat in range(args.repeats):
                names = tuple(compiled)[::1 if repeat % 2 == 0 else -1]
                for name in names:
                    if not name.endswith("single") and repeat >= max(3, args.repeats // 20):
                        continue
                    started = time.perf_counter()
                    result = jax.block_until_ready(compiled[name](eta, xi))
                    records[name]["seconds"].append(time.perf_counter() - started)
                    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result)):
                        raise RuntimeError(f"Nonfinite {name} output")
            with jax.profiler.trace(str(args.output / "trace")):
                for name, executable in compiled.items():
                    calls = 20 if name.endswith("single") else 1
                    records[name]["traced_calls"] = calls
                    with jax.profiler.TraceAnnotation(name):
                        for _ in range(calls):
                            jax.block_until_ready(executable(eta, xi))
        finally:
            monitor.terminate()
            telemetry, telemetry_error = monitor.communicate(timeout=10)
        for name, record in records.items():
            record["median_us"] = float(np.median(record["seconds"]) * 1e6)
            record["median_us_per_unit"] = record["median_us"] / record["units"]
            print(f"{name}: {record['median_us_per_unit']:.3f} us/{record['unit']}", flush=True)
        packages = {}
        for package in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "numpy", "triton",
                        "nvidia-cuda-runtime-cu12", "nvidia-cufft-cu12", "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12",
                        "jax-cuda13-plugin", "jax-cuda13-pjrt", "nvidia-cufft", "nvidia-nvjitlink",
                        "nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cublas", "nvidia-cuda-cupti"):
            packages[package] = "not installed"
            with suppress(importlib.metadata.PackageNotFoundError):
                packages[package] = importlib.metadata.version(package)
        report = {"device": jax.devices()[0].device_kind, "run": str(args.run), "checkpoint_epoch": loaded.epoch,
                  "batch_size": 1, "nx": 1024, "short_steps": 16, "dt": 0.01, "gl2_iterations": 4,
                  "forward_relative_error": relative_error, "packages": packages, "timings": records,
                  "loaded_cuda_libraries": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                                   if "/libcu" in line}),
                  "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
                  "environment": {key: os.environ.get(key) for key in (
                      "XLA_FLAGS", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NVIDIA_TF32_OVERRIDE", "LD_LIBRARY_PATH")},
                  "gpu_telemetry_fields": fields.split(","),
                  "gpu_telemetry": [line.split(", ") for line in telemetry.splitlines()],
                  "gpu_telemetry_error": telemetry_error,
                  "scope": "Standalone and synthetic dependent forwards plus 16 actual GL2 steps; not full-rollout timings."}
        (args.output / "hardware.json").write_text(json.dumps(report, indent=2) + "\n")
        trace_path = next((args.output / "trace").rglob("*.trace.json.gz"))
        with gzip.open(trace_path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        processes = {e["pid"]: e.get("args", {}).get("name", "") for e in events if e.get("name") == "process_name"}
        threads = {(e["pid"], e["tid"]): e.get("args", {}).get("name", "")
                   for e in events if e.get("name") == "thread_name"}
        gpu_events = [e for e in events if e.get("ph") == "X" and "GPU" in processes.get(e["pid"], "")
                      and "Stream" in threads.get((e["pid"], e["tid"]), "")]
        scopes = [e for e in events if e.get("ph") == "X" and e.get("name") in compiled]
        summary = {"gpu_events": len(gpu_events), "processes": processes, "threads": list(threads.values()), "scopes": {}}
        for scope in scopes:
            name, start, end = scope["name"], scope["ts"], scope["ts"] + scope["dur"]
            selected = sorted((e for e in gpu_events if start <= e["ts"] < end), key=lambda e: e["ts"])
            categories, kernels = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
            busy, previous_end = 0.0, start
            for event in selected:
                kernel, duration = event["name"], event["dur"]
                kind = "Other kernels"
                if "sandwich_kernel" in kernel:
                    kind = "FP64 cuFFTDx G1"
                elif "fft" in kernel.lower() or "scal_kernel" in kernel:
                    kind = "FP64 FFTs" if "double" in kernel else "FP32 FFTs"
                elif "gemm" in kernel or "matmul" in kernel:
                    kind = "Dense GEMMs"
                elif "Memcpy" in kernel:
                    kind = "Device copies"
                categories[kind][0] += 1
                categories[kind][1] += duration
                kernels[kernel][0] += 1
                kernels[kernel][1] += duration
                busy += max(0.0, event["ts"] + duration - max(event["ts"], previous_end))
                previous_end = max(previous_end, event["ts"] + duration)
            span = previous_end - selected[0]["ts"] if selected else 0.0
            kernel_names = list(kernels)
            api = defaultdict(lambda: [0, 0.0])
            for event in events:
                if event.get("ph") == "X" and start <= event["ts"] < end and event["name"].startswith(("cu", "cuda")):
                    api[event["name"]][0] += 1
                    api[event["name"]][1] += event["dur"]
            summary["scopes"][name] = {"host_scope_us": scope["dur"], "gpu_span_us": span,
                                        "gpu_busy_us": busy, "gpu_gap_us": span - busy,
                                        "categories_count_us": dict(categories), "kernels_count_us": dict(kernels),
                                        "cuda_api_count_us": dict(api), "kernel_names": kernel_names,
                                        "events_us_kernel": [[e["ts"] - start, e["dur"], kernel_names.index(e["name"])] for e in selected]}
        (args.output / "hardware_trace.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {len(gpu_events)} GPU events in {len(scopes)} annotated workloads", flush=True)
        raise SystemExit(0)
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
