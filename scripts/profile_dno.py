"""Profile batch-one DNO inference; CUDA capture excludes compilation and warm-up.

Run directly for clean latency/HLO estimates, or use --capture 20 under
nsys --capture-range=cudaProfilerApi (ncu: --profile-from-start off).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import suppress
import csv
import ctypes
from functools import partial
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
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

from scripts.benchmark_dno_fusion import build_variant  # noqa: E402
from solver.evals.model_rollout import build_predict_gxi_batched, load_run  # noqa: E402


def summarize_jax_trace(path: Path, names: set[str]) -> dict[str, Any]:
    """Summarize GPU events inside synchronized, named JAX workloads."""
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    processes = {e["pid"]: e.get("args", {}).get("name", "") for e in events if e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e.get("args", {}).get("name", "")
               for e in events if e.get("name") == "thread_name"}
    gpu_events = [e for e in events if e.get("ph") == "X" and "GPU" in processes.get(e["pid"], "")
                  and "Stream" in threads.get((e["pid"], e["tid"]), "")]
    scopes = [e for e in events if e.get("ph") == "X" and e.get("name") in names]
    summary: dict[str, Any] = {
        "gpu_events": len(gpu_events), "processes": processes, "threads": list(threads.values()), "scopes": {},
    }
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
        summary["scopes"][name] = {
            "host_scope_us": scope["dur"], "gpu_span_us": span, "gpu_busy_us": busy, "gpu_gap_us": span - busy,
            "categories_count_us": dict(categories), "kernels_count_us": dict(kernels),
            "cuda_api_count_us": dict(api), "kernel_names": kernel_names,
            "events_us_kernel": [[e["ts"] - start, e["dur"], kernel_names.index(e["name"])] for e in selected],
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/c27_w320_b4_h80_tanaka_hard128_20260924")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dno_profile_20260924")
    parser.add_argument("--depth-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--variant", choices=("original", "cufftdx", "fourier_round",
        "fourier_fused_features_product_overlap", "classical_m2", "classical_m6"), default="original")
    parser.add_argument("--stages", type=int, choices=(1, 2), default=1,
                        help="One forward or two vmap-paired GL2 stage forwards; trajectory batch stays one.")
    parser.add_argument("--capture", type=int, default=0, help="Capture this many warmed calls instead of benchmarking.")
    parser.add_argument("--trace", type=int, default=0, help="Trace this many native forwards after clean timing.")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--analyze", action="store_true", help="Plot measured rooflines and forward traces from the hardware/model result tree (no GPU work).")
    parser.add_argument("--analyze-current", action="store_true", help="Plot current Ada network/M2/M6 kernel timings (no GPU work).")
    parser.add_argument("--hardware-diagnosis", action="store_true", help="Compare dispatch, dependent execution, and 16 real GL2 steps with an annotated GPU trace.")
    args = parser.parse_args()
    if args.capture < 0 or args.trace < 0 or args.repeats < 1:
        parser.error("capture and trace must be nonnegative and repeats positive")
    if args.capture and args.trace:
        parser.error("capture and trace must run separately")
    if args.variant != "original" and args.depth_mode != "fixed":
        parser.error("optimized variants require fixed depth")
    if args.analyze_current:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = {"network": "Current network", "m2": "CS M2", "m6": "CS M6"}
        colors = {"FP64 cuFFTDx G1": "#c78450", "FP64 FFTs": "#bb5548", "FP32 FFTs": "#277da8",
                  "Dense GEMMs": "#589365", "Other kernels": "#909ba3", "Device copies": "#b9a336"}
        records: dict[str, Any] = {}
        for model in labels:
            for stages in (1, 2):
                directory = args.output / f"{model}_s{stages}"
                timing = json.loads((directory / "fixed.json").read_text())
                trace = json.loads((directory / "forward_trace.json").read_text())
                scope, calls = trace["scopes"]["forward"], trace["traced_calls"]
                kernels = sorted((dict(name=name, count_per_call=count / calls, us_per_call=duration / calls)
                                  for name, (count, duration) in scope["kernels_count_us"].items()),
                                 key=lambda kernel: kernel["us_per_call"], reverse=True)
                records[directory.name] = {
                    "clean_median_us": timing["median_us"], "clean_iqr_us": [timing["p25_us"], timing["p75_us"]],
                    "input_shape": timing["input_shape"], "variant": timing["variant"],
                    "reference_relative_error": timing["wrapper_relative_error"],
                    "traced_gpu_busy_us": scope["gpu_busy_us"] / calls,
                    "summed_kernel_us": sum(kernel["us_per_call"] for kernel in kernels),
                    "launches": sum(kernel["count_per_call"] for kernel in kernels), "kernels": kernels,
                    "categories": {name: dict(count_per_call=count / calls, us_per_call=duration / calls)
                                   for name, (count, duration) in scope["categories_count_us"].items()},
                }
        fig, (latency, breakdown) = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.28, wspace=0.45)
        positions = np.arange(len(labels))
        for stages, offset, color in ((1, -0.18, "#8ba0ac"), (2, 0.18, "#c45b22")):
            values = [records[f"{model}_s{stages}"]["clean_median_us"] for model in labels]
            errors = np.asarray([[value - records[f"{model}_s{stages}"]["clean_iqr_us"][0],
                                  records[f"{model}_s{stages}"]["clean_iqr_us"][1] - value]
                                 for model, value in zip(labels, values)]).T
            bars = latency.barh(positions + offset, values, height=0.32, color=color, xerr=errors,
                                error_kw={"elinewidth": 0.8, "capsize": 2},
                                label="One stage" if stages == 1 else "Two stages together")
            latency.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=9)
        latency.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.21), fontsize=9)
        latency.set(title="Synchronized forward latency", xlabel="Microseconds per call")
        latency.set_xlim(0, 1.22 * max(record["clean_median_us"] for record in records.values()))
        left = np.zeros(len(labels))
        for name, color in colors.items():
            values = np.asarray([records[f"{model}_s2"]["categories"].get(name, {}).get("us_per_call", 0)
                                 for model in labels])
            if np.any(values):
                breakdown.barh(positions, values, left=left, height=0.58, color=color,
                               label={"FP64 cuFFTDx G1": "G₁ kernel", "Dense GEMMs": "Matrix multiplies"}.get(name, name))
                left += values
        breakdown.set(title="Two-stage kernel breakdown", xlabel="Summed GPU kernel time (µs)")
        for index, model in enumerate(labels):
            breakdown.text(left[index] + 3, index, f"{int(records[f'{model}_s2']['launches'])} kernels", va="center", fontsize=9)
        breakdown.set_xlim(0, 1.30 * left.max())
        breakdown.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.21), ncol=2, fontsize=9)
        for ax in (latency, breakdown):
            ax.set(yticks=positions, yticklabels=list(labels.values()), ylim=(2.55, -0.55))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)
        fig.suptitle("Current Ada DNO profile", fontsize=20, y=0.97)
        fig.text(0.13, 0.88, "RTX 6000 Ada · N = 1024 · one trajectory · epoch 120", fontsize=11, color="#626b73")
        analysis = {
            "scope": "Native spectral DNO forwards, not complete rollouts; paired stages repeat one P128-filtered Stokes state.",
            "counters": "ERR_NVGPUCTRPERM: hardware counters unavailable; no roofline points inferred from XLA estimates.",
            "timing": "400 synchronized calls after 50 warmups; separate 20-call GPU traces. Summed kernel durations can overlap and are not wall time.",
            "records": records,
        }
        (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        for suffix in ("png", "pdf"):
            fig.savefig(ROOT / "paper-plots/figures" / f"14-current-ada-profile.{suffix}", dpi=180)
        print(f"Saved current Ada kernel profile to {args.output / 'analysis.json'}", flush=True)
        raise SystemExit(0)
    if args.analyze:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        models = dict(zip(("candidate_fourier", "candidate_previous", "small", "full"),
                          (("Fourier candidate", "o"), ("Previous candidate", "s"),
                           ("Neural — small", "^"), ("Neural — full", "D"))))
        hardware = {
            "a100": ("A100 80 GB", (19.5, 9.7, 156.0), 2039),
            "blackwell": ("RTX PRO 6000 Blackwell", (120.0, 120.0 / 64, 234.0), 1597),
            "ada": ("RTX 6000 Ada", (91.1, 91.1 / 64, 182.1), 960),
        }
        precision_colors = dict(zip(("Scalar FP32", "Scalar FP64", "TF32 tensor"),
                                    ("#277da8", "#bb5548", "#589365")))
        categories = dict(zip(("FP64 cuFFTDx G1", "FP64 FFTs", "FP32 FFTs", "Dense GEMMs", "Other kernels", "Device copies"),
                              ("#c78450", "#bb5548", "#277da8", "#589365", "#909ba3", "#b9a336")))
        analysis = {
            "scope": "Native batch-one forward passes; no time integration.",
            "roofline_definition": "Each precision independently aggregates kernels with positive work: sum(FLOPs)/sum(DRAM bytes), sum(FLOPs)/sum(kernel seconds). Mixed kernels may appear in multiple groups; groups are not additive.",
            "ceilings": "Published dense peak references, not measured sustained performance. Ada/Blackwell scalar FP64 is derived as FP32/64.",
            "work_definition": "Executed scalar add+multiply+2*FMA instructions; tensor counter already counts operations. Transcendentals and integer instructions are excluded.",
            "counter_collection": "Nsight Compute kernel replay; cache-control none; clock-control none. Traffic and duration are replay measurements, not exact unprofiled whole-pass traffic. Clean forward latency and JAX kernel traces are separate measurements.",
            "interpretation": "Distance below a roof alone does not identify a whole-pass memory bottleneck; launch overhead and uncounted transcendental work also matter.",
            "hardware": {},
        }
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"height_ratios": (3, 2)})
        fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.15, hspace=0.55, wspace=0.40)
        intensity = np.logspace(-2, 7, 400)
        measured_intensities = []
        for column, (gpu, (gpu_label, peaks, bandwidth)) in enumerate(hardware.items()):
            roof, breakdown = axes[:, column]
            gpu_report: dict[str, Any] = {"label": gpu_label, "peak_tflops": dict(zip(precision_colors, peaks)),
                                        "bandwidth_gbytes_s": bandwidth, "models": {}}
            analysis["hardware"][gpu] = gpu_report
            permission_denied = any("ERR_NVGPUCTRPERM" in path.read_text()
                                    for path in (args.output / gpu).glob("*/counter_capture.log"))
            points = 0
            for precision, peak in zip(precision_colors, peaks):
                roof.loglog(intensity, np.minimum(peak, bandwidth * intensity / 1000),
                            color=precision_colors[precision], linewidth=1.1, alpha=0.65)
            for index, (model, (label, marker)) in enumerate(models.items()):
                directory = args.output / gpu / model
                if not (directory / "fixed.json").exists():
                    gpu_report["models"][model] = {"status": "forward measurements unavailable"}
                    continue
                timing = json.loads((directory / "fixed.json").read_text())
                record: dict[str, Any] = {
                    "label": label, "device": timing["device"], "variant": timing["variant"],
                    "input_space": timing["input_space"], "output_space": timing["output_space"],
                    "clean_forward_median_us": timing["median_us"], "validation_relative_error": timing["wrapper_relative_error"],
                    "counters_status": "unavailable", "kernels": [], "precision_groups": {},
                }
                gpu_report["models"][model] = record
                trace_path = directory / "forward_trace.json"
                if trace_path.exists():
                    trace = json.loads(trace_path.read_text())
                    scope, calls = trace["scopes"]["forward"], trace["traced_calls"]
                    record["traced_gpu_busy_us"] = scope["gpu_busy_us"] / calls
                    record["traced_categories"] = {
                        name: {"count_per_call": count / calls, "us_per_call": duration / calls}
                        for name, (count, duration) in scope["categories_count_us"].items()
                    }
                    left = 0.0
                    for name, color in categories.items():
                        duration = record["traced_categories"].get(name, {}).get("us_per_call", 0.0)
                        breakdown.barh(index, duration, left=left, color=color, height=0.60)
                        left += duration
                csv_path = directory / "counters.csv"
                if not csv_path.exists():
                    record["counters_reason"] = ("GPU counter access denied (ERR_NVGPUCTRPERM); administrator permission required."
                                                 if permission_denied else "No hardware-counter CSV; no estimated traffic substituted.")
                    continue
                lines = csv_path.read_text().splitlines()
                header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
                if header is None:
                    record["counters_reason"] = "Counter collection did not produce a metric table."
                    continue
                launches: dict[str, Any] = {}
                rows = csv.DictReader(lines[header:])
                units = next(rows)
                metric_names = ("gpu__time_duration.sum", "lts__t_bytes.sum", "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
                                *(f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum"
                                  for op in ("fadd", "fmul", "ffma", "dadd", "dmul", "dfma")),
                                *(name for name in ("dram__bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum") if name in units))
                for row in rows:
                    launches[row["ID"]] = {"name": row["Kernel Name"], "metrics": {
                        name: {"value": float(row[name].replace(",", "")), "unit": units[name]}
                        for name in metric_names
                    }}
                for identifier, kernel in launches.items():
                    values = {name: item["value"] for name, item in kernel["metrics"].items()}
                    duration = kernel["metrics"]["gpu__time_duration.sum"]
                    seconds = duration["value"] * {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[duration["unit"]]
                    traffic = values["dram__bytes.sum"] if "dram__bytes.sum" in values else values["dram__bytes_read.sum"] + values["dram__bytes_write.sum"]
                    work = {}
                    for precision, prefix in (("Scalar FP32", "f"), ("Scalar FP64", "d")):
                        work[precision] = sum(factor * values[f"smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum"]
                                              for op, factor in (("add", 1), ("mul", 1), ("fma", 2)))
                    work["TF32 tensor"] = values["sm__ops_path_tensor_src_tf32_dst_fp32.sum"]
                    record["kernels"].append(dict(kernel, id=identifier, duration_s=seconds, dram_bytes=traffic,
                                                   l2_bytes=values["lts__t_bytes.sum"], flops=work))
                record["counters_status"] = "available" if launches else "empty"
                record["kernel_count"] = len(launches)
                for precision, color in precision_colors.items():
                    selected = [k for k in record["kernels"] if k["flops"][precision] > 0]
                    work = sum(k["flops"][precision] for k in selected)
                    traffic = sum(k["dram_bytes"] for k in selected)
                    seconds = sum(k["duration_s"] for k in selected)
                    group: dict[str, Any] = {"kernel_count": len(selected), "flops": work, "dram_bytes": traffic,
                                             "l2_bytes": sum(k["l2_bytes"] for k in selected), "duration_s": seconds}
                    record["precision_groups"][precision] = group
                    if not work or not traffic or not seconds:
                        group["omitted_reason"] = "No positive work, DRAM bytes, or kernel duration; intensity is undefined."
                        continue
                    group.update(intensity_flop_per_dram_byte=work / traffic, tflops=work / seconds / 1e12)
                    roof.scatter(work / traffic, work / seconds / 1e12, marker=marker, s=65,
                                 facecolors="none" if marker == "s" else color,
                                 edgecolors=color if marker == "s" else "white",
                                 linewidths=1.3 if marker == "s" else 0.5, zorder=3)
                    measured_intensities.append(work / traffic)
                    points += 1
            roof.set(title=gpu_label, xlabel="FLOP / DRAM byte", ylabel="TFLOP/s", xlim=(1e-2, 1e7), ylim=(1e-4, 400))
            roof.grid(alpha=0.16, which="major")
            if not points:
                roof.text(0.5, 0.35, "Hardware counters unavailable" + ("\nPermission denied" if permission_denied else ""), transform=roof.transAxes,
                          ha="center", va="center", fontsize=12, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
            breakdown.set(yticks=range(len(models)), yticklabels=[v[0] for v in models.values()],
                          xlabel="Traced GPU time per forward (µs)", ylim=(len(models) - 0.4, -0.6))
            breakdown.grid(axis="x", alpha=0.16)
            breakdown.set_axisbelow(True)
            for ax in (roof, breakdown):
                ax.spines[["top", "right"]].set_visible(False)
        left_limit = 10 ** np.floor(np.log10(min([0.1, *measured_intensities])))
        right_limit = 10 ** np.ceil(np.log10(max([1000, *measured_intensities])))
        for roof in axes[0]:
            roof.set_xlim(left_limit, right_limit)
        maximum_time = max(sum(c["us_per_call"] for c in model.get("traced_categories", {}).values())
                           for gpu in analysis["hardware"].values() for model in gpu["models"].values())
        for breakdown in axes[1]:
            breakdown.set_xlim(0, 1.05 * maximum_time)
        fig.suptitle("Model forward-pass rooflines", fontsize=23, y=0.98)
        fig.legend(handles=[Line2D([], [], color=color, label=name, linewidth=2) for name, color in precision_colors.items()],
                   loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
        fig.legend(handles=[Line2D([], [], color="#555555", marker=marker, linestyle="none", label=label,
                                   markerfacecolor="none" if marker == "s" else "#555555")
                            for label, marker in models.values()], loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4, frameon=False)
        fig.legend(handles=[Patch(color=color, label=name) for name, color in categories.items()],
                   loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False)
        (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        for suffix in ("png", "pdf"):
            fig.savefig(ROOT / "paper-plots/figures" / f"13-model-roofline.{suffix}", dpi=180)
        print(f"Saved measured forward analysis to {args.output / 'analysis.json'}", flush=True)
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
        summary = summarize_jax_trace(trace_path, set(compiled))
        (args.output / "hardware_trace.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {summary['gpu_events']} GPU events in {len(summary['scopes'])} annotated workloads", flush=True)
        raise SystemExit(0)
    predict = build_predict_gxi_batched(loaded)
    nx = eta.shape[-1]
    classical_order = int(args.variant[-1]) if args.variant.startswith("classical_") else None
    spectral_io = args.variant.startswith(("fourier", "classical_"))
    current = classical_order is not None or args.variant == "fourier_fused_features_product_overlap"
    # Rollouts hold depth fixed: its multiplier MLPs can be constant-folded.
    function = jax.jit((lambda eta, xi: predict(eta, xi, depth)) if args.depth_mode == "fixed" else predict)
    reference_function = jax.jit(lambda e, x: predict(e, x, depth))
    inputs = (eta, xi) if args.depth_mode == "fixed" else (eta, xi, depth)
    if classical_order is not None:
        from scripts.benchmark_classical import classical_hat
        from solver.solvers import time_integrator as ti

        params = ti.make_solver_params(nx, 2 * np.pi, jnp.exp(depth)[:, None],
                                      dno_order=classical_order, pad_factor=2, filter_fraction=0.25)
        classical = partial(classical_hat, k=params.k[:nx // 2 + 1],
                            g0=params.g0[..., :nx // 2 + 1], nx=nx, order=classical_order, precision="fp64")
        function = jax.jit(partial(classical,
            backend="specialized_cufftdx" if classical_order == 2 else "interleaved_cufftdx",
            pad_factor=1 if classical_order == 2 else 2, ept=16 if classical_order == 2 else 8))
        reference_function = jax.jit(partial(classical, backend="cufft", pad_factor=2))
    elif args.variant != "original":
        function = build_variant(loaded, depth, nx, args.variant, fft_ept=8)
        if args.variant == "fourier_round":
            reference_function = build_variant(loaded, depth, nx, "cufftdx")
    if spectral_io:
        inputs = tuple(jnp.fft.rfft(field) for field in (eta, xi))
        if current:
            # Match the P128 state support of the current steady rollout kernels.
            inputs = tuple(jnp.where(jnp.arange(nx // 2 + 1) <= nx // 8, field, 0) for field in inputs)
    reference_inputs = (inputs if classical_order is not None else
                        tuple(jnp.fft.irfft(field, n=nx) for field in inputs) if spectral_io else (eta, xi))
    if args.stages == 2:
        function = jax.jit(jax.vmap(function, in_axes=(0, 0) if args.depth_mode == "fixed" else (0, 0, None)))
        reference_function = jax.jit(jax.vmap(reference_function))
        inputs = (*tuple(jnp.stack((field, field)) for field in inputs[:2]), *inputs[2:])
        reference_inputs = tuple(jnp.stack((field, field)) for field in reference_inputs)
    label = args.depth_mode
    jax.block_until_ready(inputs)
    started = time.perf_counter()
    executable = function.lower(*inputs).compile()
    compile_s = time.perf_counter() - started
    for _ in range(50):
        jax.block_until_ready(executable(*inputs))
    actual = jax.block_until_ready(executable(*inputs))
    if spectral_io and classical_order is None:
        actual = jnp.fft.irfft(actual, n=nx)
    reference = reference_function(*reference_inputs).block_until_ready()
    relative_error = float(jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference))
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(actual)) or relative_error > (1e-9 if classical_order else 1e-5):
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
        print(f"Captured {args.capture} batch-one {label} calls, {args.stages} stage(s) each", flush=True)
    else:
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            jax.block_until_ready(executable(*inputs))
            durations.append(time.perf_counter() - started)
        report = {
            "run": None if classical_order else str(args.run), "checkpoint_epoch": None if classical_order else loaded.epoch,
            "parameters": 0 if classical_order else sum(leaf.size for leaf in jax.tree.leaves(loaded.params)),
            "device": jax.devices()[0].device_kind, "jax_version": jax.__version__,
            "batch_size": 1, "stages": args.stages, "nx": nx, "depth_mode": args.depth_mode,
            "variant": args.variant,
            "input_space": "unnormalized rFFT" if spectral_io else "physical",
            "output_space": "unnormalized rFFT" if spectral_io else "physical",
            "input_shape": list(inputs[0].shape), "input_filter_cutoff": nx // 8 if current else None,
            "workload": f"{args.stages} DNO stage forward(s), one trajectory; no time integrator",
            "probe": "Stokes initial state, repeated for paired-stage shape",
            "reference": f"FP64 cuFFT M{classical_order}, padding2" if classical_order else "physical model forward",
            "compile_s": compile_s, "seconds": durations,
            "median_us": float(np.median(durations) * 1e6),
            "p25_us": float(np.percentile(durations, 25) * 1e6),
            "p75_us": float(np.percentile(durations, 75) * 1e6),
            "wrapper_relative_error": relative_error,
            "xla_cost_estimate": executable.cost_analysis(),
            "memory_analysis": str(executable.memory_analysis()),
            "caveat": "XLA estimates omit native FFI kernel work/traffic and are not a measured roofline; FFT and transcendental accounting is incomplete.",
        }
        (args.output / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / f"{label}.hlo").write_text(executable.as_text())
        print(json.dumps({key: value for key, value in report.items() if key != "seconds"}, indent=2), flush=True)
        if args.trace:
            with jax.profiler.trace(str(args.output / "trace")):
                with jax.profiler.TraceAnnotation("forward"):
                    for _ in range(args.trace):
                        jax.block_until_ready(executable(*inputs))
            summary = summarize_jax_trace(next((args.output / "trace").rglob("*.trace.json.gz")), {"forward"})
            summary.update(variant=args.variant, traced_calls=args.trace)
            (args.output / "forward_trace.json").write_text(json.dumps(summary, indent=2) + "\n")
