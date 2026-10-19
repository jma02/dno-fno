"""Trace one trained-hero DNO call with GPU-only or threaded CPU G0+G1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gzip
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts import benchmark_dno_fusion as fusion  # noqa: E402
from scripts.benchmark_cufft_library import graph_fft  # noqa: E402
from solver.evals.model_rollout import load_run  # noqa: E402

RUN = ROOT / "outputs/c27_nested_b16_ema_low_lr_ep60_20261014"
VARIANT = ("fourier_fused_features_product_overlap_nestedfused_contextbatch_"
           "projectiontranspose_fieldstride_gelureuse_branchmajor_contextreal")


def probe_builder(part: str) -> Callable:
    """Split the existing forward at its shared preparation, without copying its layers."""
    source = inspect.getsource(fusion.build_variant).replace("def build_variant(", "def build_probe(", 1)
    if part == "prepare":
        source = source.replace("            if overlap_g1:\n",
            "            return normalized_inputs, normalized_hat, normalized_hat[:, 1]\n"
            "            if overlap_g1:\n", 1)
    else:
        start = source.index("        @jax.jit\n        def forward_hat(")
        middle = source.index("            phase_hat =", start)
        end = source.index("            if not overlap_g1:", middle)
        source = source[:start] + """        @jax.jit
        def forward_hat(normalized_inputs: jax.Array, normalized_hat: jax.Array) -> jax.Array:
            eta_norm, xi_phys = jnp.moveaxis(normalized_inputs, 1, 0)
            eta_norm_hat, xi_phys_hat = jnp.moveaxis(normalized_hat, 1, 0)
""" + source[middle:end] + "            return correction_hat\n\n        return forward_hat\n"
    namespace = vars(fusion).copy()
    exec(compile(source, "<offload_" + part + ">", "exec"), namespace)
    return namespace["build_probe"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hero_speed_20261019/cpu_offload_profile")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        jax.config.update("jax_enable_x64", True)
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        loaded = load_run(RUN, checkpoint="best_weight_ema", checkpoint_dir=RUN / "rollout_best_epoch37_ckpt")
        model = loaded.model
        with np.load(RUN / "eval_epoch37_weight_ema/tanaka_trajs.npz") as saved:
            eta, xi = map(jnp.asarray, (saved["truth_eta"][0, :1], saved["truth_xi"][0, :1]))
            depth = jnp.log(jnp.asarray(saved["depths"][:1]))
        nx = eta.shape[-1]
        inputs = jax.block_until_ready((jnp.fft.rfft(eta), jnp.fft.rfft(xi)))
        prepare_builder, correction_builder = map(probe_builder, ("prepare", "correction"))
        with patch.object(jnp.fft, "rfft", graph_fft), patch.object(jnp.fft, "irfft", partial(graph_fft, inverse=True)):
            native = fusion.build_variant(loaded, depth, nx, VARIANT).lower(*inputs).compile()
            prepare = prepare_builder(loaded, depth, nx, VARIANT).lower(*inputs).compile()
            prepared = jax.block_until_ready(prepare(*inputs))
            correction = correction_builder(loaded, depth, nx, VARIANT).lower(*prepared[:2]).compile()
        k32 = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float32)
        log_depth = depth.astype(jnp.float32).reshape(-1, 1)
        h = jnp.exp(log_depth if model.baseline_true_depth else jnp.minimum(log_depth, jnp.log(model.h_clip_max)))
        k64 = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=jnp.float64)
        g0 = jax.device_put(np.asarray(k32[None] * jnp.tanh(h * k32[None])), cpu)
        operators = jax.device_put(np.asarray(jnp.stack((k64[None] * jnp.tanh(h.astype(jnp.float64) * k64[None]),
            jnp.broadcast_to(1j * k64, (1, len(k64)))), axis=1)), cpu)

        @jax.jit
        def cpu_baseline(values: jax.Array, xi_hat: jax.Array) -> jax.Array:
            eta64 = (values[:, 0] * model.eta_scale).astype(jnp.float64)
            xi64_hat = jnp.fft.rfft(values[:, 1].astype(jnp.float64))
            fields = jnp.fft.irfft(operators * xi64_hat[:, None], n=nx)
            g1 = -jnp.sum(operators * jnp.fft.rfft(eta64[:, None] * fields), axis=1)
            return g0 * xi_hat + g1

        cpu_values = tuple(jax.device_put(value, cpu) for value in (prepared[0], prepared[2]))
        cpu_executable = cpu_baseline.lower(*cpu_values).compile()
        output_scale = float(loaded.stats["target_absmax"]) / model.target_scale
        join = jax.jit(lambda base, residual: (output_scale * (base + residual)).at[..., 0].set(0)
            .at[..., -1].set((output_scale * (base + residual))[..., -1].real))
        joined = join.lower(jax.device_put(cpu_executable(*cpu_values), gpu), correction(*prepared[:2])).compile()

        def worker(values: jax.Array, xi_hat: jax.Array) -> jax.Array:
            with jax.profiler.TraceAnnotation("offload/d2h_dispatch"):
                host_values, host_xi = jax.device_put(values, cpu), jax.device_put(xi_hat, cpu)
            with jax.profiler.TraceAnnotation("offload/cpu_compute_wait"):
                base = cpu_executable(host_values, host_xi)
                base.block_until_ready()
            with jax.profiler.TraceAnnotation("offload/h2d_dispatch"):
                return jax.device_put(base, gpu)

        with ThreadPoolExecutor(max_workers=1) as pool:
            def evaluate(mode: str) -> jax.Array:
                if mode == "all_gpu":
                    with jax.profiler.TraceAnnotation("offload/native_dispatch"):
                        result = native(*inputs)
                else:
                    with jax.profiler.TraceAnnotation("offload/prepare_dispatch"):
                        values, spectra, xi_hat = prepare(*inputs)
                    with jax.profiler.TraceAnnotation("offload/thread_dispatch"):
                        future = pool.submit(worker, values, xi_hat)
                    with jax.profiler.TraceAnnotation("offload/correction_dispatch"):
                        learned = correction(values, spectra)
                    with jax.profiler.TraceAnnotation("offload/worker_wait"):
                        base = future.result()
                    with jax.profiler.TraceAnnotation("offload/join_dispatch"):
                        result = joined(base, learned)
                with jax.profiler.TraceAnnotation("offload/final_wait"):
                    return result.block_until_ready()

            expected, actual = map(lambda mode: np.asarray(evaluate(mode)), ("all_gpu", "cpu_overlap"))
            relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            correction_norm = float(np.linalg.norm(np.asarray(correction(*prepared[:2]))))
            if not np.isfinite(actual).all() or relative > 1e-9 or correction_norm == 0:
                raise AssertionError(f"Offload changed the trained forward: {relative=}, {correction_norm=}")
            for _ in range(5):
                for mode in ("all_gpu", "cpu_overlap"):
                    evaluate(mode)
            clean_us = {}
            for mode in ("all_gpu", "cpu_overlap"):
                start = perf_counter()
                evaluate(mode)
                clean_us[mode] = (perf_counter() - start) * 1e6
            with jax.profiler.trace(str(args.output / "trace")):
                for mode in ("all_gpu", "cpu_overlap"):
                    evaluate(mode)
                    with jax.profiler.TraceAnnotation("offload/" + mode):
                        evaluate(mode)
        report = {
            "device": gpu.device_kind, "batch": 1, "nx": nx, "variant": VARIANT,
            "checkpoint": str(RUN / "rollout_best_epoch37_ckpt"), "relative_l2": relative,
            "trained_correction_norm": correction_norm, "clean_us": clean_us,
            "scope": "One synchronized DNO call, GPU-resident Fourier inputs; not a full rollout.",
            "timing": "Five warmups, one clean timing per path; separate instrumented trace. No internal synchronization added.",
        }
        (args.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    report = json.loads((args.output / "timing.json").read_text())
    with gzip.open(next((args.output / "trace").rglob("*.trace.json.gz")), "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu_pids = {event["pid"] for event in events if event.get("name") == "process_name"
                and "GPU" in event.get("args", {}).get("name", "")}
    scopes = {event["name"].removeprefix("offload/"): event for event in events
              if event.get("ph") == "X" and event.get("name") in ("offload/all_gpu", "offload/cpu_overlap")}
    timeline_limit = max(scope["dur"] for scope in scopes.values()) * 1.02
    copy_names = {"MemcpyH2D", "MemcpyD2H", "MemcpyD2D"}
    labels = {"all_gpu": "All GPU", "cpu_overlap": "GPU + CPU G₀+G₁"}
    colors = {"dispatch": "#4c78a8", "wait": "#b8c0cc", "transfer": "#dd8452",
              "cpu": "#8c6bb1", "gpu": "#17867b", "copy": "#d54f57"}
    phases = {
        "native_dispatch": ("dispatch", "GPU call"), "prepare_dispatch": ("dispatch", "Prepare"),
        "thread_dispatch": ("dispatch", "Submit"), "correction_dispatch": ("dispatch", "Correction"),
        "worker_wait": ("wait", "Waiting for CPU worker"), "join_dispatch": ("dispatch", "Join"),
        "final_wait": ("wait", "Final wait"), "d2h_dispatch": ("transfer", "D→H dispatch"),
        "cpu_compute_wait": ("cpu", "CPU + wait"), "h2d_dispatch": ("transfer", "H→D dispatch"),
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.6), gridspec_kw={"height_ratios": (1, 1.7, 1.7)})
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.85, hspace=0.72)
    fig.suptitle("One DNO evaluation: GPU vs CPU offload", x=0.18, ha="left", fontsize=19, y=0.98)
    fig.text(0.18, 0.94, "RTX 6000 Ada  ·  Hero epoch 37 EMA  ·  1 state × 1024 points", color="#58616d")
    clean = report["clean_us"]
    axes[0].barh((1, 0), (clean["all_gpu"], clean["cpu_overlap"]), height=0.48,
                 color=(colors["gpu"], colors["transfer"]))
    for row, mode in ((1, "all_gpu"), (0, "cpu_overlap")):
        axes[0].text(clean[mode] + 8, row, f"{clean[mode]:.1f} µs", va="center")
    axes[0].set(yticks=(1, 0), yticklabels=tuple(labels.values()), xlim=(0, max(clean.values()) * 1.22),
                xlabel="Synchronized latency (µs)", ylim=(-0.55, 1.55))
    axes[0].set_title("Unprofiled measurement", loc="left", fontsize=12, pad=10)
    axes[0].text(1, 1.12, f"CPU offload: {clean['cpu_overlap'] / clean['all_gpu']:.2f}× longer",
                 transform=axes[0].transAxes, ha="right", color="#58616d")
    analysis = {"clean_us": clean, "profiled_calls": {}}
    for ax, mode in zip(axes[1:], labels, strict=True):
        scope = scopes[mode]
        start, duration = scope["ts"], scope["dur"]
        selected = [event for event in events if event.get("ph") == "X" and "dur" in event
                    and start <= event["ts"] < start + duration]
        gpu_events = [event for event in selected if event["pid"] in gpu_pids]
        main_stream = next(event["tid"] for event in gpu_events if "field_inverse_kernel" in event["name"])
        host_events = [event for event in selected if event["name"].removeprefix("offload/") in phases
                       and event["name"].startswith("offload/")]
        rows = {"host": 3, "worker": 2, "gpu": 1, "copy": 0} if mode == "cpu_overlap" else {
            "host": 3, "gpu": 2, "baseline": 1, "copy": 0}
        for event in host_events:
            name = event["name"].removeprefix("offload/")
            category, label = phases[name]
            row = rows["host"] if event["tid"] == scope["tid"] else rows["worker"]
            left, width = event["ts"] - start, event["dur"]
            ax.broken_barh([(left, width)], (row - 0.25, 0.5), facecolors=colors[category])
            if width / timeline_limit > 0.065:
                ax.text(left + width / 2, row, label, ha="center", va="center", fontsize=8,
                        color="#27313d" if category == "wait" else "white")
        for event in gpu_events:
            copy = event["name"] in copy_names
            row = rows["copy"] if copy else rows["gpu"]
            if not copy and mode == "all_gpu" and event["tid"] != main_stream:
                row = rows["baseline"]
            ax.broken_barh([(event["ts"] - start, event["dur"])], (row - 0.25, 0.5),
                          facecolors=colors["copy" if copy else "gpu"])
        ylabels = ("Host calls / waits", "CPU worker calls / waits", "GPU kernels", "Device copies") if mode == "cpu_overlap" else (
            "Host calls / waits", "GPU main stream", "GPU G₁ stream", "Device copies")
        ax.set(yticks=(3, 2, 1, 0), yticklabels=ylabels, xlim=(0, timeline_limit), ylim=(-0.6, 3.6),
               xlabel="Time from call start (µs; instrumented)")
        ax.set_title(f"Instrumented timeline — {labels[mode]}", loc="left", fontsize=12, pad=12)
        ax.text(1, 1.07, f"{duration:.1f} µs", transform=ax.transAxes, ha="right", color="#58616d")
        ax.axvline(duration, color="#7b8794", linestyle=":", linewidth=1)
        analysis["profiled_calls"][mode] = {
            "elapsed_us": duration,
            "host_scopes": {event["name"]: {"start_us": event["ts"] - start, "duration_us": event["dur"]}
                            for event in host_events},
            "gpu_kernel_count": sum(event["name"] not in copy_names for event in gpu_events),
            "gpu_copy_events": [{"name": event["name"], "start_us": event["ts"] - start,
                                 "duration_us": event["dur"]} for event in gpu_events
                                if event["name"] in copy_names],
        }
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e8ebef", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.legend([Patch(facecolor=colors[key]) for key in ("dispatch", "wait", "transfer", "cpu", "gpu", "copy")],
               ("Host dispatch", "Host wait", "Transfer dispatch", "CPU call + ready wait", "GPU kernels", "Device copies"),
               loc="lower center", bbox_to_anchor=(0.57, 0.015), ncol=3, frameon=False, fontsize=9)
    output = ROOT / "paper-plots/figures/gpu-vs-cpu-offload-profile"
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(output.with_suffix(".png"), flush=True)
