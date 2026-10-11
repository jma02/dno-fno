"""Plot saved rollout comparisons; no training, inference, or new timing runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure, SubFigure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator, StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-plots/figures"
FAMILIES = (("stokes", "Stokes"), ("tanaka", "Tanaka"),
            ("benjamin_feir", "Benjamin–Feir"), ("jonswap_tma", "JONSWAP / TMA"))
TRAINED_STYLES = (("c27_rollout_teacher_m6_ep120_gpu0_20261001", "Rollout-supervised network", "#d34a87", "P"),
                  ("c27_surface_rms_context8_rollout_ep120_gpu0_20261006", "RMS + Fourier context", "#007da6", "s"))


def draw_comparison(args: argparse.Namespace, fig: Figure | SubFigure, heading: str = "") -> None:
    warm = json.loads(args.timings.read_text()) if args.timings else {}
    if warm:
        for path in args.classical_timings or ():
            for family, result in json.loads(path.read_text())["families"].items():
                warm["families"].setdefault(family, result)["methods"].update(result["methods"])
        for result in warm["families"].values():
            result["methods"] = {name.split("_", 1)[0] if name.startswith("M") else name: record
                                 for name, record in result["methods"].items()}
    candidate = {"families": {}} if args.candidate_timings else {}
    for path in args.candidate_timings or ():
        for family, result in json.loads(path.read_text())["families"].items():
            candidate["families"].setdefault(family, result)["methods"].update(result["methods"])
    trained = {}
    trained_method = "fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local"
    for path in args.trained_timings or ():
        result = json.loads(path.read_text())
        run = Path(result["neural_runs"][trained_method]).name
        trained.setdefault(run, {}).update({family: timing["methods"][trained_method]
                                           for family, timing in result["families"].items()})
    baseline_free = json.loads(args.no_baseline_timings.read_text()) if args.no_baseline_timings and not args.single_case else None
    spectral = {}
    for path in args.spectral_timings or ():
        result = json.loads(path.read_text())
        if result["device"] != warm["device"]:
            continue
        if result["batch_size"] != 1:
            parser.error(f"{path}: spectral overlays require batch-one timings")
        for family, timing in result["families"].items():
            reference = warm["families"][family]
            if any(timing[key] != reference[key] for key in ("horizon", "simulation_id")):
                parser.error(f"{path}: {family} spectral trajectory does not match the classical comparison")
            spectral[family] = timing["methods"]["spectral_ema"]
    overlays = (
        (candidate, "#c45b22", "New candidate optimized network", args.candidate_method, "--", 1.3),
        (baseline_free, "#178178", "New candidate optimized network without G₀+G₁", "fused", "--", 1.3),
    )
    interleaved_m6 = bool(warm and "previous_seconds" in warm["families"]["stokes"]["methods"]["M6"])
    compact = args.stack_hardware
    per_step = False
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2", "grid.linewidth": 0.6,
        "xtick.color": "#59616a", "ytick.color": "#59616a",
    })
    axes = fig.subplots(1, 4) if compact else fig.subplots(2, 2)
    fig.subplots_adjust(left=0.065 if compact else 0.10, right=0.98, bottom=0.13,
                        top=0.67 if compact else 0.74 if trained else 0.78 if baseline_free else 0.80,
                        hspace=0.37, wspace=0.32 if compact else 0.25)
    for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
        if args.single_case:
            timing = warm["families"][family]
            orders = np.asarray(sorted(int(name[1:]) for name in timing["methods"] if name.startswith("M")))
            seconds = np.asarray([np.median(timing["methods"][f"M{order}"]["seconds"]) for order in orders])
            errors = 100 * np.asarray([timing["methods"][f"M{order}"]["cached_reference_terminal_eta_error"] for order in orders])
            horizon = timing["horizon"]
        else:
            with np.load(ROOT / "outputs/cs_order_sweep_20260923" / f"{family}.npz") as data:
                selected = np.argsort(data["orders"])
                orders = data["orders"][selected]
                seconds = np.median(data["timings_s"][selected], axis=1)
                errors = 100 * np.median(data["eta_rel_l2"][selected, -1], axis=1)
                horizon = float(data["times"][-1])
        scale = 1.0
        if warm:
            timing = warm["families"][family]
            per_step = timing["horizon"] != horizon
            if (candidate or baseline_free) and per_step:
                parser.error("the candidate overlay requires full-trajectory timings")
            scale = 1000 / timing["steps"] if per_step else 1.0
            seconds = scale * np.asarray([np.median(timing["methods"][f"M{order}"]["seconds"]) for order in orders])
        ax.set(yscale="log", xscale="log" if compact else "linear")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5), numticks=12) if compact else MaxNLocator(4))
        if compact:
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", zorder=0)
        ax.plot(seconds[:-1], errors[:-1], "o-", color="#606970", ms=4,
                markerfacecolor="white", lw=1.1, zorder=2)
        for order, x, y in zip(orders[:-1], seconds[:-1], errors[:-1], strict=True):
            offset = (6, -8) if family == "benjamin_feir" and order == 3 else (6, 3)
            if (candidate or baseline_free) and order == 1:
                offset = (0, 10)
            if trained and order == 2:
                offset = (6, -10)
            ax.annotate(f"M{order}", (x, y), xytext=offset, textcoords="offset points",
                        color="#606970", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})
        ax.axvline(seconds[-1], color="#aab2b9", ls=":", lw=1.2, zorder=1)
        if interleaved_m6 and not compact:
            previous_seconds = float(np.median(warm["families"][family]["methods"]["M6"]["previous_seconds"]))
            savings = 100 * (1 - seconds[-1] / previous_seconds)
            ax.text(0.43, 0.035, f"M6: {previous_seconds:.2f} → {seconds[-1]:.2f} s  (−{savings:.1f}%)",
                    transform=ax.transAxes, color="#606970", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})
        right = float(seconds.max())
        left = float(seconds.min())
        for comparison, color, label, method, style, width in overlays:
            if not comparison or family not in comparison["families"]:
                continue
            methods = comparison["families"][family]["methods"]
            record = methods[method]
            new_seconds = float(np.median(record["seconds"]))
            if comparison is candidate:
                ax.scatter(new_seconds, 100 * record["cached_reference_terminal_eta_error"],
                           color=color, marker="o", s=65, edgecolors="white", linewidths=0.8, zorder=3)
            else:
                ax.axvline(new_seconds, color=color, ls=style, lw=width, zorder=1)
            if comparison is candidate and method == args.candidate_method and args.candidate_previous_method and not interleaved_m6 and not compact:
                previous = methods[args.candidate_previous_method]
                previous_seconds = float(np.median(previous["seconds"]))
                savings = 100 * (1 - new_seconds / previous_seconds)
                ax.text(0.43, 0.035, f"{previous_seconds:.2f} → {new_seconds:.2f} s  (−{savings:.1f}%)",
                        transform=ax.transAxes, color=color, fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})
            right = max(right, new_seconds)
            left = min(left, new_seconds)
            print(f"{title}, {label}: {new_seconds:.3f}s; finite={record['finite']}")
        for run, label, color, marker in TRAINED_STYLES:
            if family not in trained.get(run, {}):
                continue
            record = trained[run][family]
            trained_seconds = float(np.median(record["seconds"]))
            trained_error = 100 * record["cached_reference_terminal_eta_error"]
            ax.scatter(trained_seconds, trained_error, color=color, marker=marker,
                       s=65, edgecolors="white", linewidths=0.8, zorder=4)
            right = max(right, trained_seconds)
            left = min(left, trained_seconds)
            print(f"{title}, {label}: {trained_seconds:.3f}s; error={trained_error:.6g}%; finite={record['finite']}")
        if family in spectral:
            record = spectral[family]
            spectral_seconds = float(np.median(record["seconds"]))
            if not np.isfinite(spectral_seconds) or spectral_seconds <= 0:
                parser.error(f"{family}: spectral runtime must be finite and positive")
            spectral_error = record["cached_reference_terminal_eta_error"]
            finite = record["finite"] and spectral_error is not None and np.isfinite(spectral_error)
            if not args.single_case or not finite:
                ax.axvline(spectral_seconds, color="#7444a5", ls="-.", lw=1.3, zorder=1)
            if args.single_case and spectral_error is not None and finite:
                ax.scatter(spectral_seconds, 100 * spectral_error, color="#7444a5", marker="D",
                           s=65, edgecolors="white", linewidths=0.8, zorder=3)
            elif not finite:
                ax.plot(spectral_seconds, 0.92, marker="x", color="#7444a5", ms=7,
                        transform=ax.get_xaxis_transform(), zorder=3)
                ax.annotate("failed", (spectral_seconds, 0.92), xycoords=ax.get_xaxis_transform(),
                            xytext=(6, -3), textcoords="offset points", color="#7444a5", fontsize=8)
            right = max(right, spectral_seconds)
            left = min(left, spectral_seconds)
            print(f"{title}, Spectral DNO weight EMA: {spectral_seconds:.3f}s; finite={finite}; error={spectral_error}")
        ax.set_xlim(left / 1.28 if compact else 0, 1.28 * right if compact else 1.13 * right)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_title(f"T = {horizon:g}", loc="right", fontsize=9, color="#777f86", pad=12)

    fig.suptitle(heading or ("Single-rollout runtime vs error" if warm else "Batch runtime vs error"),
                 y=0.97 if compact else 0.94 if heading else 0.96, fontsize=14 if heading else 18,
                 ha="left", x=0.065 if compact else 0.10)
    if args.single_case and not heading:
        fig.text(0.10, 0.921, warm["device"], color="#59616a", fontsize=10)
    fig.legend(handles=[
        Line2D([], [], color="#606970", marker="o", markerfacecolor="white", ms=4, lw=1, label="Classical"),
        Line2D([], [], color="#aab2b9", ls=":", lw=1.2, label="M6 time"),
        *[Line2D([], [], color=color, marker="o" if comparison is candidate else None,
                  ls="none" if comparison is candidate else style, ms=7, lw=width, label=label)
          for comparison, color, label, _method, style, width in overlays if comparison],
        *([Line2D([], [], color="#7444a5", marker="D" if args.single_case else None, ls="none" if args.single_case else "-.", ms=6, lw=1.3,
                  label="Spectral DNO — weight EMA (PyTorch)" if args.single_case else
                  "Spectral DNO — weight EMA (runtime only)")] if spectral else []),
        *[Line2D([], [], color=color, marker=marker, ls="none", ms=7, label=label)
          for run, label, color, marker in TRAINED_STYLES if run in trained],
    ], loc="upper left", bbox_to_anchor=(0.06, 0.90) if compact else (0.09, 0.895 if args.single_case and not heading else 0.91),
        ncols=3 if compact else 2 if candidate or baseline_free else 4, frameon=False, fontsize=10)
    xlabel = "Time per step (ms)" if per_step else "One rollout (seconds)" if warm else "32 rollouts together (seconds)"
    if not compact:
        fig.supxlabel(xlabel, y=0.06, fontsize=11)
        fig.supylabel("Final surface error (%)" if args.single_case else "Median final surface error (%)", x=0.015, y=0.50, fontsize=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, help="Use fresh single-rollout timings instead of archived batch-32 times.")
    parser.add_argument("--classical-timings", type=Path, nargs="+", help="Replace classical runtimes with these full-rollout measurements.")
    parser.add_argument("--candidate-timings", type=Path, nargs="+", help="Overlay measured fused-model runtimes from these files; requires --timings.")
    parser.add_argument("--trained-timings", type=Path, nargs="+", help="Add measured trained-model comparisons to the Ada panels.")
    parser.add_argument("--candidate-method", default="fused")
    parser.add_argument("--candidate-previous-method", help="Show the runtime reduction against this method in the same timing files.")
    parser.add_argument("--no-baseline-timings", type=Path, help="Overlay speed-only full rollouts with model G0+G1 omitted.")
    parser.add_argument("--spectral-timings", type=Path, nargs="+", help="Overlay spectral weight-EMA timings on matching Figure 12 hardware; show errors only on single-case panels.")
    parser.add_argument("--stack-hardware", action="store_true", help="Stack the four-family Blackwell, Ada, and A100 comparisons.")
    parser.add_argument("--single-case", action="store_true", help="Plot single-case runtime/error measurements.")
    parser.add_argument("--one-epoch", action="store_true", help="Figure 12b: completed full-data, one-epoch screens.")
    parser.add_argument("--momentum-comparison", action="store_true", help="Figure 12c: measured original and momentum-corrected rollouts.")
    parser.add_argument("--output-name", type=Path, help="Output stem, relative to paper-plots/figures or absolute.")
    args = parser.parse_args()
    if args.momentum_comparison:
        momentum_dir = ROOT / "outputs/momentum_state_projection_diagnostic_20261010"
        momentum_neural = ROOT / "outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_full_20261010/eval_fast_weight_ema"
        momentum_timings = json.loads((momentum_dir / "single_rollout_timings.json").read_text())
        momentum_method = "fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local"
        momentum_styles = ((0, "New candidate optimized network", "#007da6", "o"),
                           (2, "Optimized M2", "#c45b22", "s"), (3, "Optimized M3", "#7855ad", "D"))
        momentum_points = {}
        for family, _ in FAMILIES:
            for order, _, _, _ in momentum_styles:
                records = (momentum_timings["families"][family]["methods"] if order == 0 else json.loads(
                    (ROOT / f"outputs/momentum_classical_M{order}_timing_20261010/{family}_timing.json").read_text(),
                )["methods"])
                paired_points = []
                for correction in ("none", "state"):
                    source = ((momentum_neural if correction == "none" else momentum_dir) if order == 0 else
                              ROOT / f"outputs/momentum_classical_M{order}_{correction}_20261010")
                    summary = json.loads((source / f"{family}_summary.json").read_text())
                    timing_key = (momentum_method + ("_momentum" if correction == "state" else "")
                                  if order == 0 else correction)
                    paired_points.append((float(np.median(records[timing_key]["seconds"])),
                                          100 * summary["rel_l2_eta_median_conditional_finite_tfinal"]))
                momentum_points[family, order] = np.asarray(paired_points)
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2",
            "xtick.color": "#59616a", "ytick.color": "#59616a",
        })
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.subplots_adjust(left=0.11, right=0.98, top=0.75, bottom=0.11, hspace=0.38, wspace=0.28)
        for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
            for order, _, color, marker in momentum_styles:
                pair = momentum_points[family, order]
                ax.plot(pair[:, 0], pair[:, 1], color=color, lw=1.2, zorder=2)
                ax.plot(*pair[0], marker=marker, color=color, mfc="white", ms=8, ls="none", zorder=3)
                ax.plot(*pair[1], marker=marker, color=color, ms=5, ls="none", zorder=4)
            ax.set(yscale="log", xlim=(0, 1.15 * max(momentum_points[family, order][:, 0].max()
                                                   for order, _, _, _ in momentum_styles)))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5), numticks=6))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", lw=0.6)
            ax.margins(y=0.20)
            ax.set_title(title, loc="left", fontsize=13, pad=12)
            ax.set_title(f"T = {momentum_timings['families'][family]['horizon']:g}", loc="right",
                         fontsize=10, color="#777f86", pad=12)
        fig.suptitle("Momentum correction · rollout accuracy and runtime", x=0.11, y=0.97, ha="left", fontsize=17)
        fig.text(0.11, 0.925, "RTX 6000 Ada · Neural: one epoch, weight + gradient EMA, same weights",
                 color="#59616a", fontsize=10)
        fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls="none", label=label)
                            for _, label, color, marker in momentum_styles],
                   loc="upper left", bbox_to_anchor=(0.10, 0.90), ncols=3, frameon=False)
        fig.legend(handles=[Line2D([], [], color="#59616a", marker="o", mfc=face, ls="none", label=label)
                            for face, label in (("white", "Original integrator"),
                                                ("#59616a", "+ state momentum correction"))],
                   loc="upper left", bbox_to_anchor=(0.10, 0.86), ncols=2, frameon=False)
        fig.supxlabel("Full single-rollout runtime (s) · median of 3 warm repeats", y=0.04, fontsize=11)
        fig.supylabel("Median terminal η error (%) · 32 cases", x=0.025, y=0.43, fontsize=11)
        stem = OUT / (args.output_name or "12c-momentum-projection")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(f"{stem}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)
        raise SystemExit(0)
    if args.one_epoch:
        classical = json.loads((ROOT / "outputs/classical_all_orders_20260928/ada/classical.json").read_text())
        neural = json.loads((ROOT / "outputs/rms_context_timing_20261007/ada/timings.json").read_text())
        method = "fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local"
        screens = (
            ("control", "Control", "#343b43", "o"),
            ("reflection", "Reflection augmentation", "#cf6327", "s"),
            ("polynomial_skip", "Polynomial shortcut", "#008772", "D"),
            ("momentum_half", "Momentum × ½", "#7855ad", "^"),
            ("momentum_quarter", "Momentum × ¼", "#007da6", "v"),
            ("mb2", "Mode balancing × ⅓", "#ba3e75", "P"),
            ("even_features", "Even surface features", "#9b761d", "X"),
            ("lr_low", "Learning rate 1e−5", "#627b34", "<"),
            ("lr_high", "Learning rate 4e−5", "#a4473b", ">"),
            ("multiplier_skip", "Multiplier skip", "#4169a1", "h"),
            ("lr_high_momentum_half", "LR 4e−5 + momentum × ½", "#156e55", "*"),
            ("lr_high_momentum_quarter", "LR 4e−5 + momentum × ¼", "#bd6b10", "p"),
            ("lr6_momentum_half", "LR 6e−5 + momentum × ½", "#9c416a", "H"),
            ("lr_high_momentum_half_ema995", "Combined + EMA 0.995", "#395ea8", "d"),
            ("lr_high_momentum_half_ramp025", "Combined + ¼-epoch ramp", "#66852e", "8"),
            ("lr6_momentum_half_ema995", "LR 6e−5 + EMA .995", "#b24d24", "D"),
            ("lr6_momentum_half_ema995_hold", "Longer LR plateau", "#517f2f", "v"),
            ("lr6_momentum_half_ema995_momentum_ramp", "Momentum ramp", "#79549e", "s"),
            ("lr6_momentum_half_ema995_grad05", "Gradient EMA .5", "#197b88", "P"),
            ("lr6_momentum_half_ema995_head2", "Head learning rate × 2", "#934c79", "^"),
            ("lr6_momentum_half_ema995_band128", "Data modes ≤ 128", "#967228", "h"),
            ("lr6_momentum_half_ema995_log_inputs", "Log multiplier inputs", "#3e668e", "*"),
            ("lr6_momentum_half_ema995_seed1", "Parent · seed 1", "#b24d24", "s"),
            ("lr6_momentum_half_ema995_grad05_seed1", "Gradient EMA .5 · seed 1", "#197b88", "X"),
            ("lr6_momentum_half_ema995_grad025", "Gradient EMA .25", "#548226", "v"),
            ("lr6_momentum_half_ema995_grad05_head2", "Gradient EMA .5 + head × 2", "#934c79", "D"),
            ("lr6_momentum_half_ema995_grad05_adam05", "Gradient EMA .5 + Adam β₁=.5", "#4d58a6", "p"),
            ("lr6_momentum_half_ema995_grad05_adam05_seed1", "Grad EMA .5 + β₁=.5 · seed 1", "#4d58a6", "X"),
            ("lr6_momentum_half_ema995_grad05_lr8", "Gradient EMA .5 + LR 8e−5", "#a4473b", ">"),
            ("lr6_momentum_half_ema995_grad05_teacher20", "Gradient EMA .5 + 20-step teacher", "#967228", "h"),
            ("lr6_momentum_half_ema995_grad05_adam07", "Adam β₁=.7", "#4d58a6", "v"),
            ("lr8_momentum_half_ema995_grad05_adam05", "LR 8e−5 + Adam β₁=.5", "#a4473b", "D"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099", "Adam β₂=.99", "#387a66", "o"),
            ("lr8_momentum_half_ema995_grad05_adam05_b2099", "Adam β₂=.99 + LR 8e−5", "#994e36", "D"),
            ("lr8_momentum_half_ema995_grad05_adam05_b2099_batch512", "LR 8e−5 + batch 512", "#207981", "s"),
            ("lr8_momentum_half_ema995_grad05_adam05_b2099_batch512_seed1", "Batch 512 · seed 1", "#207981", "X"),
            ("lr8_momentum_full_ema995_grad05_adam05_b2099", "LR 8e−5 + momentum × 2", "#6952a0", "^"),
            ("lr10_momentum_half_ema995_grad05_adam05_b2099", "Adam β₂=.99 + LR 1e−4", "#b46d32", "P"),
            ("lr8_momentum_half_ema995_grad05_adam05_b2099_seed1", "Adam β₂=.99 + LR 8e−5 · seed 1", "#994e36", "X"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_seed1", "Adam β₂=.99 · seed 1", "#387a66", "X"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_context16", "Context width 16", "#85569c", "^"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_mbfloor1e5", "Adam β₂=.99 + MB floor × 10", "#2f6d4e", "s"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_amp025", "Amplitude targets · full band", "#a15f32", "v"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_amp025k128", "Amplitude targets · ≤128", "#834126", "P"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_depthgate", "Depth-conditioned surface head", "#42688f", "p"),
            ("lr6_momentum_half_ema995_grad05_adam05_b2099_datum025", "Datum augmentation · 2.5% RMS", "#796235", "h"),
            ("lr6_momentum_half_ema995_grad05_adam05_whiten", "Whitened surface inputs", "#826234", "h"),
            ("lr6_momentum_half_ema995_grad05_adam05_mbfloor1e5", "Mode-balance floor × 10", "#52794b", "v"),
        )
        completed = []
        for suffix, label, color, marker in screens:
            run_date = "20261010" if suffix.startswith(("lr6_momentum_half_ema995", "lr8_momentum_half_ema995", "lr8_momentum_full_ema995", "lr10_momentum_half_ema995")) else "20261009"
            run = ROOT / f"outputs/c27_phase1_{suffix}_full_{run_date}"
            paths = {family: run / f"eval_epoch1_weight_ema_test_stratified_n32_gpu{0 if family in ('stokes', 'tanaka') else 1}/{family}_summary.json"
                     for family, _ in FAMILIES}
            if all(path.exists() for path in paths.values()):
                results = {family: json.loads(path.read_text()) for family, path in paths.items()}
                timing_path = run / "timing/timings.json"
                if not timing_path.exists() and suffix in (
                    "polynomial_skip", "even_features", "lr6_momentum_half_ema995_grad05_adam05_b2099_context16",
                    "lr6_momentum_half_ema995_grad05_adam05_b2099_depthgate",
                ):
                    print(f"Awaiting architecture timing: {label}")
                else:
                    timings = json.loads(timing_path.read_text()) if timing_path.exists() else neural
                    completed.append((results, timings, label, color, marker))
            else:
                print(f"Pending: {label}")
            fast_dir = run / "eval_fast_weight_ema"
            fast_results = fast_dir / "all_summaries.json"
            fast_timings = fast_dir / "single_rollout_timings.json"
            if fast_results.exists() and fast_timings.exists():
                measured = json.loads(fast_timings.read_text())
                if all(family in measured["families"] for family, _ in FAMILIES):
                    completed.append((json.loads(fast_results.read_text()), measured,
                                      label.removeprefix("Gradient EMA .5 + ") + " · fast", color, "D"))
        if not completed:
            parser.error("No completed four-family one-epoch weight-EMA rollout evaluation with an available runtime yet.")
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2",
            "xtick.color": "#59616a", "ytick.color": "#59616a",
        })
        fig, axes = plt.subplots(2, 2, figsize=(17, max(9.5, 2.5 + 1.2 * len(completed))))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.12, hspace=0.38, wspace=0.27)
        for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
            timing = neural["families"][family]
            with np.load(ROOT / f"outputs/cs_order_sweep_20260923/{family}.npz") as data:
                orders = (2, 3)
                classical_errors = [100 * float(np.median(data["eta_rel_l2"][np.flatnonzero(data["orders"] == order)[0], -1]))
                                    for order in orders]
            classical_seconds = [float(np.median(classical["families"][family]["methods"][f"M{order}"]["seconds"]))
                                 for order in orders]
            ax.plot(classical_seconds, classical_errors, "o-", color="#899198", markerfacecolor="white", ms=5, lw=1)
            for order, x, y in zip(orders, classical_seconds, classical_errors, strict=True):
                ax.annotate(f"M{order}", (x, y), xytext=(6, -2), textcoords="offset points", color="#667078", fontsize=10)
            points = []
            for results, timings, label, color, marker in completed:
                result = results[family]
                seconds = float(np.median(timings["families"][family]["methods"][method]["seconds"]))
                failures = result["model_nonfinite_any_count_truth_valid"]
                if failures:
                    label += f" · {failures}/{result['n_truth_valid']} failed"
                error = 100 * result["rel_l2_eta_median_conditional_finite_tfinal"]
                points.append((error, seconds, label, color))
                ax.scatter(seconds, error, s=45, marker=marker, color=color, edgecolors="white", linewidths=0.7, zorder=4)
                print(f"{family}, {label}: median error {error:.8g}%; architecture runtime {seconds:.6f}s; "
                      f"finite {result['n_model_finite_truth_valid']}/{result['n_truth_valid']}")
            label_positions = np.linspace(0.90, max(0.08, 0.90 - 0.0975 * (len(points) - 1)), len(points))
            for (error, seconds, label, color), label_y in zip(sorted(points, reverse=True), label_positions, strict=True):
                ax.annotate(f"{label}\n{error:.4g}%", (seconds, error), xytext=(0.65, label_y), textcoords="axes fraction",
                            ha="left", va="center", color=color, fontsize=9,
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.6, "relpos": (0, 0.5)},
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=5)
            values = classical_errors + [point[0] for point in points]
            log_low, log_high = np.log10([min(values), max(values)])
            span = max(log_high - log_low, 0.5)
            ax.set(yscale="log", xlim=(0, 2.0 * max(*classical_seconds, *(point[1] for point in points))),
                   ylim=(10**(log_low - 0.13 * span), 10**(log_high + 0.20 * span)))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(LogLocator(numticks=5))
            if span < 1.0:
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", lw=0.6)
            ax.set_title(title, loc="left", fontsize=13, pad=12)
            ax.set_title(f"T = {timing['horizon']:g}", loc="right", fontsize=10, color="#777f86", pad=12)
        fig.suptitle("Figure 12b · One epoch, full data", x=0.10, y=0.965, ha="left", fontsize=19)
        fig.text(0.10, 0.923, "RTX 6000 Ada · 32 test cases per family · Weight EMA", color="#59616a", fontsize=11)
        fig.text(0.10, 0.891, "Native screens: shared timing proxy. Fast: directly evaluated and timed.",
                 color="#59616a", fontsize=10)
        fig.supxlabel("Single-rollout runtime (seconds)", y=0.045, fontsize=12)
        fig.supylabel("Median final surface error (%)", x=0.02, fontsize=12)
        stem = OUT / (args.output_name or "12b-one-epoch-full-data")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(f"{stem}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)
        raise SystemExit(0)
    if (args.classical_timings or args.candidate_timings or args.no_baseline_timings) and not args.timings:
        parser.error("runtime overlays require --timings")
    if args.single_case and not (args.timings and args.candidate_timings):
        parser.error("--single-case requires --timings and --candidate-timings")
    if args.stack_hardware and (args.single_case or not args.timings or not args.candidate_timings):
        parser.error("--stack-hardware requires local --timings and --candidate-timings without --single-case")
    if args.spectral_timings and not (args.single_case or args.stack_hardware):
        parser.error("--spectral-timings requires --single-case or --stack-hardware")
    if args.trained_timings and not args.stack_hardware:
        parser.error("--trained-timings requires --stack-hardware")
    if args.stack_hardware:
        modal = argparse.Namespace(**vars(args))
        modal.single_case = True
        ada_config = argparse.Namespace(**vars(modal))
        a100 = argparse.Namespace(**vars(modal))
        for config, hardware in ((modal, "blackwell"), (ada_config, "ada"), (a100, "a100")):
            classical = ROOT / "outputs" / {
                "ada": "classical_all_orders_20260928/ada/classical.json",
                "blackwell": "classical_all_orders_20260928/blackwell/classical.json",
                "a100": "classical_all_orders_20260928/a100_sxm/classical.json",
            }[hardware]
            config.timings = classical
            config.classical_timings = [classical]
            config.candidate_timings = [ROOT / "outputs" / {
                "ada": "ema_grad_timing_20260928/ada/timings.json",
                "blackwell": "ema_grad_timing_20261001/blackwell/timings.json",
                "a100": "ema_grad_timing_20261001/a100_sxm/timings.json",
            }[hardware]]
            config.candidate_method = "fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local"
            config.candidate_previous_method = None
            config.no_baseline_timings = None
            config.trained_timings = args.trained_timings if hardware == "ada" else None
        fig = plt.figure(figsize=(18, 13.5))
        fig.suptitle("Single-rollout runtime vs error", y=0.99, fontsize=18, ha="right", x=0.98)
        grid = fig.add_gridspec(3, 1, top=0.945, bottom=0.04, hspace=0.04)
        blackwell, ada, ampere = (fig.add_subfigure(grid[row]) for row in range(3))
        draw_comparison(modal, blackwell, "Blackwell · Modal — RTX PRO 6000")
        draw_comparison(ada_config, ada, "Ada · Local — RTX 6000")
        draw_comparison(a100, ampere, "A100 · Modal — A100-SXM4-80GB")
        for column in range(4):
            panels = fig.axes[column::4]
            limits = [(min(ax.get_xlim()[0] for ax in panels), max(ax.get_xlim()[1] for ax in panels)),
                      (min(ax.get_ylim()[0] for ax in panels), max(ax.get_ylim()[1] for ax in panels))]
            for ax in panels:
                ax.set(xlim=limits[0], ylim=limits[1])
        fig.supxlabel("One rollout (seconds, log scale)", y=0.012, fontsize=12)
        fig.supylabel("Final surface error (%)", x=0.012, fontsize=12)
    else:
        fig = plt.figure(figsize=(10.5, 8))
        draw_comparison(args, fig)
    name = ("12-potential-new-nn-modal" if args.single_case else "12-potential-new-nn" if args.candidate_timings or args.no_baseline_timings
            else "11-single-rollout-warm" if args.timings else "10-neural-vs-classical")
    stem = OUT / (args.output_name or name)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
