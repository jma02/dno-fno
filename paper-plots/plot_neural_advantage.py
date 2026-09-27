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
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-plots/figures"
RUNS = (
    ("c27_tanaka_hard128_full_equal_local_20260918", "full", "larger", "#8b5ea8", "s"),
    ("c27_branches16_tanaka_hard128_20260923", "small", "small", "#087eaa", "o"),
)
FAMILIES = (("stokes", "Stokes"), ("tanaka", "Tanaka"),
            ("benjamin_feir", "Benjamin–Feir"), ("jonswap_tma", "JONSWAP / TMA"))


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
    if args.stack_hardware:
        for family, result in candidate["families"].items():
            for method, record in result["methods"].items():
                classical = method.split("_", 1)[0]
                if classical in ("M1", "M2"):
                    warm["families"][family]["methods"][classical]["seconds"] = record["seconds"]
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
    per_step = False
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2", "grid.linewidth": 0.6,
        "xtick.color": "#59616a", "ytick.color": "#59616a",
    })
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14,
                        top=0.78 if candidate or baseline_free else 0.81, hspace=0.37, wspace=0.25)
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
        ax.set(yscale="log")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", zorder=0)
        ax.plot(seconds[:-1], errors[:-1], "o-", color="#606970", ms=4,
                markerfacecolor="white", lw=1.1, zorder=2)
        for order, x, y in zip(orders[:-1], seconds[:-1], errors[:-1], strict=True):
            offset = ((6, -8) if args.single_case else (8, 9)) if family == "benjamin_feir" and order == 3 else (6, 3)
            if (candidate or baseline_free) and order == 1:
                offset = (0, 10)
            ax.annotate(f"M{order}", (x, y), xytext=offset, textcoords="offset points",
                        color="#606970", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})
        ax.axvline(seconds[-1], color="#aab2b9", ls=":", lw=1.2, zorder=1)
        right = float(seconds.max())
        for run, name, _label, color, marker in RUNS:
            source = next((ROOT / "outputs" / run).glob(
                f"eval_best_current_test_stratified_n32*/{family}_summary.json"))
            if args.single_case:
                with np.load(source.with_name(f"{family}_trajs.npz")) as saved:
                    selected = np.flatnonzero(saved["simulation_ids"] == warm["families"][family]["simulation_id"])
                    if len(selected) != 1 or saved["times"][-1] != horizon:
                        parser.error(f"{source}: saved accuracy case or horizon does not match")
                    neural_error = 100 * float(saved["rel_l2_eta"][-1, selected[0]])
                ax.axhline(neural_error, color=color, ls="--", lw=1.0, zorder=1)
                print(f"{title}, {name}: saved final surface error={neural_error:.6g}%")
                continue
            result = json.loads(source.read_text())
            neural_seconds = result["surrogate_wall_s"]
            if warm:
                neural_seconds = scale * np.median(warm["families"][family]["methods"][name]["seconds"])
            neural_error = 100 * result["rel_l2_eta_median_conditional_finite_tfinal"]
            ax.scatter(neural_seconds, neural_error, color=color, marker=marker,
                       s=65, edgecolors="white", linewidths=0.8, zorder=3)
            right = max(right, neural_seconds)
            print(f"{title}, {name}: runtime axis={neural_seconds:.3f}, error={neural_error:.6g}%")
        for comparison, color, label, method, style, width in overlays:
            if not comparison or family not in comparison["families"]:
                continue
            methods = comparison["families"][family]["methods"]
            record = methods[method]
            new_seconds = float(np.median(record["seconds"]))
            ax.axvline(new_seconds, color=color, ls=style, lw=width, zorder=1)
            if comparison is candidate and method == args.candidate_method and args.candidate_previous_method:
                previous = methods[args.candidate_previous_method]
                previous_seconds = float(np.median(previous["seconds"]))
                savings = 100 * (1 - new_seconds / previous_seconds)
                ax.text(0.43, 0.035, f"{previous_seconds:.2f} → {new_seconds:.2f} s  (−{savings:.1f}%)",
                        transform=ax.transAxes, color=color, fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})
            right = max(right, new_seconds)
            print(f"{title}, {label}: {new_seconds:.3f}s; finite={record['finite']}")
        if family in spectral:
            record = spectral[family]
            spectral_seconds = float(np.median(record["seconds"]))
            if not np.isfinite(spectral_seconds) or spectral_seconds <= 0:
                parser.error(f"{family}: spectral runtime must be finite and positive")
            spectral_error = record["cached_reference_terminal_eta_error"]
            finite = record["finite"] and spectral_error is not None and np.isfinite(spectral_error)
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
            print(f"{title}, Spectral DNO weight EMA: {spectral_seconds:.3f}s; finite={finite}; error={spectral_error}")
        ax.set_xlim(0, 1.13 * right)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_title(f"T = {horizon:g}", loc="right", fontsize=9, color="#777f86", pad=12)

    fig.suptitle(heading or ("Single-rollout runtime vs error" if warm else "Batch runtime vs error"),
                 y=0.94 if heading else 0.96, fontsize=14 if heading else 18, ha="left", x=0.10)
    if args.single_case and not heading:
        fig.text(0.10, 0.921, warm["device"], color="#59616a", fontsize=10)
    fig.legend(handles=[
        Line2D([], [], color="#606970", marker="o", markerfacecolor="white", ms=4, lw=1, label="Classical"),
        *[Line2D([], [], color=color, marker=None if args.single_case else marker,
                  ls="--" if args.single_case else "none", lw=1.0, ms=7,
                  label=f"Neural — {name}" + (" (saved accuracy)" if args.single_case else ""))
          for _, name, _, color, marker in reversed(RUNS)],
        Line2D([], [], color="#aab2b9", ls=":", lw=1.2, label="M6 time"),
        *[Line2D([], [], color=color, ls=style, lw=width, label=label)
          for comparison, color, label, _method, style, width in overlays if comparison],
        *([Line2D([], [], color="#7444a5", marker="D" if args.single_case else None, ls="-.", ms=6, lw=1.3,
                  label="Spectral DNO — weight EMA (PyTorch)" if args.single_case else
                  "Spectral DNO — weight EMA (runtime only)")] if spectral else []),
    ], loc="upper left", bbox_to_anchor=(0.09, 0.895 if args.single_case and not heading else 0.91),
        ncols=2 if candidate or baseline_free else 4, frameon=False, fontsize=10)
    xlabel = "Time per step (ms)" if per_step else "One rollout (seconds)" if warm else "32 rollouts together (seconds)"
    fig.supxlabel(xlabel, y=0.06, fontsize=11)
    fig.supylabel("Final surface error (%)" if args.single_case else "Median final surface error (%)", x=0.015, y=0.50, fontsize=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, help="Use fresh single-rollout timings instead of archived batch-32 times.")
    parser.add_argument("--classical-timings", type=Path, nargs="+", help="Replace classical runtimes with these full-rollout measurements.")
    parser.add_argument("--candidate-timings", type=Path, nargs="+", help="Overlay measured fused-model runtimes from these files; requires --timings.")
    parser.add_argument("--candidate-method", default="fused")
    parser.add_argument("--candidate-previous-method", help="Show the runtime reduction against this method in the same timing files.")
    parser.add_argument("--no-baseline-timings", type=Path, help="Overlay speed-only full rollouts with model G0+G1 omitted.")
    parser.add_argument("--spectral-timings", type=Path, nargs="+", help="Overlay spectral weight-EMA timings on matching Figure 12 hardware; show errors only on single-case panels.")
    parser.add_argument("--stack-hardware", action="store_true", help="Stack the four-family Blackwell, Ada, and A100 comparisons.")
    parser.add_argument("--single-case", action="store_true", help="Plot single-case errors, saved trained-model accuracy references, and candidate timing lines.")
    parser.add_argument("--output-name", type=Path, help="Output stem, relative to paper-plots/figures or absolute.")
    args = parser.parse_args()
    if (args.classical_timings or args.candidate_timings or args.no_baseline_timings) and not args.timings:
        parser.error("runtime overlays require --timings")
    if args.single_case and not (args.timings and args.candidate_timings):
        parser.error("--single-case requires --timings and --candidate-timings")
    if args.stack_hardware and (args.single_case or not args.timings or not args.candidate_timings):
        parser.error("--stack-hardware requires local --timings and --candidate-timings without --single-case")
    if args.spectral_timings and not (args.single_case or args.stack_hardware):
        parser.error("--spectral-timings requires --single-case or --stack-hardware")
    if args.stack_hardware:
        modal = argparse.Namespace(**vars(args))
        modal.single_case = True
        a100 = argparse.Namespace(**vars(modal))
        for config, hardware in ((modal, "blackwell"), (args, "ada"), (a100, "a100")):
            directory = ROOT / "outputs/spectral_carry_20260927"
            classical = directory / {
                "ada": "ada_classical.json", "blackwell": "blackwell_classical/classical.json",
                "a100": "a100_combined_sxm/classical.json",
            }[hardware]
            if config.single_case:
                config.timings = classical
            config.classical_timings = [classical]
            config.candidate_timings = [directory / {
                "ada": "ada_full.json", "blackwell": "blackwell_full/rollouts.json",
                "a100": "a100_combined_sxm/rollouts.json",
            }[hardware]]
            config.candidate_method = "fourier_fused_features_overlap_spectral_all_boundaries_stages_carry_local"
            config.candidate_previous_method = "fourier_fused_fields_overlap_spectral_all_boundaries_stages"
            config.no_baseline_timings = None
        fig = plt.figure(figsize=(10.5, 24.75))
        fig.suptitle("Single-rollout runtime vs error", y=0.995, fontsize=18, ha="left", x=0.10)
        blackwell, ada, ampere = fig.subfigures(3, 1)
        draw_comparison(modal, blackwell, "Blackwell · Modal — RTX PRO 6000")
        draw_comparison(args, ada, "Ada · Local — RTX 6000")
        draw_comparison(a100, ampere, "A100 · Modal — A100-SXM4-80GB")
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
