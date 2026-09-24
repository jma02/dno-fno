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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, help="Use fresh single-rollout timings instead of archived batch-32 times.")
    parser.add_argument("--classical-timings", type=Path, nargs="+", help="Replace classical runtimes with these full-rollout measurements.")
    parser.add_argument("--candidate-timings", type=Path, nargs="+", help="Overlay measured fused-model runtimes from these files; requires --timings.")
    parser.add_argument("--candidate-method", default="fused")
    parser.add_argument("--candidate-previous-method", help="Show the runtime reduction against this method in the same timing files.")
    parser.add_argument("--no-baseline-timings", type=Path, help="Overlay speed-only full rollouts with model G0+G1 omitted.")
    args = parser.parse_args()
    if (args.classical_timings or args.candidate_timings or args.no_baseline_timings) and not args.timings:
        parser.error("runtime overlays require --timings")
    warm = json.loads(args.timings.read_text()) if args.timings else None
    if warm:
        for path in args.classical_timings or ():
            for family, result in json.loads(path.read_text())["families"].items():
                warm["families"][family]["methods"].update(result["methods"])
    candidate = ({"families": {family: result for path in args.candidate_timings
                              for family, result in json.loads(path.read_text())["families"].items()}}
                 if args.candidate_timings else None)
    baseline_free = json.loads(args.no_baseline_timings.read_text()) if args.no_baseline_timings else None
    overlays = (
        (candidate, "#c45b22", "New candidate optimized network"),
        (baseline_free, "#178178", "New candidate optimized network without G₀+G₁"),
    )
    per_step = False
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2", "grid.linewidth": 0.6,
        "xtick.color": "#59616a", "ytick.color": "#59616a",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14,
                        top=0.78 if candidate or baseline_free else 0.81, hspace=0.37, wspace=0.25)
    for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
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
            offset = (8, 9) if family == "benjamin_feir" and order == 3 else (6, 3)
            if (candidate or baseline_free) and order == 1:
                offset = (-22, 5)
            ax.annotate(f"M{order}", (x, y), xytext=offset, textcoords="offset points",
                        color="#606970", fontsize=8)
        ax.axvline(seconds[-1], color="#aab2b9", ls=":", lw=1.2, zorder=1)
        right = float(seconds.max())
        for run, name, _label, color, marker in RUNS:
            source = next((ROOT / "outputs" / run).glob(
                f"eval_best_current_test_stratified_n32*/{family}_summary.json"))
            result = json.loads(source.read_text())
            neural_seconds = result["surrogate_wall_s"]
            if warm:
                neural_seconds = scale * np.median(warm["families"][family]["methods"][name]["seconds"])
            neural_error = 100 * result["rel_l2_eta_median_conditional_finite_tfinal"]
            ax.scatter(neural_seconds, neural_error, color=color, marker=marker,
                       s=65, edgecolors="white", linewidths=0.8, zorder=3)
            right = max(right, neural_seconds)
            print(f"{title}, {name}: runtime axis={neural_seconds:.3f}, error={neural_error:.6g}%")
        for comparison, color, label in overlays:
            if not comparison or family not in comparison["families"]:
                continue
            methods = comparison["families"][family]["methods"]
            record = methods[args.candidate_method if comparison is candidate else "fused"]
            new_seconds = float(np.median(record["seconds"]))
            ax.axvline(new_seconds, color=color, ls="--", lw=1.3, zorder=1)
            if comparison is candidate and args.candidate_previous_method:
                previous_seconds = float(np.median(methods[args.candidate_previous_method]["seconds"]))
                savings = 100 * (1 - new_seconds / previous_seconds)
                ax.text(0.03, 0.035, f"{new_seconds:.2f} s  (−{savings:.1f}%)",
                        transform=ax.transAxes, color=color, fontsize=9)
            right = max(right, new_seconds)
            print(f"{title}, {label}: {new_seconds:.3f}s; finite={record['finite']}")
        ax.set_xlim(0, 1.13 * right)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_title(f"T = {horizon:g}", loc="right", fontsize=9, color="#777f86", pad=12)

    fig.suptitle("Single-rollout runtime vs error" if warm else "Batch runtime vs error", y=0.96, fontsize=18, ha="left", x=0.10)
    fig.legend(handles=[
        Line2D([], [], color="#606970", marker="o", markerfacecolor="white", ms=4, lw=1, label="Classical"),
        *[Line2D([], [], color=color, marker=marker, ls="none", ms=7, label=f"Neural — {name}")
          for _, name, _, color, marker in reversed(RUNS)],
        Line2D([], [], color="#aab2b9", ls=":", lw=1.2, label="M6 time"),
        *[Line2D([], [], color=color, ls="--", lw=1.3, label=label)
          for comparison, color, label in overlays if comparison],
    ], loc="upper left", bbox_to_anchor=(0.09, 0.91),
        ncols=2 if candidate or baseline_free else 4, frameon=False, fontsize=10)
    xlabel = "Time per step (ms)" if per_step else "One rollout (seconds)" if warm else "32 rollouts together (seconds)"
    fig.supxlabel(xlabel, y=0.06, fontsize=11)
    fig.supylabel("Median final surface error (%)", x=0.015, y=0.50, fontsize=11)
    OUT.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        name = "12-potential-new-nn" if candidate or baseline_free else "11-single-rollout-warm" if warm else "10-neural-vs-classical"
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
