"""Plot saved rollout comparisons; no training, inference, or new timing runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-plots/figures"
RUNS = (
    ("c27_tanaka_hard128_full_equal_local_20260918", "N320", "#8b5ea8", "s"),
    ("c27_branches16_tanaka_hard128_20260923", "N16", "#087eaa", "o"),
)
FAMILIES = (("stokes", "Stokes"), ("tanaka", "Tanaka"),
            ("benjamin_feir", "Benjamin–Feir"), ("jonswap_tma", "JONSWAP / TMA"))


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#bcc2c9", "grid.color": "#e6e9ec", "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.765, hspace=0.44, wspace=0.22)
    for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
        with np.load(ROOT / "outputs/cs_order_sweep_20260923" / f"{family}.npz") as data:
            selected = np.argsort(data["orders"])
            orders = data["orders"][selected]
            seconds = np.median(data["timings_s"][selected], axis=1)
            errors = np.median(data["eta_rel_l2"][selected, -1], axis=1)
            horizon = float(data["times"][-1])
        ax.set(xscale="log", xlim=(0.06, 3.2), ylim=(0, 1600))
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
        ax.fill_between([1, 3.2], 1, 1600, color="#dff1e5", zorder=0)
        ax.axvline(1, color="#55616d", lw=1.1)
        ax.axhline(1, color="#55616d", lw=1.1)
        ax.set_xticks([0.1, 0.25, 0.5, 1, 2], ["0.1×", "0.25×", "0.5×", "1×", "2×"])
        ax.set_yticks([0, 0.1, 1, 10, 100, 1000], ["0", "0.1×", "1×", "10×", "100×", "1000×"])
        ax.minorticks_off()
        ax.grid(alpha=0.75, zorder=0)
        ax.text(0.97, 0.96, "NEURAL BETTER\nON BOTH", transform=ax.transAxes,
                ha="right", va="top", color="#26734c", fontsize=9, weight="bold")
        notes = []
        for run, name, color, marker in RUNS:
            source = next((ROOT / "outputs" / run).glob(
                f"eval_best_current_test_stratified_n32*/{family}_summary.json"))
            result = json.loads(source.read_text())
            neural_seconds = result["surrogate_wall_s"]
            neural_error = result["rel_l2_eta_median_conditional_finite_tfinal"]
            speed_gain, accuracy_gain = seconds / neural_seconds, errors / neural_error
            ax.plot(speed_gain, accuracy_gain, color=color, alpha=0.45, lw=1, zorder=2)
            ax.scatter(speed_gain, accuracy_gain, color=color, marker=marker,
                       s=43, edgecolors="white", linewidths=0.8, zorder=3, clip_on=False)
            for order, x, y in zip(orders, speed_gain, accuracy_gain, strict=True):
                ax.annotate(f"M{order}", (x, y), xytext=(0, 7), textcoords="offset points",
                            ha="center", color=color, fontsize=8, zorder=4)
            notes.append(f"{name}: {neural_seconds:.1f} s / {100 * neural_error:.3g}%")
            winners = orders[(speed_gain > 1) & (accuracy_gain > 1)]
            print(f"{title} {name}: lower recorded cost AND median error vs {winners.tolist()}")
            if family == "benjamin_feir" and name == "N16":
                index = int(np.flatnonzero(orders == 3)[0])
                ax.annotate(
                    f"N16 vs M3\n{speed_gain[index]:.2f}× time gain\n{accuracy_gain[index]:.2f}× error gain",
                    (speed_gain[index], accuracy_gain[index]), xytext=(1.35, 6),
                    color="#226343", fontsize=9, va="bottom",
                    arrowprops={"arrowstyle": "->", "color": "#226343", "lw": 1},
                )
        ax.set_title(f"{title}  ·  T = {horizon:g}", loc="left", fontsize=12, weight="bold", pad=27)
        ax.text(0, 1.035, "    ".join(notes), transform=ax.transAxes, fontsize=8.5, color="#56616c")

    fig.suptitle("Where does the neural DNO beat classical truncation?", y=0.975,
                 fontsize=19, weight="bold", ha="left", x=0.035)
    fig.text(0.035, 0.932, "Right of 1× = lower recorded runtime.  Above 1× = lower median surface error.", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, lw=1,
                              label=f"{name}: {16 if name == 'N16' else 320} branches/group" + (" (new)" if name == "N16" else ""))
                        for _, name, color, marker in reversed(RUNS)],
               loc="upper left", bbox_to_anchor=(0.028, 0.916), ncols=2, frameon=False)
    fig.text(0.035, 0.85,
             "PROVISIONAL TIMING: classical excludes compilation/transfers; neural includes them. Not a matched speed benchmark.",
             fontsize=10, color="#88590c", bbox={"facecolor": "#fff4df", "edgecolor": "none", "pad": 7})
    fig.supxlabel("Recorded speed gain = classical runtime / neural runtime  →", y=0.095, fontsize=12)
    fig.supylabel("Accuracy gain = classical median error / neural median error  →", x=0.015, y=0.48, fontsize=12)
    fig.text(0.035, 0.060, "Same 32 held-out cases / family · batch-32 GPU rollout · terminal surface relative L² error vs M6 · N = 1024 · dt = 0.01", fontsize=9)
    fig.text(0.035, 0.039, "M6 has zero reference self-error, not zero exact-solution error. X is logarithmic; Y is linear below 0.1× and logarithmic above.", fontsize=9)
    fig.text(0.035, 0.018, "Panel subtitles give each neural model’s recorded seconds / median error (%). Lines only connect tested orders; they are not fitted curves.", fontsize=9)
    OUT.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"10-neural-vs-classical.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
