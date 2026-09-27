"""Plot measured family rollout timing and final surface relative L2 error."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "torch_ema_rollout_family_benchmark_20260926"


def main() -> None:
    report = json.loads((ROOT / "experiments" / f"{NAME}.json").read_text())
    families = list(report["families"].values())
    labels = ["Stokes", "Tanaka", "Benjamin–Feir", "JONSWAP-TMA"]
    colors = ["#277da8", "#258978", "#7461ac", "#d17b29"]
    output = ROOT / "outputs" / NAME
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.145, right=.97, bottom=.25, top=.74, wspace=.28)
    fig.suptitle("Rollout runtime and final surface error by family", x=.04, ha="left", y=.96, fontsize=19)
    fig.text(.04, .87, "Epoch-4 weight EMA · one H100 · 16 held-out trajectories per family · horizon t = 4", fontsize=11)
    timing, error = axes
    for i, (row, color) in enumerate(zip(families, colors, strict=True)):
        seconds = np.asarray(row["seconds"])
        median = row["median_seconds"]
        timing.barh(i, median, height=.48, color=color, alpha=.85)
        timing.errorbar(median, i, xerr=[[median - seconds.min()], [seconds.max() - median]],
                        color="#222222", capsize=4, linewidth=1.3)
        timing.text(seconds.max() + .18, i, f"{median:.2f} s", va="center", fontsize=10)
        errors = np.asarray(row["final_surface_relative_l2"]).mean(axis=0) * 100
        error.scatter(errors, i + np.linspace(-.18, .18, len(errors)), s=24,
                      color=color, alpha=.45, edgecolors="none")
        mean = row["mean_final_surface_error_percent"]
        error.scatter(mean, i, marker="D", s=68, color=color, edgecolors="#222222", linewidths=.6, zorder=5)
        error.annotate(f"{mean:.4f}%", (mean, i), xytext=(8, 11), textcoords="offset points", fontsize=10)
    timing.set(yticks=range(4), yticklabels=labels, xlabel="Wall time per 16-trajectory batch (seconds)")
    timing.set_xlim(0, max(max(row["seconds"]) for row in families) * 1.22)
    timing.set_title("Runtime: median of 3 warmed repeats", loc="left", fontsize=11, pad=13)
    error.set(xscale="log", yticks=range(4), yticklabels=[], xlabel="Final surface relative L2 error (%) · log scale")
    error.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    error.set_title("Error: family mean ◆ and individual cases •", loc="left", fontsize=11, pad=13)
    for ax in axes:
        ax.set_ylim(3.5, -.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.text(.04, .13, "Error = 100 × ‖predicted surface − reference surface‖₂ / ‖reference surface‖₂ at t = 4.", fontsize=10)
    fig.text(.04, .085, "Timing whiskers: repeat min–max. BF16 model; FP64 integrator; 400 GL2 steps; 51 saved frames.", fontsize=9)
    fig.text(.04, .045, "Excludes loading, input/output transfers, warmup, reference generation and plotting. These 64 cases are a validation subset.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"runtime_and_error.{extension}", dpi=180)
    plt.close(fig)
    print(output / "runtime_and_error.png")


if __name__ == "__main__":
    main()
