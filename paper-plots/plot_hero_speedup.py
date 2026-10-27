"""Figure 12d variant: the epoch-37 hero before and after the inference speedups, against classical CS orders."""

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
FAMILIES = (("stokes", "Stokes"), ("tanaka", "Tanaka"),
            ("benjamin_feir", "Benjamin–Feir"), ("jonswap_tma", "JONSWAP / TMA"))
SUFFIX = "_spectral_all_boundaries_stages_carry_local"
PUBLISHED = ("fourier_fused_features_product_overlap_nestedfused_contextbatch_projectiontranspose_fieldstride_"
             "gelureuse_branchmajor_contextreal")
OPTIMIZED = ("fourier_fused_features_product_overlap_exactg0_nestedfused_contextbatch_projectiontranspose_fieldstride_"
             "gelureuse_branchmajor_contextreal_cudacontext_cudahead_cudatail_cudaexactg1_cudafields")
HEROES = (("published", "Trained hero · epoch 37 · published inference", "hero37_pinned_baseline", PUBLISHED),
          ("optimized", "Trained hero · epoch 37 · optimized inference",
           "hero37_pinned_exactg0_cudacontext_cudahead_cudatail_cudaexactg1_cudafields", OPTIMIZED))
ORDERS = (1, 2, 3, 4, 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, default=ROOT / "outputs/hero37_fig12d_timing_20261027")
    parser.add_argument("--classical", type=Path, default=ROOT / "outputs/hero37_classical_panel_20261027")
    parser.add_argument("--output-name", default="12d-hero-optimized")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c6cbd0", "grid.color": "#edf0f2",
        "xtick.color": "#59616a", "ytick.color": "#59616a",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.12, hspace=0.38, wspace=0.27)
    for ax, (family, title) in zip(axes.flat, FAMILIES, strict=True):
        timing = json.loads((args.timings / f"{family}.json").read_text())["families"][family]
        seconds = {name.removesuffix(SUFFIX).split("_boundaries")[0]: float(np.median(record["seconds"]))
                   for name, record in timing["methods"].items()}
        sweep = json.loads((args.classical / f"{family}.json").read_text())["orders"]
        classical = [(seconds[f"M{order}"], 100 * sweep[str(order)]["median_finite"], sweep[str(order)]["finite"])
                     for order in ORDERS]
        midpoint = (seconds["M2"] + seconds["M3"]) / 2
        ax.axvline(seconds["M6"], color="#aab2b9", ls=":", lw=1.2, zorder=1)
        ax.axvline(midpoint, color="#c45b22", ls="--", lw=0.9, alpha=0.6, zorder=1)
        ax.plot(*np.asarray(classical)[:, :2].T, "o-", color="#899198", markerfacecolor="white", ms=5, lw=1)
        for order, (x, y, finite) in zip(ORDERS, classical, strict=True):
            ax.annotate(f"M{order}" + ("" if finite == 32 else f" ({finite}/32 finite)"), (x, y),
                        xytext=(6, -12 if order == 3 else -2), textcoords="offset points", color="#667078", fontsize=10)
        heroes = {}
        for key, _, evaluation, method in HEROES:
            summary = json.loads((ROOT / f"outputs/{evaluation}_{family}/{family}_summary.json").read_text())
            heroes[key] = (seconds[method], 100 * summary["rel_l2_eta_median_conditional_finite_tfinal"])
        ax.annotate("", xy=heroes["optimized"], xytext=heroes["published"],
                    arrowprops={"arrowstyle": "->", "color": "#222831", "lw": 1}, zorder=3)
        ax.scatter(*heroes["published"], s=110, marker="*", facecolors="white", edgecolors="#222831",
                   linewidths=0.9, zorder=4)
        ax.scatter(*heroes["optimized"], s=130, marker="*", color="#222831", edgecolors="white", linewidths=0.6,
                   zorder=5)
        before, after = heroes["published"][0], heroes["optimized"][0]
        ax.text(0.03, 0.085, f"Hero: {before:.2f} → {after:.2f} s ({100 * (after / before - 1):+.1f}%)",
                transform=ax.transAxes, color="#222831", fontsize=9)
        ax.text(0.03, 0.025, f"vs M2/M3 midpoint {midpoint:.2f} s: {100 * (after / midpoint - 1):+.1f}%",
                transform=ax.transAxes, color="#c45b22", fontsize=9)
        errors = [y for _, y, _ in classical] + [y for _, y in heroes.values()]
        log_low, log_high = np.log10([min(errors), max(errors)])
        span = max(log_high - log_low, 0.5)
        ax.set(yscale="log", xlim=(0, 1.12 * seconds["M6"]),
               ylim=(10**(log_low - 0.25 * span), 10**(log_high + 0.20 * span)))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", lw=0.6)
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.set_title(f"T = {timing['horizon']:g}", loc="right", fontsize=10, color="#777f86", pad=12)
        print(f"{family}: " + ", ".join(f"M{order} {x:.4f}s {y:.4g}%" for order, (x, y, _) in zip(ORDERS, classical))
              + f"; midpoint {midpoint:.4f}s; " + ", ".join(f"{key} {x:.4f}s {y:.4g}%" for key, (x, y) in heroes.items()))
    fig.suptitle("Figure 12d · Optimized hero inference", x=0.10, y=0.965, ha="left", fontsize=19)
    fig.text(0.10, 0.91, "RTX 6000 Ada · 32-case median error · Single-rollout timing · Same epoch-37 weight EMA",
             color="#59616a", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color="#899198", marker="o", mfc="white", lw=1, label="Classical"),
        Line2D([], [], color="#aab2b9", ls=":", lw=1.2, label="M6 time"),
        Line2D([], [], color="#c45b22", ls="--", lw=0.9, alpha=0.6, label="M2/M3 runtime midpoint"),
        Line2D([], [], color="#222831", marker="*", mfc="white", ls="none", ms=11, label=HEROES[0][1]),
        Line2D([], [], color="#222831", marker="*", ls="none", ms=11, label=HEROES[1][1]),
    ], loc="upper left", bbox_to_anchor=(0.09, 0.895), ncols=2, frameon=False, fontsize=10)
    fig.supxlabel("Single-rollout runtime (seconds)", y=0.045, fontsize=12)
    fig.supylabel("Median final surface error (%)", x=0.02, fontsize=12)
    stem = OUT / args.output_name
    for extension in ("png", "pdf"):
        fig.savefig(f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
