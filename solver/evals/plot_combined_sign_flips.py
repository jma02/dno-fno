"""Find and plot the highest eta_x sign-flip samples in the combined dataset.

Outputs:
    /tmp/combined_top_sign_flips.png       top N samples (eta, xi, gxi)
    /tmp/combined_sign_flip_histogram.png  histogram per source
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC = Path("data/combined_dataset.npz")
META = Path("data/combined_dataset.meta.json")
TOP_PLOT = Path("/tmp/combined_top_sign_flips.png")
HIST_PLOT = Path("/tmp/combined_sign_flip_histogram.png")
N_TOP_SHOW = 16
SIGN_REL_THRESHOLD = 0.05
CHUNK = 50_000


def count_thresholded_sign_changes(diff: np.ndarray, rel_threshold: float) -> np.ndarray:
    row_scales = np.max(np.abs(diff), axis=1, keepdims=True)
    thresholds = rel_threshold * row_scales
    signs = np.where(diff > thresholds, 1, np.where(diff < -thresholds, -1, 0)).astype(np.int8)
    nonzero = signs != 0
    n = diff.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = signs[i][nonzero[i]]
        if s.size > 1:
            counts[i] = int(np.sum(s[1:] != s[:-1]) + (s[-1] != s[0]))
    return counts


def main() -> None:
    meta = json.loads(META.read_text())
    legend: dict[int, str] = {int(k): v for k, v in meta["source_legend"].items()}
    nx = int(meta["nx"]); length = float(meta["length"])
    dx = length / nx
    n_total = int(meta["n_samples"])
    print(f"combined: N={n_total:,}, NX={nx}, L={length:.4f}")

    with np.load(SRC) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
        source_all = np.asarray(archive["source"], dtype=np.int8)

        all_counts = np.empty(n_total, dtype=np.int32)
        # streaming top-K with full payload
        top: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int, float]] = []
        # (sc, global_idx, eta, xi, gxi, source_id, depth)

        eta_all = archive["eta"]      # mmap
        xi_all = archive["xi"]
        gxi_all = archive["gxi"]
        depth_all = archive["depth"]

        for s in range(0, n_total, CHUNK):
            e = s + min(CHUNK, n_total - s)
            eta = np.asarray(eta_all[s:e], dtype=np.float32)
            diff = (np.roll(eta, -1, axis=-1) - eta) / dx
            sc = count_thresholded_sign_changes(diff, SIGN_REL_THRESHOLD)
            all_counts[s:e] = sc
            # promote candidates
            top_idx = np.argsort(-sc)[: N_TOP_SHOW * 2]
            xi_chunk = np.asarray(xi_all[s:e], dtype=np.float32)
            gxi_chunk = np.asarray(gxi_all[s:e], dtype=np.float32)
            depth_chunk = np.asarray(depth_all[s:e], dtype=np.float32)
            for j in top_idx:
                top.append((int(sc[j]), int(s + j),
                            eta[j], xi_chunk[j], gxi_chunk[j],
                            int(source_all[s + j]), float(depth_chunk[j])))
            top.sort(key=lambda r: -r[0])
            top = top[:N_TOP_SHOW]
            if (s // CHUNK) % 20 == 0:
                print(f"  scanned {e:>9,}/{n_total:,}  best={top[0][0]}")

    print(f"  global top: {[r[0] for r in top[:5]]}")

    # ---- top-K plot ----
    rows = N_TOP_SHOW
    fig, axes = plt.subplots(rows, 3, figsize=(13, 1.5 * rows), sharex=True)
    for r, (sc, gi, eta_r, xi_r, gxi_r, src, h) in enumerate(top):
        ax_e, ax_x, ax_g = axes[r]
        ax_e.plot(x, eta_r, color="tab:blue", lw=0.8)
        ax_x.plot(x, xi_r, color="tab:orange", lw=0.8)
        ax_g.plot(x, gxi_r, color="tab:green", lw=0.8)
        for ax in (ax_e, ax_x, ax_g):
            ax.axhline(0, color="k", lw=0.3, alpha=0.4)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax_e.set_ylabel(f"sc={sc}\n{legend[src]}\nidx={gi}\nh={h:.3g}",
                        fontsize=7, rotation=0, ha="right", va="center")
    axes[0, 0].set_title("eta", fontsize=10)
    axes[0, 1].set_title("xi", fontsize=10)
    axes[0, 2].set_title("G(eta) xi", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle(f"combined_dataset — top {rows} samples by eta_x sign-flip count "
                 f"(rel threshold {SIGN_REL_THRESHOLD})", fontsize=10)
    fig.tight_layout(rect=[0.07, 0, 1, 0.98])
    fig.savefig(TOP_PLOT, dpi=110); plt.close(fig)
    print(f"  -> {TOP_PLOT}")

    # ---- histogram per source ----
    sources = sorted(legend.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.arange(0, max(60, all_counts.max() + 2), 1)
    for sid in sources:
        mask = source_all == sid
        if not mask.any():
            continue
        ax.hist(all_counts[mask], bins=bins, histtype="step",
                density=True, label=f"{legend[sid]} (n={int(mask.sum()):,})", lw=1.2)
    ax.set_xlabel("eta_x sign-flip count")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title(f"sign-flip distribution per source (rel threshold {SIGN_REL_THRESHOLD})")
    fig.tight_layout()
    fig.savefig(HIST_PLOT, dpi=110); plt.close(fig)
    print(f"  -> {HIST_PLOT}")

    # ---- per-source quantile table ----
    print()
    print(f"{'source':<22} | {'n':>9} | {'p50':>5} {'p95':>5} {'p99':>5} {'p99.9':>5} {'max':>5}")
    print("-" * 70)
    for sid in sources:
        mask = source_all == sid
        if not mask.any():
            continue
        c = all_counts[mask]
        q = np.quantile(c, [0.5, 0.95, 0.99, 0.999])
        print(f"{legend[sid]:<22} | {int(mask.sum()):>9,} | "
              f"{int(q[0]):>5} {int(q[1]):>5} {int(q[2]):>5} {int(q[3]):>5} {int(c.max()):>5}")


if __name__ == "__main__":
    main()
