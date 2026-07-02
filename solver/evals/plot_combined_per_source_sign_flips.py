"""Single-process scan of combined_dataset.npz: top-K eta_x sign-flip samples per source.

Vectorized sign-flip counting (no per-row Python loop) and we capture full row
payloads during the scan itself so the renderer doesn't have to revisit the
65 GB npz file (NpzFile.__getitem__ has no row-level access; each lookup
re-reads the full channel).

Outputs:
    /tmp/combined_top_sign_flips_per_source.png   one row per source
    /tmp/combined_sign_flip_quantiles.txt         per-source quantile table
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC = Path("data/combined_dataset.npz")
META = Path("data/combined_dataset.meta.json")
OUT_PNG = Path("/tmp/combined_top_sign_flips_per_source.png")
QUANT_TXT = Path("/tmp/combined_sign_flip_quantiles.txt")
PROGRESS = Path("/tmp/combined_sign_flip_progress.txt")

N_TOP_PER_SOURCE = 8
SIGN_REL_THRESHOLD = 0.05
CHUNK = 50_000


def count_sign_changes_vec(diff: np.ndarray, rel_threshold: float) -> np.ndarray:
    """Count thresholded sign-flips per row, vectorized.

    A sign-flip is a transition between consecutive *nonzero* signs along the
    row (with periodic wrap-around). We forward-fill zero positions with the
    previous nonzero sign so adjacent comparisons handle runs of small values.
    """
    n, w = diff.shape
    row_scales = np.max(np.abs(diff), axis=1, keepdims=True)
    row_scales = np.where(row_scales > 0, row_scales, 1.0)
    thresholds = rel_threshold * row_scales
    signs = np.where(diff > thresholds, 1, np.where(diff < -thresholds, -1, 0)).astype(np.int8)

    mask = signs != 0
    idx = np.broadcast_to(np.arange(w, dtype=np.int32), (n, w))
    last_nz_idx = np.where(mask, idx, np.int32(-1))
    last_nz_idx = np.maximum.accumulate(last_nz_idx, axis=1)
    safe_idx = np.maximum(last_nz_idx, 0)
    filled = np.take_along_axis(signs, safe_idx, axis=1)
    filled = np.where(last_nz_idx >= 0, filled, np.int8(0))

    a = filled[:, :-1]; b = filled[:, 1:]
    transitions = ((a != b) & (a != 0) & (b != 0)).sum(axis=1)
    wrap = ((filled[:, -1] != filled[:, 0]) & (filled[:, -1] != 0) & (filled[:, 0] != 0)).astype(np.int32)
    return transitions.astype(np.int32) + wrap


def main() -> None:
    meta = json.loads(META.read_text())
    legend: dict[int, str] = {int(k): v for k, v in meta["source_legend"].items()}
    nx = int(meta["nx"]); length = float(meta["length"])
    dx = length / nx
    n_total = int(meta["n_samples"])
    n_chunks = (n_total + CHUNK - 1) // CHUNK
    print(f"combined: N={n_total:,}, NX={nx}, L={length:.4f}, chunks={n_chunks}", flush=True)

    archive = np.load(SRC, mmap_mode="r")
    eta_all = archive["eta"]
    xi_all = archive["xi"]
    gxi_all = archive["gxi"]
    src_all = archive["source"]
    depth_all = archive["depth"]
    x = np.asarray(archive["x"], dtype=np.float32)

    # tops[sid] keeps the running best N_TOP rows. Each entry is
    # (sc, gi, eta_row, xi_row, gxi_row, depth)
    tops: dict[int, list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]]] = {sid: [] for sid in legend}
    quant_buckets: dict[int, list[np.ndarray]] = {sid: [] for sid in legend}

    t_start = time.time()
    for ci, s in enumerate(range(0, n_total, CHUNK)):
        e = min(s + CHUNK, n_total)
        eta = np.asarray(eta_all[s:e], dtype=np.float32)
        src_chunk = np.asarray(src_all[s:e], dtype=np.int8)
        diff = (np.roll(eta, -1, axis=-1) - eta) / dx
        sc = count_sign_changes_vec(diff, SIGN_REL_THRESHOLD)

        # group by source within the chunk to extract top-K candidates cheaply
        unique_src = np.unique(src_chunk)
        xi_chunk = None  # lazy — only read when a source has a contender

        for sid in unique_src:
            mask = src_chunk == sid
            local_idx = np.where(mask)[0]
            local_sc = sc[local_idx]
            quant_buckets[int(sid)].append(local_sc.astype(np.int32))
            order = np.argsort(-local_sc)[:N_TOP_PER_SOURCE * 2]
            sid_int = int(sid)
            current_min = (
                tops[sid_int][-1][0] if len(tops[sid_int]) >= N_TOP_PER_SOURCE else -1
            )
            best_in_chunk = int(local_sc[order[0]]) if order.size else -1
            if best_in_chunk <= current_min and len(tops[sid_int]) >= N_TOP_PER_SOURCE:
                continue  # nothing here can displace the running top-K

            if xi_chunk is None:
                xi_chunk = np.asarray(xi_all[s:e], dtype=np.float32)
                gxi_chunk = np.asarray(gxi_all[s:e], dtype=np.float32)
                depth_chunk = np.asarray(depth_all[s:e], dtype=np.float32)

            for j in order:
                row_in_chunk = int(local_idx[j])
                cand = (
                    int(local_sc[j]),
                    int(s + row_in_chunk),
                    eta[row_in_chunk].copy(),
                    xi_chunk[row_in_chunk].copy(),
                    gxi_chunk[row_in_chunk].copy(),
                    float(depth_chunk[row_in_chunk]),
                )
                tops[sid_int].append(cand)
            tops[sid_int].sort(key=lambda r: -r[0])
            tops[sid_int] = tops[sid_int][:N_TOP_PER_SOURCE]

        if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            dt = time.time() - t_start
            eta_total = dt * n_chunks / (ci + 1) - dt
            line = (f"  chunk {ci+1}/{n_chunks} ({e:,}/{n_total:,})  "
                    f"elapsed {dt:.0f}s  eta {eta_total:.0f}s")
            print(line, flush=True)
            PROGRESS.write_text(line + "\n", encoding="utf-8")

    print("  scan done; rendering top-K figure ...", flush=True)
    sources_sorted = sorted(legend.keys())
    rows = sum(len(tops[sid]) for sid in sources_sorted)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 1.4 * rows), sharex=True)
    r = 0
    for sid in sources_sorted:
        for sc_val, gi, eta_r, xi_r, gxi_r, h in tops[sid]:
            ax_e, ax_x, ax_g = axes[r]
            ax_e.plot(x, eta_r, color="tab:blue", lw=0.8)
            ax_x.plot(x, xi_r, color="tab:orange", lw=0.8)
            ax_g.plot(x, gxi_r, color="tab:green", lw=0.8)
            for ax in (ax_e, ax_x, ax_g):
                ax.axhline(0, color="k", lw=0.3, alpha=0.4)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            ax_e.set_ylabel(f"sc={sc_val}\n{legend[sid]}\nidx={gi}\nh={h:.3g}",
                            fontsize=6.5, rotation=0, ha="right", va="center")
            r += 1
    axes[0, 0].set_title("eta", fontsize=10)
    axes[0, 1].set_title("xi", fontsize=10)
    axes[0, 2].set_title("G(eta) xi", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle(f"combined_dataset — top {N_TOP_PER_SOURCE} per source by eta_x sign-flip "
                 f"(rel threshold {SIGN_REL_THRESHOLD})", fontsize=10)
    fig.tight_layout(rect=[0.10, 0, 1, 0.99])
    fig.savefig(OUT_PNG, dpi=110); plt.close(fig)
    print(f"  -> {OUT_PNG}", flush=True)

    lines = [f"{'source':<22} | {'n':>10} | {'p50':>5} {'p95':>5} {'p99':>5} {'p99.9':>5} {'max':>5}",
             "-" * 80]
    for sid in sources_sorted:
        bucket = quant_buckets[sid]
        if not bucket:
            continue
        c = np.concatenate(bucket)
        q = np.quantile(c, [0.5, 0.95, 0.99, 0.999])
        lines.append(f"{legend[sid]:<22} | {c.size:>10,} | "
                     f"{int(q[0]):>5} {int(q[1]):>5} {int(q[2]):>5} {int(q[3]):>5} {int(c.max()):>5}")
    QUANT_TXT.write_text("\n".join(lines) + "\n")
    print(f"  -> {QUANT_TXT}", flush=True)
    print()
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
