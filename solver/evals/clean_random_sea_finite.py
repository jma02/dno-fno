"""Inspect random_sea_finite for unstable rollouts and produce a NaN-filtered copy.

Outputs:
    /tmp/random_sea_finite_top_sign_flips.png   visual diagnostic
    data/random_sea_finite.clean.npz             same layout, NaN samples dropped
    data/random_sea_finite.clean.json            sidecar listing dropped (batch, idx) tuples
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC = Path("data/random_sea_finite.npz")
OUT_NPZ = Path("data/random_sea_finite.clean.npz")
OUT_JSON = Path("data/random_sea_finite.clean.json")
PLOT_PATH = Path("/tmp/random_sea_finite_top_sign_flips.png")
N_TOP_SHOW = 12
SIGN_REL_THRESHOLD = 0.05


def periodic_forward_diff(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=-1) - values) / dx


def count_thresholded_sign_changes(diff: np.ndarray, rel_threshold: float) -> np.ndarray:
    """Count direction changes in eta_x along x, ignoring magnitudes below
    rel_threshold * max-abs(eta_x) for that sample."""
    row_scales = np.max(np.abs(diff), axis=1, keepdims=True)
    thresholds = rel_threshold * row_scales
    signs = np.where(diff > thresholds, 1, np.where(diff < -thresholds, -1, 0)).astype(np.int8)
    nonzero = signs != 0
    nonzero_counts = np.count_nonzero(nonzero, axis=1)
    n = diff.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = signs[i][nonzero[i]]
        if s.size > 1:
            counts[i] = int(np.sum(s[1:] != s[:-1]) + (s[-1] != s[0]))
    return counts


def main() -> None:
    with zipfile.ZipFile(SRC) as zf:
        names = zf.namelist()
        meta = json.loads(zf.read("meta.json"))
        with zf.open("x.npy") as f:
            x = np.load(f, allow_pickle=False).astype(np.float32)
    nx = int(meta["nx"])
    L = float(meta["length"])
    dx = L / nx
    batch_ids = sorted({int(n[len("eta_batch_"):-len(".npy")])
                        for n in names if n.startswith("eta_batch_") and n.endswith(".npy")})

    bad_records: list[tuple[int, int]] = []  # (batch_id, sample_idx)
    top_candidates: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    # (sign_changes, batch_id, sample_idx_in_batch, eta_row, xi_row, gxi_row, depth)

    print(f"Scanning {SRC} ...")
    with np.load(SRC) as archive:
        for bid in batch_ids:
            tag = f"{bid:04d}"
            eta = np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32)
            xi = np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32)
            gxi = np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32)
            depth = np.asarray(archive[f"depth_batch_{tag}"], dtype=np.float32)

            bad_mask = (~np.isfinite(eta).all(axis=-1)
                        | ~np.isfinite(xi).all(axis=-1)
                        | ~np.isfinite(gxi).all(axis=-1))
            for i in np.where(bad_mask)[0]:
                bad_records.append((int(bid), int(i)))

            good_mask = ~bad_mask
            if not good_mask.any():
                continue
            eta_g = eta[good_mask]
            diff = periodic_forward_diff(eta_g, dx)
            sc = count_thresholded_sign_changes(diff, SIGN_REL_THRESHOLD)
            # keep top samples globally
            local_idx = np.where(good_mask)[0]
            order = np.argsort(-sc)[:N_TOP_SHOW * 2]
            for j in order:
                top_candidates.append((int(sc[j]), int(bid), int(local_idx[j]),
                                       eta_g[j], xi[good_mask][j], gxi[good_mask][j],
                                       float(depth[good_mask][j])))
            top_candidates.sort(key=lambda r: -r[0])
            top_candidates = top_candidates[:N_TOP_SHOW]

    print(f"  bad samples (NaN/inf in any of eta/xi/gxi): {len(bad_records)}")
    print(f"  top sign-flip count among finite samples: "
          f"{[r[0] for r in top_candidates[:5]]}")

    # ---- plot top sign-flip samples ----
    rows = N_TOP_SHOW
    fig, axes = plt.subplots(rows, 3, figsize=(13, 1.7 * rows), sharex=True)
    for r, (sc, bid, idx, eta_r, xi_r, gxi_r, h) in enumerate(top_candidates):
        ax_e, ax_x, ax_g = axes[r]
        ax_e.plot(x, eta_r, color="tab:blue", lw=0.8)
        ax_x.plot(x, xi_r, color="tab:orange", lw=0.8)
        ax_g.plot(x, gxi_r, color="tab:green", lw=0.8)
        for ax in (ax_e, ax_x, ax_g):
            ax.axhline(0, color="k", lw=0.3, alpha=0.4); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax_e.set_ylabel(f"sc={sc}\nb{bid:04d}/{idx}\nh={h:.3f}", fontsize=7, rotation=0, ha="right", va="center")
    axes[0, 0].set_title("eta", fontsize=10)
    axes[0, 1].set_title("xi", fontsize=10)
    axes[0, 2].set_title("G(eta) xi", fontsize=10)
    for ax in axes[-1]: ax.set_xlabel("x")
    fig.suptitle(f"random_sea_finite — top {rows} samples by eta_x sign-flip count "
                 f"(rel threshold {SIGN_REL_THRESHOLD}, finite samples only)", fontsize=10)
    fig.tight_layout(rect=[0.06, 0, 1, 0.97])
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=110); plt.close(fig)
    print(f"  saved diagnostic plot -> {PLOT_PATH}")

    # ---- write NaN-filtered npz ----
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SRC) as in_zf, \
         zipfile.ZipFile(OUT_NPZ, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as out_zf:
        # carry over x and meta
        with in_zf.open("x.npy") as f:
            x_arr = np.load(f, allow_pickle=False)
        with out_zf.open("x.npy", mode="w", force_zip64=True) as h:
            npy_format.write_array(h, np.asarray(x_arr), allow_pickle=False)
        new_meta = dict(meta)
        new_meta["filtered_from"] = str(SRC)
        new_meta["nan_filter_relative_threshold"] = SIGN_REL_THRESHOLD
        new_meta["dropped_samples"] = len(bad_records)
        out_zf.writestr("meta.json", json.dumps(new_meta, indent=2))

        with np.load(SRC) as archive:
            kept_total = 0
            for bid in batch_ids:
                tag = f"{bid:04d}"
                eta = np.asarray(archive[f"eta_batch_{tag}"])
                xi = np.asarray(archive[f"xi_batch_{tag}"])
                gxi = np.asarray(archive[f"gxi_batch_{tag}"])
                depth = np.asarray(archive[f"depth_batch_{tag}"])
                time = np.asarray(archive[f"time_batch_{tag}"])
                cid = np.asarray(archive[f"case_id_batch_{tag}"])
                specs_raw = archive[f"specs_batch_{tag}.json"]
                # specs may be a 0-dim numpy array of bytes; pull as text
                if hasattr(specs_raw, "tobytes"):
                    specs_text = specs_raw.tobytes().decode("utf-8")
                elif isinstance(specs_raw, bytes):
                    specs_text = specs_raw.decode("utf-8")
                else:
                    specs_text = str(specs_raw)
                specs = json.loads(specs_text)

                good = (np.isfinite(eta).all(axis=-1)
                        & np.isfinite(xi).all(axis=-1)
                        & np.isfinite(gxi).all(axis=-1))
                if good.all():
                    eta_o, xi_o, gxi_o, depth_o = eta, xi, gxi, depth
                    time_o, cid_o, specs_o = time, cid, specs
                else:
                    keep_idx = np.where(good)[0]
                    eta_o = eta[keep_idx]; xi_o = xi[keep_idx]; gxi_o = gxi[keep_idx]
                    depth_o = depth[keep_idx]; time_o = time[keep_idx]; cid_o = cid[keep_idx]
                    specs_o = [specs[i] for i in keep_idx.tolist()] if isinstance(specs, list) and len(specs) == eta.shape[0] else specs

                kept_total += eta_o.shape[0]
                for name, arr in (("eta", eta_o), ("xi", xi_o), ("gxi", gxi_o),
                                   ("depth", depth_o), ("time", time_o), ("case_id", cid_o)):
                    with out_zf.open(f"{name}_batch_{tag}.npy", mode="w", force_zip64=True) as h:
                        npy_format.write_array(h, np.asarray(arr), allow_pickle=False)
                out_zf.writestr(f"specs_batch_{tag}.json", json.dumps(specs_o, indent=2))

    OUT_JSON.write_text(json.dumps({"dropped_samples": bad_records,
                                    "kept_total": kept_total,
                                    "source_total": sum(int(np.asarray(np.load(SRC)[f'eta_batch_{bid:04d}']).shape[0]) for bid in batch_ids)},
                                   indent=2), encoding="utf-8")
    print(f"  wrote filtered: {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e9:.2f} GB)  kept={kept_total}")
    print(f"  drop list:     {OUT_JSON}")


if __name__ == "__main__":
    main()
