"""Compute per-family (h, kh_peak, a/h) coverage in physical coordinates.

Pure analysis on data/combined_dataset_v5.npz. No GPU, no model.

Strategy: load only the 1D arrays (depth, time, source) up front from the
STORED npy members in the zip. Pick a sample of row indices per family
(preferring time==0 rows when available). For each sampled row, seek into
the STORED eta.npy member and read just that row to compute the spectrum.
"""

from __future__ import annotations

import json
import os
import struct
import time
import zipfile
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # avoid accidental jax GPU import

import numpy as np
from numpy.lib import format as npy_format


NPZ_PATH = Path("data/combined_dataset_v5.npz")
META_PATH = Path("data/combined_dataset_v5.meta.json")
OUT_NPZ = Path("notes/family_coverage_v5.npz")
OUT_JSON = Path("notes/family_coverage_v5.json")
OUT_FIG = Path("notes/figures/family_coverage_v5.png")

NX = 1024
L = 2.0 * np.pi
SAMPLES_PER_FAMILY = 5000
SEED = 20260616

FAMILY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]


class FamilyCoverage(TypedDict):
    name: str
    indices: np.ndarray
    h: np.ndarray
    kh_peak: np.ndarray
    a_over_h: np.ndarray
    n_t0: int


def payload_start_abs(path: Path, member: str) -> tuple[int, tuple[int, ...], np.dtype]:
    """Return the absolute byte offset of the npy payload of `member`
    inside the (STORED) zip, plus the npy shape and dtype.

    zipfile.ZipExtFile.seek() on STORED entries is implemented as a re-decompress
    walk, which is O(distance) -- catastrophically slow for far random seeks.
    Seeking the underlying file handle directly is O(1).
    """
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} is not STORED; cannot raw-seek")
    with open(path, "rb") as fp:
        fp.seek(info.header_offset)
        sig = fp.read(4)
        if sig != b"PK\x03\x04":
            raise ValueError(f"bad local file header at offset {info.header_offset}")
        raw = fp.read(26)
        fnlen = struct.unpack("<H", raw[22:24])[0]
        exlen = struct.unpack("<H", raw[24:26])[0]
        data_offset = info.header_offset + 30 + fnlen + exlen
        fp.seek(data_offset)
        version = npy_format.read_magic(fp)
        reader = {
            (1, 0): npy_format.read_array_header_1_0,
            (2, 0): npy_format.read_array_header_2_0,
        }[version]
        shape, _, dtype = reader(fp)
        payload_off = fp.tell()
    return payload_off, shape, dtype


def read_rows(
    path: Path,
    payload_off: int,
    row_indices: np.ndarray,
    row_bytes: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Random-access rows from a STORED npy member by raw-file seeking.

    row_indices must be 1D int array (sorted is faster on cold cache, but for
    a STORED zip on a fast filesystem it does not really matter). Returns
    (len(row_indices), row_bytes/itemsize).
    """
    n_rows = len(row_indices)
    out = np.empty((n_rows, row_bytes // dtype.itemsize), dtype=dtype)
    with open(path, "rb") as fp:
        for i, ridx in enumerate(row_indices):
            fp.seek(payload_off + int(ridx) * row_bytes)
            out[i] = np.frombuffer(fp.read(row_bytes), dtype=dtype)
    return out


def pick_indices(
    rng: np.random.Generator, source_id: int, source: np.ndarray, t: np.ndarray, n: int
) -> np.ndarray:
    """Pick up to n row indices for the given family. Prefer time==0 rows."""
    mask = source == source_id
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    t0_idx = idx[t[idx] == 0.0]
    take_n = min(n, idx.size)
    # If we have enough t=0 ICs, use them only
    if t0_idx.size >= take_n:
        return rng.choice(t0_idx, size=take_n, replace=False)
    # Otherwise prefer all t=0 + top up with non-t=0
    if t0_idx.size > 0:
        rest = np.setdiff1d(idx, t0_idx, assume_unique=True)
        extra_n = take_n - t0_idx.size
        extra = rng.choice(rest, size=min(extra_n, rest.size), replace=False)
        return np.concatenate([t0_idx, extra])
    # No t=0 at all (stokes_*)
    return rng.choice(idx, size=take_n, replace=False)


def compute_features(
    eta_rows: np.ndarray, depth_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (h, kh_peak, a_over_h) per row."""
    # a = sup |eta|
    a = np.abs(eta_rows).max(axis=1)
    h = depth_rows.astype(np.float64)
    aoh = a.astype(np.float64) / h
    # rfft along the spatial axis. k index = bin index; k_phys = idx since L = 2pi
    eta_hat = np.fft.rfft(eta_rows, axis=1)
    mag = np.abs(eta_hat)
    mag[:, 0] = 0.0  # skip k=0
    k_idx = mag.argmax(axis=1).astype(np.float64)
    # k = 2*pi*idx/L = idx because L = 2*pi
    k_peak = k_idx
    kh = k_peak * h
    return h, kh, aoh


def main() -> None:
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    print(f"Loading 1D arrays from {NPZ_PATH} ...")
    with zipfile.ZipFile(NPZ_PATH) as zf:
        # depth, time are <f4 (1D); source is <i1 (1D). Just np.load is fine; 84 MB total.
        with zf.open("depth.npy") as f:
            depth = np.load(f)
        with zf.open("time.npy") as f:
            tt = np.load(f)
        with zf.open("source.npy") as f:
            source = np.load(f)
    eta_payload_off, eta_shape, eta_dtype = payload_start_abs(NPZ_PATH, "eta.npy")
    print(f"  depth: {depth.shape} {depth.dtype}")
    print(f"  time:  {tt.shape} {tt.dtype}")
    print(f"  source: {source.shape} {source.dtype}")
    print(
        f"  eta header: shape={eta_shape} dtype={eta_dtype} payload_off={eta_payload_off}"
    )

    row_bytes = NX * eta_dtype.itemsize
    legend: dict[str, str] = json.loads(META_PATH.read_text())["source_legend"]

    per_family: dict[int, FamilyCoverage] = {}

    for fid in FAMILY_IDS:
        name = legend[str(fid)]
        idx = pick_indices(rng, fid, source, tt, SAMPLES_PER_FAMILY)
        if idx.size == 0:
            print(f"[{fid:>2}] {name}: no rows; skipping")
            continue
        # Sort indices for sequential-ish seeks (mild win for OS readahead)
        idx_sorted = np.sort(idx)
        t_read = time.perf_counter()
        eta_rows = read_rows(
            NPZ_PATH, eta_payload_off, idx_sorted, row_bytes, eta_dtype
        )
        h, kh, aoh = compute_features(eta_rows, depth[idx_sorted])
        dt_read = time.perf_counter() - t_read
        n_t0 = int(np.count_nonzero(tt[idx_sorted] == 0.0))
        print(
            f"[{fid:>2}] {name:<22s}: n={idx_sorted.size} (t0={n_t0}) read {dt_read:.1f}s"
        )
        per_family[fid] = {
            "name": name,
            "indices": idx_sorted,
            "h": h.astype(np.float32),
            "kh_peak": kh.astype(np.float32),
            "a_over_h": aoh.astype(np.float32),
            "n_t0": n_t0,
        }

    # Persist arrays
    save_kwargs: dict[str, np.ndarray] = {}
    json_summary: dict[str, dict[str, object]] = {}
    for fid, d in per_family.items():
        pfx = f"f{fid:02d}_{d['name']}"
        save_kwargs[f"{pfx}__h"] = d["h"]
        save_kwargs[f"{pfx}__kh"] = d["kh_peak"]
        save_kwargs[f"{pfx}__aoh"] = d["a_over_h"]
        save_kwargs[f"{pfx}__idx"] = d["indices"]
        h = d["h"]
        kh = d["kh_peak"]
        aoh = d["a_over_h"]
        json_summary[str(fid)] = {
            "name": d["name"],
            "n_sampled": int(h.size),
            "n_t0": d["n_t0"],
            "h_min": float(np.min(h)),
            "h_max": float(np.max(h)),
            "kh_p05": float(np.percentile(kh, 5)),
            "kh_p50": float(np.percentile(kh, 50)),
            "kh_p95": float(np.percentile(kh, 95)),
            "aoh_p05": float(np.percentile(aoh, 5)),
            "aoh_p50": float(np.percentile(aoh, 50)),
            "aoh_p95": float(np.percentile(aoh, 95)),
        }
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, allow_pickle=False, **save_kwargs)
    OUT_JSON.write_text(json.dumps(json_summary, indent=2))
    print(f"Wrote {OUT_NPZ}")
    print(f"Wrote {OUT_JSON}")

    # ---- table ----
    print()
    hdr = (
        f"{'id':>3} {'family':<22s} {'n':>5} {'h_min':>9s} {'h_max':>9s} "
        f"{'kh_p05':>8s} {'kh_p95':>8s} {'aoh_p05':>9s} {'aoh_p95':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for fid in FAMILY_IDS:
        if fid not in per_family:
            continue
        d = per_family[fid]
        h = d["h"]
        kh = d["kh_peak"]
        aoh = d["a_over_h"]
        print(
            f"{fid:>3} {d['name']:<22s} {h.size:>5d} {float(np.min(h)):>9.4f} {float(np.max(h)):>9.4f} "
            f"{float(np.percentile(kh, 5)):>8.3f} {float(np.percentile(kh, 95)):>8.3f} "
            f"{float(np.percentile(aoh, 5)):>9.4f} {float(np.percentile(aoh, 95)):>9.4f}"
        )

    # ---- figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    palette = [matplotlib.colormaps["tab20"](i / 19.0) for i in range(20)]
    # Map each family to its own distinct color from tab20 (skip lighter pairs for clarity)
    # Pick the 12 strong (even-indexed) colors first.
    strong = [palette[i] for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3)]
    color_map: dict[int, tuple[float, float, float, float]] = {}
    for i, fid in enumerate([f for f in FAMILY_IDS if f in per_family]):
        color_map[fid] = to_rgba(strong[i % len(strong)], alpha=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        r"$\log_{10}(h)$ vs $\log_{10}(a/h)$",
        r"$\log_{10}(h)$ vs $\log_{10}(kh_{\mathrm{peak}})$",
        r"$\log_{10}(kh_{\mathrm{peak}})$ vs $\log_{10}(a/h)$",
    ]
    xlabels = [r"$\log_{10}(h)$", r"$\log_{10}(h)$", r"$\log_{10}(kh_{\mathrm{peak}})$"]
    ylabels = [
        r"$\log_{10}(a/h)$",
        r"$\log_{10}(kh_{\mathrm{peak}})$",
        r"$\log_{10}(a/h)$",
    ]

    eps = 1e-12
    for fid in FAMILY_IDS:
        if fid not in per_family:
            continue
        d = per_family[fid]
        h = np.asarray(d["h"], dtype=np.float64)
        kh = np.asarray(d["kh_peak"], dtype=np.float64)
        aoh = np.asarray(d["a_over_h"], dtype=np.float64)
        lh = np.log10(np.maximum(h, eps))
        lkh = np.log10(np.maximum(kh, eps))
        laoh = np.log10(np.maximum(aoh, eps))
        c = color_map[fid]
        label = f"{fid}:{d['name']}"
        axes[0].scatter(lh, laoh, s=2, alpha=0.3, color=c, label=label, linewidths=0)
        axes[1].scatter(lh, lkh, s=2, alpha=0.3, color=c, label=label, linewidths=0)
        axes[2].scatter(lkh, laoh, s=2, alpha=0.3, color=c, label=label, linewidths=0)

    for ax, title, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)

    # legend with full-opacity markers; place outside on the right of axes[2]
    handles = []
    labels = []
    for fid in FAMILY_IDS:
        if fid not in per_family:
            continue
        d = per_family[fid]
        c = color_map[fid]
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=c, markersize=6)
        )
        labels.append(f"{fid}: {d['name']}")
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        fontsize=9,
        title="family",
    )
    fig.suptitle(
        "v5 combined dataset: per-family coverage in (h, kh_peak, a/h)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
