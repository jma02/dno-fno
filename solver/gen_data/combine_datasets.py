"""Combine the five generated datasets into one sample-major .npz.

Output layout (single .npz, uncompressed):
    eta:      (N, NX)  float32
    xi:       (N, NX)  float32
    gxi:      (N, NX)  float32
    depth:    (N,)     float32
    time:     (N,)     float32   (0.0 where source is linear_shallow)
    source:   (N,)     int8      enumerates source dataset (see source_legend in meta.json)
    x:        (NX,)    float32   (grid, single copy)

A meta.json sidecar lists the per-source sample counts, the source legend,
and the global shuffle seed so the same combined dataset is reproducible.

Usage:
    uv run python -m solver.gen_data.combine_datasets \\
        --output data/combined_dataset.npz --shuffle-seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.lib import format as npy_format


SOURCES: tuple[tuple[str, str], ...] = (
    ("random_sea_deep", "data/random_sea_deep.clean.npz"),
    ("random_sea_finite", "data/random_sea_finite.clean.npz"),
    ("linear", "data/linear.npz"),
    ("stokes_deep", "data/stokes_deep.npz"),
    ("stokes_finite", "data/stokes_finite.npz"),
    ("tanaka_g0", "data/tanaka_2_adaptive_g0.npz"),
    ("tanaka_g1", "data/tanaka_2_adaptive_g1.npz"),
    ("bf_g0", "data/bf_2_adaptive_g0.npz"),
    ("bf_g1", "data/bf_2_adaptive_g1.npz"),
    ("bf_modal", "data/bf_2_adaptive_modal.npz"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/combined_dataset.npz")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true",
                   help="emit samples in source-then-batch order (debugging)")
    p.add_argument("--data-root", default=".",
                   help="resolve relative source paths against this dir")
    p.add_argument("--limit-per-source", type=int, default=0,
                   help="if >0, take at most this many samples per source")
    return p.parse_args()


def load_one(path: Path) -> dict[str, np.ndarray]:
    """Concatenate every batch in one source npz into flat sample-major arrays."""
    with zipfile.ZipFile(path, mode="r") as zf:
        names = zf.namelist()
        meta = json.loads(zf.read("meta.json"))
        with zf.open("x.npy") as f:
            x = np.load(f, allow_pickle=False).astype(np.float32, copy=False)
    eta_parts: list[np.ndarray] = []
    xi_parts: list[np.ndarray] = []
    gxi_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []

    has_time = any(n.startswith("time_batch_") and n.endswith(".npy") for n in names)

    batch_ids = sorted({int(n[len("eta_batch_"):-len(".npy")])
                        for n in names if n.startswith("eta_batch_") and n.endswith(".npy")})
    with np.load(path) as archive:
        for bid in batch_ids:
            tag = f"{bid:04d}"
            eta_parts.append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            xi_parts.append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            gxi_parts.append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
            depth_parts.append(np.asarray(archive[f"depth_batch_{tag}"], dtype=np.float32))
            if has_time:
                time_parts.append(np.asarray(archive[f"time_batch_{tag}"], dtype=np.float32))

    eta = np.concatenate(eta_parts, axis=0)
    xi = np.concatenate(xi_parts, axis=0)
    gxi = np.concatenate(gxi_parts, axis=0)
    depth = np.concatenate(depth_parts, axis=0)
    if has_time:
        time = np.concatenate(time_parts, axis=0)
    else:
        time = np.zeros((eta.shape[0],), dtype=np.float32)

    # Drop non-finite rows (BF adaptive runs occasionally NaN, ~0.7% of BF)
    # and rows with extreme |gxi|. The loamp linear/stokes datasets have a
    # thin tail where shallow + steep produces |gxi| of O(10-10^3); those
    # samples blow up target_scale and destabilize training. The healthy
    # bulk of every source has |gxi| < 1, matching the modal model's regime.
    GXI_MAX = 1.0
    finite = (
        np.isfinite(eta).all(axis=1)
        & np.isfinite(xi).all(axis=1)
        & np.isfinite(gxi).all(axis=1)
        & np.isfinite(depth)
        & np.isfinite(time)
    )
    bounded = np.max(np.abs(gxi), axis=1) <= GXI_MAX
    keep = finite & bounded
    n_drop = int((~keep).sum())
    if n_drop:
        eta = eta[keep]
        xi = xi[keep]
        gxi = gxi[keep]
        depth = depth[keep]
        time = time[keep]
    return {
        "eta": eta, "xi": xi, "gxi": gxi,
        "depth": depth, "time": time, "x": x,
        "_meta": meta,
        "_n_dropped": n_drop,
    }


def write_npz_streaming(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez writes via tempfiles + concurrent zipfile; do it manually to keep
    peak memory low and fail loudly on disk-full."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays.items():
            entry = name + ".npy"
            with zf.open(entry, mode="w", force_zip64=True) as h:
                npy_format.write_array(h, np.ascontiguousarray(arr), allow_pickle=False)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    out = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Combining {len(SOURCES)} datasets into {out}")
    per_source: dict[str, dict[str, np.ndarray]] = {}
    counts: list[tuple[str, int]] = []
    x_canonical: np.ndarray | None = None

    paths = [(label, root / rel) for label, rel in SOURCES]
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing source dataset: {path}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t_load = perf_counter()
    print(f"  loading {len(paths)} sources in parallel ...")
    with ProcessPoolExecutor(max_workers=min(len(paths), 16)) as pool:
        futures = {pool.submit(load_one, p): (i, label) for i, (label, p) in enumerate(paths)}
        for fut in as_completed(futures):
            i, label = futures[fut]
            d = fut.result()
            n = d["eta"].shape[0]
            if args.limit_per_source and n > args.limit_per_source:
                n = args.limit_per_source
                for k in ("eta", "xi", "gxi", "depth", "time"):
                    d[k] = d[k][:n]
            per_source[label] = d
            drop_note = f" (dropped {d.get('_n_dropped', 0):,} non-finite)" if d.get("_n_dropped") else ""
            print(f"  [{i}] {label}: {n:,} samples{drop_note}")
    # validate shared grid (tolerate 1-ULP float32 rounding across generators)
    for label, _ in paths:
        d = per_source[label]
        if x_canonical is None:
            x_canonical = d["x"]
        else:
            if d["x"].shape != x_canonical.shape:
                raise ValueError(f"grid shape mismatch in {label}: {d['x'].shape} vs {x_canonical.shape}")
            max_diff = float(np.max(np.abs(x_canonical.astype(np.float64) - d["x"].astype(np.float64))))
            if max_diff > 1e-5:
                raise ValueError(f"grid mismatch in {label}; max |dx|={max_diff:.3e}")
    counts = [(label, per_source[label]["eta"].shape[0]) for label, _ in paths]
    print(f"  parallel load done in {perf_counter()-t_load:.1f}s")

    total = sum(n for _, n in counts)
    nx = x_canonical.shape[0]
    print(f"  total: {total:,} samples, NX={nx}")

    eta = np.empty((total, nx), dtype=np.float32)
    xi = np.empty((total, nx), dtype=np.float32)
    gxi = np.empty((total, nx), dtype=np.float32)
    depth = np.empty((total,), dtype=np.float32)
    time = np.empty((total,), dtype=np.float32)
    source = np.empty((total,), dtype=np.int8)

    cursor = 0
    for source_idx, (label, _rel) in enumerate(SOURCES):
        d = per_source[label]
        n = d["eta"].shape[0]
        sl = slice(cursor, cursor + n)
        eta[sl] = d["eta"]
        xi[sl] = d["xi"]
        gxi[sl] = d["gxi"]
        depth[sl] = d["depth"]
        time[sl] = d["time"]
        source[sl] = source_idx
        cursor += n
        # free the per-source copy now that it's been placed
        per_source[label] = {}

    if not args.no_shuffle:
        rng = np.random.default_rng(args.shuffle_seed)
        perm = rng.permutation(total)
        print(f"  shuffling {total:,} samples with seed={args.shuffle_seed}")
        eta = eta[perm]
        xi = xi[perm]
        gxi = gxi[perm]
        depth = depth[perm]
        time = time[perm]
        source = source[perm]

    legend = {idx: label for idx, (label, _) in enumerate(SOURCES)}
    meta = {
        "kind": "combined_dataset",
        "nx": int(nx),
        "length": float(2.0 * math.pi),
        "n_samples": int(total),
        "shuffled": not args.no_shuffle,
        "shuffle_seed": int(args.shuffle_seed) if not args.no_shuffle else None,
        "source_legend": legend,
        "source_counts": dict(counts),
        "fields": {
            "eta":    {"shape": [int(total), int(nx)], "dtype": "float32"},
            "xi":     {"shape": [int(total), int(nx)], "dtype": "float32"},
            "gxi":    {"shape": [int(total), int(nx)], "dtype": "float32"},
            "depth":  {"shape": [int(total)],          "dtype": "float32"},
            "time":   {"shape": [int(total)],          "dtype": "float32",
                       "note": "0.0 for linear_shallow (no time semantics)"},
            "source": {"shape": [int(total)],          "dtype": "int8",
                       "note": "see source_legend"},
            "x":      {"shape": [int(nx)],             "dtype": "float32"},
        },
    }

    print(f"  writing {out} ...")
    t0 = perf_counter()
    write_npz_streaming(out, {
        "eta": eta, "xi": xi, "gxi": gxi,
        "depth": depth, "time": time, "source": source,
        "x": x_canonical,
    })
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sz = out.stat().st_size / 1e9
    print(f"  wrote {out} ({sz:.2f} GB) in {perf_counter()-t0:.1f}s")
    print(f"  sidecar: {out.with_suffix('.meta.json')}")
    print()
    print("source counts:")
    for label, n in counts:
        print(f"  {label:24s} {n:>10,}")
    print(f"  {'TOTAL':24s} {total:>10,}")


if __name__ == "__main__":
    main()
