"""Convert legacy `data/dno_dataset.npz` (L=164, h=1) into a test set on the
canonical L=2π, h≈0.0383 domain.

We apply the *Zakharov-invariant* rescaling (x, y, t) -> (alpha X, alpha Y, sqrt(alpha) tau)
with g preserved. This keeps numerical magnitudes in the same units as our
training data (Zakharov-generated on L=2pi with g=1). The static-BVP rescaling
in notes/rescaling/rescaling.tex (eta /= a, xi unchanged, gxi *= a) is also geometrically
self-consistent but leaves xi at legacy magnitudes (~300x our training xi).

For alpha = 164 / (2pi):
    eta_new   = eta_old / alpha               ([eta] = L)
    xi_new    = xi_old  / alpha^(3/2)         ([xi]  = L^2/T)
    gxi_new   = gxi_old / alpha^(1/2)         ([G(eta)xi] = L/T)
    h_new     = h_old   / alpha               (depth scaled with y)
    x grid    = same NX=1024, just relabeled to span [0, 2pi]

Self-consistency: G is linear in xi, so scaling xi -> xi/a^(3/2) and gxi -> gxi/a^(1/2)
preserves G(eta_new)xi_new = gxi_new on the new domain (the static BVP closes
under any common multiplicative factor on (xi, gxi)).

The legacy file stores soliton/stokes/linear sub-arrays with no batch structure
and no recorded depth. We assume h_old=1.0 (Tanaka-style convention) and tag
each sample with a `source` label so it can be evaluated separately.

Output schema mirrors `combine_datasets.py`:
    eta, xi, gxi   (N, NX)  float32
    depth          (N,)     float32   (constant 1/α ≈ 0.0383)
    time           (N,)     float32   (0.0; legacy file has none)
    source         (N,)     int8      0=soliton, 1=stokes, 2=linear
    x              (NX,)    float32   on [0, 2π]
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


# Each entry: (label, eta_npy, xi_npy, gxi_npy, h_old). The legacy MATLAB stokes
# generator runs both ichoi=0 (deep, h_use=1000) and ichoi=1 (finite, h=1) and
# concatenates them; regenerate_dno_stokes.py splits them into stokes_deep_* and
# stokes_finite_* with their true physical depths so the rescaled labels are correct.
SUBSETS: tuple[tuple[str, str, str, str, float], ...] = (
    ("soliton",       "soliton_eta.npy",        "soliton_xi.npy",        "soliton_Gxi.npy",        1.0),
    ("stokes_deep",   "stokes_deep_eta.npy",    "stokes_deep_xi.npy",    "stokes_deep_Gxi.npy",    1000.0),
    ("stokes_finite", "stokes_finite_eta.npy",  "stokes_finite_xi.npy",  "stokes_finite_Gxi.npy",  1.0),
    ("linear",        "linear_eta.npy",         "linear_xi.npy",         "linear_Gxi.npy",         1.0),
)

L_OLD = 164.0
L_NEW = 2.0 * math.pi
ALPHA = L_OLD / L_NEW                # ≈ 26.10
ALPHA_HALF = ALPHA ** 0.5            # ≈ 5.109
ALPHA_THREE_HALF = ALPHA ** 1.5      # ≈ 133.35


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/dno_dataset.npz")
    p.add_argument("--output", default="data/test_dno_rescaled.npz")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="0 to disable shuffle; otherwise interleave subsets")
    return p.parse_args()


def load_subset(zf: zipfile.ZipFile, eta_name: str, xi_name: str, gxi_name: str, h_old: float) -> dict[str, np.ndarray | float]:
    with zf.open(eta_name) as f:
        eta = np.load(f, allow_pickle=False).astype(np.float32, copy=False)
    with zf.open(xi_name) as f:
        xi = np.load(f, allow_pickle=False).astype(np.float32, copy=False)
    with zf.open(gxi_name) as f:
        gxi = np.load(f, allow_pickle=False).astype(np.float32, copy=False)
    return {"eta": eta, "xi": xi, "gxi": gxi, "h_old": float(h_old)}


def write_npz_streaming(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", mode="w", force_zip64=True) as h:
                npy_format.write_array(h, np.ascontiguousarray(arr), allow_pickle=False)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    print(f"Reading legacy test set: {in_path}")
    print(f"  rescaling: alpha = L_old / L_new = {L_OLD} / {L_NEW:.6f} = {ALPHA:.4f}")

    counts: list[tuple[str, int]] = []
    subset_payloads: list[dict[str, object]] = []
    with zipfile.ZipFile(in_path, mode="r") as zf:
        names = set(zf.namelist())
        if "x.npy" not in names:
            raise RuntimeError("expected x.npy in legacy dno_dataset.npz")
        with zf.open("x.npy") as f:
            x_old = np.load(f, allow_pickle=False).astype(np.float64, copy=False)
        for label, eta_n, xi_n, gxi_n, h_old in SUBSETS:
            if eta_n not in names:
                print(f"  skip {label} (not present)")
                continue
            t0 = perf_counter()
            d = load_subset(zf, eta_n, xi_n, gxi_n, h_old)
            n = d["eta"].shape[0]
            print(f"  {label:14s}  loaded {n:,} samples (h_old={h_old}) in {perf_counter()-t0:.1f}s")
            d["label"] = label
            subset_payloads.append(d)
            counts.append((label, n))

    # Sanity-check the legacy grid: should span [0, L_old) with NX=1024
    nx = x_old.shape[0]
    spacing = float(x_old[1] - x_old[0])
    span = float(spacing * nx)
    if not math.isclose(span, L_OLD, rel_tol=1e-4):
        print(f"  WARN: legacy x.npy spans {span:.4f}, expected {L_OLD}")

    total = sum(n for _, n in counts)
    print(f"  total samples: {total:,}, NX={nx}")

    eta = np.empty((total, nx), dtype=np.float32)
    xi  = np.empty((total, nx), dtype=np.float32)
    gxi = np.empty((total, nx), dtype=np.float32)
    source = np.empty((total,), dtype=np.int8)
    depth = np.empty((total,), dtype=np.float32)

    cursor = 0
    label_to_idx = {label: i for i, (label, _, _, _, _) in enumerate(SUBSETS)}
    for d in subset_payloads:
        n = d["eta"].shape[0]
        sl = slice(cursor, cursor + n)
        # Apply Zakharov-invariant rescaling under (x,y,t) -> (a X, a Y, sqrt(a) tau):
        #   eta /= a, xi /= a^(3/2), gxi /= a^(1/2), h /= a
        eta[sl] = (d["eta"] / ALPHA).astype(np.float32, copy=False)
        xi[sl]  = (d["xi"]  / ALPHA_THREE_HALF).astype(np.float32, copy=False)
        gxi[sl] = (d["gxi"] / ALPHA_HALF).astype(np.float32, copy=False)
        source[sl] = label_to_idx[d["label"]]
        depth[sl] = np.float32(float(d["h_old"]) / ALPHA)
        cursor += n

    time = np.zeros((total,), dtype=np.float32)
    x_new = np.linspace(0.0, L_NEW, nx, endpoint=False, dtype=np.float32)

    if args.shuffle_seed:
        rng = np.random.default_rng(args.shuffle_seed)
        perm = rng.permutation(total)
        eta = eta[perm]; xi = xi[perm]; gxi = gxi[perm]
        depth = depth[perm]; time = time[perm]; source = source[perm]
        print(f"  shuffled with seed={args.shuffle_seed}")

    legend = {label_to_idx[label]: label for label, _, _, _, _ in SUBSETS if label in {l for l, _ in counts}}
    h_old_per_label = {label: h_old for label, _, _, _, h_old in SUBSETS if label in {l for l, _ in counts}}
    meta = {
        "kind": "test_dno_rescaled",
        "source_file": str(in_path),
        "L_old": L_OLD, "L_new": L_NEW,
        "alpha": ALPHA,
        "h_old_per_label": h_old_per_label,
        "h_new_per_label": {k: v / ALPHA for k, v in h_old_per_label.items()},
        "rescaling": "Zakharov-invariant: eta /= alpha; xi /= alpha^(3/2); "
                     "gxi /= alpha^(1/2); h /= alpha; x relabeled to [0, 2pi].",
        "nx": int(nx), "length": L_NEW,
        "n_samples": int(total),
        "shuffled": bool(args.shuffle_seed),
        "shuffle_seed": int(args.shuffle_seed) if args.shuffle_seed else None,
        "source_legend": {str(k): v for k, v in legend.items()},
        "source_counts": dict(counts),
    }

    print(f"  writing {out_path} ...")
    t0 = perf_counter()
    write_npz_streaming(out_path, {
        "eta": eta, "xi": xi, "gxi": gxi,
        "depth": depth, "time": time, "source": source, "x": x_new,
    })
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sz = out_path.stat().st_size / 1e9
    print(f"  wrote {out_path} ({sz:.2f} GB) in {perf_counter()-t0:.1f}s")
    print(f"  sidecar: {out_path.with_suffix('.meta.json')}")
    print()
    print("test set source counts (after rescale):")
    for label, n in counts:
        print(f"  {label:24s} {n:>10,}")
    print(f"  {'TOTAL':24s} {total:>10,}")


if __name__ == "__main__":
    main()
