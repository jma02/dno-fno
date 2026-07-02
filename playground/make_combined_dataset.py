"""Build a combined (deep random sea + shallow Tanaka) dataset for depth-conditioned FNO.

Produces a flat .npz with fields: eta, xi, gxi, x, depth (per-sample log-depth).

Usage:
    uv run python playground/make_combined_dataset.py
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np


def load_tanaka_subset(path: Path, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path, mode="r") as zf:
        meta = json.loads(zf.read("meta.json"))

    with np.load(path) as archive:
        eta_parts, xi_parts, gxi_parts = [], [], []
        for shard_id in range(int(meta["n_batches_planned"])):
            tag = f"{shard_id:04d}"
            eta_parts.append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            xi_parts.append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            gxi_parts.append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
        x = np.asarray(archive["x"], dtype=np.float32)

    eta = np.concatenate(eta_parts, axis=0)
    xi = np.concatenate(xi_parts, axis=0)
    gxi = np.concatenate(gxi_parts, axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eta), size=min(n, len(eta)), replace=False)
    return {"eta": eta[idx], "xi": xi[idx], "gxi": gxi[idx], "x": x}


def load_random_sea(path: Path, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        eta = np.asarray(f["eta"], dtype=np.float32)
        xi = np.asarray(f["xi"], dtype=np.float32)
        gxi = np.asarray(f["gxi"], dtype=np.float32)
        x = np.asarray(f["x"], dtype=np.float32)

    if len(eta) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eta), size=n, replace=False)
        eta, xi, gxi = eta[idx], xi[idx], gxi[idx]
    return {"eta": eta, "xi": xi, "gxi": gxi, "x": x}


def main() -> None:
    tanaka_path = Path("data/tanaka_1_clean_sub500k.npz")
    deep_path = Path("playground/data/random_sea_deep_100k.npz")
    out_path = Path("playground/data/combined_200k.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_each = 100_000

    print(f"Loading {n_each} Tanaka samples (h=1) ...", flush=True)
    tanaka = load_tanaka_subset(tanaka_path, n_each)
    n_tanaka = tanaka["eta"].shape[0]
    print(f"  got {n_tanaka}", flush=True)

    print(f"Loading {n_each} deep random sea samples (h=1000) ...", flush=True)
    deep = load_random_sea(deep_path, n_each)
    n_deep = deep["eta"].shape[0]
    print(f"  got {n_deep}", flush=True)

    assert np.allclose(tanaka["x"], deep["x"]), "grids must match"

    eta = np.concatenate([tanaka["eta"], deep["eta"]], axis=0)
    xi = np.concatenate([tanaka["xi"], deep["xi"]], axis=0)
    gxi = np.concatenate([tanaka["gxi"], deep["gxi"]], axis=0)
    log_depth = np.concatenate([
        np.full(n_tanaka, np.log(1.0), dtype=np.float32),
        np.full(n_deep, np.log(1000.0), dtype=np.float32),
    ])

    print(f"Combined: {len(eta)} samples", flush=True)
    print(f"  eta range: [{eta.min():.4f}, {eta.max():.4f}]", flush=True)
    print(f"  gxi range: [{gxi.min():.4f}, {gxi.max():.4f}]", flush=True)

    np.savez(out_path, eta=eta, xi=xi, gxi=gxi, x=tanaka["x"], log_depth=log_depth)
    print(f"Saved → {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    # Stats sidecar (compute over full combined set)
    stats = {
        "n_samples": len(eta),
        "nx": int(eta.shape[1]),
        "feature_min": [float(eta.min()), float(xi.min())],
        "feature_max": [float(eta.max()), float(xi.max())],
        "feature_absmax": [float(np.abs(eta).max()), float(np.abs(xi).max())],
        "target_min": float(gxi.min()),
        "target_max": float(gxi.max()),
        "target_absmax": float(np.abs(gxi).max()),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats → {stats_path}", flush=True)


if __name__ == "__main__":
    main()
