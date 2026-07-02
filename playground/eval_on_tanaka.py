"""Evaluate a trained FNO on the Tanaka soliton test set.

Usage:
    uv run python playground/eval_on_tanaka.py --run_dir playground/runs/combined_fno
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from fno_jax.fno1d import FNO1d


def _load_norm(run_dir: Path) -> dict:
    return json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: dict) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32)
    if ns["mode"] == "scale":
        absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
        return stacked / np.where(absmax > 0, absmax, 1.0)
    fmin = np.asarray(ns["feature_min"], dtype=np.float32).reshape(1, 1, 2)
    fmax = np.asarray(ns["feature_max"], dtype=np.float32).reshape(1, 1, 2)
    return (stacked - fmin) / (fmax - fmin + 1e-8) * 2 - 1


def denormalize_targets(arr: np.ndarray, ns: dict) -> np.ndarray:
    if ns["mode"] == "scale":
        ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
        return arr * ta
    tmin, tmax = float(ns["target_min"]), float(ns["target_max"])
    return ((arr + 1.0) * 0.5) * (tmax - tmin + 1e-8) + tmin


def load_tanaka_dataset(path: Path, max_samples: int = 5000) -> dict[str, np.ndarray]:
    import zipfile
    with zipfile.ZipFile(path, mode="r") as zf:
        meta = json.loads(zf.read("meta.json"))

    with np.load(path) as archive:
        eta_parts, xi_parts, gxi_parts = [], [], []
        n_shards = int(meta["n_batches_planned"])
        for shard_id in range(n_shards):
            tag = f"{shard_id:04d}"
            eta_parts.append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            xi_parts.append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            gxi_parts.append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
        x = np.asarray(archive["x"], dtype=np.float32)

    eta = np.concatenate(eta_parts, axis=0)
    xi = np.concatenate(xi_parts, axis=0)
    gxi = np.concatenate(gxi_parts, axis=0)

    if len(eta) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(eta), size=max_samples, replace=False)
        eta, xi, gxi = eta[idx], xi[idx], gxi[idx]

    return {"eta": eta, "xi": xi, "gxi": gxi, "x": x}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FNO on Tanaka solitons.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--tanaka_data", type=str, default="data/tanaka_1_clean_sub500k.npz")
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = _load_norm(run_dir)
    log_depth = np.log(args.depth).astype(np.float32)

    # ---- Load model ----
    fa = np.asarray(ns["feature_absmax"], dtype=np.float32)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    model = FNO1d(
        modes=config["modes"],
        width=config["width"],
        n_blocks=config["n_blocks"],
        xi_scale=float(fa[1]),
        target_scale=ta,
    )

    with np.load(run_dir / "best_params.npz") as f:
        flat_arrays = [jnp.asarray(f[k]) for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    with open(run_dir / "tree_def.pkl", "rb") as f:
        tree_def = pickle.load(f)
    params = tree_def.unflatten(flat_arrays)

    @jax.jit
    def predict(p: dict, x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": p}, x, d)

    # ---- Load Tanaka ----
    print(f"Loading Tanaka data from {args.tanaka_data} ...", flush=True)
    tanaka = load_tanaka_dataset(Path(args.tanaka_data), max_samples=args.max_samples)
    n_samples = tanaka["eta"].shape[0]
    print(f"  {n_samples} samples, depth={args.depth}", flush=True)

    # ---- Evaluate in batches ----
    print("Running inference...", flush=True)
    rel_l2_parts, rel_l1_parts, pred_parts = [], [], []

    for start in range(0, n_samples, args.batch_size):
        end = min(start + args.batch_size, n_samples)
        bs = end - start

        inp = normalize_features(tanaka["eta"][start:end], tanaka["xi"][start:end], ns)
        depth_arr = jnp.full((bs, 1), log_depth)
        pred_norm = predict(params, jnp.asarray(inp), depth_arr)
        pred_raw = denormalize_targets(np.asarray(jax.device_get(pred_norm)), ns)
        pred_parts.append(pred_raw)

        flat_pred = pred_raw[..., 0]
        gxi_true = tanaka["gxi"][start:end]
        diff = flat_pred - gxi_true
        rel_l2_parts.append(np.linalg.norm(diff, axis=1) / (np.linalg.norm(gxi_true, axis=1) + 1e-12))
        rel_l1_parts.append(np.sum(np.abs(diff), axis=1) / (np.sum(np.abs(gxi_true), axis=1) + 1e-12))

    predictions = np.concatenate(pred_parts, axis=0)  # (N, nx, 1)
    rel_l2 = np.concatenate(rel_l2_parts)
    rel_l1 = np.concatenate(rel_l1_parts)

    summary = {
        "n_samples": n_samples,
        "depth": args.depth,
        "mean_rel_l2": float(np.mean(rel_l2)),
        "median_rel_l2": float(np.median(rel_l2)),
        "std_rel_l2": float(np.std(rel_l2)),
        "p95_rel_l2": float(np.percentile(rel_l2, 95)),
        "p99_rel_l2": float(np.percentile(rel_l2, 99)),
        "mean_rel_l1": float(np.mean(rel_l1)),
        "median_rel_l1": float(np.median(rel_l1)),
    }

    print(f"\nTanaka eval ({n_samples} samples, h={args.depth}):", flush=True)
    print(f"  mean   rel_l2 = {summary['mean_rel_l2']:.4f}", flush=True)
    print(f"  median rel_l2 = {summary['median_rel_l2']:.4f}", flush=True)
    print(f"  p95    rel_l2 = {summary['p95_rel_l2']:.4f}", flush=True)
    print(f"  p99    rel_l2 = {summary['p99_rel_l2']:.4f}", flush=True)

    out_path = run_dir / "eval_tanaka.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}", flush=True)

    # ---- Plot ----
    from jax_training_util import plot_representative_samples

    plot_path = run_dir / "eval_tanaka.png"
    plot_representative_samples(
        plot_path,
        f"Tanaka solitons (h={args.depth}): mean L2={summary['mean_rel_l2']:.4f}",
        tanaka["x"],
        tanaka["eta"],
        tanaka["xi"],
        tanaka["gxi"][:, :, None],
        predictions,
        rel_l2,
        rel_l1,
        random_seed=42,
    )
    print(f"Plot -> {plot_path}", flush=True)


if __name__ == "__main__":
    main()
