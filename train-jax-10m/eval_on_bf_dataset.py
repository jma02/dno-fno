"""Evaluate a checkpoint on `data/stokes_bf_dataset.npz` (BF wavetrains, 512-grid).

The dataset has no depth metadata; pass --depth h (or --depth_scan to try a list).
Since FNO is discretization-invariant, the 512-grid is fed straight in.

Usage:
    uv run python train-jax-10m/eval_on_bf_dataset.py \\
        --run_dir outputs/<run> --checkpoint best --depth 1.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from util import (
    NormStats,
    compute_log_depth,
    denormalize_targets,
    normalize_features,
    require_jax_devices,
)
from jax_training_util import plot_labeled_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--dataset", default="data/stokes_bf_dataset.npz")
    p.add_argument("--depth", type=float, default=None,
                   help="Single depth to evaluate (default 1.4 = median of training BF).")
    p.add_argument("--depth_scan", default=None,
                   help="Comma-separated list of depths to scan, e.g. '0.5,1.0,1.4,2.0,4.0'.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--plot", action="store_true",
                   help="Save best/median/worst/random representative-sample plots "
                        "at the first depth in --depth_scan (or --depth).")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def _load_checkpoint(run_dir: Path, which: str):
    ckpt_dir = run_dir / ("best_val_ckpt" if which == "best" else "final_ckpt")
    if not (ckpt_dir / "metadata.json").exists():
        raise FileNotFoundError(ckpt_dir)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=None, prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    return ckpt_dir, jax.tree_util.tree_map(jnp.asarray, restored["params"]), metadata


def _build_model(config: dict[str, object], stats: dict[str, object]):
    ns = NormStats.from_dict(stats, mode=config.get("norm", "minmax"))
    if config.get("model", "fno") == "spectral_dno":
        return SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(ns.feature_absmax).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )
    return FNO1d(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
        domain_length=float(stats.get("domain_length", 2.0 * np.pi)),
        xi_scale=float(np.asarray(ns.feature_absmax).reshape(-1)[1]),
        target_scale=float(ns.target_absmax),
    )


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)
    device = devices[0]

    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_dir, params, metadata = _load_checkpoint(run_dir, args.checkpoint)
    stats = metadata["stats"]
    norm_mode = config.get("norm", "minmax")

    dataset_path = (REPO_ROOT / args.dataset).resolve()
    with np.load(dataset_path) as d:
        eta = d["stokes_eta"].astype(np.float32)
        xi = d["stokes_xi"].astype(np.float32)
        gxi = d["stokes_Gxi"].astype(np.float32)
        x = d["x"].astype(np.float32)
    if args.max_examples is not None:
        sel = np.arange(min(args.max_examples, eta.shape[0]))
        eta, xi, gxi = eta[sel], xi[sel], gxi[sel]
    n, nx = eta.shape
    xi = xi - xi.mean(axis=1, keepdims=True)
    print(f"backend={backend}  dataset={dataset_path.name}  N={n}  nx={nx}  "
          f"x=[{x.min():.4f},{x.max():.4f}]  ckpt={ckpt_dir.name}")
    print(f"  target gxi: std={gxi.std():.4f}  median|.|={np.median(np.abs(gxi)):.5f}")

    if args.depth_scan:
        depths = [float(s) for s in args.depth_scan.split(",")]
    elif args.depth is not None:
        depths = [float(args.depth)]
    else:
        depths = [1.4]

    ns = NormStats.from_dict(stats, mode=norm_mode)
    model = _build_model(config, stats)

    @jax.jit
    def predict(p, inputs, depth):
        return model.apply({"params": p}, inputs, depth)

    target_norm = np.linalg.norm(gxi, axis=1)
    target_norm_safe = target_norm + 1e-12

    print()
    print(f"{'h':>7s}  {'mean':>8s} {'median':>8s} {'p95':>8s} {'max':>8s}  (rel-L2)")
    print("-" * 56)
    last_preds = None
    last_rel_l2 = None
    last_rel_l1 = None
    last_h = None
    for h in depths:
        depth_arr = np.full((n, 1), float(h), dtype=np.float32)
        log_depth = compute_log_depth(depth_arr)
        preds = np.empty_like(gxi)
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            batch_inputs = normalize_features(eta[start:end], xi[start:end], ns)
            batch_depth = log_depth[start:end]
            pred_norm = predict(
                params,
                jax.device_put(batch_inputs, device),
                jax.device_put(batch_depth, device),
            )
            preds[start:end] = np.asarray(jax.device_get(
                denormalize_targets(pred_norm, ns)
            )).reshape(end - start, nx)
        residual = np.linalg.norm(preds - gxi, axis=1)
        rel_l2 = residual / target_norm_safe
        rel_l1 = (np.sum(np.abs(preds - gxi), axis=1)
                  / (np.sum(np.abs(gxi), axis=1) + 1e-12))
        print(f"{h:>7.3f}  {rel_l2.mean():>8.4f} {np.median(rel_l2):>8.4f} "
              f"{np.quantile(rel_l2,0.95):>8.4f} {rel_l2.max():>8.4f}")
        last_preds, last_rel_l2, last_rel_l1, last_h = preds, rel_l2, rel_l1, h

    if args.plot and last_preds is not None:
        output_dir = (
            Path(args.output_dir).resolve() if args.output_dir
            else run_dir / "eval_on_bf_dataset"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        order = np.argsort(last_rel_l2)
        rng = np.random.default_rng(0)
        chosen = np.asarray([
            int(order[0]),
            int(order[len(order) // 2]),
            int(order[-1]),
            int(rng.integers(0, len(order))),
        ], dtype=np.int32)
        labels = ("best", "median", "worst", "random")
        out_png = output_dir / f"representative_samples_bf_h{last_h:.2f}.png"
        plot_labeled_samples(
            out_png,
            f"BF (stokes_bf_dataset, h={last_h:.2f}): representatives",
            x,
            eta[chosen], xi[chosen],
            gxi[chosen, :, None], last_preds[chosen, :, None],
            last_rel_l2[chosen], last_rel_l1[chosen],
            labels,
        )
        summary = {
            "dataset": str(dataset_path),
            "checkpoint": str(ckpt_dir),
            "depth": float(last_h),
            "n_examples": int(n),
            "rel_l2": {
                "mean": float(last_rel_l2.mean()),
                "median": float(np.median(last_rel_l2)),
                "p95": float(np.quantile(last_rel_l2, 0.95)),
                "max": float(last_rel_l2.max()),
            },
            "chosen_indices": chosen.tolist(),
            "chosen_rel_l2": last_rel_l2[chosen].tolist(),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"saved -> {out_png}")
        print(f"saved -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
