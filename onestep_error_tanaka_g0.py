"""One-step model error on saved snapshots (no rollout, no compounding).

For each saved truth snapshot (eta_n, xi_n), applies the trained model once to
predict gxi_n and reports rel_L2 vs truth_gxi_n. Runs on CPU.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import build_predict_gxi_with_depth, load_run


def rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-row relative L2. pred/true are (..., N)."""
    diff = pred - true
    num = np.sqrt(np.sum(diff * diff, axis=-1))
    den = np.sqrt(np.sum(true * true, axis=-1)) + 1e-30
    return num / den


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--trajs", required=True, help="Path to a *_trajs.npz produced by eval_suite.")
    p.add_argument("--checkpoint", default="best", choices=("best", "final"))
    p.add_argument("--out", default=None, help="Where to save a summary npz (default: alongside trajs).")
    args = p.parse_args()

    trajs_path = Path(args.trajs).resolve()
    out_path = Path(args.out).resolve() if args.out else trajs_path.with_name(trajs_path.stem.replace("_trajs", "_onestep") + ".npz")

    d = np.load(trajs_path)
    times = np.asarray(d["times"])
    depths = np.asarray(d["depths"])
    truth_eta = np.asarray(d["truth_eta"])
    truth_xi = np.asarray(d["truth_xi"])
    truth_gxi = np.asarray(d["truth_gxi"])
    n_t, n_ic, nx = truth_eta.shape
    print(f"loaded {trajs_path.name}: n_t={n_t} n_ic={n_ic} nx={nx}")
    print(f"depths: {depths.tolist()}")

    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    predict = build_predict_gxi_with_depth(loaded)
    log_depths = np.log(np.maximum(depths.astype(np.float64), 1e-12))

    # vmap predict over the IC axis for one time step at a time.
    predict_batched = jax.jit(jax.vmap(predict, in_axes=(0, 0, 0)))

    pred_gxi = np.empty_like(truth_gxi, dtype=np.float32)
    for k in range(n_t):
        eta_k = jnp.asarray(truth_eta[k], dtype=jnp.float32)
        xi_k = jnp.asarray(truth_xi[k], dtype=jnp.float32)
        ld_k = jnp.asarray(log_depths, dtype=jnp.float32)
        out = predict_batched(eta_k, xi_k, ld_k)
        pred_gxi[k] = np.asarray(out)
        if k in (0, 1, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1):
            print(f"  t={times[k]:.2f} snapshot {k}/{n_t-1} done")

    # (n_t, n_ic) rel L2
    r = rel_l2(pred_gxi.reshape(-1, nx), truth_gxi.reshape(-1, nx)).reshape(n_t, n_ic)
    print(f"\nrel_l2(model gxi, truth gxi) at truth snapshots  [(n_t, n_ic) = {r.shape}]")

    # Per-IC summary
    print(f"\n{'IC':>3} {'depth':>7} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10}")
    for j in range(n_ic):
        col = r[:, j]
        col = col[np.isfinite(col)]
        print(f"{j:>3d} {depths[j]:>7.3f} {col.mean():>10.4g} {np.median(col):>10.4g} {np.percentile(col, 95):>10.4g} {col.max():>10.4g}")

    # For the failing ICs (5, 11 in tanaka_g0), print per-snapshot growth
    failing = [5, 11] if r.shape[1] > 11 else []
    if failing:
        print(f"\nfailing ICs one-step error (snapshot-by-snapshot):")
        header = f"{'t':>7} " + " ".join(f"IC{j:>2d}" for j in failing)
        print(header)
        for k in range(0, n_t, 5):
            row = f"{times[k]:>7.2f} " + " ".join(f"{r[k, j]:>7.4g}" for j in failing)
            print(row)

    np.savez(out_path, times=times, depths=depths, rel_l2_onestep=r, pred_gxi=pred_gxi)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
