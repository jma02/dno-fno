"""Build the fine-tune dataset: perturbed tanaka + replay mix from v3.

Replay rows come from the v3 TRAIN split only (the v3 val split stays honest
for before/after regression checks). Replay rows from --relabel_sources get
their gxi recomputed with the on-grid order-6 f64 Craig-Sulem series: the
stored v3 labels disagree with the on-grid operator by up to ~2% in the
shallow/steep tanaka corner, and mixing both label conventions in one
fine-tune would supervise the model with contradictions.

The .stats.json sidecar is copied from v3 so the fine-tune reuses the
baseline checkpoint's exact normalization.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -u -m solver.gen_data.build_finetune_mix \
        --perturbed data/tanaka_perturbed_v1.npz --output data/finetune_tanaka_v1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train-jax-10m"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="combined_dataset_v3.npz")
    parser.add_argument("--perturbed", default="data/tanaka_perturbed_v1.npz")
    parser.add_argument("--output", default="data/finetune_tanaka_v1.npz")
    parser.add_argument("--replay_per_family", type=int, default=150000)
    parser.add_argument("--relabel_sources", default="5,6")
    parser.add_argument("--tail_tol", type=float, default=1e-4)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from solver.gen_data.perturbed_tanaka import dno_series_with_tail
    from solver.solvers.dno_series_jax import build_grid
    from util import build_split_indices

    relabel_ids = {int(s) for s in args.relabel_sources.split(",") if s.strip()}
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    with np.load(REPO_ROOT / "data" / args.dataset) as d:
        src = np.asarray(d["source"])
        train_idx, _, _ = build_split_indices(src.shape[0], 0)
        src_train = src[train_idx]
        replay_list = [
            rng.choice(train_idx[src_train == sid],
                       size=min(args.replay_per_family, int((src_train == sid).sum())),
                       replace=False)
            for sid in np.unique(src)
        ]
        replay_rows = np.sort(np.concatenate(replay_list))
        print(f"replay rows: {replay_rows.size} from train split", flush=True)
        eta_r = d["eta"][replay_rows].astype(np.float32)
        xi_r = d["xi"][replay_rows].astype(np.float32)
        gxi_r = d["gxi"][replay_rows].astype(np.float32)
        depth_r = np.asarray(d["depth"])[replay_rows].astype(np.float32)
        time_r = np.asarray(d["time"])[replay_rows].astype(np.float32)
        src_r = src[replay_rows]
        x_grid = np.asarray(d["x"], dtype=np.float32)
    print(f"replay slice loaded in {time.time()-t0:.0f}s", flush=True)

    nx = eta_r.shape[-1]
    _, k_np = build_grid(nx, 2.0 * np.pi)
    k_grid = jnp.asarray(k_np, dtype=jnp.float64)
    label_fn = jax.jit(lambda e, x, h: dno_series_with_tail(e, x, k_grid, h, 6, 8))

    relabel_mask = np.isin(src_r, list(relabel_ids))
    rel_pos = np.where(relabel_mask)[0]
    print(f"relabeling {rel_pos.size} replay rows (sources {sorted(relabel_ids)}) "
          f"with on-grid order-6 f64 ...", flush=True)
    drop = np.zeros(replay_rows.size, dtype=bool)
    diffs = []
    for i in range(0, rel_pos.size, args.chunk):
        p = rel_pos[i:i + args.chunk]
        gxi_new, tail = label_fn(
            jnp.asarray(eta_r[p], jnp.float64), jnp.asarray(xi_r[p], jnp.float64),
            jnp.asarray(depth_r[p, None], jnp.float64),
        )
        gxi_new, tail = np.asarray(gxi_new), np.asarray(tail)
        norm = np.linalg.norm(gxi_new, axis=-1) + 1e-30
        bad = ~(np.isfinite(gxi_new).all(axis=-1)
                & (np.linalg.norm(tail, axis=-1) / norm < args.tail_tol))
        diffs.append(np.linalg.norm(gxi_new - gxi_r[p], axis=-1) / norm)
        drop[p[bad]] = True
        gxi_r[p] = gxi_new.astype(np.float32)
    diffs_all = np.concatenate(diffs) if diffs else np.zeros(0)
    print(f"  relabel delta vs stored: med={np.median(diffs_all):.2e} "
          f"p99={np.quantile(diffs_all, 0.99):.2e}; dropped {int(drop.sum())} non-converged",
          flush=True)
    keep = ~drop
    eta_r, xi_r, gxi_r = eta_r[keep], xi_r[keep], gxi_r[keep]
    depth_r, time_r, src_r = depth_r[keep], time_r[keep], src_r[keep]

    with np.load(REPO_ROOT / args.perturbed) as p:
        eta_p = np.asarray(p["eta"])
        xi_p = np.asarray(p["xi"])
        gxi_p = np.asarray(p["gxi"])
        depth_p = np.asarray(p["depth"])
        time_p = np.asarray(p["time"])
        src_p = np.asarray(p["source"])
    print(f"perturbed rows: {eta_p.shape[0]}", flush=True)

    eta = np.concatenate([eta_r, eta_p])
    xi = np.concatenate([xi_r, xi_p])
    gxi = np.concatenate([gxi_r, gxi_p])
    depth = np.concatenate([depth_r, depth_p])
    tcol = np.concatenate([time_r, time_p])
    source = np.concatenate([src_r.astype(np.int8), src_p.astype(np.int8)])
    perm = rng.permutation(eta.shape[0])
    eta, xi, gxi = eta[perm], xi[perm], gxi[perm]
    depth, tcol, source = depth[perm], tcol[perm], source[perm]

    out_path = REPO_ROOT / args.output
    np.savez(out_path, eta=eta, xi=xi, gxi=gxi, depth=depth, time=tcol,
             source=source, x=x_grid)

    v3_stats = json.loads((REPO_ROOT / "data" / args.dataset).with_suffix(".stats.json").read_text())
    for name, arr, idx in (("eta", eta, 0), ("xi", xi, 1)):
        margin = float(np.abs(arr).max()) / v3_stats["feature_absmax"][idx]
        print(f"  absmax({name}) = {margin:.3f} x v3 stats", flush=True)
    print(f"  absmax(gxi) = {float(np.abs(gxi).max()) / v3_stats['target_absmax']:.3f} x v3 stats",
          flush=True)
    stats = dict(v3_stats)
    stats["dataset"] = str(out_path)
    stats["num_examples"] = int(eta.shape[0])
    Path(str(out_path).replace(".npz", ".stats.json")).write_text(json.dumps(stats, indent=2))

    meta = {
        "kind": "finetune_mix",
        "nx": int(nx), "n_samples": int(eta.shape[0]),
        "replay_dataset": args.dataset,
        "replay_per_family": args.replay_per_family,
        "relabeled_sources": sorted(relabel_ids),
        "relabel": "on-grid order-6 pad-8 CS series, f64, tail-rejected",
        "perturbed": args.perturbed,
        "source_legend_extra": {"10": "tanaka_perturbed"},
        "norm_stats": "copied from v3 (baseline checkpoint compatibility)",
        "seed": args.seed,
    }
    Path(str(out_path).replace(".npz", ".meta.json")).write_text(json.dumps(meta, indent=2))
    counts = {int(s): int(c) for s, c in zip(*np.unique(source, return_counts=True))}
    print(f"wrote {out_path}: {eta.shape[0]} rows, source counts {counts}", flush=True)


if __name__ == "__main__":
    main()
