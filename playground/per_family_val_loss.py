"""Per-family single-step H1 val loss for a trained run on the combined dataset.

Uses the combined npz's per-sample `source` column and the same val split as
training (build_split_indices, seed 0) to break the global val loss out by
source family. Answers: is the rollout-drifting family (tanaka/bf) also the
family with elevated single-step error, or is the error flat across families?

Usage:
    JAX_PLATFORMS=cuda uv run python -m playground.per_family_val_loss \
        --run_dir outputs/cs_dno_w512b8_l256_v3_20260605_032608 \
        --per_family 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train-jax-10m"))

SOURCE_LEGEND = {
    0: "random_sea_deep", 1: "random_sea_finite", 2: "linear",
    3: "stokes_deep", 4: "stokes_finite", 5: "tanaka_g0",
    6: "tanaka_g1", 7: "bf_g0", 8: "bf_g1", 9: "bf_modal",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", default="combined_dataset_v3.npz")
    parser.add_argument("--per_family", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    legend: dict[int, str] = dict(SOURCE_LEGEND)
    with suppress(FileNotFoundError, KeyError, json.JSONDecodeError):
        meta_path = (REPO_ROOT / "data" / args.dataset).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        legend = {int(k): str(v) for k, v in meta["source_legend"].items()}

    import jax.numpy as jnp

    sys.path.insert(0, str(REPO_ROOT))
    from solver.evals.model_rollout import load_run
    from util import build_split_indices

    sys.path.insert(0, str(REPO_ROOT / "models" / "fno-jax"))
    from losses import build_loss

    loaded = load_run(args.run_dir)
    stats = loaded.stats
    fam = np.asarray(stats["feature_absmax"], dtype=np.float32)
    tam = np.float32(stats["target_absmax"])
    loss_fn = build_loss(sobolev_k=int(loaded.config.get("sobolev_k", 1)))

    print(f"run: {args.run_dir}", flush=True)
    # npz members are not mmap-able (np.load silently ignores mmap_mode for
    # archives), so indexing d["eta"] per batch re-reads the full member.
    # Select rows per family first, then read each big member exactly once.
    rng = np.random.default_rng(0)
    with np.load(REPO_ROOT / "data" / args.dataset) as d:
        src = np.asarray(d["source"])
        _, val_idx, _ = build_split_indices(src.shape[0], 0)
        src_val = src[val_idx]
        fam_rows: dict[int, np.ndarray] = {}
        for sid in legend:
            rows = val_idx[src_val == sid]
            if rows.size:
                take = min(args.per_family, rows.size)
                fam_rows[sid] = np.sort(rng.choice(rows, size=take, replace=False))
        all_rows = np.sort(np.concatenate(list(fam_rows.values())))
        eta_sel = d["eta"][all_rows].astype(np.float32, copy=False)
        xi_sel = d["xi"][all_rows].astype(np.float32, copy=False)
        gxi_sel = d["gxi"][all_rows].astype(np.float32, copy=False)
        dep_sel = d["depth"][all_rows].astype(np.float32, copy=False)
    print(f"val samples total: {val_idx.size}; selected {all_rows.size}", flush=True)

    results: dict[str, tuple[float, int]] = {}
    for sid, name in legend.items():
        if sid not in fam_rows:
            continue
        pos = np.searchsorted(all_rows, fam_rows[sid])
        losses = []
        for i in range(0, pos.size, args.batch_size):
            p = pos[i:i + args.batch_size]
            inp = jnp.asarray(np.stack([eta_sel[p] / fam[0], xi_sel[p] / fam[1]], axis=-1))
            tgt = jnp.asarray((gxi_sel[p] / tam)[..., None])
            logh = jnp.asarray(np.log(dep_sel[p]).reshape(-1, 1).astype(np.float32))
            pred = loaded.model.apply({"params": loaded.params}, inp, logh)
            losses.append((float(loss_fn(pred, tgt)), p.size))
        total_n = sum(n for _, n in losses)
        mean_loss = sum(v * n for v, n in losses) / total_n
        results[name] = (mean_loss, total_n)
        print(f"  {name:20s} n={total_n:6d}  H1_rel={mean_loss:.5f}")

    global_mean = sum(v * n for v, n in results.values()) / sum(n for _, n in results.values())
    print(f"  {'(sampled global)':20s} H1_rel={global_mean:.5f}")
    out = Path(args.run_dir) / "per_family_val_loss.json"
    out.write_text(json.dumps({k: {"loss": v, "n": n} for k, (v, n) in results.items()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
