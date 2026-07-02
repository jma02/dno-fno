"""Train a depth-conditioned FNO on combined dataset (CPU-friendly, self-contained).

Usage:
    uv run python playground/train_fno.py --data playground/data/combined_200k.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Force CPU unless --gpu is passed (checked before JAX import)
if "--gpu" not in __import__("sys").argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from fno_jax.fno1d import FNO1d
from fno_jax.losses import build_loss, count_params

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormStats:
    feature_min: np.ndarray   # (1, 1, 2)
    feature_max: np.ndarray   # (1, 1, 2)
    feature_absmax: np.ndarray  # (1, 1, 2)
    target_min: float
    target_max: float
    target_absmax: float
    mode: str

    @staticmethod
    def from_stats_dict(s: dict, mode: str = "scale") -> NormStats:
        fa = np.asarray(s.get("feature_absmax", [1.0, 1.0]), dtype=np.float32).reshape(1, 1, 2)
        ta = float(s.get("target_absmax", 1.0))
        return NormStats(
            feature_min=np.asarray(s["feature_min"], dtype=np.float32).reshape(1, 1, 2),
            feature_max=np.asarray(s["feature_max"], dtype=np.float32).reshape(1, 1, 2),
            feature_absmax=np.where(fa > 0, fa, 1.0),
            target_min=float(s["target_min"]),
            target_max=float(s["target_max"]),
            target_absmax=ta if ta > 0 else 1.0,
            mode=mode,
        )


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: NormStats) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32)
    if ns.mode == "scale":
        return stacked / ns.feature_absmax
    return (stacked - ns.feature_min) / (ns.feature_max - ns.feature_min + 1e-8) * 2 - 1


def normalize_targets(gxi: np.ndarray, ns: NormStats) -> np.ndarray:
    t = gxi[..., None].astype(np.float32)
    if ns.mode == "scale":
        return t / ns.target_absmax
    return (t - ns.target_min) / (ns.target_max - ns.target_min + 1e-8) * 2 - 1


def denormalize_targets(arr: np.ndarray, ns: NormStats) -> np.ndarray:
    if ns.mode == "scale":
        return arr * ns.target_absmax
    return ((arr + 1.0) * 0.5) * (ns.target_max - ns.target_min + 1e-8) + ns.target_min


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_flat_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        out = {k: np.asarray(f[k], dtype=np.float32) for k in ("eta", "xi", "gxi", "x")}
        if "log_depth" in f:
            out["log_depth"] = np.asarray(f["log_depth"], dtype=np.float32)
    return out


def split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    val_n = int(n * 0.1)
    test_n = int(n * 0.1)
    train_n = n - val_n - test_n
    return perm[:train_n], perm[train_n:train_n + val_n], perm[train_n + val_n:]


def prenormalize(
    eta: np.ndarray, xi: np.ndarray, gxi: np.ndarray, ns: NormStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize entire dataset once upfront. Returns (features, targets)."""
    return normalize_features(eta, xi, ns), normalize_targets(gxi, ns)


def get_batches(
    features: np.ndarray, targets: np.ndarray, log_depth: np.ndarray,
    indices: np.ndarray, batch_size: int,
    rng: np.random.Generator | None, *, shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    idx = np.array(indices, copy=True)
    if shuffle and rng is not None:
        rng.shuffle(idx)
    limit = (len(idx) // batch_size) * batch_size
    idx = idx[:limit]
    f_e, t_e, ld_e = features[idx], targets[idx], log_depth[idx]
    for s in range(0, limit, batch_size):
        e = s + batch_size
        yield f_e[s:e], t_e[s:e], ld_e[s:e, None]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train depth-conditioned FNO.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--norm", type=str, default="scale", choices=["scale", "minmax"])
    parser.add_argument("--sobolev_k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="playground/runs/combined_fno")
    parser.add_argument("--depth", type=float, default=None,
                        help="Override depth for all samples (e.g. 1.0 or 1000.0)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (default: CPU)")
    args = parser.parse_args()

    data_path = Path(args.data)
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print(f"Loading {data_path} ...", flush=True)
    ds = load_flat_dataset(data_path)
    n_samples, nx = ds["eta"].shape
    if args.depth is not None:
        log_depth = np.full(n_samples, np.log(args.depth), dtype=np.float32)
    else:
        log_depth = ds.get("log_depth", np.zeros(n_samples, dtype=np.float32))
    print(f"  {n_samples} samples, nx={nx}", flush=True)

    stats_path = data_path.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    ns = NormStats.from_stats_dict(stats, mode=args.norm)

    print("  Pre-normalizing ...", flush=True)
    features, targets = prenormalize(ds["eta"], ds["xi"], ds["gxi"], ns)
    del ds  # free raw arrays

    train_idx, val_idx, test_idx = split_indices(n_samples, args.seed)
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)

    # ---- Model ----
    model = FNO1d(
        modes=args.modes,
        width=args.width,
        n_blocks=args.n_blocks,
        xi_scale=float(ns.feature_absmax[0, 0, 1]),
        target_scale=float(ns.target_absmax),
    )
    dummy_x = jnp.zeros((1, nx, 2), dtype=jnp.float32)
    dummy_d = jnp.zeros((1, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(args.seed), dummy_x, dummy_d)["params"]
    n_params = count_params(params)
    print(f"  FNO: modes={args.modes}, width={args.width}, blocks={args.n_blocks}, params={n_params:,}", flush=True)

    # ---- Optimiser ----
    train_steps = len(train_idx) // args.batch_size
    total_steps = args.epochs * train_steps
    lr_sched = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=total_steps)
    optimizer = optax.adamw(learning_rate=lr_sched, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    loss_fn = build_loss(sobolev_k=args.sobolev_k)

    @jax.jit
    def train_step(
        ts: train_state.TrainState,
        batch_in: jnp.ndarray,
        batch_tgt: jnp.ndarray,
        batch_depth: jnp.ndarray,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        def loss_for_params(p: dict) -> jnp.ndarray:
            pred = ts.apply_fn({"params": p}, batch_in, batch_depth)
            return loss_fn(pred, batch_tgt)
        loss_val, grads = jax.value_and_grad(loss_for_params)(ts.params)
        return ts.apply_gradients(grads=grads), loss_val

    @jax.jit
    def eval_step(
        p: dict, batch_in: jnp.ndarray, batch_tgt: jnp.ndarray, batch_depth: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pred = model.apply({"params": p}, batch_in, batch_depth)
        return loss_fn(pred, batch_tgt), pred

    # ---- Config dump ----
    config = {k: v for k, v in vars(args).items()}
    config["n_params"] = n_params
    config["n_train"] = int(len(train_idx))
    config["n_val"] = int(len(val_idx))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- Training loop ----
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    best_params = None
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_losses: list[float] = []
        for bi, bt, bd in get_batches(features, targets, log_depth,
                                      train_idx, args.batch_size, rng, shuffle=True):
            bi_d = jax.device_put(jnp.asarray(bi))
            bt_d = jax.device_put(jnp.asarray(bt))
            bd_d = jax.device_put(jnp.asarray(bd))
            state, loss_val = train_step(state, bi_d, bt_d, bd_d)
            train_losses.append(float(loss_val))

        # Validate
        val_losses: list[float] = []
        for bi, bt, bd in get_batches(features, targets, log_depth,
                                      val_idx, args.batch_size, None, shuffle=False):
            bi_d = jax.device_put(jnp.asarray(bi))
            bt_d = jax.device_put(jnp.asarray(bt))
            bd_d = jax.device_put(jnp.asarray(bd))
            vl, _ = eval_step(state.params, bi_d, bt_d, bd_d)
            val_losses.append(float(vl))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_params = jax.tree.map(lambda a: np.asarray(jax.device_get(a)), state.params)
            improved = " *"

        print(f"  epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({elapsed:.1f}s){improved}", flush=True)

    # ---- Save ----
    flat_params, tree_def = jax.tree.flatten(best_params)
    param_dict = {f"param_{i}": np.asarray(p) for i, p in enumerate(flat_params)}
    np.savez(run_dir / "best_params.npz", **param_dict)

    import pickle
    with open(run_dir / "tree_def.pkl", "wb") as f:
        pickle.dump(tree_def, f)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "n_params": n_params,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (run_dir / "norm_stats.json").write_text(json.dumps({
        "feature_min": stats["feature_min"],
        "feature_max": stats["feature_max"],
        "feature_absmax": stats["feature_absmax"],
        "target_min": stats["target_min"],
        "target_max": stats["target_max"],
        "target_absmax": stats["target_absmax"],
        "mode": args.norm,
    }, indent=2), encoding="utf-8")

    # Loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_arr = [h["epoch"] for h in history]
        ax.plot(epochs_arr, [h["train_loss"] for h in history], label="train")
        ax.plot(epochs_arr, [h["val_loss"] for h in history], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"FNO training (best val={best_val:.6f} @ epoch {best_epoch})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass

    print(f"\nDone. best_val={best_val:.6f} @ epoch {best_epoch}", flush=True)
    print(f"Saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
