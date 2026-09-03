from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "fno-jax"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fno1d import FNO1d  # noqa: E402
from losses import build_loss, count_params  # noqa: E402
from util import (  # noqa: E402
    iterate_batches,
    load_training_arrays,
    normalize_to_range,
    plot_epoch_summary,
    plot_loss_history,
    split_indices,
)


def require_gpu_backend() -> str:
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(
            f"JAX GPU backend is required for this trainer. Found {backend!r}."
        )
    return backend


def save_checkpoint(
    output_path: Path,
    epoch: int,
    params,
    train_loss: float,
    val_loss: float,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
) -> None:
    payload = {
        "epoch": epoch,
        "params": jax.tree_util.tree_map(
            lambda value: np.asarray(value), jax.device_get(params)
        ),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)


def evaluate(
    predict_batch,
    params,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    loss_fn,
) -> float:
    total_loss = 0.0
    total_examples = 0
    rng = np.random.default_rng(0)
    for batch_inputs, batch_targets in iterate_batches(
        inputs,
        targets,
        batch_size=batch_size,
        shuffle=False,
        rng=rng,
    ):
        predictions = predict_batch(params, batch_inputs)
        batch_loss = float(loss_fn(predictions, batch_targets))
        total_loss += batch_loss * batch_inputs.shape[0]
        total_examples += batch_inputs.shape[0]
    return total_loss / max(1, total_examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 1D JAX FNO on the DNO dataset"
    )
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=10)
    parser.add_argument("--sobolev_k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()
    np.random.seed(args.seed)

    backend = require_gpu_backend()
    dataset_path = REPO_ROOT / "data" / args.dataset
    outputs_dir = REPO_ROOT / args.output_root
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("fno_jax_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_arrays(dataset_path, args.sources)

    features_raw = np.stack((dataset["eta"], dataset["xi"]), axis=-1).astype(np.float32)
    targets_raw = dataset["gxi"][..., None].astype(np.float32)
    features_normalized, feature_stats = normalize_to_range(features_raw)
    targets_normalized, target_stats = normalize_to_range(targets_raw)

    train_indices, val_indices, _ = split_indices(len(features_normalized), args.seed)
    train_inputs = features_normalized[train_indices]
    train_targets = targets_normalized[train_indices]
    val_inputs = features_normalized[val_indices]
    val_targets = targets_normalized[val_indices]
    fixed_random_idx = int(
        np.random.default_rng(args.seed).integers(0, len(val_inputs))
    )

    model = FNO1d(args.modes, args.width, args.n_blocks)
    init_inputs = jnp.asarray(train_inputs[:1])
    params = model.init(jax.random.PRNGKey(args.seed), init_inputs)["params"]
    loss_fn = build_loss(sobolev_k=args.sobolev_k)

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    config_payload = {
        "dataset": args.dataset,
        "sources": list(dataset["source_names"]),
        "device": backend,
        "modes": args.modes,
        "width": args.width,
        "n_blocks": args.n_blocks,
        "sobolev_k": args.sobolev_k,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "param_count": count_params(params),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    @jax.jit
    def train_step(current_state, batch_inputs, batch_targets):
        def loss_for_params(current_params):
            predictions = current_state.apply_fn(
                {"params": current_params}, batch_inputs
            )
            return loss_fn(predictions, batch_targets)

        loss_value, grads = jax.value_and_grad(loss_for_params)(current_state.params)
        next_state = current_state.apply_gradients(grads=grads)
        return next_state, loss_value

    @jax.jit
    def predict_batch(current_params, batch_inputs):
        return model.apply({"params": current_params}, batch_inputs)

    best_val_loss = float("inf")
    best_epoch = 0
    best_params = state.params
    history: list[dict[str, float]] = []
    train_loss = 0.0

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        total_loss = 0.0
        total_examples = 0
        epoch_rng = np.random.default_rng(args.seed + epoch)

        for batch_inputs, batch_targets in iterate_batches(
            train_inputs,
            train_targets,
            batch_size=args.batch_size,
            shuffle=True,
            rng=epoch_rng,
        ):
            state, batch_loss = train_step(state, batch_inputs, batch_targets)
            total_loss += float(batch_loss) * batch_inputs.shape[0]
            total_examples += batch_inputs.shape[0]

        train_loss = total_loss / max(1, total_examples)
        val_loss = evaluate(
            predict_batch,
            state.params,
            val_inputs,
            val_targets,
            args.batch_size,
            loss_fn,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        if not args.skip_plots:
            plot_loss_history(history, run_dir / "loss_curve.png")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = state.params
            save_checkpoint(
                run_dir / "best_val_ckpt.pkl",
                epoch=epoch,
                params=best_params,
                train_loss=train_loss,
                val_loss=val_loss,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

        if not args.skip_plots:
            plot_epoch_summary(
                predict_batch,
                state.params,
                val_inputs,
                val_targets,
                dataset["x"],
                feature_stats,
                target_stats,
                fixed_random_idx,
                plots_dir / f"epoch_{epoch}.png",
                args.batch_size,
            )

    with open(run_dir / "train_log.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    save_checkpoint(
        run_dir / "final_ckpt.pkl",
        epoch=int(history[-1]["epoch"]),
        params=state.params,
        train_loss=train_loss,
        val_loss=history[-1]["val_loss"],
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )

    if not args.skip_plots:
        plot_epoch_summary(
            predict_batch,
            best_params,
            val_inputs,
            val_targets,
            dataset["x"],
            feature_stats,
            target_stats,
            fixed_random_idx,
            plots_dir / "final.png",
            args.batch_size,
        )

    summary_payload = {
        "dataset": args.dataset,
        "sources": list(dataset["source_names"]),
        "device": backend,
        "run_dir": str(run_dir),
        "hparams": config_payload,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "runtime_seconds": perf_counter() - start_time,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
