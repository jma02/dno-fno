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
MODEL_DIR = REPO_ROOT / "models" / "dno-net"
LOSS_DIR = REPO_ROOT / "models" / "fno-jax"
for path in (REPO_ROOT, MODEL_DIR, LOSS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dno_net import DNONet
from jax_training_util import (
    denormalize_by_max_abs,
    iterate_batches,
    load_training_arrays,
    normalize_by_max_abs,
    normalize_to_range,
    plot_loss_history,
    plot_representative_samples,
    split_indices,
)
from losses import build_loss, count_params


def require_gpu_backend() -> str:
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"JAX GPU backend is required for this trainer. Found {backend!r}.")
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
        "params": jax.tree_util.tree_map(lambda value: np.asarray(value), jax.device_get(params)),
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


def collect_raw_predictions(
    predict_batch,
    params,
    inputs: np.ndarray,
    targets: np.ndarray,
    target_stats: dict[str, float],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions_raw: list[np.ndarray] = []
    targets_raw: list[np.ndarray] = []
    rel_l2_batches: list[np.ndarray] = []
    rel_l1_batches: list[np.ndarray] = []

    rng = np.random.default_rng(0)
    for batch_inputs, batch_targets in iterate_batches(
        inputs,
        targets,
        batch_size=batch_size,
        shuffle=False,
        rng=rng,
    ):
        predicted_normalized = np.asarray(predict_batch(params, batch_inputs))
        predicted_raw = denormalize_by_max_abs(predicted_normalized, target_stats)
        target_raw = denormalize_by_max_abs(batch_targets, target_stats)

        flat_pred = predicted_raw.reshape((predicted_raw.shape[0], -1))
        flat_target = target_raw.reshape((target_raw.shape[0], -1))
        rel_l2 = np.linalg.norm(flat_pred - flat_target, axis=1) / (
            np.linalg.norm(flat_target, axis=1) + 1e-12
        )
        rel_l1 = np.sum(np.abs(flat_pred - flat_target), axis=1) / (
            np.sum(np.abs(flat_target), axis=1) + 1e-12
        )

        predictions_raw.append(predicted_raw)
        targets_raw.append(target_raw)
        rel_l2_batches.append(rel_l2)
        rel_l1_batches.append(rel_l1)

    return (
        np.concatenate(predictions_raw, axis=0),
        np.concatenate(targets_raw, axis=0),
        np.concatenate(rel_l2_batches, axis=0),
        np.concatenate(rel_l1_batches, axis=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a structure-preserving JAX DNONet on the DNO dataset")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spectral_width", type=int, default=None)
    parser.add_argument("--n_blocks", type=int, default=8)
    parser.add_argument("--spectral_floor", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loss", choices=("sobolev", "lp"), default="sobolev")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()
    np.random.seed(args.seed)
    spectral_width = args.spectral_width or args.width
    spectral_floor = args.epsilon if args.epsilon is not None else args.spectral_floor

    backend = require_gpu_backend()
    dataset_path = REPO_ROOT / "data" / args.dataset
    outputs_dir = REPO_ROOT / args.output_root
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("dnonet_jax_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_arrays(dataset_path, args.sources)
    eta_raw = dataset["eta"].astype(np.float32)
    xi_raw = dataset["xi"].astype(np.float32)
    targets_raw = dataset["gxi"][..., None].astype(np.float32)

    eta_normalized, eta_stats = normalize_to_range(eta_raw[..., None])
    xi_normalized, xi_stats = normalize_by_max_abs(xi_raw[..., None])
    targets_normalized, target_stats = normalize_by_max_abs(targets_raw)
    features_normalized = np.concatenate((eta_normalized, xi_normalized), axis=-1)

    train_indices, val_indices, _ = split_indices(len(features_normalized), args.seed)
    train_inputs = features_normalized[train_indices]
    train_targets = targets_normalized[train_indices]
    val_inputs = features_normalized[val_indices]
    val_targets = targets_normalized[val_indices]
    val_eta_raw = eta_raw[val_indices]
    val_xi_raw = xi_raw[val_indices]

    model = DNONet(
        rank=args.rank,
        modes=args.modes,
        width=args.width,
        n_blocks=args.n_blocks,
        spectral_width=spectral_width,
        spectral_floor=spectral_floor,
        x=jnp.asarray(dataset["x"], dtype=jnp.float32),
    )
    init_inputs = jnp.asarray(train_inputs[:1])
    params = model.init(jax.random.PRNGKey(args.seed), init_inputs)["params"]
    loss_fn = build_loss(args.loss)

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
        "rank": args.rank,
        "modes": args.modes,
        "width": args.width,
        "spectral_width": spectral_width,
        "n_blocks": args.n_blocks,
        "spectral_floor": spectral_floor,
        "loss": args.loss,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "param_count": count_params(params),
        "eta_normalization": "range[-1,1]",
        "xi_normalization": "max_abs",
        "target_normalization": "max_abs",
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    @jax.jit
    def train_step(current_state, batch_inputs, batch_targets):
        def loss_for_params(current_params):
            predictions = current_state.apply_fn({"params": current_params}, batch_inputs)
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
            predictions_raw, targets_raw_val, rel_l2, rel_l1 = collect_raw_predictions(
                predict_batch,
                state.params,
                val_inputs,
                val_targets,
                target_stats,
                args.batch_size,
            )
            plot_representative_samples(
                plots_dir / f"epoch_{epoch}.png",
                "Validation: Representative Samples",
                dataset["x"],
                val_eta_raw,
                val_xi_raw,
                targets_raw_val,
                predictions_raw,
                rel_l2,
                rel_l1,
                args.seed + epoch,
            )

    with open(run_dir / "train_log.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    save_checkpoint(
        run_dir / "final_ckpt.pkl",
        epoch=history[-1]["epoch"],
        params=state.params,
        train_loss=train_loss,
        val_loss=history[-1]["val_loss"],
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )

    if not args.skip_plots:
        predictions_raw, targets_raw_val, rel_l2, rel_l1 = collect_raw_predictions(
            predict_batch,
            best_params,
            val_inputs,
            val_targets,
            target_stats,
            args.batch_size,
        )
        plot_representative_samples(
            plots_dir / "final.png",
            "Validation: Representative Samples",
            dataset["x"],
            val_eta_raw,
            val_xi_raw,
            targets_raw_val,
            predictions_raw,
            rel_l2,
            rel_l1,
            args.seed,
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
