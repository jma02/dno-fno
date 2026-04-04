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
MODEL_DIR = REPO_ROOT / "models" / "fno-jax"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fno1d import FNO1d
from losses import count_params
from util import NormStats, denormalize_targets, normalize_features, require_jax_devices
from jax_training_util import (
    build_source_labels,
    load_training_arrays,
    parse_sources,
    plot_labeled_samples,
    split_indices,
    summarize_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-jax-10m checkpoint on dno_dataset and save per-source representative plots."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--zero_mean_xi", action="store_true")
    return parser.parse_args()


def load_checkpoint(run_dir: Path, checkpoint_name: str):
    checkpoint_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    metadata_path = checkpoint_dir / "metadata.json"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return checkpoint_dir, params, metadata


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint_dir, params, metadata = load_checkpoint(run_dir, args.checkpoint)

    dataset_path = REPO_ROOT / "data" / args.dataset
    selected_sources = parse_sources(args.sources)
    arrays = load_training_arrays(dataset_path, selected_sources)

    eta = np.asarray(arrays["eta"], dtype=np.float32)
    xi = np.asarray(arrays["xi"], dtype=np.float32)
    gxi = np.asarray(arrays["gxi"], dtype=np.float32)
    x = np.asarray(arrays["x"], dtype=np.float32)
    source_labels = build_source_labels(dataset_path, arrays["source_names"])

    _, _, test_indices = split_indices(int(eta.shape[0]), args.seed)
    eta = eta[test_indices]
    xi = xi[test_indices]
    gxi = gxi[test_indices]
    source_labels = source_labels[test_indices]

    if args.zero_mean_xi:
        xi = xi - np.mean(xi, axis=1, keepdims=True)

    train_stats = metadata.get("stats")
    if train_stats is None:
        raise ValueError(
            f"Checkpoint metadata at {run_dir} has no 'stats' entry — "
            "cannot recover training-time normalization."
        )
    ns = NormStats.from_dict(train_stats)

    model = FNO1d(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
    )

    @jax.jit
    def predict_batch(current_params, batch_inputs):
        return model.apply({"params": current_params}, batch_inputs)

    prediction_chunks: list[np.ndarray] = []
    for start in range(0, eta.shape[0], args.batch_size):
        batch_inputs = normalize_features(
            eta[start : start + args.batch_size],
            xi[start : start + args.batch_size],
            ns,
        )
        batch_inputs_device = jax.device_put(batch_inputs, devices[0])
        predicted_norm = np.asarray(jax.device_get(predict_batch(params, batch_inputs_device)))
        predicted_raw = denormalize_targets(predicted_norm, ns)
        prediction_chunks.append(predicted_raw.astype(np.float32))

    predictions = np.concatenate(prediction_chunks, axis=0)
    flat_pred = predictions.reshape((predictions.shape[0], -1))
    flat_target = gxi[..., None].reshape((gxi.shape[0], -1))
    rel_l2 = np.linalg.norm(flat_pred - flat_target, axis=1) / (
        np.linalg.norm(flat_target, axis=1) + 1e-12
    )
    rel_l1 = np.sum(np.abs(flat_pred - flat_target), axis=1) / (
        np.sum(np.abs(flat_target), axis=1) + 1e-12
    )
    prediction_std = np.std(flat_pred, axis=1)
    target_std = np.std(flat_target, axis=1)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / ("eval_on_dno_dataset_zero_mean_xi" if args.zero_mean_xi else "eval_on_dno_dataset")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_epoch": int(metadata["epoch"]),
        "trained_on_dataset": config["dataset"],
        "evaluated_on_dataset": dataset_path.name,
        "split": "test",
        "split_seed": args.seed,
        "zero_mean_xi": args.zero_mean_xi,
        "num_test_examples": int(eta.shape[0]),
        "param_count": count_params(params),
        "backend": backend,
        "overall": summarize_errors(rel_l2, rel_l1),
        "prediction_std": {
            "mean": float(np.mean(prediction_std)),
            "median": float(np.median(prediction_std)),
            "min": float(np.min(prediction_std)),
            "max": float(np.max(prediction_std)),
        },
        "per_source": {},
    }

    rng = np.random.default_rng(args.seed)
    for source in selected_sources:
        mask = source_labels == source
        source_rel_l2 = rel_l2[mask]
        source_rel_l1 = rel_l1[mask]
        source_eta = eta[mask]
        source_xi = xi[mask]
        source_targets = gxi[mask]
        source_predictions = predictions[mask]
        source_prediction_std = prediction_std[mask]
        source_target_std = target_std[mask]

        order = np.argsort(source_rel_l2)
        chosen_positions = np.asarray(
            [
                int(order[0]),
                int(order[len(order) // 2]),
                int(order[-1]),
                int(rng.integers(0, len(order))),
            ],
            dtype=np.int32,
        )
        labels = ("best", "median", "worst", "random")

        plot_labeled_samples(
            output_dir / f"representative_samples_{source}.png",
            f"{source.title()}: dno_dataset Test Representative Samples",
            x,
            source_eta[chosen_positions],
            source_xi[chosen_positions],
            source_targets[chosen_positions, :, None],
            source_predictions[chosen_positions],
            source_rel_l2[chosen_positions],
            source_rel_l1[chosen_positions],
            labels,
        )

        summary["per_source"][source] = {
            "num_examples": int(mask.sum()),
            "chosen_positions_in_source_subset": chosen_positions.tolist(),
            "chosen_rel_l2": source_rel_l2[chosen_positions].tolist(),
            "chosen_rel_l1": source_rel_l1[chosen_positions].tolist(),
            "prediction_std_mean": float(np.mean(source_prediction_std)),
            "prediction_std_median": float(np.median(source_prediction_std)),
            "target_std_mean": float(np.mean(source_target_std)),
            "target_std_median": float(np.median(source_target_std)),
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "representative_samples_per_source.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
