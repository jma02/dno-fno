from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "dno-net"
for path in (REPO_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dno_net import DNONet
from jax_training_util import (
    build_source_labels,
    denormalize_by_max_abs,
    load_training_arrays,
    normalize_by_max_abs,
    normalize_to_range,
    plot_representative_samples,
    split_indices,
    summarize_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a JAX DNONet checkpoint on the held-out DNO test split")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--carbs_root", default=None)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def choose_carbs_root(raw_carbs_root: str | None) -> Path:
    if raw_carbs_root is not None:
        return (REPO_ROOT / raw_carbs_root).resolve()

    outputs_dir = REPO_ROOT / "outputs"
    candidates = sorted(
        (
            path
            for path in outputs_dir.glob("carbs_dnonet_jax*")
            if path.is_dir() and ((path / "best_result.json").exists() or (path / "carbs_history.json").exists())
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find a DNONet JAX CARBS output directory under outputs/")
    return candidates[0]


def resolve_run_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir is not None:
        run_dir = (REPO_ROOT / args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, None

    carbs_root = choose_carbs_root(args.carbs_root)
    best_result_path = carbs_root / "best_result.json"
    if best_result_path.exists():
        with open(best_result_path, "r", encoding="utf-8") as handle:
            best_result = json.load(handle)
        return Path(best_result["result"]["run_dir"]).resolve(), carbs_root

    history_path = carbs_root / "carbs_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Could not find best_result.json or carbs_history.json in {carbs_root}")

    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    successes = [record for record in history if record["result"].get("success")]
    if not successes:
        raise ValueError(f"No successful CARBS trials found in {history_path}")
    best_record = min(successes, key=lambda record: float(record["result"]["best_val_loss"]))
    return Path(best_record["result"]["run_dir"]).resolve(), carbs_root


def require_backend(allow_cpu: bool) -> str:
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise RuntimeError(f"JAX GPU backend is required for this evaluator. Found {backend!r}.")
    return backend


def load_checkpoint(run_dir: Path, checkpoint_name: str):
    checkpoint_file = "best_val_ckpt.pkl" if checkpoint_name == "best" else "final_ckpt.pkl"
    checkpoint_path = run_dir / checkpoint_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as handle:
        payload = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return checkpoint_path, params


def evaluate_model(
    predict_batch,
    params,
    inputs: np.ndarray,
    targets_normalized: np.ndarray,
    target_stats: dict[str, float],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions_raw: list[np.ndarray] = []
    targets_raw: list[np.ndarray] = []
    rel_l2_batches: list[np.ndarray] = []
    rel_l1_batches: list[np.ndarray] = []

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets_norm = targets_normalized[start : start + batch_size]

        predicted_normalized = np.asarray(predict_batch(params, batch_inputs))
        predicted_raw = denormalize_by_max_abs(predicted_normalized, target_stats)
        target_raw = denormalize_by_max_abs(batch_targets_norm, target_stats)

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


def main() -> None:
    args = parse_args()
    backend = require_backend(args.allow_cpu)
    run_dir, carbs_root = resolve_run_dir(args)

    config_path = run_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    checkpoint_path, params = load_checkpoint(run_dir, args.checkpoint)

    dataset_path = REPO_ROOT / "data" / args.dataset
    dataset = load_training_arrays(dataset_path, args.sources)
    source_labels = build_source_labels(dataset_path, list(dataset["source_names"]))

    eta_raw = dataset["eta"].astype(np.float32)
    xi_raw = dataset["xi"].astype(np.float32)
    targets_raw = dataset["gxi"][..., None].astype(np.float32)

    eta_normalized, _ = normalize_to_range(eta_raw[..., None])
    xi_normalized, _ = normalize_by_max_abs(xi_raw[..., None])
    targets_normalized, target_stats = normalize_by_max_abs(targets_raw)
    features_normalized = np.concatenate((eta_normalized, xi_normalized), axis=-1)

    _, _, test_indices = split_indices(len(features_normalized), args.seed)
    test_inputs = features_normalized[test_indices]
    test_targets = targets_normalized[test_indices]
    test_eta_raw = eta_raw[test_indices]
    test_xi_raw = xi_raw[test_indices]
    test_source_labels = source_labels[test_indices]

    model = DNONet(
        rank=int(config["rank"]),
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
        spectral_width=int(config.get("spectral_width", config["width"])),
        spectral_floor=float(config.get("spectral_floor", config.get("epsilon", 1e-3))),
        x=jnp.asarray(dataset["x"], dtype=jnp.float32),
    )

    @jax.jit
    def predict_batch(current_params, batch_inputs):
        return model.apply({"params": current_params}, batch_inputs)

    predictions_raw, targets_raw_test, rel_l2, rel_l1 = evaluate_model(
        predict_batch,
        params,
        test_inputs,
        test_targets,
        target_stats,
        args.batch_size,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_dnonet_jax"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        plot_representative_samples(
            output_dir / "representative_samples.png",
            "All Sources: Representative Test Samples",
            dataset["x"],
            test_eta_raw,
            test_xi_raw,
            targets_raw_test,
            predictions_raw,
            rel_l2,
            rel_l1,
            args.seed,
        )

    overall_metrics = summarize_errors(rel_l2, rel_l1)
    source_metrics: dict[str, dict[str, float | int]] = {}
    for source_offset, source_name in enumerate(dataset["source_names"]):
        source_mask = test_source_labels == source_name
        if not np.any(source_mask):
            continue

        source_rel_l2 = rel_l2[source_mask]
        source_rel_l1 = rel_l1[source_mask]
        source_metrics[source_name] = summarize_errors(source_rel_l2, source_rel_l1)

        if not args.skip_plots:
            plot_representative_samples(
                output_dir / f"representative_samples_{source_name}.png",
                f"{source_name.title()}: Representative Test Samples",
                dataset["x"],
                test_eta_raw[source_mask],
                test_xi_raw[source_mask],
                targets_raw_test[source_mask],
                predictions_raw[source_mask],
                source_rel_l2,
                source_rel_l1,
                args.seed + source_offset + 1,
            )

    summary = {
        "backend": backend,
        "dataset": args.dataset,
        "sources": list(dataset["source_names"]),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "run_dir": str(run_dir),
        "carbs_root": str(carbs_root) if carbs_root is not None else None,
        "checkpoint": str(checkpoint_path),
        "num_test_examples": int(len(test_indices)),
        "param_count": int(config["param_count"]),
        **overall_metrics,
        "per_source": source_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        output_dir / "per_sample_metrics.npz",
        rel_l2=rel_l2,
        rel_l1=rel_l1,
        predictions=predictions_raw,
        targets=targets_raw_test,
        test_indices=test_indices,
        x=dataset["x"],
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
