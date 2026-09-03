from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "fno-jax"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fno1d import FNO1d  # noqa: E402
from losses import count_params  # noqa: E402
from util import (  # noqa: E402
    denormalize_from_range,
    load_training_arrays,
    normalize_to_range,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a JAX FNO checkpoint on the held-out DNO test split"
    )
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
            for path in outputs_dir.glob("carbs_jax*")
            if path.is_dir()
            and (
                (path / "best_result.json").exists()
                or (path / "carbs_history.json").exists()
            )
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find a JAX CARBS output directory under outputs/"
        )
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
        raise FileNotFoundError(
            f"Could not find best_result.json or carbs_history.json in {carbs_root}"
        )

    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    successes = [record for record in history if record["result"].get("success")]
    if not successes:
        raise ValueError(f"No successful CARBS trials found in {history_path}")
    best_record = min(
        successes, key=lambda record: float(record["result"]["best_val_loss"])
    )
    return Path(best_record["result"]["run_dir"]).resolve(), carbs_root


def require_backend(allow_cpu: bool) -> str:
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise RuntimeError(
            f"JAX GPU backend is required for this evaluator. Found {backend!r}."
        )
    return backend


def load_checkpoint(run_dir: Path, checkpoint_name: str):
    checkpoint_file = (
        "best_val_ckpt.pkl" if checkpoint_name == "best" else "final_ckpt.pkl"
    )
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
        predicted_raw = denormalize_from_range(predicted_normalized, target_stats)
        target_raw = denormalize_from_range(batch_targets_norm, target_stats)

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


def build_source_labels(dataset_path: Path, selected_sources: list[str]) -> np.ndarray:
    labels: list[str] = []
    with np.load(dataset_path) as npz:
        for source_name in selected_sources:
            labels.extend([source_name] * int(npz[f"{source_name}_eta"].shape[0]))
    return np.asarray(labels, dtype=object)


def summarize_errors(rel_l2: np.ndarray, rel_l1: np.ndarray) -> dict[str, float | int]:
    return {
        "num_examples": int(rel_l2.shape[0]),
        "mean_rel_l2": float(np.mean(rel_l2)),
        "median_rel_l2": float(np.median(rel_l2)),
        "std_rel_l2": float(np.std(rel_l2)),
        "mean_rel_l1": float(np.mean(rel_l1)),
        "median_rel_l1": float(np.median(rel_l1)),
        "std_rel_l1": float(np.std(rel_l1)),
    }


def plot_representative_samples(
    output_path: Path,
    title_prefix: str,
    x: np.ndarray,
    eta_values: np.ndarray,
    xi_values: np.ndarray,
    targets_raw: np.ndarray,
    predictions_raw: np.ndarray,
    rel_l2: np.ndarray,
    rel_l1: np.ndarray,
    random_seed: int,
) -> None:
    if len(rel_l2) == 0:
        raise ValueError("Cannot plot representative samples for an empty selection")

    sorted_indices = np.argsort(rel_l2)
    sample_indices = {
        "best": int(sorted_indices[0]),
        "median": int(sorted_indices[len(sorted_indices) // 2]),
        "worst": int(sorted_indices[-1]),
        "random": int(np.random.default_rng(random_seed).integers(0, len(rel_l2))),
    }

    figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex="col")
    for column, label in enumerate(("best", "median", "worst", "random")):
        sample_idx = sample_indices[label]
        l2_error = float(rel_l2[sample_idx])
        l1_error = float(rel_l1[sample_idx])

        axes[0, column].plot(x, eta_values[sample_idx], color="tab:blue", linewidth=1.0)
        axes[0, column].set_title(
            f"{label.title()} eta(x)", fontdict={"weight": "bold", "size": 12}
        )
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].plot(x, xi_values[sample_idx], color="tab:green", linewidth=1.0)
        axes[1, column].set_title("xi(x)", fontdict={"weight": "bold", "size": 12})
        axes[1, column].grid(True, alpha=0.3)

        axes[2, column].plot(
            x,
            targets_raw[sample_idx].squeeze(),
            label="Ground Truth",
            color="black",
            linewidth=1.0,
        )
        axes[2, column].plot(
            x,
            predictions_raw[sample_idx].squeeze(),
            label="Prediction",
            color="tab:red",
            linewidth=1.0,
        )
        axes[2, column].set_title(
            f"G(eta)xi  L2={l2_error:.2e}  L1={l1_error:.2e}",
            fontdict={"weight": "bold", "size": 12},
        )
        axes[2, column].grid(True, alpha=0.3)
        axes[2, column].legend(fontsize=9)
        axes[2, column].set_xlabel("x")

    figure.suptitle(title_prefix, fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


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
    features_raw = np.stack((dataset["eta"], dataset["xi"]), axis=-1).astype(np.float32)
    targets_raw_full = dataset["gxi"][..., None].astype(np.float32)
    features_normalized, _ = normalize_to_range(features_raw)
    targets_normalized, target_stats = normalize_to_range(targets_raw_full)
    source_labels = build_source_labels(dataset_path, list(dataset["source_names"]))

    _, _, test_indices = split_indices(len(features_normalized), args.seed)
    test_inputs = features_normalized[test_indices]
    test_targets = targets_normalized[test_indices]
    test_eta_raw = features_raw[test_indices, :, 0]
    test_xi_raw = features_raw[test_indices, :, 1]
    test_source_labels = source_labels[test_indices]

    model = FNO1d(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
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

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_jax"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics = summarize_errors(rel_l2, rel_l1)
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
        "param_count": count_params(params),
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
