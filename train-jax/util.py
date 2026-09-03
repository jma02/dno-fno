from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ALL_SOURCES = ("soliton", "stokes", "linear")
TITLE_FONT = {"weight": "bold", "size": 12}


class TrainingArrays(TypedDict):
    eta: np.ndarray
    xi: np.ndarray
    gxi: np.ndarray
    x: np.ndarray
    source_names: tuple[str, ...]


matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "axes.unicode_minus": False,
    }
)


def parse_sources(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ALL_SOURCES

    if isinstance(raw, str):
        cleaned = raw.strip().lower()
        if cleaned in {"", "all"}:
            return ALL_SOURCES
        parts = [part.strip().lower() for part in cleaned.split(",") if part.strip()]
    else:
        parts = [str(part).strip().lower() for part in raw if str(part).strip()]

    invalid = sorted(set(parts).difference(ALL_SOURCES))
    if invalid:
        raise ValueError(
            f"Unknown sources {invalid}; expected a subset of {ALL_SOURCES}"
        )

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            ordered_unique.append(part)
            seen.add(part)
    return tuple(ordered_unique)


def load_training_arrays(
    path: Path,
    sources: Sequence[str] | str | None,
) -> TrainingArrays:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    selected_sources = parse_sources(sources)
    with np.load(path) as npz:
        eta_parts: list[np.ndarray] = []
        xi_parts: list[np.ndarray] = []
        gxi_parts: list[np.ndarray] = []

        for source_name in selected_sources:
            eta_key = f"{source_name}_eta"
            xi_key = f"{source_name}_xi"
            gxi_key = f"{source_name}_Gxi"
            missing = [
                key for key in (eta_key, xi_key, gxi_key) if key not in npz.files
            ]
            if missing:
                raise KeyError(
                    f"Dataset missing keys for source {source_name}: {missing}"
                )

            eta_parts.append(npz[eta_key])
            xi_parts.append(npz[xi_key])
            gxi_parts.append(npz[gxi_key])

        x = (
            npz["x"]
            if "x" in npz.files
            else np.arange(eta_parts[0].shape[1], dtype=np.float64)
        )

    return {
        "eta": np.concatenate(eta_parts, axis=0),
        "xi": np.concatenate(xi_parts, axis=0),
        "gxi": np.concatenate(gxi_parts, axis=0),
        "x": x,
        "source_names": selected_sources,
    }


def normalize_to_range(
    array: np.ndarray,
    lower: float = -1.0,
    upper: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    array_min = float(np.min(array))
    array_max = float(np.max(array))
    scaled = (array - array_min) / (array_max - array_min + 1e-8)
    normalized = scaled * (upper - lower) + lower
    return normalized.astype(np.float32), {"min": array_min, "max": array_max}


def denormalize_from_range(
    array: np.ndarray,
    stats: dict[str, float],
    lower: float = -1.0,
    upper: float = 1.0,
) -> np.ndarray:
    return (array - lower) / (upper - lower) * (
        stats["max"] - stats["min"] + 1e-8
    ) + stats["min"]


def split_indices(
    num_examples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(num_examples)
    val_count = int(num_examples * 0.1)
    test_count = int(num_examples * 0.1)
    train_count = num_examples - val_count - test_count

    train_indices = permutation[:train_count]
    val_indices = permutation[train_count : train_count + val_count]
    test_indices = permutation[train_count + val_count :]
    return train_indices, val_indices, test_indices


def iterate_batches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    indices = np.arange(len(inputs))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield inputs[batch_indices], targets[batch_indices]


def plot_loss_history(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    axis.plot(epochs, train_losses, label="Train", linewidth=2.0)
    axis.plot(epochs, val_losses, label="Val", linewidth=2.0)
    axis.set_title("Train/Val Loss", fontdict=TITLE_FONT)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=110)
    plt.close(figure)


def collect_plot_predictions(
    predict_batch,
    params,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    feature_stats: dict[str, float],
    target_stats: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    eta_values: list[np.ndarray] = []
    xi_values: list[np.ndarray] = []

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        predicted_normalized = np.asarray(predict_batch(params, batch_inputs))

        raw_inputs = denormalize_from_range(batch_inputs, feature_stats)
        predicted_raw = denormalize_from_range(predicted_normalized, target_stats)
        target_raw = denormalize_from_range(batch_targets, target_stats)

        predictions.append(predicted_raw)
        raw_targets.append(target_raw)
        eta_values.append(raw_inputs[..., 0])
        xi_values.append(raw_inputs[..., 1])

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(raw_targets, axis=0),
        np.concatenate(eta_values, axis=0),
        np.concatenate(xi_values, axis=0),
    )


def plot_epoch_summary(
    predict_batch,
    params,
    inputs: np.ndarray,
    targets: np.ndarray,
    x: np.ndarray,
    feature_stats: dict[str, float],
    target_stats: dict[str, float],
    fixed_random_idx: int,
    output_path: Path,
    batch_size: int,
) -> None:
    predictions, raw_targets, eta_values, xi_values = collect_plot_predictions(
        predict_batch,
        params,
        inputs,
        targets,
        batch_size,
        feature_stats,
        target_stats,
    )

    flat_predictions = predictions.reshape((predictions.shape[0], -1))
    flat_targets = raw_targets.reshape((raw_targets.shape[0], -1))
    rel_l2 = np.linalg.norm(flat_predictions - flat_targets, axis=1) / (
        np.linalg.norm(flat_targets, axis=1) + 1e-12
    )
    rel_l1 = np.sum(np.abs(flat_predictions - flat_targets), axis=1) / (
        np.sum(np.abs(flat_targets), axis=1) + 1e-12
    )

    sorted_indices = np.argsort(rel_l2)
    sample_indices = {
        "best": int(sorted_indices[0]),
        "median": int(sorted_indices[len(sorted_indices) // 2]),
        "worst": int(sorted_indices[-1]),
        "random": int(fixed_random_idx),
    }

    figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex="col")
    for column, label in enumerate(("best", "median", "worst", "random")):
        sample_idx = sample_indices[label]
        l2_error = float(rel_l2[sample_idx])
        l1_error = float(rel_l1[sample_idx])

        axes[0, column].plot(x, eta_values[sample_idx], color="tab:blue", linewidth=1.0)
        axes[0, column].set_title(rf"{label.title()} $\eta(x)$", fontdict=TITLE_FONT)
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].plot(x, xi_values[sample_idx], color="tab:green", linewidth=1.0)
        axes[1, column].set_title(r"$\xi(x)$", fontdict=TITLE_FONT)
        axes[1, column].grid(True, alpha=0.3)

        axes[2, column].plot(
            x,
            raw_targets[sample_idx].squeeze(),
            label="Ground Truth",
            color="black",
            linewidth=1.0,
        )
        axes[2, column].plot(
            x,
            predictions[sample_idx].squeeze(),
            label="Prediction",
            color="tab:red",
            linewidth=1.0,
        )
        axes[2, column].set_title(
            rf"$G(\eta)\xi$  $L^2$={l2_error:.2e}  $L^1$={l1_error:.2e}",
            fontdict=TITLE_FONT,
        )
        axes[2, column].grid(True, alpha=0.3)
        axes[2, column].legend(fontsize=9)
        axes[2, column].set_xlabel("x")

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
