from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ALL_SOURCES = ("soliton", "stokes", "linear")
TITLE_FONT = {"weight": "bold", "size": 12}

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
        raise ValueError(f"Unknown sources {invalid}; expected a subset of {ALL_SOURCES}")

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
) -> dict[str, np.ndarray | tuple[str, ...]]:
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
            missing = [key for key in (eta_key, xi_key, gxi_key) if key not in npz.files]
            if missing:
                raise KeyError(f"Dataset missing keys for source {source_name}: {missing}")

            eta_parts.append(npz[eta_key])
            xi_parts.append(npz[xi_key])
            gxi_parts.append(npz[gxi_key])

        x = npz["x"] if "x" in npz.files else np.arange(eta_parts[0].shape[1], dtype=np.float64)

    return {
        "eta": np.concatenate(eta_parts, axis=0),
        "xi": np.concatenate(xi_parts, axis=0),
        "gxi": np.concatenate(gxi_parts, axis=0),
        "x": x,
        "source_names": selected_sources,
    }


def build_source_labels(dataset_path: Path, selected_sources: Sequence[str]) -> np.ndarray:
    labels: list[str] = []
    with np.load(dataset_path) as npz:
        for source_name in selected_sources:
            labels.extend([source_name] * int(npz[f"{source_name}_eta"].shape[0]))
    return np.asarray(labels, dtype=object)


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
    return (array - lower) / (upper - lower) * (stats["max"] - stats["min"] + 1e-8) + stats["min"]


def normalize_by_max_abs(
    array: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    scale = float(np.max(np.abs(array)))
    scale = max(scale, 1e-8)
    return (array / scale).astype(np.float32), {"scale": scale}


def denormalize_by_max_abs(
    array: np.ndarray,
    stats: dict[str, float],
) -> np.ndarray:
    return array * stats["scale"]


def split_indices(num_examples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    axis.semilogy(epochs, train_losses, label="Train", linewidth=2.0)
    axis.semilogy(epochs, val_losses, label="Val", linewidth=2.0)
    axis.set_title("Train/Val Loss", fontdict=TITLE_FONT)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=110)
    plt.close(figure)


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


def plot_labeled_samples(
    output_path: Path,
    title_prefix: str,
    x: np.ndarray,
    eta_values: np.ndarray,
    xi_values: np.ndarray,
    targets_raw: np.ndarray,
    predictions_raw: np.ndarray,
    rel_l2: np.ndarray,
    rel_l1: np.ndarray,
    labels: Sequence[str],
    depth_values: np.ndarray | None = None,
) -> None:
    if len(labels) == 0:
        raise ValueError("Cannot plot labeled samples for an empty selection")

    figure, axes = plt.subplots(3, len(labels), figsize=(4 * len(labels), 8), sharex="col")
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None]

    for column, label in enumerate(labels):
        l2_error = float(rel_l2[column])
        l1_error = float(rel_l1[column])

        eta_title = f"{label.title()} eta(x)"
        if depth_values is not None:
            eta_title = f"{eta_title}  h={float(depth_values[column]):.3g}"
        axes[0, column].plot(x, eta_values[column], color="tab:blue", linewidth=1.0)
        axes[0, column].set_title(eta_title, fontdict=TITLE_FONT)
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].plot(x, xi_values[column], color="tab:green", linewidth=1.0)
        axes[1, column].set_title("xi(x)", fontdict=TITLE_FONT)
        axes[1, column].grid(True, alpha=0.3)

        axes[2, column].plot(
            x,
            targets_raw[column].squeeze(),
            label="Ground Truth",
            color="black",
            linewidth=1.0,
        )
        axes[2, column].plot(
            x,
            predictions_raw[column].squeeze(),
            label="Prediction",
            color="tab:red",
            linewidth=1.0,
        )
        axes[2, column].set_title(
            f"G(eta)xi  L2={l2_error:.2e}  L1={l1_error:.2e}",
            fontdict=TITLE_FONT,
        )
        axes[2, column].grid(True, alpha=0.3)
        axes[2, column].legend(fontsize=9)
        axes[2, column].set_xlabel("x")

    figure.suptitle(title_prefix, fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


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
    labels = ("best", "median", "worst", "random")
    sample_indices = np.asarray(
        [
            int(sorted_indices[0]),
            int(sorted_indices[len(sorted_indices) // 2]),
            int(sorted_indices[-1]),
            int(np.random.default_rng(random_seed).integers(0, len(rel_l2))),
        ],
        dtype=np.int32,
    )
    plot_labeled_samples(
        output_path,
        title_prefix,
        x,
        eta_values[sample_indices],
        xi_values[sample_indices],
        targets_raw[sample_indices],
        predictions_raw[sample_indices],
        rel_l2[sample_indices],
        rel_l1[sample_indices],
        labels,
    )
