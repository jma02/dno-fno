from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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


def normalize_to_range(
    tensor: torch.Tensor,
    lower: float = -1.0,
    upper: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    normalized = scaled * (upper - lower) + lower
    return normalized, {"min": tensor_min, "max": tensor_max}


def denormalize_from_range(
    tensor: torch.Tensor,
    stats: dict[str, torch.Tensor],
    lower: float = -1.0,
    upper: float = 1.0,
) -> torch.Tensor:
    tensor_min = stats["min"].to(tensor.device)
    tensor_max = stats["max"].to(tensor.device)
    return (tensor - lower) / (upper - lower) * (tensor_max - tensor_min + 1e-8) + tensor_min


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def configure_runtime(device: torch.device, mps_memory_fraction: float | None = None) -> None:
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return

    if device.type == "mps":
        if (
            mps_memory_fraction is not None
            and hasattr(torch.mps, "set_per_process_memory_fraction")
        ):
            torch.mps.set_per_process_memory_fraction(mps_memory_fraction)


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
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_stats: dict[str, torch.Tensor],
    target_stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    eta_values: list[torch.Tensor] = []
    xi_values: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for normalized_inputs, normalized_targets in loader:
            normalized_inputs = normalized_inputs.to(device)
            normalized_targets = normalized_targets.to(device)
            predicted_normalized = model(normalized_inputs)

            raw_inputs = denormalize_from_range(normalized_inputs, feature_stats)
            predicted_raw = denormalize_from_range(predicted_normalized, target_stats)
            target_raw = denormalize_from_range(normalized_targets, target_stats)

            predictions.append(predicted_raw.cpu())
            targets.append(target_raw.cpu())
            eta_values.append(raw_inputs[..., 0].cpu())
            xi_values.append(raw_inputs[..., 1].cpu())

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(eta_values, dim=0),
        torch.cat(xi_values, dim=0),
    )


def plot_epoch_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    x: np.ndarray,
    feature_stats: dict[str, torch.Tensor],
    target_stats: dict[str, torch.Tensor],
    fixed_random_idx: int,
    output_path: Path,
    device: torch.device,
) -> None:
    predictions, targets, eta_values, xi_values = collect_plot_predictions(
        model,
        loader,
        device,
        feature_stats,
        target_stats,
    )

    flat_predictions = predictions.view(predictions.shape[0], -1)
    flat_targets = targets.view(targets.shape[0], -1)
    rel_l2 = torch.norm(flat_predictions - flat_targets, dim=1) / (
        torch.norm(flat_targets, dim=1) + 1e-12
    )
    rel_l1 = torch.sum(torch.abs(flat_predictions - flat_targets), dim=1) / (
        torch.sum(torch.abs(flat_targets), dim=1) + 1e-12
    )

    sorted_indices = torch.argsort(rel_l2)
    sample_indices = {
        "best": sorted_indices[0].item(),
        "median": sorted_indices[len(sorted_indices) // 2].item(),
        "worst": sorted_indices[-1].item(),
        "random": fixed_random_idx,
    }

    figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex="col")
    for column, label in enumerate(("best", "median", "worst", "random")):
        sample_idx = sample_indices[label]
        l2_error = rel_l2[sample_idx].item()
        l1_error = rel_l1[sample_idx].item()

        axes[0, column].plot(x, eta_values[sample_idx].numpy(), color="tab:blue", linewidth=1.0)
        axes[0, column].set_title(rf"{label.title()} $\eta(x)$", fontdict=TITLE_FONT)
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].plot(x, xi_values[sample_idx].numpy(), color="tab:green", linewidth=1.0)
        axes[1, column].set_title(r"$\xi(x)$", fontdict=TITLE_FONT)
        axes[1, column].grid(True, alpha=0.3)

        axes[2, column].plot(
            x,
            targets[sample_idx].squeeze().numpy(),
            label="Ground Truth",
            color="black",
            linewidth=1.0,
        )
        axes[2, column].plot(
            x,
            predictions[sample_idx].squeeze().numpy(),
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
