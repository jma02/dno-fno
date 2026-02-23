from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from models.fno import FNO1d
from models.fno import HsLoss, LpLoss, count_params
from models.fno.util import UnitGaussianNormalizer

torch.set_float32_matmul_precision("high")

plt.rcParams["font.family"] = "DejaVu Serif"
TITLE_FONT = {"family": "DejaVu Serif", "weight": "bold", "size": 12}


def load_processed_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find processed dataset at {path}")
    data = np.load(path)
    required = {"xi", "eta", "Gxi"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")
    return {key: data[key] for key in data.files}


def build_feature_target_tensors(dataset: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Use eta + xi as inputs; predict Gxi
    eta = dataset["eta"][..., None]
    xi = dataset["xi"][..., None]
    inputs = np.concatenate([eta, xi], axis=-1)  # (N, Nx, 2)
    targets = dataset["Gxi"][..., None]  # (N, Nx, 1)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def normalize_to_range(data: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Simple [-1, 1] normalization across all dimensions except batch and channel
    # Using global min/max for the entire dataset feature/target
    d_min = data.min()
    d_max = data.max()
    norm_data = (data - d_min) / (d_max - d_min + 1e-8) * (max_val - min_val) + min_val
    stats = {"min": d_min, "max": d_max}
    return norm_data, stats


def denormalize_from_range(norm_data: torch.Tensor, stats: Dict[str, torch.Tensor], min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    d_min = stats["min"].to(norm_data.device)
    d_max = stats["max"].to(norm_data.device)
    return (norm_data - min_val) / (max_val - min_val) * (d_max - d_min + 1e-8) + d_min


def build_dataloader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def build_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "sobolev":
        return HsLoss(d=1, p=2, k=1, size_average=True)
    if loss_name == "lp":
        return LpLoss(size_average=True)
    raise ValueError(f"Unknown loss_name: {loss_name}")


def plot_epoch_summary(
    model: torch.nn.Module,
    plot_loader: DataLoader,
    x_plot: np.ndarray,
    target_stats: Dict[str, torch.Tensor],
    history: List[Dict[str, float]],
    epoch: int,
    fixed_random_idx: int,
    output_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    preds = []
    targets = []
    eta_list = []
    xi_list = []
    with torch.no_grad():
        for x_norm, y_raw, eta_raw, xi_raw in plot_loader:
            x_norm = x_norm.to(device)
            pred_norm = model(x_norm)
            pred = denormalize_from_range(pred_norm, target_stats)
            preds.append(pred.cpu())
            targets.append(y_raw)
            eta_list.append(eta_raw)
            xi_list.append(xi_raw)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    eta_raw = torch.cat(eta_list, dim=0)
    xi_raw = torch.cat(xi_list, dim=0)

    flat_pred = preds.view(preds.size(0), -1)
    flat_true = targets.view(targets.size(0), -1)
    rel_l2 = torch.norm(flat_pred - flat_true, dim=1) / (torch.norm(flat_true, dim=1) + 1e-12)
    rel_l1 = torch.sum(torch.abs(flat_pred - flat_true), dim=1) / (
        torch.sum(torch.abs(flat_true), dim=1) + 1e-12
    )

    sorted_idx = torch.argsort(rel_l2)
    num_samples = preds.size(0)
    sample_indices = {
        "best": sorted_idx[0].item(),
        "median": sorted_idx[num_samples // 2].item(),
        "worst": sorted_idx[-1].item(),
        "random": fixed_random_idx,
    }

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8])
    order = ["best", "median", "worst", "random"]

    for col, key in enumerate(order):
        idx = sample_indices[key]
        err_l2 = rel_l2[idx].item()
        err_l1 = rel_l1[idx].item()

        ax_eta = fig.add_subplot(gs[0, col])
        ax_eta.plot(x_plot, eta_raw[idx].numpy(), color="tab:blue", linewidth=1.0)
        ax_eta.set_title(f"{key.title()} η(x)", fontdict=TITLE_FONT)
        ax_eta.set_xlabel("x")
        ax_eta.grid(True, alpha=0.3)

        ax_xi = fig.add_subplot(gs[1, col])
        ax_xi.plot(x_plot, xi_raw[idx].numpy(), color="tab:green", linewidth=1.0)
        ax_xi.set_title("ξ(x)", fontdict=TITLE_FONT)
        ax_xi.set_xlabel("x")
        ax_xi.grid(True, alpha=0.3)

        ax_gxi = fig.add_subplot(gs[2, col])
        ax_gxi.plot(x_plot, targets[idx].squeeze().numpy(), label="Ground Truth", color="black", linewidth=1.0)
        ax_gxi.plot(x_plot, preds[idx].squeeze().numpy(), label="Prediction", color="tab:red", linewidth=1.0)
        ax_gxi.set_title(f"G(η, ξ)  L2={err_l2:.2e}  L1={err_l1:.2e}", fontdict=TITLE_FONT)
        ax_gxi.set_xlabel("x")
        ax_gxi.grid(True, alpha=0.3)
        ax_gxi.legend(fontsize=9)

    ax_loss = fig.add_subplot(gs[3, :])
    epochs = [h["epoch"] for h in history]
    train_vals = [h["train_loss"] for h in history]
    val_vals = [h["val_loss"] for h in history]
    ax_loss.plot(epochs, train_vals, label="Train")
    ax_loss.plot(epochs, val_vals, label="Val")
    ax_loss.set_title(f"Loss Curve (Epoch {epoch})", fontdict=TITLE_FONT)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    loss_name: str,
    plot_loader: DataLoader | None,
    x_plot: np.ndarray | None,
    target_stats: Dict[str, torch.Tensor],
    plot_dir: Path | None,
    fixed_random_idx: int,
) -> Tuple[float, float, Dict[str, torch.Tensor], List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        div_factor=1e4,
        final_div_factor=1e4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    myloss = build_loss(loss_name)

    epoch_bar = tqdm(range(epochs), desc="Epochs", leave=False)
    running_loss = 0.0
    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []
    for ep in epoch_bar:
        model.train()
        total_loss = 0.0
        n_examples = 0
        batch_bar = tqdm(train_loader, desc=f"Epoch {ep + 1} batches", leave=False)
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out.view(out.size(0), -1), y.view(y.size(0), -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            n_examples += x.size(0)
            batch_bar.set_postfix(batch_loss=loss.item())
        running_loss = total_loss / max(1, n_examples)
        val_loss = evaluate(model, val_loader, device, loss_name)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch_bar.set_postfix(train_loss=running_loss, val_loss=val_loss)
        history.append({"epoch": ep + 1, "train_loss": running_loss, "val_loss": val_loss})
        if plot_loader is not None and x_plot is not None and plot_dir is not None:
            plot_epoch_summary(
                model,
                plot_loader,
                x_plot,
                target_stats,
                history,
                ep + 1,
                fixed_random_idx,
                plot_dir / f"epoch_{ep + 1}.png",
                device,
            )
    return running_loss, best_val, best_state, history


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, loss_name: str) -> float:
    myloss = build_loss(loss_name)
    model.eval()
    total = 0.0
    n_examples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            batch_loss = myloss(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()
            total += batch_loss * x.size(0)
            n_examples += x.size(0)
    return total / max(1, n_examples)


def main():
    parser = argparse.ArgumentParser(description="Train FNO on processed Stokes dataset")
    parser.add_argument("--dataset", default="stokes_dno_training_ichoi0.npz")
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / args.dataset
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = outputs_dir / datetime.now().strftime("fno_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_np = load_processed_dataset(data_path)
    features_raw, targets_raw = build_feature_target_tensors(dataset_np)
    
    features, feat_stats = normalize_to_range(features_raw)
    targets, target_stats = normalize_to_range(targets_raw)

    base_dataset = TensorDataset(features, targets)
    n_samples = len(base_dataset)

    indices = np.arange(n_samples)
    test_size = int(n_samples * args.test_fraction)
    val_size = int(n_samples * args.val_fraction)
    test_idx = indices[:test_size]
    val_idx = indices[test_size : test_size + val_size]
    train_idx = indices[test_size + val_size :]
    
    device = torch.device(args.device)

    raw_eta = torch.tensor(dataset_np["eta"], dtype=torch.float32)
    raw_xi = torch.tensor(dataset_np["xi"], dtype=torch.float32)
    raw_gxi = torch.tensor(dataset_np["Gxi"], dtype=torch.float32)
    nx = raw_gxi.size(1)
    x_plot = np.arange(nx)

    hparams = {
        "modes": 128,
        "width": 64,
        "n_blocks": 10,
        "loss": "sobolev",
        "batch_size": 128,
        "lr": 5e-3,
        "epochs": 600,
        "weight_decay": 1e-4,
    }

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    model = FNO1d(
        hparams["modes"],
        hparams["width"],
        hparams["n_blocks"],
    ).to(device)
    
    config_payload = {**hparams, "param_count": count_params(model)}
    with open(run_dir / "config.json", "w", encoding="utf-8") as cfg:
        json.dump(config_payload, cfg, indent=2)

    train_loader = build_dataloader(
        base_dataset,
        train_idx,
        batch_size=hparams["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(base_dataset, val_idx, batch_size=hparams["batch_size"], shuffle=False)
    test_loader = build_dataloader(base_dataset, test_idx, batch_size=hparams["batch_size"], shuffle=False)

    rng = np.random.default_rng(args.seed)
    fixed_random_idx = int(rng.integers(0, len(val_idx)))
    
    # Plot dataset with raw values for visualization
    plot_dataset = TensorDataset(
        features[val_idx],
        targets_raw[val_idx],
        raw_eta[val_idx],
        raw_xi[val_idx],
    )
    plot_loader = DataLoader(plot_dataset, batch_size=hparams["batch_size"], shuffle=False)

    train_loss, best_val, best_state, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        device=device,
        loss_name=hparams["loss"],
        plot_loader=plot_loader,
        x_plot=x_plot,
        target_stats=target_stats,
        plot_dir=plots_dir,
        fixed_random_idx=fixed_random_idx,
    )

    with open(run_dir / "train_log.json", "w", encoding="utf-8") as log_file:
        json.dump(history, log_file, indent=2)

    if best_state:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, device, hparams["loss"])
    torch.save(model.state_dict(), run_dir / "best_model.pt")

    summary = {
        "dataset": args.dataset,
        "hparams": hparams,
        "train_loss": train_loss,
        "best_val_loss": best_val,
        "test_loss": test_loss,
    }
    
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()