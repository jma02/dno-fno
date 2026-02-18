from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from torch.nn import functional as F

from models.wno.wno1d import WNO1d
from models.fno import count_params

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


plt.rcParams["font.family"] = "DejaVu Serif"
TITLE_FONT = {"family": "DejaVu Serif", "weight": "bold", "size": 12}



def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    plot_loader: DataLoader | None,
    x_plot: np.ndarray | None,
    plot_dir: Path | None,
    fixed_random_idx: int,
    best_ckpt_path: Path,
) -> Tuple[float, float, Dict[str, torch.Tensor], List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
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
            pred = model(x).squeeze(-1)
            loss = F.smooth_l1_loss(pred, y, beta=0.05)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_examples += x.size(0)
            batch_bar.set_postfix(batch_loss=loss.item())
        running_loss = total_loss / max(1, n_examples)
        val_loss = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, best_ckpt_path)
        torch.save(model.state_dict(), plot_dir.parent / "last_train.pt")
        epoch_bar.set_postfix(train_loss=running_loss, val_loss=val_loss)
        history.append({"epoch": ep + 1, "train_loss": running_loss, "val_loss": val_loss})
        scheduler.step()
        plot_epoch_summary(
            model,
            plot_loader,
            x_plot,
            history,
            ep + 1,
            fixed_random_idx,
            plot_dir / f"epoch_{ep + 1}.png",
            device,
        )
    return running_loss, best_val, best_state, history


def lbfgs_finetune(
    model: torch.nn.Module,
    train_dataset: Subset,
    val_loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    plot_loader: DataLoader,
    x_plot: np.ndarray,
    plot_dir: Path,
    fixed_random_idx: int,
    start_epoch: int,
    best_val: float,
    best_ckpt_path: Path,
    max_iter: int = 20,
    history: List[Dict[str, float]] | None = None,
    chunk_size: int = 64,
) -> List[Dict[str, float]]:
    model.train()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1e-4,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )
    
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=chunk_size, sampler=sampler)
    
    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        n_total = 0
        total_samples = max(1, len(train_dataset))
        batch_bar = tqdm(train_loader, desc="LBFGS closure", leave=False)
        for x_batch, y_batch in batch_bar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch).squeeze(-1)
            raw_loss = F.smooth_l1_loss(pred, y_batch, beta=0.05, reduction="sum")
            loss = raw_loss / total_samples
            loss.backward()
            total_loss += raw_loss.item()
            n_total += x_batch.size(0)
            batch_bar.set_postfix(avg_loss=total_loss / max(1, n_total))
        return total_loss / max(1, n_total)
    
    for step in range(max_iter):
        loss = optimizer.step(closure)
        val_loss = evaluate(model, val_loader, device)
        if history is not None:
            epoch = (history[-1]["epoch"] if history else start_epoch) + 1
            history.append({"epoch": epoch, "train_loss": loss, "val_loss": val_loss})
        plot_epoch_summary(
            model,
            plot_loader,
            x_plot,
            history or [],
            (history[-1]["epoch"] if history else start_epoch) + 1,
            fixed_random_idx,
            plot_dir / f"lbfgs_epoch_{step + 1}.png",
            device,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, best_ckpt_path)

    return history if history else []


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_examples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            batch_loss = F.mse_loss(pred, y).item()
            total += batch_loss * x.size(0)
            n_examples += x.size(0)
    return total / max(1, n_examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="stokes_dno_training_ichoi0.npz")
    parser.add_argument("--train_percent", type=float, default=80.0)
    parser.add_argument("--val_percent", type=float, default=10.0)
    parser.add_argument("--test_percent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--overfit_samples", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--J_blocks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--lbfgs_steps", type=int, default=0, help="LBFGS finetuning steps after main training (0 to disable)")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / args.dataset
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = outputs_dir / datetime.now().strftime("run_wno_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_np = np.load(data_path)
    eta = dataset_np["eta"][..., None]
    xi = dataset_np["xi"][..., None]
    inputs = np.concatenate([eta, xi], axis=-1)
    features = torch.tensor(inputs, dtype=torch.float32)
    nx = features.size(1)
    raw_gxi = torch.tensor(dataset_np["Gxi"], dtype=torch.float32)
    n_samples = features.size(0)

    total_percent = args.train_percent + args.val_percent + args.test_percent
    if total_percent > 100.0 + 1e-6:
        raise ValueError("train_percent + val_percent + args.test_percent must be <= 100")

    train_count = int(n_samples * (args.train_percent / 100.0))
    val_count = int(n_samples * (args.val_percent / 100.0))
    test_count = n_samples - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Each split must have at least one sample")

    train_idx = np.arange(0, train_count)
    val_idx = np.arange(train_count, train_count + val_count)
    test_idx = np.arange(train_count + val_count, train_count + val_count + test_count)
    if args.overfit_samples > 0:
        k = min(args.overfit_samples, len(train_idx))
        subset = train_idx[:k]
        train_idx = subset
        val_idx = subset
        test_idx = subset
        print(f"[Overfit] Restricting to {k} samples for train/val/test.")
    device = torch.device(args.device)

    base_dataset = TensorDataset(features, raw_gxi)

    raw_eta = torch.tensor(dataset_np["eta"], dtype=torch.float32)
    raw_xi = torch.tensor(dataset_np["xi"], dtype=torch.float32)
    x_plot = np.arange(nx)

    config_payload = {
        "model": "wno",
        "width": args.width,
        "depth": args.depth,
        "J_blocks": args.J_blocks,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
    }
    config_payload["param_count"] = count_params(WNO1d(
        width=args.width,
        depth=args.depth,
        J_blocks=args.J_blocks,
    ))
    with open(run_dir / "config.json", "w", encoding="utf-8") as cfg:
        json.dump(config_payload, cfg, indent=2)

    model = WNO1d(
        width=args.width,
        depth=args.depth,
        J_blocks=args.J_blocks,
    ).to(device)

    best_train_ckpt = run_dir / "best_train.pt"
    best_finetune_ckpt = run_dir / "best_fine_tune.pt"

    # Compute weights for training sampler (bias toward high-norm samples)
    train_targets_norm = torch.norm(raw_gxi[train_idx].view(len(train_idx), -1), dim=1)
    weights = train_targets_norm / train_targets_norm.sum()
    sampler = WeightedRandomSampler(weights, len(train_idx), replacement=True)
    train_loader = DataLoader(Subset(base_dataset, train_idx), batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(Subset(base_dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(base_dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    rng = np.random.default_rng(args.seed)
    fixed_random_idx = int(rng.integers(0, max(1, len(val_idx))))
    plot_dataset = TensorDataset(
        features[val_idx],
        raw_gxi[val_idx],
        raw_eta[val_idx],
        raw_xi[val_idx],
    )
    plot_loader = DataLoader(plot_dataset, batch_size=args.batch_size, shuffle=False)

    train_loss, val_loss, best_state, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        plot_loader=plot_loader,
        x_plot=x_plot,
        plot_dir=plots_dir,
        fixed_random_idx=fixed_random_idx,
        best_ckpt_path=best_train_ckpt,
    )

    with open(run_dir / "train_log.json", "w", encoding="utf-8") as log_file:
        json.dump(history, log_file, indent=2)

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), run_dir / "last_train.pt")

    if args.lbfgs_steps > 0:
        print(f"Running LBFGS finetuning for {args.lbfgs_steps} steps...")
        train_subset = Subset(base_dataset, train_idx)
        history = lbfgs_finetune(
            model,
            train_subset,
            val_loader,
            device,
            weights,
            plot_loader,
            x_plot,
            plots_dir,
            fixed_random_idx,
            history[-1]["epoch"] if history else 0,
            val_loss,
            best_finetune_ckpt,
            max_iter=args.lbfgs_steps,
            history=history,
        )

    test_loss = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), run_dir / "best_model.pt")

    summary = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {run_dir}")

    plot_epoch_summary(
        model,
        plot_loader,
        x_plot,
        history,
        history[-1]["epoch"],
        fixed_random_idx,
        plots_dir / "final.png",
        device,
    )


def plot_epoch_summary(
    model: torch.nn.Module,
    plot_loader: DataLoader,
    x_plot: np.ndarray,
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
        for x_in, y_raw, eta_raw, xi_raw in plot_loader:
            x_in = x_in.to(device)
            pred = model(x_in).squeeze(-1)  # Raw output, no denormalization needed
            preds.append(pred.unsqueeze(-1).cpu())
            targets.append(y_raw)
            eta_list.append(eta_raw)
            xi_list.append(xi_raw)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    eta_raw = torch.cat(eta_list, dim=0)
    xi_raw = torch.cat(xi_list, dim=0)
    
    # Compute global y-limits for inputs (eta and xi on same scale)
    eta_min = eta_raw.min().item()
    eta_max = eta_raw.max().item()
    xi_min = xi_raw.min().item()
    xi_max = xi_raw.max().item()
    input_ymin = min(eta_min, xi_min)
    input_ymax = max(eta_max, xi_max)
    
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
        ax_eta.set_ylim(input_ymin, input_ymax)
        ax_eta.grid(True, alpha=0.3)

        ax_xi = fig.add_subplot(gs[1, col])
        ax_xi.plot(x_plot, xi_raw[idx].numpy(), color="tab:green", linewidth=1.0)
        ax_xi.set_title("ξ(x)", fontdict=TITLE_FONT)
        ax_xi.set_xlabel("x")
        ax_xi.set_ylim(input_ymin, input_ymax)
        ax_xi.grid(True, alpha=0.3)

        ax_gxi = fig.add_subplot(gs[2, col])
        ax_gxi.plot(x_plot, targets[idx].numpy(), label="Ground Truth", color="black", linewidth=1.0)
        ax_gxi.plot(x_plot, preds[idx].numpy(), label="Prediction", color="blue", linewidth=1.0, alpha=0.7)
        ax_gxi.set_title(f"G(η)ξ L2={err_l2:.2e}  L1={err_l1:.2e}", fontdict=TITLE_FONT)
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
    ax_loss.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    main()