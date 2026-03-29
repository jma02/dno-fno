from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.fno import FNO1d, HsLoss, LpLoss, count_params
from train.util import (
    configure_runtime,
    load_training_arrays,
    normalize_to_range,
    plot_epoch_summary,
    plot_loss_history,
    resolve_device,
)


def build_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "sobolev":
        return HsLoss(d=1, p=2, k=1, size_average=True)
    if loss_name == "lp":
        return LpLoss(size_average=True)
    raise ValueError(f"Unknown loss_name: {loss_name}")


def build_loader(
    dataset: TensorDataset | Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)

def save_checkpoint(
    output_path: Path,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    train_loss: float,
    val_loss: float,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
) -> None:
    payload: dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    torch.save(payload, output_path)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            batch_loss = loss_fn(
                predictions.reshape(predictions.shape[0], -1),
                targets.reshape(targets.shape[0], -1),
            ).item()
            total_loss += batch_loss * inputs.shape[0]
            total_examples += inputs.shape[0]
    return total_loss / max(1, total_examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D FNO on the DNO dataset")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=10)
    parser.add_argument("--loss", choices=("sobolev", "lp"), default="sobolev")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mps_memory_fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = REPO_ROOT / "data" / args.dataset
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = outputs_dir / datetime.now().strftime("fno_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_arrays(dataset_path, args.sources)

    features_raw = torch.tensor(
        np.stack((dataset["eta"], dataset["xi"]), axis=-1),
        dtype=torch.float32,
    )
    targets_raw = torch.tensor(
        dataset["gxi"][..., None],
        dtype=torch.float32,
    )
    features_normalized, feature_stats = normalize_to_range(features_raw)
    targets_normalized, target_stats = normalize_to_range(targets_raw)

    base_dataset = TensorDataset(features_normalized, targets_normalized)
    val_count = int(len(base_dataset) * 0.1)
    test_count = int(len(base_dataset) * 0.1)
    train_count = len(base_dataset) - val_count - test_count
    split_generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset, _ = random_split(
        base_dataset,
        (train_count, val_count, test_count),
        generator=split_generator,
    )

    device = resolve_device(args.device)
    configure_runtime(device, args.mps_memory_fraction)

    model = FNO1d(args.modes, args.width, args.n_blocks).to(device)
    loss_fn = build_loss(args.loss)

    config_payload = {
        "dataset": args.dataset,
        "sources": list(dataset["source_names"]),
        "device": str(device),
        "modes": args.modes,
        "width": args.width,
        "n_blocks": args.n_blocks,
        "loss": args.loss,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "param_count": count_params(model),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    train_loader = build_loader(
        train_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = build_loader(
        val_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)
    fixed_random_idx = int(np.random.default_rng(args.seed).integers(0, len(val_indices)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        div_factor=1e4,
        final_div_factor=1e4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    train_loss = 0.0

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total_examples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(
                predictions.reshape(predictions.shape[0], -1),
                targets.reshape(targets.shape[0], -1),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * inputs.shape[0]
            total_examples += inputs.shape[0]

        train_loss = total_loss / max(1, total_examples)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        plot_loss_history(history, run_dir / "loss_curve.png")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            save_checkpoint(
                run_dir / "best_val_ckpt.pt",
                epoch=epoch,
                model_state=best_state,
                train_loss=train_loss,
                val_loss=val_loss,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

        plot_epoch_summary(
            model,
            val_loader,
            dataset["x"],
            feature_stats,
            target_stats,
            fixed_random_idx,
            plots_dir / f"epoch_{epoch}.png",
            device,
        )

    with open(run_dir / "train_log.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    final_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    save_checkpoint(
        run_dir / "final_ckpt.pt",
        epoch=history[-1]["epoch"],
        model_state=final_state,
        train_loss=train_loss,
        val_loss=history[-1]["val_loss"],
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )

    plot_epoch_summary(
        model,
        val_loader,
        dataset["x"],
        feature_stats,
        target_stats,
        fixed_random_idx,
        plots_dir / "final.png",
        device,
    )

    summary_payload = {
        "dataset": args.dataset,
        "sources": list(dataset["source_names"]),
        "device": str(device),
        "hparams": config_payload,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
