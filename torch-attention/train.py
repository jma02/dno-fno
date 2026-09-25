# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "numpy>=2"]
# ///
"""Minimal shuffled training, validation, and gradient EMA before AdamW."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import DNO
from spectral import SpectralDNO, surface_features


class Waves(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        self.arrays = [np.load(root / f"{name}.npy", mmap_mode="r") for name in ("eta", "xi", "depth", "gxi")]
        self.rows = np.flatnonzero(np.load(root / "dataset_split.npy") == split)
        if not len(self.rows):
            raise ValueError(f"No {split!r} rows in {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        eta, xi, depth, target = (torch.tensor(np.array(a[self.rows[index]], copy=True), dtype=torch.float32).reshape(-1) for a in self.arrays)
        return eta, xi - xi.mean(), depth, target


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    # Match the existing unweighted rFFT-bin relative-L2 data term.
    error = torch.fft.rfft(prediction - target)
    reference = torch.fft.rfft(target)
    return (error.abs().norm(dim=-1) / reference.abs().norm(dim=-1).clamp_min(1e-6)).mean()


@torch.no_grad()
def build_model(architecture: str, dataset: Dataset, n: int, length: float,
                depth: int, bf16: bool) -> DNO | SpectralDNO:
    if architecture == "attention":
        return DNO(n=n, length=length, depth=depth, bf16=bf16)
    # Fit seven RMS scales from training data only, without advancing its shuffle generator.
    sums = torch.zeros(7, dtype=torch.float64)
    for eta, *_ in DataLoader(dataset, batch_size=256):
        sums += surface_features(eta, length).double().square().sum((0, 1))
    scales = (sums / (len(dataset) * n)).sqrt().float().clamp_min(1e-12)
    return SpectralDNO(n=n, length=length, depth=depth, bf16=bf16, feature_scales=scales)


def build_optimizers(model: DNO | SpectralDNO, kind: str, lr: float) -> tuple[torch.optim.Optimizer, ...]:
    if kind == "adamw":
        return (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4),)
    matrices, other = [], []
    for name, p in model.named_parameters():
        group = matrices if p.ndim == 2 and name.startswith(("spatial.", "frequency.")) else other
        group.append(p)
    return (torch.optim.Muon(matrices, lr=lr, weight_decay=1e-4, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(other, lr=lr, weight_decay=1e-4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory of physical eta/xi/depth/gxi, x, dataset_split .npy files")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--architecture", choices=("attention", "spectral"), default="attention")
    parser.add_argument("--bf16", action="store_true", help="BF16 encoder/projections/MLPs; FP32 FFTs and decoder (MPS attention core upcasts)")
    parser.add_argument("--depth", type=int, default=1, help="Attention blocks per spatial/frequency stage")
    parser.add_argument("--ema", type=float, default=.9, help="Gradient EMA decay; 0 disables smoothing")
    parser.add_argument("--out", type=Path, default=Path("outputs/torch-attention.pt"))
    args = parser.parse_args()
    if not 0 <= args.ema < 1:
        parser.error("--ema must be in [0,1)")
    torch.manual_seed(0)
    train = DataLoader(Waves(args.data, "train"), batch_size=args.batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(0))
    validation = DataLoader(Waves(args.data, "validation"), batch_size=args.batch_size)
    x = np.load(args.data / "x.npy")
    model = build_model(args.architecture, train.dataset, len(x), float((x[1] - x[0]) * len(x)),
                        args.depth, args.bf16).to(args.device)
    optimizers = build_optimizers(model, args.optimizer, args.lr)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    step = 0
    print(f"device={args.device} architecture={args.architecture} parameters={sum(p.numel() for p in model.parameters())} batch={args.batch_size} lr={args.lr} EMA={args.ema} optimizer={args.optimizer} bf16={args.bf16} depth={args.depth}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        total = 0.
        for batch in train:
            eta, xi, depth, target = (v.to(args.device) for v in batch)
            model.zero_grad(set_to_none=True)
            loss = relative_l2(model(eta, xi, depth), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            step += 1
            with torch.no_grad():
                for p, average in zip(model.parameters(), averages, strict=True):
                    average.mul_(args.ema).add_(p.grad, alpha=1 - args.ema)
                    p.grad.copy_(average / (1 - args.ema**step))
            for optimizer in optimizers:
                optimizer.step()
            total += loss.item() * len(eta)
        train_seconds = perf_counter() - started
        model.eval()
        val = 0.
        with torch.no_grad():
            for batch in validation:
                eta, xi, depth, target = (v.to(args.device) for v in batch)
                val += relative_l2(model(eta, xi, depth), target).item() * len(eta)
        print(f"epoch={epoch} step={step} train_L2={total / len(train.dataset):.6g} val_L2={val / len(validation.dataset):.6g} train_seconds={train_seconds:.2f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "gradient_ema": averages, "step": step, "epoch": epoch,
                    "n": model.n, "length": model.length, "args": vars(args)}, args.out)


if __name__ == "__main__":
    main()
