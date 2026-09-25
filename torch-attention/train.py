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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory of physical eta/xi/depth/gxi, x, dataset_split .npy files")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    model = DNO(n=len(x), length=float((x[1] - x[0]) * len(x))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    step = 0
    print(f"device={args.device} parameters={sum(p.numel() for p in model.parameters())} batch={args.batch_size} lr={args.lr} EMA={args.ema}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        total = 0.
        for batch in train:
            eta, xi, depth, target = (v.to(args.device) for v in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = relative_l2(model(eta, xi, depth), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            step += 1
            with torch.no_grad():
                for p, average in zip(model.parameters(), averages, strict=True):
                    average.mul_(args.ema).add_(p.grad, alpha=1 - args.ema)
                    p.grad.copy_(average / (1 - args.ema**step))
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
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "gradient_ema": averages, "step": step, "epoch": epoch,
                    "n": model.n, "length": model.length, "args": vars(args)}, args.out)


if __name__ == "__main__":
    main()
