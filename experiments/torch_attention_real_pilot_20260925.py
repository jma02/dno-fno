"""Fifty fresh Muon/AdamW steps on the downloaded real subset, on Metal or CUDA."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from model import DNO  # noqa: E402
from train import Waves, build_optimizers, relative_l2  # noqa: E402


@torch.no_grad()
def evaluate(model: DNO, loader: DataLoader) -> float:
    model.eval()
    total = 0.
    device = next(model.parameters()).device
    for batch in loader:
        eta, xi, depth, target = (v.to(device) for v in batch)
        total += relative_l2(model(eta, xi, depth), target).item() * len(eta)
    value = total / len(loader.dataset)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite validation loss")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--tag", default="torch_attention_real_pilot_20260925")
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT.parent / "pilot-checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    synchronize = torch.mps.synchronize if args.device == "mps" else torch.cuda.synchronize
    torch.manual_seed(0)
    data = ROOT.parent / "local-data/paper_equal_subset_20260924"
    train = DataLoader(Waves(data, "train"), batch_size=64, shuffle=True,
                       generator=torch.Generator().manual_seed(0))
    validation = DataLoader(Waves(data, "validation"), batch_size=64)
    x = np.load(data / "x.npy")
    model = DNO(n=len(x), length=float((x[1] - x[0]) * len(x)), depth=args.depth, bf16=args.bf16).to(args.device)
    optimizers = build_optimizers(model, "muon", 1e-5)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    history, validations = [], []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    validations.append({"step": 0, "relative_l2": evaluate(model, validation)})
    print("validation", validations[-1], flush=True)
    iterator = iter(train)
    for step in range(1, 51):
        model.train()
        synchronize()
        tick = perf_counter()
        eta, xi, depth, target = (v.to(args.device) for v in next(iterator))
        model.zero_grad(set_to_none=True)
        loss = relative_l2(model(eta, xi, depth), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        with torch.no_grad():
            for p, average in zip(model.parameters(), averages, strict=True):
                average.mul_(.8).add_(p.grad, alpha=.2)
                p.grad.copy_(average / (1 - .8**step))
        for optimizer in optimizers:
            optimizer.step()
        value = loss.item()
        synchronize()
        history.append({"step": step, "loss_before_update": value, "seconds": perf_counter() - tick})
        if step % 10 == 0:
            validations.append({"step": step, "relative_l2": evaluate(model, validation)})
            print("train", history[-1], "validation", validations[-1], flush=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(a).all() for a in averages)
    assert all(torch.isfinite(v).all() for o in optimizers for s in o.state.values()
               for v in s.values() if isinstance(v, torch.Tensor))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                "gradient_ema": averages, "step": 50, "lr": 1e-5, "ema": .8,
                "seed": 0, "n": model.n, "length": model.length, "depth": args.depth,
                "bf16": args.bf16, "data": str(data)}, checkpoint)
    result = {
        "source": str(data), "device": args.device, "torch": torch.__version__, "seed": 0,
        "optimizer": "Muon transformer matrices + AdamW remainder", "lr": 1e-5,
        "gradient_ema_decay": .8, "batch_size": 64, "optimizer_steps": 50,
        "bf16": args.bf16, "depth_per_stage": args.depth,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_rows": len(train.dataset), "training_samples_seen": 3200,
        "validation_rows": len(validation.dataset),
        "scope": "Fresh initialization, first3200 unique rows of a shuffled4096-row real training split. All1024 held-out validation rows after every10 updates. Relative-L2 data loss only, no normalization or physics losses.",
        "history": history, "validation": validations,
        "training_seconds": sum(row["seconds"] for row in history),
        "steady_median_ms": 1000 * median(row["seconds"] for row in history[3:]),
        "wall_seconds": perf_counter() - started, "finite": True, "checkpoint": str(checkpoint),
    }
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        # Inspect backend selection after timing; this takes no optimizer update.
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            relative_l2(model(eta, xi, depth), target).backward()
            synchronize()
        result["attention_operators"] = sorted({event.key for event in profile.key_averages()
                                                if "attention" in event.key.lower()})
    (args.output_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
