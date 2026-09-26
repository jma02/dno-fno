"""Fresh Muon/AdamW training on the downloaded real subset, on Metal or CUDA."""

import argparse
from contextlib import nullcontext
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
from spectral import SpectralDNO  # noqa: E402
from train import CudaStep, Waves, build_model, build_optimizers, make_loader, prefetch_batches, relative_l2  # noqa: E402


@torch.no_grad()
def evaluate(model: DNO | SpectralDNO, loader: DataLoader, prefetch: bool = False) -> float:
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device) if device.type == "cuda" else 0.
    for batch in prefetch_batches(loader) if prefetch else loader:
        eta, xi, depth, target = (v.to(device) for v in batch)
        loss = relative_l2(model(eta, xi, depth), target)
        if device.type == "cuda":
            total.add_(loss, alpha=len(eta))
        else:
            total += loss.item() * len(eta)
    value = (total.item() if device.type == "cuda" else total) / len(loader.dataset)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite validation loss")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--architecture", choices=("attention", "spectral"), default="attention")
    parser.add_argument("--fast-step", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, help="Run complete passes and validate each epoch; otherwise run 50 steps")
    parser.add_argument("--tag", default="torch_attention_real_pilot_20260925")
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT.parent / "pilot-checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.fast_step and args.device != "cuda":
        parser.error("--fast-step requires CUDA")
    if args.prefetch and not args.fast_step:
        parser.error("--prefetch requires --fast-step")
    synchronize = torch.mps.synchronize if args.device == "mps" else torch.cuda.synchronize
    torch.manual_seed(0)
    data = ROOT.parent / "local-data/paper_equal_subset_20260924"
    train = make_loader(Waves(data, "train"), args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0), bulk=args.fast_step, pin_memory=args.prefetch)
    validation = make_loader(Waves(data, "validation"), 64, bulk=args.fast_step, pin_memory=args.prefetch)
    steps = args.epochs * len(train) if args.epochs is not None else 50
    validation_every = len(train) if args.epochs is not None else 10
    x = np.load(data / "x.npy")
    model = build_model(args.architecture, train.dataset, len(x), float((x[1] - x[0]) * len(x)),
                        args.depth, args.bf16).to(args.device)
    optimizers = build_optimizers(model, "muon", 1e-5)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    history, validations = [], []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    validations.append({"step": 0, "relative_l2": evaluate(model, validation, args.prefetch)})
    print("validation", validations[-1], flush=True)
    samples_seen = 0
    fast_step = None
    setup_seconds = 0.
    if args.fast_step:
        model.train()
        tick = perf_counter()
        fast_step = CudaStep(model, optimizers, averages,
                             train.dataset[list(range(min(args.batch_size, len(train.dataset))))], .8)
        setup_seconds = perf_counter() - tick
    iterator = iter(prefetch_batches(train) if args.prefetch else train)
    pending = []
    segment_started = perf_counter()
    for step in range(1, steps + 1):
        model.train()
        if not args.fast_step:
            synchronize()
        tick = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(prefetch_batches(train) if args.prefetch else train)
            batch = next(iterator)
        samples_seen += len(batch[0])
        if args.fast_step:
            loss = fast_step(batch)
            pending.append((step, loss.detach().clone()))
        else:
            eta, xi, depth, target = (v.to(args.device) for v in batch)
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
        flush = step % 32 == 0 or step % validation_every == 0 or step == steps
        if args.fast_step:
            if flush:
                values = torch.stack([value for _, value in pending]).cpu().tolist()
                seconds = (perf_counter() - segment_started) / len(pending)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite training loss")
                history.extend({"step": index, "loss_before_update": value, "seconds": seconds}
                               for (index, _), value in zip(pending, values, strict=True))
                pending.clear()
        else:
            value = loss.item()
            synchronize()
            history.append({"step": step, "loss_before_update": value, "seconds": perf_counter() - tick})
        if step % validation_every == 0:
            validations.append({"step": step, "relative_l2": evaluate(model, validation, args.prefetch)})
            print("train", history[-1], "validation", validations[-1], flush=True)
        if flush:
            segment_started = perf_counter()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(a).all() for a in averages)
    assert all(torch.isfinite(v).all() for o in optimizers for s in o.state.values()
               for v in s.values() if isinstance(v, torch.Tensor))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                "gradient_ema": averages, "step": steps, "epochs": args.epochs, "lr": 1e-5, "ema": .8,
                "seed": 0, "n": model.n, "length": model.length, "depth": args.depth,
                "bf16": args.bf16, "architecture": args.architecture,
                "fast_step": args.fast_step,
                "prefetch": args.prefetch,
                "batch_size": args.batch_size, "data": str(data)}, checkpoint)
    result = {
        "source": str(data), "device": args.device, "torch": torch.__version__, "seed": 0,
        "optimizer": "Muon transformer matrices + AdamW remainder", "lr": 1e-5,
        "gradient_ema_decay": .8, "batch_size": args.batch_size, "optimizer_steps": steps,
        "epochs": args.epochs, "steps_per_epoch": len(train), "validation_every": validation_every,
        "bf16": args.bf16, "depth_per_stage": args.depth,
        "architecture": args.architecture,
        "fast_step": args.fast_step,
        "prefetch": args.prefetch,
        "compile_capture_seconds": setup_seconds,
        "step_timing": "Amortized wall time across synchronized groups of at most32 updates; excludes validation and compile/capture" if args.fast_step else "Synchronized per-step wall time",
        "feature_scales": model.feature_scales.tolist() if isinstance(model, SpectralDNO) else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_rows": len(train.dataset), "training_samples_seen": samples_seen,
        "validation_rows": len(validation.dataset),
        "scope": f"Fresh initialization; 4096 real training rows reshuffled each pass. All 1024 held-out validation rows, in batches of 64, after every {validation_every} updates. Relative-L2 data loss only; spectral variant fits surface-feature RMS scales on training rows. Physical targets; no physics losses.",
        "history": history, "validation": validations,
        "training_seconds": sum(row["seconds"] for row in history),
        "steady_median_ms": 1000 * median(row["seconds"] for row in history[3:]),
        "wall_seconds": perf_counter() - started, "finite": True, "checkpoint": str(checkpoint),
    }
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        # Inspect backend selection after timing; this takes no optimizer update.
        model.train()
        eta, xi, depth, target = (v.to(args.device) for v in batch)
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            with torch.cuda.stream(fast_step.stream) if fast_step is not None else nullcontext():
                relative_l2(model(eta, xi, depth), target).backward()
                synchronize()
        result["attention_operators"] = sorted({event.key for event in profile.key_averages()
                                                if "attention" in event.key.lower()})
    (args.output_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
