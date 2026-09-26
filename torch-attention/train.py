# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "numpy>=2"]
# ///
"""Minimal shuffled training, validation, and gradient EMA before AdamW."""

import argparse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import math
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

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

    def __getitem__(self, index: int | list[int]) -> tuple[Tensor, ...]:
        shape = (len(index), -1) if isinstance(index, list) else (-1,)
        eta, xi, depth, target = (torch.as_tensor(np.array(a[self.rows[index]], copy=True), dtype=torch.float32).reshape(shape) for a in self.arrays)
        return eta, xi - xi.mean(-1, keepdim=True), depth, target


def cache_data(source: Path, destination: Path) -> Path:
    """Stage immutable arrays with sequential reads; reuse only a matching cache."""
    names = tuple(f"{name}.npy" for name in ("eta", "xi", "depth", "gxi", "x", "dataset_split"))
    signature = {"source": str(source.resolve()), "files": {
        name: {"bytes": (source / name).stat().st_size, "mtime_ns": (source / name).stat().st_mtime_ns}
        for name in names}}
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "cache_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) == signature:
        if all((destination / name).is_file() and (destination / name).stat().st_size == info["bytes"]
               for name, info in signature["files"].items()):
            return destination
    if source.resolve() == destination.resolve():
        raise ValueError("Data cache must be separate from the source arrays")
    # Remove the completion marker before any replacement; interrupted copies
    # cannot be mistaken for a complete cache. Source files are never modified.
    manifest.unlink(missing_ok=True)
    with TemporaryDirectory(prefix="stage-", dir=destination) as temporary:
        staging = Path(temporary)
        with ThreadPoolExecutor(max_workers=4) as executor:
            copies = [executor.submit(shutil.copyfile, source / name, staging / name) for name in names]
            for future in copies:
                future.result()
        for name in names:
            if (staging / name).stat().st_size != signature["files"][name]["bytes"]:
                raise OSError(f"Incomplete data cache copy: {name}")
            (staging / name).replace(destination / name)
        marker = staging / "cache_manifest.json"
        marker.write_text(json.dumps(signature, indent=2) + "\n")
        marker.replace(manifest)
    return destination


def make_loader(dataset: Waves, batch_size: int, *, shuffle: bool = False,
                generator: torch.Generator | None = None, bulk: bool = False,
                pin_memory: bool = False) -> DataLoader:
    if not bulk:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, pin_memory=pin_memory)
    sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=None, generator=generator, pin_memory=pin_memory,
                      sampler=BatchSampler(sampler, batch_size, drop_last=False))


def prefetch_batches(loader: DataLoader) -> Iterator[tuple[Tensor, ...]]:
    """Fetch/pin one CPU batch ahead and overlap CUDA copies with queued compute."""
    iterator = iter(loader)
    transfer = torch.cuda.Stream()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-prefetch") as executor:
        pending = executor.submit(next, iterator, None)
        while (batch := pending.result()) is not None:
            pending = executor.submit(next, iterator, None)
            with torch.cuda.stream(transfer):
                device_batch = tuple(value.to("cuda", non_blocking=True) for value in batch)
            consumer = torch.cuda.current_stream()
            consumer.wait_stream(transfer)
            for value in device_batch:
                value.record_stream(consumer)
            yield device_batch


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
    loader = make_loader(dataset, 256, bulk=True) if isinstance(dataset, Waves) else DataLoader(dataset, batch_size=256)
    for eta, *_ in loader:
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


class CudaStep:
    """Replay forward/backward/EMA/optimizer work; warmup never advances training."""

    def __init__(self, model: DNO | SpectralDNO, optimizers: tuple[torch.optim.Optimizer, ...],
                 averages: list[Tensor], batch: tuple[Tensor, ...], ema: float, step: int = 0) -> None:
        self.model, self.optimizers, self.averages, self.ema = model, optimizers, averages, ema
        self.parameters = list(model.parameters())
        self.inputs = tuple(value.to("cuda").clone() for value in batch)
        self.compiled = torch.compile(model, fullgraph=True, options={"triton.cudagraphs": False})
        self.counter = torch.tensor(float(step), dtype=torch.float64, device="cuda")
        saved_model = {name: value.clone() for name, value in model.state_dict().items()}
        saved_averages = [value.clone() for value in averages]
        saved_states = [{p: {k: v.clone() if isinstance(v, Tensor) else v for k, v in state.items()}
                         for p, state in optimizer.state.items()} for optimizer in optimizers]
        for optimizer in optimizers:
            if isinstance(optimizer, torch.optim.AdamW):
                for group in optimizer.param_groups:
                    group["capturable"] = True
                for state in optimizer.state.values():
                    state["step"] = state["step"].cuda()
        self.compiled_updates = torch.compile(self.apply_updates, options={"triton.cudagraphs": False})
        # Stable gradient buffers allow a partial last batch to use eager execution
        # without changing the addresses retained by the full-batch CUDA graph.
        for p in self.parameters:
            p.grad = torch.zeros_like(p)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.update(self.inputs)
        torch.cuda.current_stream().wait_stream(self.stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = self.update(self.inputs)
        with torch.no_grad():
            model.load_state_dict(saved_model)
            for optimizer, saved in zip(optimizers, saved_states, strict=True):
                for p, state in optimizer.state.items():
                    for key, value in state.items():
                        if isinstance(value, Tensor):
                            if key in saved.get(p, {}):
                                value.copy_(saved[p][key])
                            else:
                                value.zero_()
            for current, saved in zip(averages, saved_averages, strict=True):
                current.copy_(saved)
            self.counter.fill_(step)
            model.zero_grad(set_to_none=False)
        torch.cuda.synchronize()

    def update(self, batch: tuple[Tensor, ...]) -> Tensor:
        torch._foreach_zero_([p.grad for p in self.parameters])
        forward = self.compiled if batch[0].shape == self.inputs[0].shape else self.model
        loss = relative_l2(forward(*batch[:3]), batch[3])
        loss.backward()
        self.compiled_updates()
        return loss

    @torch.no_grad()
    def apply_updates(self) -> None:
        self.counter.add_(1)
        correction = (1 - self.ema**self.counter).float()
        gradients = [p.grad for p in self.parameters]
        torch._foreach_mul_(self.averages, self.ema)
        torch._foreach_add_(self.averages, gradients, alpha=1 - self.ema)
        torch._foreach_copy_(gradients, torch._foreach_div(self.averages, correction))
        for optimizer in self.optimizers:
            optimizer.step()

    def __call__(self, batch: tuple[Tensor, ...]) -> Tensor:
        if batch[0].shape != self.inputs[0].shape:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                loss = self.update(tuple(value.to("cuda") for value in batch))
            torch.cuda.current_stream().wait_stream(self.stream)
            return loss
        for buffer, value in zip(self.inputs, batch, strict=True):
            buffer.copy_(value, non_blocking=True)
        self.graph.replay()
        return self.loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory of physical eta/xi/depth/gxi, x, dataset_split .npy files")
    parser.add_argument("--cache-data", type=Path, help="Stage required arrays here before training; use local ephemeral storage, not another remote-volume path")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--architecture", choices=("attention", "spectral"), default="attention")
    parser.add_argument("--fast-step", action="store_true", help="Compile and CUDA-graph training with bulk batch fetch; preserves model and optimizer settings")
    parser.add_argument("--prefetch", action="store_true", help="With --fast-step, fetch/pin on a CPU thread and transfer on a separate CUDA stream")
    parser.add_argument("--bf16", action="store_true", help="BF16 encoder/projections/MLPs; FP32 FFTs and decoder (MPS attention core upcasts)")
    parser.add_argument("--depth", type=int, default=1, help="Attention blocks per spatial/frequency stage")
    parser.add_argument("--ema", type=float, default=.9, help="Gradient EMA decay; 0 disables smoothing")
    parser.add_argument("--out", type=Path, default=Path("outputs/torch-attention.pt"))
    args = parser.parse_args()
    if not 0 <= args.ema < 1:
        parser.error("--ema must be in [0,1)")
    if args.fast_step and args.device != "cuda":
        parser.error("--fast-step requires --device cuda")
    if args.prefetch and not args.fast_step:
        parser.error("--prefetch requires --fast-step")
    data = args.data
    if args.cache_data is not None:
        started = perf_counter()
        data = cache_data(data, args.cache_data)
        print(f"data_cache={data} cache_seconds={perf_counter() - started:.2f}", flush=True)
    torch.manual_seed(0)
    train = make_loader(Waves(data, "train"), args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0), bulk=args.fast_step, pin_memory=args.prefetch)
    validation = make_loader(Waves(data, "validation"), args.batch_size,
                             bulk=args.fast_step, pin_memory=args.prefetch)
    x = np.load(data / "x.npy")
    model = build_model(args.architecture, train.dataset, len(x), float((x[1] - x[0]) * len(x)),
                        args.depth, args.bf16).to(args.device)
    optimizers = build_optimizers(model, args.optimizer, args.lr)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    step = 0
    fast_step = None
    print(f"device={args.device} architecture={args.architecture} parameters={sum(p.numel() for p in model.parameters())} batch={args.batch_size} lr={args.lr} EMA={args.ema} optimizer={args.optimizer} bf16={args.bf16} depth={args.depth}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        total = torch.zeros((), dtype=torch.float64, device=args.device) if args.fast_step else 0.
        for batch in prefetch_batches(train) if args.prefetch else train:
            if args.fast_step:
                if fast_step is None:
                    fast_step = CudaStep(model, optimizers, averages, batch, args.ema, step)
                loss = fast_step(batch)
                step += 1
                total.add_(loss.detach(), alpha=len(batch[0]))
                # Reading every loss would serialize loading/copies with GPU work.
                # Bound queued work and detect nonfinite training within 32 updates.
                if step % 32 == 0 and not math.isfinite(total.item()):
                    raise FloatingPointError("Nonfinite training loss")
                continue
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
        if args.fast_step:
            total = total.item()
            if not math.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
        train_seconds = perf_counter() - started
        model.eval()
        val = torch.zeros((), dtype=torch.float64, device=args.device) if args.fast_step else 0.
        with torch.no_grad():
            for batch in prefetch_batches(validation) if args.prefetch else validation:
                eta, xi, depth, target = (v.to(args.device) for v in batch)
                loss = relative_l2(model(eta, xi, depth), target)
                if args.fast_step:
                    val.add_(loss, alpha=len(eta))
                else:
                    val += loss.item() * len(eta)
        if args.fast_step:
            val = val.item()
        print(f"epoch={epoch} step={step} train_L2={total / len(train.dataset):.6g} val_L2={val / len(validation.dataset):.6g} train_seconds={train_seconds:.2f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "gradient_ema": averages, "step": step, "epoch": epoch,
                    "n": model.n, "length": model.length, "args": vars(args)}, args.out)


if __name__ == "__main__":
    main()
