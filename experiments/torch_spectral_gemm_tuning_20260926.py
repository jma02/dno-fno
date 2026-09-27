"""Matched H100 tests for GEMM autotuning and grouped Muon matrix products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-gemm-tuning")


def grouped_muon_step(self: torch.optim.Muon) -> None:
    import torch
    from torch.optim._muon import _adjust_lr

    with torch.no_grad():
        for group in self.param_groups:
            parameters, gradients, buffers = [], [], []
            self._init_group(group, parameters, gradients, buffers)
            buckets = {}
            for parameter, gradient, buffer in zip(parameters, gradients, buffers, strict=True):
                buffer.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                if update.shape[0] > update.shape[1]:
                    update = update.T
                buckets.setdefault(tuple(update.shape), []).append((parameter, update))
            for entries in buckets.values():
                x = torch.stack([update for _, update in entries]).bfloat16()
                x.div_(x.norm(dim=(-2, -1), keepdim=True).clamp_min(group["eps"]))
                a, b, c = group["ns_coefficients"]
                for _ in range(group["ns_steps"]):
                    gram = x @ x.mT
                    polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                    x = torch.baddbmm(x, polynomial, x, beta=a)
                for index, (parameter, _) in enumerate(entries):
                    update = x[index].T if parameter.shape[0] > parameter.shape[1] else x[index]
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-_adjust_lr(group["lr"], group["adjust_lr_fn"], parameter.shape))


def benchmark(variant: str, checkpoint: dict) -> tuple[dict, dict]:
    import copy
    from statistics import median
    from time import perf_counter
    from types import MethodType

    import torch
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader, relative_l2

    torch.manual_seed(0)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    optimizers = build_optimizers(model, "muon", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(copy.deepcopy(saved))
    if "grouped" in variant:
        optimizers[0].step = MethodType(grouped_muon_step, optimizers[0])
    averages = [value.cuda().clone() for value in checkpoint["gradient_ema"]]
    dataset = Waves(Path("/subset"), "train")
    loader = make_loader(dataset, 256, shuffle=True, generator=torch.Generator().manual_seed(0), bulk=True)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    fast = CudaStep(model, optimizers, averages, dataset[list(range(256))], .8,
                    checkpoint["step"], autotune="autotune" in variant)
    setup = perf_counter() - started
    losses = torch.empty(80, device="cuda")
    epochs = []
    step = 0
    for _ in range(5):
        torch.cuda.synchronize()
        started = perf_counter()
        for batch in loader:
            losses[step].copy_(fast(batch).detach())
            step += 1
        torch.cuda.synchronize()
        epochs.append(perf_counter() - started)
    assert fast.counter.item() == checkpoint["step"] + step
    assert torch.isfinite(losses).all().item()
    state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    validation = make_loader(Waves(Path("/subset"), "validation"), 64, bulk=True)
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for batch in validation:
            device = tuple(value.cuda() for value in batch)
            total.add_(relative_l2(model(*device[:3]), device[3]), alpha=len(device[0]))
    val = total.item() / len(validation.dataset)
    # Isolate full device step and optimizer update only after matched accuracy checks.
    device_times = []
    optimizer_times = []
    optimizer_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(optimizer_graph, stream=fast.stream):
        fast.compiled_updates()
    for graph, measurements in ((fast.graph, device_times), (optimizer_graph, optimizer_times)):
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            measurements.append(start.elapsed_time(end) / 10)
    ms = sum(epochs[1:]) / 64 * 1000
    return {"variant": variant, "amortized_step_ms": ms, "samples_per_second": 256000 / ms,
            "epoch_seconds": epochs, "compile_capture_seconds": setup,
            "device_step_ms": median(device_times), "optimizer_only_ms": median(optimizer_times),
            "validation_relative_l2": val, "losses": losses.cpu().tolist(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, state


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run() -> dict:
    import gc
    import os
    import sys
    from time import perf_counter

    import torch

    sys.path.insert(0, "/repo/torch-attention")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    started = perf_counter()
    checkpoint = torch.load("/data/experiments/torch_spectral_fast_h100_20260925/"
                            "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt",
                            map_location="cpu", weights_only=True)
    output = Path("/data/experiments/torch_spectral_gemm_tuning_20260926")
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "batch_size": 256, "results": []}
    reference = None
    for variant in ("current", "grouped", "autotune", "autotune_grouped"):
        result, state = benchmark(variant, checkpoint)
        if reference is None:
            reference = result, state
        else:
            previous, parameters = reference
            result["max_loss_difference"] = max(abs(a - b) for a, b in zip(previous["losses"], result["losses"], strict=True))
            result["max_parameter_difference"] = max((p - state[name]).abs().max().item() for name, p in parameters.items())
            assert result["max_loss_difference"] < 1e-5
        report["results"].append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "losses"}), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
