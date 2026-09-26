"""Same-GPU width sweep, keeping batch, heads, depth, branches and precision fixed."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-widths")


def benchmark(width: int, checkpoint: dict) -> dict:
    from statistics import median
    from time import perf_counter

    import torch
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader

    torch.manual_seed(0)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], width=width,
                        branches=32, heads=4, depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    optimizers = build_optimizers(model, "muon-grouped", 1e-5)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    dataset = Waves(Path("/subset"), "train")
    loader = make_loader(dataset, 256, shuffle=True, generator=torch.Generator().manual_seed(0), bulk=True)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    fast = CudaStep(model, optimizers, averages, dataset[list(range(256))], .8, autotune=True)
    setup = perf_counter() - started
    losses = torch.empty(144, device="cuda")
    epochs = []
    step = 0
    for _ in range(9):
        torch.cuda.synchronize()
        started = perf_counter()
        for batch in loader:
            losses[step].copy_(fast(batch).detach())
            step += 1
        torch.cuda.synchronize()
        epochs.append(perf_counter() - started)
    assert fast.counter.item() == step == 144
    assert torch.isfinite(losses).all().item()
    assert all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(a).all() for a in averages)
    # Training loss is a finite-update diagnostic, not a convergence comparison.
    final_loss = losses[-1].item()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fast.graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 20)
    ms = sum(epochs[1:]) / 128 * 1000
    return {"width": width, "heads": 4, "head_dim": width // 4,
            "parameters": sum(p.numel() for p in model.parameters()),
            "amortized_step_ms": ms, "samples_per_second": 256000 / ms,
            "epoch_seconds": epochs, "compile_capture_seconds": setup,
            "device_step_ms": median(times), "device_samples_ms": times,
            "last_training_relative_l2": final_loss,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checked_training_steps": step, "finite_updates": True}


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(widths: tuple[int, ...], run_name: str) -> dict:
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
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "batch_size": 256,
              "depth": 2, "branches": 32, "bf16": True, "autotune": True,
              "optimizer": "grouped Muon + AdamW", "lr": 1e-5, "gradient_ema": .8,
              "initialization": "fresh seed0; checkpoint used only for grid and training feature scales",
              "timing": "8 shuffled cached-subset epochs after one warmup epoch; excludes compile and validation",
              "results": []}
    # Repeat the original width last to expose timing drift across compilation.
    for width in widths:
        result = benchmark(width, checkpoint)
        report["results"].append(result)
        print("WIDTH_RESULT " + json.dumps(result), flush=True)
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
def main(widths: str = "64,96,128,64", run_name: str = "torch_spectral_widths_aligned_20260926") -> None:
    result = run.remote(tuple(map(int, widths.split(","))), run_name)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
