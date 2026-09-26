"""Compare full optimized training steps across batch sizes on one H100."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-fast-batch-sweep")


def benchmark(batch_size: int, checkpoint: dict) -> dict:
    import copy
    from statistics import median
    from time import perf_counter

    import torch

    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader

    torch.manual_seed(0)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    optimizers = build_optimizers(model, "muon", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(copy.deepcopy(saved))
    averages = [value.cuda().clone() for value in checkpoint["gradient_ema"]]
    loader = make_loader(Waves(Path("/subset"), "train"), batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(0), bulk=True)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    fast = CudaStep(model, optimizers, averages, next(iterator), .8, checkpoint["step"])
    setup_seconds = perf_counter() - started
    times, losses = [], []
    # Four replay warmups followed by 32 whole steps, including bulk CPU loading.
    for step in range(36):
        torch.cuda.synchronize()
        started = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loss = fast(batch)
        value = loss.item()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss at batch={batch_size}, step={step}")
        if step >= 4:
            times.append(elapsed)
            losses.append(value)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    assert fast.counter.item() == checkpoint["step"] + 36
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    assert all(torch.isfinite(value).all().item() for optimizer in optimizers
               for state in optimizer.state.values() for value in state.values()
               if isinstance(value, torch.Tensor))
    return {"batch_size": batch_size, "median_step_ms": median(times) * 1000,
            "samples_per_second": batch_size / median(times),
            "aggregate_samples_per_second": batch_size * len(times) / sum(times),
            "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "compile_capture_seconds": setup_seconds, "timed_steps": len(times),
            "step_seconds": times, "losses": losses, "states_finite": True}


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(run_name: str) -> dict:
    import gc
    import os
    import sys
    from time import perf_counter

    import torch

    sys.path.insert(0, "/repo/torch-attention")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    started = perf_counter()
    gpu = torch.cuda.get_device_name()
    assert "H100" in gpu, gpu
    source = ("/data/experiments/torch_spectral_fast_h100_20260925/"
              "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": gpu, "source_checkpoint": source, "lr": 1e-5, "gradient_ema": .8,
              "timing": "bulk fetch, host-to-device copy, forward/backward, EMA, Muon/AdamW; compile excluded",
              "results": []}
    for size in (256, 512, 1024, 2048):
        result = benchmark(size, checkpoint)
        report["results"].append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("step_seconds", "losses")}), flush=True)
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
def main(run_name: str = "torch_spectral_fast_batch_sweep_20260925") -> None:
    result = run.remote(run_name)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
