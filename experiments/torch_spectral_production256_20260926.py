"""256 width256 training updates with timed, prefetched full-volume random reads."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-production256")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=8,
              memory=32768, timeout=600, retries=0, scaledown_window=2)
def run(run_name: str, loader_workers: int = 4) -> dict:
    import mmap
    import os
    import sys
    from statistics import median
    from time import perf_counter

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/repo/torch-attention")
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, cpu_batches, make_loader, relative_l2

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    torch.set_num_threads(1)  # I/O workers must not each spawn a CPU compute pool.
    started = perf_counter()
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    source = Path("/data/outputs/paper_dataset_full_equal_20260916/arrays")
    print("Opening production training split", flush=True)
    tick = perf_counter()
    data = Waves(source, "train")
    for array in data.arrays:
        array._mmap.madvise(mmap.MADV_RANDOM)
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(0))[:256 * 256].reshape(256, 256)
    batches = indices.tolist()
    assert len(torch.unique(indices)) == 65536
    torch.save(torch.as_tensor(data.rows[indices.numpy()]), output / "source_rows.pt")
    setup = perf_counter() - tick
    print(f"Production rows={len(data)}; setup={setup:.3f}s", flush=True)
    reference = torch.load("/data/experiments/torch_spectral_fast_h100_20260925/"
                           "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt",
                           map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    model = SpectralDNO(n=reference["n"], length=reference["length"], width=256,
                        branches=32, heads=4, depth=2, bf16=True,
                        feature_scales=reference["model"]["feature_scales"]).cuda()
    optimizers = build_optimizers(model, "muon-grouped", 1e-5)
    averages = [torch.zeros_like(p) for p in model.parameters()]
    warmup = Waves(Path("/subset"), "train")[list(range(256))]
    tick = perf_counter()
    fast = CudaStep(model, optimizers, averages, warmup, .8, autotune=True)
    compile_seconds = perf_counter() - tick
    validation = make_loader(Waves(Path("/subset"), "validation"), 64, bulk=True)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        for batch in validation:
            device = tuple(value.cuda() for value in batch)
            total.add_(relative_l2(model(*device[:3]), device[3]), alpha=len(device[0]))
        return total.item() / len(validation.dataset)

    tick = perf_counter()
    initial_val = evaluate()
    initial_val_seconds = perf_counter() - tick
    model.train()
    report = {"run_name": run_name, "source": str(source), "training_rows": len(data),
              "gpu": torch.cuda.get_device_name(), "batch_size": 256, "width": 256,
              "parameters": sum(p.numel() for p in model.parameters()), "depth": 2, "heads": 4,
              "branches": 32, "bf16": True, "lr": 1e-5, "gradient_ema": .8,
              "optimizer": "grouped Muon transformer + AdamW remainder",
              "initialization": "fresh seed0; feature RMS scales reused from previous training-only4096-row fit",
              "sampling": "First65536 unique rows of seed0 torch.randperm over full production training split; order preserved",
              "loading": f"Direct production mmap gathers, MADV_RANDOM,{loader_workers} concurrent prefetched batches; no staged subset",
              "validation": [{"step": 0, "relative_l2": initial_val}], "validation_rows": 1024,
              "validation_scope": "Existing fixed production-derived held-out subset; original validation split, disjoint from training",
              "data_setup_seconds": setup, "compile_capture_seconds": compile_seconds,
              "initial_validation_seconds": initial_val_seconds, "history": [], "windows": []}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "windows")}), flush=True)

    loader = DataLoader(data, batch_size=None, sampler=batches)
    iterator = iter(cpu_batches(loader, workers=loader_workers))
    losses = torch.empty(256, device="cuda")
    events = []
    completed = 0
    torch.cuda.synchronize()
    training_started = perf_counter()
    window_start = training_started
    try:
        for index in range(256):
            wait_start = perf_counter()
            # Stop at a batch boundary before the hard600s function limit.
            remaining = 510 - (perf_counter() - started)
            if remaining <= 0:
                report["stopped_reason"] = "soft510s runtime limit"
                break
            batch = next(iterator)
            waiting = perf_counter() - wait_start
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            losses[index].copy_(fast(batch).detach())
            end.record()
            events.append((begin, end))
            completed += 1
            report["history"].append({"step": completed, "consumer_wait_seconds": waiting})
            if completed == 1 or completed % 16 == 0:
                torch.cuda.synchronize()
                values = losses[:completed].cpu().tolist()
                assert all(torch.isfinite(torch.tensor(values)))
                for row, value in zip(report["history"], values, strict=True):
                    row["training_relative_l2"] = value
                window = {"step": completed, "elapsed_training_seconds": perf_counter() - training_started,
                          "window_seconds": perf_counter() - window_start,
                          "last_loss": values[-1], "recent_mean_loss": sum(values[-16:]) / len(values[-16:])}
                report["windows"].append(window)
                print("PRODUCTION_PROGRESS " + json.dumps(window), flush=True)
                (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                window_start = perf_counter()
    finally:
        iterator.close()
    torch.cuda.synchronize()
    training_seconds = perf_counter() - training_started
    values = losses[:completed].cpu().tolist()
    for row, value, (begin, end) in zip(report["history"], values, events, strict=True):
        row.update(training_relative_l2=value, gpu_update_ms=begin.elapsed_time(end))
    assert fast.counter.item() == completed
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    tick = perf_counter()
    final_val = evaluate()
    report["validation"].append({"step": completed, "relative_l2": final_val})
    report.update(final_validation_seconds=perf_counter() - tick, completed_steps=completed,
                  training_seconds=training_seconds,
                  samples_per_second=completed * 256 / training_seconds,
                  amortized_step_ms=training_seconds * 1000 / completed if completed else None,
                  median_gpu_update_ms=median(row["gpu_update_ms"] for row in report["history"]) if completed else None,
                  total_consumer_wait_seconds=sum(row["consumer_wait_seconds"] for row in report["history"]),
                  function_seconds=perf_counter() - started)
    torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                "gradient_ema": averages, "step": completed, "n": model.n, "length": model.length,
                "args": {"width": 256, "depth": 2, "architecture": "spectral", "bf16": True,
                         "batch_size": 256, "lr": 1e-5, "ema": .8, "optimizer": "muon-grouped",
                         "loader_workers": loader_workers},
                "source_rows": "source_rows.pt"}, output / "checkpoint.pt")
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("PRODUCTION_RESULT " + json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_production256_prefetch4_20260926", loader_workers: int = 4) -> None:
    result = run.remote(run_name, loader_workers)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
