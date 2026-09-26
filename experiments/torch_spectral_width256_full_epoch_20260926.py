"""One full production epoch of width256 spectral DNO with ordered CPU readers."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-width256-full-epoch")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=8,
              memory=32768, timeout=3600, retries=0, scaledown_window=2)
def run(run_name: str) -> dict:
    import math
    import os
    import sys
    from statistics import median
    from time import perf_counter

    import torch

    sys.path.insert(0, "/repo/torch-attention")
    from model import baseline
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, cpu_batches, make_loader, relative_l2

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    torch.set_num_threads(1)
    started = perf_counter()
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    source = Path("/data/outputs/paper_dataset_full_equal_20260916/arrays")
    print("FULL_EPOCH Opening production arrays", flush=True)
    tick = perf_counter()
    train_data = Waves(source, "train")
    val_data = Waves(source, "validation")
    loader = make_loader(train_data, 256, shuffle=True, generator=torch.Generator().manual_seed(0), bulk=True)
    setup_seconds = perf_counter() - tick
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
    report = {"run_name": run_name, "source": str(source), "gpu": torch.cuda.get_device_name(),
              "training_rows": len(train_data), "validation_rows": len(val_data),
              "expected_steps": len(loader), "batch_size": 256, "width": 256, "depth": 2,
              "heads": 4, "branches": 32, "parameters": sum(p.numel() for p in model.parameters()),
              "lr": 1e-5, "gradient_ema": .8, "optimizer": "grouped Muon transformer + AdamW remainder",
              "weight_decay": 1e-4, "bf16": True, "loader_workers": 4, "seed": 0,
              "initialization": "Fresh seed0; unchanged physical feature scales from prior4096 training-only rows",
              "loss": "Mean per-example unweighted rFFT-bin relative L2; no physics regularizers",
              "sampling": "One full globally shuffled training epoch; tail retained; DataLoader generator seed0",
              "data_setup_seconds": setup_seconds, "compile_capture_seconds": compile_seconds,
              "progress": []}

    @torch.no_grad()
    def evaluate(data: Waves, analytic: bool = False) -> dict:
        tick = perf_counter()
        model.eval()
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        base_total = torch.zeros_like(total)
        count = 0
        evaluation = make_loader(data, 256, bulk=True)
        for index, batch in enumerate(cpu_batches(evaluation, workers=4), start=1):
            device = tuple(value.cuda() for value in batch)
            prediction = model(*device[:3])
            total.add_(relative_l2(prediction, device[3]), alpha=len(device[0]))
            if analytic:
                base = baseline(*device[:3], model.length)
                base_total.add_(relative_l2(base, device[3]), alpha=len(device[0]))
            count += len(device[0])
            if index % 512 == 0:
                print("VALIDATION_PROGRESS " + json.dumps({"batches": index, "examples": count,
                      "relative_l2_so_far": total.item() / count,
                      "seconds": perf_counter() - tick}), flush=True)
        value = total.item() / count
        assert count == len(data) and math.isfinite(value)
        result = {"examples": count, "relative_l2": value, "seconds": perf_counter() - tick}
        if analytic:
            result["analytic_baseline_relative_l2"] = base_total.item() / count
        return result

    report["initial_subset_validation"] = evaluate(Waves(Path("/subset"), "validation"))
    model.train()
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("FULL_EPOCH_CONFIG " + json.dumps(report), flush=True)

    def save_checkpoint(step: int, examples: int, complete: bool) -> None:
        target = output / "checkpoint.pt"
        temporary = output / "checkpoint.tmp"
        torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                    "gradient_ema": averages, "step": step, "examples_seen": examples,
                    "epoch": 1 if complete else 0, "n": model.n, "length": model.length,
                    "args": {k: report[k] for k in ("width", "depth", "heads", "branches", "bf16", "lr",
                                                   "gradient_ema", "batch_size", "loader_workers", "seed")},
                    "architecture": "spectral", "source": str(source),
                    "sampler_recovery": "Recreate seed0 bulk DataLoader and skip step batches in its sampler, preserving base-seed draw"}, temporary)
        temporary.replace(target)
        volume.commit()

    losses = torch.empty(len(loader), device="cuda")
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    waits = 0.
    checkpoint_seconds = 0.
    examples = 0
    events = []
    iterator = iter(cpu_batches(loader, workers=4))
    torch.cuda.synchronize()
    training_started = perf_counter()
    window_start = training_started
    for index in range(len(loader)):
        tick = perf_counter()
        batch = next(iterator)
        waits += perf_counter() - tick
        sample_event = index % 512 == 0 and len(batch[0]) == 256
        if sample_event:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        loss = fast(batch)
        losses[index].copy_(loss.detach())
        total.add_(loss.detach(), alpha=len(batch[0]))
        if sample_event:
            end.record()
            events.append((begin, end))
        step = index + 1
        examples += len(batch[0])
        if step % 32 == 0 and not torch.isfinite(losses[step - 32:step]).all().item():
            save_checkpoint(step, examples, False)
            raise FloatingPointError(f"Nonfinite training loss at step{step}")
        if step == 1 or step % 512 == 0 or step == len(loader):
            torch.cuda.synchronize()
            recent = losses[max(0, step - 512):step].mean().item()
            progress = {"step": step, "examples": examples, "last_loss": loss.item(),
                        "recent_mean_loss": recent, "mean_train_loss": total.item() / examples,
                        "training_wall_seconds": perf_counter() - training_started,
                        "window_seconds": perf_counter() - window_start,
                        "consumer_wait_seconds": waits}
            report["progress"].append(progress)
            print("TRAIN_PROGRESS " + json.dumps(progress), flush=True)
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            window_start = perf_counter()
        if step % 8192 == 0 or step == len(loader):
            tick = perf_counter()
            save_checkpoint(step, examples, step == len(loader))
            checkpoint_seconds += perf_counter() - tick
    assert next(iterator, None) is None
    torch.cuda.synchronize()
    training_seconds = perf_counter() - training_started
    assert examples == len(train_data)
    assert fast.counter.item() == step == len(loader)
    assert torch.isfinite(losses).all().item()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    report.update(completed_epochs=1, completed_steps=step, examples_seen=examples,
                  train_relative_l2=total.item() / examples,
                  training_wall_seconds=training_seconds, checkpoint_seconds=checkpoint_seconds,
                  training_seconds_excluding_checkpoints=training_seconds - checkpoint_seconds,
                  amortized_batch_ms=(training_seconds - checkpoint_seconds) * 1000 / step,
                  samples_per_second=examples / (training_seconds - checkpoint_seconds),
                  median_sampled_gpu_update_ms=median(begin.elapsed_time(end) for begin, end in events),
                  consumer_wait_seconds=waits, losses=losses.cpu().tolist())
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("TRAIN_FINISHED " + json.dumps({k: v for k, v in report.items() if k not in ("losses", "progress")}), flush=True)
    report["full_validation"] = evaluate(val_data, analytic=True)
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("FULL_EPOCH_RESULT " + json.dumps({k: v for k, v in report.items() if k not in ("losses", "progress")}), flush=True)
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_width256_full_epoch_20260926") -> None:
    result = run.remote(run_name)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
