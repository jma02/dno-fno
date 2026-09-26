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
def run(run_name: str, resume_run: str = "", max_mode: int = -1,
        weight_ema_decay: float = 0., weight_ema_start: float = .8) -> dict:
    import math
    import os
    import sys
    from statistics import median
    from time import perf_counter

    import numpy as np
    import torch

    sys.path.insert(0, "/repo/torch-attention")
    from model import baseline
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, cpu_batches, make_loader, relative_l2
    from weight_ema import WeightEMA

    if not 0 <= weight_ema_decay < 1 or not 0 <= weight_ema_start < 1:
        raise ValueError("Weight EMA decay/start must be in [0,1)")

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
    generator = torch.Generator().manual_seed(0)
    loader = make_loader(train_data, 256, shuffle=True, generator=generator, bulk=True)
    weight_ema = None
    weight_ema_start_index = math.floor(weight_ema_start * len(loader))
    reference_path = (f"/data/experiments/{resume_run}/checkpoint.pt" if resume_run else
                      "/data/experiments/torch_spectral_fast_h100_20260925/"
                      "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    previous_max_mode = reference["args"].get("max_mode") if resume_run else None
    cutoff = previous_max_mode if max_mode < 0 else max_mode
    start_epoch = int(reference["epoch"]) if resume_run else 0
    start_step = int(reference["step"]) if resume_run else 0
    start_examples = int(reference["examples_seen"]) if resume_run else 0
    if resume_run:
        if (start_step != start_epoch * len(loader) or start_examples != start_epoch * len(train_data)
                or reference.get("epoch_complete") is False):
            raise ValueError("This runner resumes completed epochs only")
        assert reference["source"] == str(source) and reference["architecture"] == "spectral"
        for key, value in {"width": 256, "depth": 2, "heads": 4, "branches": 32,
                           "bf16": True, "lr": 1e-5, "gradient_ema": .8, "batch_size": 256, "seed": 0}.items():
            assert reference["args"][key] == value, (key, reference["args"][key])
        if "sampler_state_after_epoch" in reference:
            generator.set_state(reference["sampler_state_after_epoch"])
        else:
            # Recreate each prior DataLoader base-seed draw and exhaust its
            # sampler, including RandomSampler's final zero-length randperm.
            # This advances exactly as a completed epoch, without reading data.
            for _ in range(start_epoch):
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                for _batch in loader.sampler:
                    pass
    epoch_start_sampler_state = generator.get_state().clone()
    setup_seconds = perf_counter() - tick
    torch.manual_seed(0)
    model = SpectralDNO(n=reference["n"], length=reference["length"], width=256,
                        branches=32, heads=4, depth=2, bf16=True,
                        feature_scales=reference["model"]["feature_scales"], max_mode=cutoff).cuda()
    optimizers = build_optimizers(model, "muon-grouped", 1e-5)
    if resume_run:
        model.load_state_dict(reference["model"])
        for optimizer, saved in zip(optimizers, reference["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        averages = [value.cuda().clone() for value in reference["gradient_ema"]]
        assert all(group["lr"] == 1e-5 for optimizer in optimizers for group in optimizer.param_groups)
    else:
        averages = [torch.zeros_like(p) for p in model.parameters()]
    warmup = Waves(Path("/subset"), "train")[list(range(256))]
    tick = perf_counter()
    fast = CudaStep(model, optimizers, averages, warmup, .8, step=start_step, autotune=True)
    compile_seconds = perf_counter() - tick
    assert fast.counter.item() == start_step
    if resume_run:
        assert all(torch.equal(value.cpu(), reference["model"][name]) for name, value in model.state_dict().items())
        assert all(torch.equal(a.cpu(), b) for a, b in zip(averages, reference["gradient_ema"], strict=True))
        for optimizer, saved in zip(optimizers, reference["optimizers"], strict=True):
            current = optimizer.state_dict()["state"]
            for key, state in saved["state"].items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(current[key][name].cpu(), value.cpu())
    report = {"run_name": run_name, "source": str(source), "gpu": torch.cuda.get_device_name(),
              "training_rows": len(train_data), "validation_rows": len(val_data),
              "expected_steps": len(loader), "batch_size": 256, "width": 256, "depth": 2,
              "heads": 4, "branches": 32, "max_mode": cutoff, "previous_max_mode": previous_max_mode, "parameters": sum(p.numel() for p in model.parameters()),
              "lr": 1e-5, "gradient_ema": .8, "optimizer": "grouped Muon transformer + AdamW remainder",
              "weight_decay": 1e-4, "bf16": True, "loader_workers": 4, "seed": 0,
              "initialization": (f"Resume {reference_path}; model, optimizers, EMA and counters restored" if resume_run else
                                 "Fresh seed0; unchanged physical feature scales from prior4096 training-only rows"),
              "resume_run": resume_run, "start_epoch": start_epoch, "start_step": start_step,
              "loss": "Mean per-example unweighted rFFT-bin relative L2; no physics regularizers",
              "sampling": f"Globally shuffled epoch{start_epoch + 1}; tail retained; continuous seed0 DataLoader sequence",
              "data_setup_seconds": setup_seconds, "compile_capture_seconds": compile_seconds,
              "weight_ema_decay": weight_ema_decay, "weight_ema_start_fraction": weight_ema_start,
              "weight_ema_first_update": weight_ema_start_index + 1 if weight_ema_decay else None,
              "progress": []}

    @torch.no_grad()
    def evaluate(data: Waves, analytic: bool = False) -> dict:
        tick = perf_counter()
        model.eval()
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        base_total = torch.zeros_like(total)
        family_ids = torch.as_tensor(np.load(Path(data.arrays[0].filename).parent / "family_id.npy",
                                             mmap_mode="r")[data.rows].astype(np.int64), device="cuda")
        family_total = torch.zeros(5, dtype=torch.float64, device="cuda")
        family_count = torch.bincount(family_ids, minlength=5)
        max_outband_fraction = torch.zeros((), device="cuda")
        count = 0
        evaluation = make_loader(data, 256, bulk=True)
        for index, batch in enumerate(cpu_batches(evaluation, workers=4), start=1):
            device = tuple(value.cuda() for value in batch)
            prediction = model(*device[:3])
            predicted_spectrum = torch.fft.rfft(prediction)
            error = torch.fft.rfft(prediction - device[3]).abs().norm(dim=-1)
            error /= torch.fft.rfft(device[3]).abs().norm(dim=-1).clamp_min(1e-6)
            total.add_(error.double().sum())
            family_total.scatter_add_(0, family_ids[count:count + len(error)], error.double())
            if model.max_mode is not None:
                leakage = predicted_spectrum[:, model.max_mode + 1:].abs().norm(dim=-1)
                leakage /= predicted_spectrum.abs().norm(dim=-1).clamp_min(1e-6)
                max_outband_fraction = torch.maximum(max_outband_fraction, leakage.max())
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
        names = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
        result["per_family"] = {name: {"examples": int(family_count[i].item()),
                                       "relative_l2": (family_total[i] / family_count[i]).item()}
                                for i, name in enumerate(names, start=1) if family_count[i].item()}
        if model.max_mode is not None:
            result["max_outband_relative_norm"] = max_outband_fraction.item()
            assert result["max_outband_relative_norm"] < 1e-6
        return result

    report["initial_subset_validation"] = evaluate(Waves(Path("/subset"), "validation"))
    if cutoff != previous_max_mode:
        report["initial_full_validation"] = evaluate(val_data)
        print("INITIAL_FULL_VALIDATION " + json.dumps(report["initial_full_validation"]), flush=True)
    model.train()
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("FULL_EPOCH_CONFIG " + json.dumps(report), flush=True)

    def save_checkpoint(step: int, examples: int, complete: bool) -> None:
        target = output / "checkpoint.pt"
        temporary = output / "checkpoint.tmp"
        torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                    "gradient_ema": averages, "step": start_step + step, "examples_seen": start_examples + examples,
                    "epoch": start_epoch + int(complete), "epoch_complete": complete, "epoch_step": step,
                    "n": model.n, "length": model.length,
                    "args": {k: report[k] for k in ("width", "depth", "heads", "branches", "bf16", "lr",
                                                   "gradient_ema", "batch_size", "loader_workers", "seed", "max_mode")},
                    "architecture": "spectral", "source": str(source),
                    **({"weight_ema": {"model": weight_ema.state_dict(), "decay": weight_ema.decay,
                                       "updates": weight_ema.updates,
                                       "first_epoch_update": weight_ema_start_index + 1}}
                       if weight_ema is not None else {}),
                    "epoch_start_sampler_state": epoch_start_sampler_state,
                    **({"sampler_state_after_epoch": generator.get_state()} if complete else {}),
                    "sampler_recovery": "At epoch boundary use sampler_state_after_epoch; mid-epoch recovery needs epoch_start_sampler_state and epoch_step"}, temporary)
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
        if weight_ema_decay and index == weight_ema_start_index:
            weight_ema = WeightEMA(model, weight_ema_decay)
            print(f"WEIGHT_EMA_START epoch_update={index + 1} decay={weight_ema_decay}", flush=True)
        loss = fast(batch)
        if weight_ema is not None:
            weight_ema.update()
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
            progress = {"step": step, "cumulative_step": start_step + step, "examples": examples, "last_loss": loss.item(),
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
    assert fast.counter.item() == start_step + step and step == len(loader)
    assert all(state["step"].item() == start_step + step for state in optimizers[-1].state.values())
    assert torch.isfinite(losses).all().item()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    report.update(completed_epochs=start_epoch + 1, completed_steps=start_step + step,
                  epoch_steps=step, examples_seen=start_examples + examples, epoch_examples=examples,
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
    if weight_ema is not None:
        assert weight_ema.updates == len(loader) - weight_ema_start_index
        assert all(torch.isfinite(value).all().item() for value in weight_ema.averages)
        raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(weight_ema.state_dict())
        report["weight_ema_updates"] = weight_ema.updates
        report["weight_ema_full_validation"] = evaluate(val_data)
        report["weight_ema_relative_improvement_percent"] = 100 * (
            1 - report["weight_ema_full_validation"]["relative_l2"] / report["full_validation"]["relative_l2"])
        # Inference artifact deliberately has no optimizer state: optimizers
        # and gradient EMA belong to the ordinary weights in checkpoint.pt.
        torch.save({"model": model.state_dict(), "n": model.n, "length": model.length,
                    "epoch": start_epoch + 1, "step": start_step + step,
                    "architecture": "spectral", "source": str(source), "inference_only": True,
                    "args": {k: report[k] for k in ("width", "depth", "heads", "branches", "bf16", "max_mode")},
                    "weight_ema_decay": weight_ema.decay, "weight_ema_updates": weight_ema.updates},
                   output / "checkpoint_weight_ema.pt")
        model.load_state_dict(raw_state)
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("FULL_EPOCH_RESULT " + json.dumps({k: v for k, v in report.items() if k not in ("losses", "progress")}), flush=True)
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_width256_full_epoch_20260926", resume_run: str = "",
         max_mode: int = -1, weight_ema_decay: float = 0., weight_ema_start: float = .8) -> None:
    result = run.remote(run_name, resume_run, max_mode, weight_ema_decay, weight_ema_start)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
