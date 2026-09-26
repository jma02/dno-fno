"""Matched old/new layout training benchmark; reference is pinned to git."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "19cf73e"
reference_path = Path("/tmp/fnotrash-layout-reference")
if modal.is_local():
    import subprocess

    reference_path.mkdir(exist_ok=True)
    for name in ("spectral", "train"):
        (reference_path / f"{name}_reference.py").write_bytes(subprocess.check_output(
            ["git", "show", f"{REFERENCE}:torch-attention/{name}.py"], cwd=ROOT))
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(reference_path, "/reference")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-layout-benchmark")


def benchmark(variant: str, checkpoint: dict, output: Path) -> tuple[dict, dict]:
    import copy
    import gzip
    from collections import defaultdict
    from statistics import median
    from time import perf_counter

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from spectral import SpectralDNO
    from spectral_reference import SpectralDNO as Reference
    from train import CudaStep, Waves, build_optimizers, make_loader, relative_l2
    from train_reference import CudaStep as ReferenceStep

    torch.manual_seed(0)
    cls = Reference if variant == "reference" else SpectralDNO
    model = cls(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(Waves(Path("/subset"), "train"), 256, shuffle=True,
                         generator=torch.Generator().manual_seed(0), bulk=True)
    iterator = iter(loader)
    batch = next(iterator)
    small = tuple(value[:16].cuda() for value in batch)
    prediction = model(*small[:3])
    loss = relative_l2(prediction, small[3])
    loss.backward()
    initial = {"prediction": prediction.detach().cpu(),
               "gradients": {name: p.grad.cpu().clone() for name, p in model.named_parameters()}}
    model.zero_grad(set_to_none=True)
    del prediction, loss, small
    optimizers = build_optimizers(model, "muon", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(copy.deepcopy(saved))
    averages = [value.cuda().clone() for value in checkpoint["gradient_ema"]]
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    step_class = CudaStep if variant == "layout_compiled_updates" else ReferenceStep
    fast = step_class(model, optimizers, averages, batch, .8, checkpoint["step"])
    setup = perf_counter() - started
    times, losses = [], []
    for step in range(36):
        torch.cuda.synchronize()
        started = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        value = fast(batch).item()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        if step >= 4:
            times.append(elapsed)
        losses.append(value)
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    assert fast.counter.item() == checkpoint["step"] + 36
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    assert all(torch.isfinite(torch.tensor(losses)))
    # Matched-update outputs include a held-out batch after the same 36 updates.
    validation = make_loader(Waves(Path("/subset"), "validation"), 64, bulk=True)
    total = 0.
    with torch.no_grad():
        for current in validation:
            device = tuple(value.cuda() for value in current)
            total += relative_l2(model(*device[:3]), device[3]).item() * len(device[0])
    initial["final_parameters"] = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    device_ms = []
    for _ in range(16):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fast.graph.replay()
        end.record()
        end.synchronize()
        device_ms.append(begin.elapsed_time(end))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        # The first replay may begin before CUPTI finishes enabling activities.
        for index in range(4):
            with record_function(f"profile_step_{index}"):
                fast.graph.replay()
                torch.cuda.synchronize()
    trace_path = output / f"{variant}_trace.json"
    trace.export_chrome_trace(str(trace_path))
    kernels = defaultdict(lambda: [0, 0.])
    events = json.loads(trace_path.read_text())["traceEvents"]
    windows = [(event["ts"], event["ts"] + event["dur"]) for event in events
               if event.get("name") in ("profile_step_1", "profile_step_2", "profile_step_3")
               and event.get("cat") == "user_annotation" and event.get("ph") == "X"]
    for event in events:
        if (event.get("cat") == "kernel" and event.get("ph") == "X"
                and any(start <= event["ts"] < end for start, end in windows)):
            kernels[event["name"]][0] += 1
            kernels[event["name"]][1] += event["dur"]
    with gzip.open(output / f"{variant}_trace.json.gz", "wb") as compressed:
        compressed.write(trace_path.read_bytes())
    trace_path.unlink()
    rows = sorted(({"name": name, "calls_per_step": count / 3, "ms_per_step": us / 3000}
                   for name, (count, us) in kernels.items()), key=lambda row: row["ms_per_step"], reverse=True)
    result = {"variant": variant, "median_step_ms": median(times) * 1000,
              "samples_per_second": 256 / median(times), "device_graph_median_ms": median(device_ms),
              "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
              "compile_capture_seconds": setup, "step_seconds": times, "losses": losses,
              "validation_relative_l2": total / len(validation.dataset),
              "profile_complete": len(windows) == 3 and bool(kernels) and all(count % 3 == 0 for count, _ in kernels.values()),
              "kernels_per_step": sum(row["calls_per_step"] for row in rows),
              "summed_kernel_ms": sum(row["ms_per_step"] for row in rows), "kernels": rows}
    return result, initial


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(run_name: str, compiled_updates: bool) -> dict:
    import gc
    import os
    import sys
    from time import perf_counter

    import torch

    sys.path[:0] = ["/repo/torch-attention", "/reference"]
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    started = perf_counter()
    checkpoint = torch.load("/data/experiments/torch_spectral_fast_h100_20260925/"
                            "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt",
                            map_location="cpu", weights_only=True)
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "reference_commit": REFERENCE,
              "batch_size": 256, "results": []}
    states = []
    for variant in ("reference", "layout_compiled_updates" if compiled_updates else "layout"):
        result, state = benchmark(variant, checkpoint, output)
        states.append(state)
        report["results"].append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("step_seconds", "losses", "kernels")}), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    reference, candidate = states
    prediction_error = ((reference["prediction"] - candidate["prediction"]).norm()
                        / reference["prediction"].norm()).item()
    gradient_errors = {name: ((value - candidate["gradients"][name]).norm() / value.norm().clamp_min(1e-12)).item()
                       for name, value in reference["gradients"].items()}
    report["correctness"] = {"initial_prediction_relative_difference": prediction_error,
        "initial_gradient_relative_differences": gradient_errors,
        "max_training_loss_difference": max(abs(a - b) for a, b in zip(
            report["results"][0]["losses"], report["results"][1]["losses"], strict=True)),
        "max_final_parameter_difference": max((value - candidate["final_parameters"][name]).abs().max().item()
                                               for name, value in reference["final_parameters"].items())}
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps(report["correctness"]), flush=True)
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_layout_20260926", compiled_updates: bool = False) -> None:
    result = run.remote(run_name, compiled_updates)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
