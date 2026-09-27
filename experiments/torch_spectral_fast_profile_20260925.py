"""Profile kernels in the current compiled, graph-replayed H100 training step."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-fast-profile")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run() -> dict:
    import gzip
    import os
    import re
    import sys
    from collections import defaultdict
    from statistics import median
    from time import perf_counter

    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, "/repo/torch-attention")
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    started = perf_counter()
    checkpoint = torch.load("/data/experiments/torch_spectral_fast_h100_20260925/"
                            "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt",
                            map_location="cuda", weights_only=True)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    optimizers = build_optimizers(model, "muon", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(saved)
    averages = [value.clone() for value in checkpoint["gradient_ema"]]
    loader = make_loader(Waves(Path("/subset"), "train"), 256, shuffle=True,
                         generator=torch.Generator().manual_seed(0), bulk=True)
    batch = next(iter(loader))
    fast = CudaStep(model, optimizers, averages, batch, .8, checkpoint["step"])
    for _ in range(4):
        fast(batch)
    torch.cuda.synchronize()
    # Device-only graph includes the full loss/backward/EMA/optimizer update.
    times = []
    for _ in range(32):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fast.graph.replay()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    output = Path("/data/experiments/torch_spectral_fast_profile_20260925")
    output.mkdir(parents=True, exist_ok=False)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(3):
            fast.graph.replay()
            torch.cuda.synchronize()
    trace_path = output / "trace.json"
    trace.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = defaultdict(lambda: [0, 0.])
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            kernels[event["name"]][0] += 1
            kernels[event["name"]][1] += event["dur"]
    rows = sorted(({"name": name, "calls_per_step": count / 3, "ms_per_step": us / 3000}
                   for name, (count, us) in kernels.items()), key=lambda row: row["ms_per_step"], reverse=True)
    assert torch.isfinite(fast.loss).item()
    result = {"gpu": torch.cuda.get_device_name(), "batch_size": 256,
              "device_graph_median_ms": median(times), "device_graph_step_ms": times,
              "profiled_steps": 3, "kernels_per_step": sum(row["calls_per_step"] for row in rows),
              "summed_kernel_ms": sum(row["ms_per_step"] for row in rows), "kernels": rows,
              "function_seconds": perf_counter() - started}
    groups = defaultdict(float)
    for row in rows:
        name = row["name"].lower()
        if "cudnn" in name or "sdpa" in name:
            category = "Attention (including helpers)"
        elif "fft" in name:
            category = "FFT kernels"
        elif any(part in name for part in ("nvjet", "gemm", "cublas", "cutlass")):
            category = "Matrix multiplies (model and optimizer)"
        elif ("direct_copy" in name or "memcpy" in name
              or re.fullmatch(r"triton_poi_fused__to_copy_\d+", name)):
            category = "Explicit copies and casts"
        elif "layer_norm" in name:
            category = "Fused layer normalization"
        elif "reduce_kernel" in name or name.startswith("triton_red_"):
            category = "Other reductions"
        else:
            category = "Other elementwise and fused kernels"
        groups[category] += row["ms_per_step"]
    result["kernel_categories"] = {
        "method": "Heuristic disjoint grouping by kernel name; fused work stays in one group. Not module attribution or hardware utilization.",
        "ms_per_step": dict(groups)}
    with gzip.open(output / "trace.json.gz", "wb") as compressed:
        compressed.write(trace_path.read_bytes())
    trace_path.unlink()
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    print(json.dumps({key: value for key, value in result.items() if key not in ("kernels", "device_graph_step_ms")}), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
