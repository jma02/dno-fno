"""Matched width256 engineering trials from the completed epoch checkpoint."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
reference_path = Path("/tmp/fnotrash-real-fft-reference")
if modal.is_local():
    import subprocess

    reference_path.mkdir(exist_ok=True)
    (reference_path / "real_fft_reference.py").write_bytes(subprocess.check_output(
        ["git", "show", "62ee731:torch-attention/real_fft.py"], cwd=ROOT))
    (reference_path / "spectral_original.py").write_bytes(subprocess.check_output(
        ["git", "show", "7d673f6:torch-attention/spectral.py"], cwd=ROOT))
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(reference_path, "/reference")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-width256-engineering")


def benchmark(variant: str, checkpoint: dict, output: Path) -> tuple[dict, dict]:
    import copy
    import gzip
    from collections import defaultdict
    from statistics import median
    from time import perf_counter

    import torch
    import spectral
    import real_fft
    import real_fft_reference
    from spectral_original import SpectralDNO as OriginalSpectralDNO
    from torch.profiler import ProfilerActivity, profile, record_function
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader, relative_l2

    torch.backends.cuda.enable_cudnn_sdp(variant != "flash")
    fft = real_fft_reference if variant.startswith("real_fft_v1") else real_fft
    use_real = variant in ("real_fft", "packed", "real_fft_repeat", "fft_norm", "final") or variant.startswith("real_fft_v1")
    spectral.rfft = fft.rfft if use_real else torch.fft.rfft
    spectral.irfft = fft.irfft if use_real else torch.fft.irfft

    class PackedSpectralDNO(SpectralDNO):
        def correction(self, eta: torch.Tensor, xi: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
            import math
            from torch.nn import functional as F

            features = spectral.surface_features(eta, self.length) / self.feature_scales
            k, h = torch.broadcast_tensors(self.k[None, :], depth.reshape(-1, 1).clamp(max=5))
            tanh_kh = torch.tanh(k * h)
            filter_features = torch.stack((k, h, tanh_kh, k * tanh_kh), -1)
            condition = torch.stack((k / self.k[-1], h / 5, tanh_kh, k / self.k[-1] * tanh_kh), -1)
            with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                local = self.encoder(features)
            spectrum = real_fft.rfft(local.transpose(1, 2).float().contiguous(), norm="ortho").transpose(1, 2)
            width = local.shape[-1]
            # Keep real/imaginary pairs together and reorder the small weight
            # matrix, rather than constructing two complex activation gradients.
            packed = torch.view_as_real(spectrum).flatten(-2)
            weight = self.frequency_in.weight
            weight = torch.cat((weight[:, :2 * width].reshape(width, 2, width)
                                .transpose(1, 2).reshape(width, 2 * width), weight[:, 2 * width:]), -1)
            with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                tokens = F.linear(torch.cat((packed, condition), -1), weight, self.frequency_in.bias)
                tokens = self.frequency(tokens)
                weight_out = self.frequency_out.weight.reshape(2, width, width).transpose(0, 1).reshape(2 * width, width)
                bias_out = self.frequency_out.bias.reshape(2, width).T.reshape(2 * width)
                coefficients = F.linear(tokens, weight_out, bias_out).float()
            packed = coefficients.reshape(*coefficients.shape[:-1], width, 2).transpose(1, 2).contiguous()
            context = real_fft.irfft(torch.view_as_complex(packed), n=self.n, norm="ortho").transpose(1, 2)
            with torch.autocast(eta.device.type, enabled=False):
                local = local.float()
                weights = self.decoder(local * local.tanh() * (1 + context.tanh()))
                filters = (self.depth_filters(filter_features) * (self.k != 0)[None, :, None]).transpose(1, 2).contiguous()
                spectrum_xi = torch.view_as_real(torch.fft.rfft(xi))[:, None]
                filtered = real_fft.irfft(torch.view_as_complex(spectrum_xi * filters[..., None]), n=self.n)
                weighted = real_fft.rfft(weights.transpose(1, 2) * filtered)
                output = (torch.view_as_real(weighted) * filters[..., None]).sum(1)
                return real_fft.irfft(torch.view_as_complex(output), n=self.n) / math.sqrt(filters.shape[1])

    torch.manual_seed(0)
    cls = OriginalSpectralDNO if variant.startswith("original") else PackedSpectralDNO if variant == "packed" else SpectralDNO
    model = cls(n=checkpoint["n"], length=checkpoint["length"], width=256,
                        branches=32, heads=4, depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    dataset = Waves(Path("/subset"), "train")
    batches = list(make_loader(dataset, 256, shuffle=True,
                              generator=torch.Generator().manual_seed(0), bulk=True))
    small = tuple(v[:16].cuda() for v in batches[0])
    prediction = model(*small[:3])
    relative_l2(prediction, small[3]).backward()
    state = {"initial_prediction": prediction.detach().cpu(),
             "initial_gradients": {k: p.grad.cpu().clone() for k, p in model.named_parameters()}}
    model.zero_grad(set_to_none=True)
    del small, prediction
    optimizers = build_optimizers(model, "muon-grouped", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(copy.deepcopy(saved))
    averages = [v.cuda().clone() for v in checkpoint["gradient_ema"]]
    torch.cuda.reset_peak_memory_stats()
    tick = perf_counter()
    fast = CudaStep(model, optimizers, averages, batches[0], .8, checkpoint["step"], autotune=True)
    setup = perf_counter() - tick
    assert fast.counter.item() == checkpoint["step"]
    assert all(torch.equal(v.cpu(), checkpoint["model"][k]) for k, v in model.state_dict().items())
    losses = torch.empty(48, device="cuda")
    times = []
    for cycle in range(3):
        torch.cuda.synchronize()
        tick = perf_counter()
        for i, batch in enumerate(batches):
            losses[cycle * len(batches) + i].copy_(fast(batch).detach())
        torch.cuda.synchronize()
        times.append((perf_counter() - tick) * 1000 / len(batches))
    assert fast.counter.item() == checkpoint["step"] + 48
    assert torch.isfinite(losses).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(a).all() for a in averages)
    state["final_parameters"] = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    validation = make_loader(Waves(Path("/subset"), "validation"), 256, bulk=True)
    total = 0.
    with torch.no_grad():
        for batch in validation:
            device = tuple(v.cuda() for v in batch)
            total += relative_l2(model(*device[:3]), device[3]).item() * len(device[0])
    device_ms = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            fast.graph.replay()
        end.record()
        end.synchronize()
        device_ms.append(start.elapsed_time(end) / 10)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for i in range(4):
            with record_function(f"measured_{i}"):
                fast.graph.replay()
                torch.cuda.synchronize()
    path = output / f"{variant}_trace.json"
    trace.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in events
               if e.get("cat") == "user_annotation" and e.get("name") in
               ("measured_1", "measured_2", "measured_3") and e.get("ph") == "X"]
    assert len(windows) == 3
    kernels = defaultdict(lambda: [0, 0.])
    counts = []
    for begin, end in windows:
        selected = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"
                    and begin <= e["ts"] < end]
        counts.append(len(selected))
        for e in selected:
            kernels[e["name"]][0] += 1
            kernels[e["name"]][1] += e["dur"]
    profile_complete = len(set(counts)) == 1 and counts[0] > 0
    rows = sorted(({"name": k, "calls_per_step": v[0] / 3, "ms_per_step": v[1] / 3000}
                   for k, v in kernels.items()), key=lambda r: r["ms_per_step"], reverse=True)
    with gzip.open(output / f"{variant}_trace.json.gz", "wb") as compressed:
        compressed.write(path.read_bytes())
    path.unlink()
    result = {"variant": variant, "batch_size": 256, "parameters": sum(p.numel() for p in model.parameters()),
              "compile_seconds": setup, "amortized_batch_ms": median(times[1:]),
              "cycle_batch_ms": times, "device_ms": median(device_ms), "device_samples_ms": device_ms,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "losses": losses.cpu().tolist(),
              "validation_relative_l2": total / len(validation.dataset),
              "checked_updates": 48, "kernels_per_step": counts[0],
              "profile_equal_kernel_counts": profile_complete, "profile_kernel_counts": counts,
              "summed_kernel_ms": sum(r["ms_per_step"] for r in rows), "kernels": rows}
    return result, state


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=8,
              memory=32768, timeout=900, retries=0, scaledown_window=2)
def run(run_name: str, variants: tuple[str, ...]) -> dict:
    import gc
    import os
    import sys
    from time import perf_counter

    import torch

    sys.path.insert(0, "/repo/torch-attention")
    sys.path.insert(0, "/reference")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    torch.set_num_threads(1)
    tick = perf_counter()
    checkpoint = torch.load("/data/experiments/torch_spectral_width256_full_epoch_20260926/checkpoint.pt",
                            map_location="cpu", weights_only=True)
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "reference_checkpoint_step": checkpoint["step"],
              "timing": "Preloaded CPU subset, matched 48 updates; profiling and graph-only diagnostic replays afterwards",
              "results": []}
    reference = None
    for variant in variants:
        result, state = benchmark(variant, checkpoint, output)
        if reference is None:
            reference = state
        else:
            result["initial_prediction_max_diff"] = (state["initial_prediction"] - reference["initial_prediction"]).abs().max().item()
            result["initial_gradient_relative_diff"] = {
                k: ((v - reference["initial_gradients"][k]).norm() /
                    reference["initial_gradients"][k].norm().clamp_min(1e-20)).item()
                for k, v in state["initial_gradients"].items()}
            result["final_parameter_max_diff"] = max((v - reference["final_parameters"][k]).abs().max().item()
                                                     for k, v in state["final_parameters"].items())
        report["results"].append(result)
        print("ENGINEERING_RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("kernels", "losses")}), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    report["seconds"] = perf_counter() - tick
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_width256_engineering_20260926", variants: str = "reference,flash,reference_repeat") -> None:
    result = run.remote(run_name, tuple(variants.split(",")))
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
