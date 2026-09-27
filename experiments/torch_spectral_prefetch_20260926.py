"""Compare serialized, queued and prefetched training; probe production-volume I/O."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-prefetch")


def benchmark(variant: str, checkpoint: dict, tail: bool = False) -> tuple[dict, dict]:
    import copy
    from time import perf_counter

    import torch

    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, make_loader, prefetch_batches, relative_l2

    torch.manual_seed(0)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"]).cuda()
    model.load_state_dict(checkpoint["model"])
    optimizers = build_optimizers(model, "muon", 1e-5)
    for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
        optimizer.load_state_dict(copy.deepcopy(saved))
    averages = [value.cuda().clone() for value in checkpoint["gradient_ema"]]
    dataset = Waves(Path("/subset"), "train")
    if tail:
        dataset.rows = dataset.rows[:513]
    loader = make_loader(dataset, 256, shuffle=True, generator=torch.Generator().manual_seed(0),
                         bulk=True, pin_memory=variant == "prefetch")
    torch.cuda.reset_peak_memory_stats()
    fast = CudaStep(model, optimizers, averages, dataset[list(range(256))], .8, checkpoint["step"])
    epochs = 2 if tail else 5
    metrics = torch.empty(epochs * len(loader), device="cuda")
    times, sizes, orders = [], [], []
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    index = 0
    for epoch in range(epochs):
        torch.cuda.synchronize()
        started = perf_counter()
        batches = prefetch_batches(loader) if variant == "prefetch" else loader
        for batch in batches:
            loss = fast(batch)
            metrics[index].copy_(loss.detach())
            total.add_(loss.detach(), alpha=len(batch[0]))
            if variant == "serial":
                assert torch.isfinite(loss).item()
                loss.item()
            elif (index + 1) % 32 == 0:
                assert torch.isfinite(total).item()
            if tail:
                # Outside performance mode, check every returned tensor and order.
                orders.append(tuple(value.cpu().clone() for value in batch))
            sizes.append(len(batch[0]))
            index += 1
        torch.cuda.synchronize()
        times.append(perf_counter() - started)
        if tail:
            model.eval()
            with torch.no_grad():
                held = tuple(value.cuda() for value in dataset[list(range(13))])
                assert torch.isfinite(model(*held[:3])).all().item()
            model.train()
    assert fast.counter.item() == checkpoint["step"] + index
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(a).all().item() for a in averages)
    assert torch.isfinite(metrics).all().item()
    validation = make_loader(Waves(Path("/subset"), "validation"), 64, bulk=True,
                             pin_memory=variant == "prefetch")
    val = torch.zeros((), dtype=torch.float64, device="cuda")
    started = perf_counter()
    with torch.no_grad():
        for batch in prefetch_batches(validation) if variant == "prefetch" else validation:
            device = tuple(value.cuda() for value in batch)
            loss = relative_l2(model(*device[:3]), device[3])
            if variant == "serial":
                val.add_(loss.item() * len(device[0]))
            else:
                val.add_(loss, alpha=len(device[0]))
    val_value = val.item() / len(validation.dataset)
    validation_seconds = perf_counter() - started
    elapsed = sum(times[1:])
    result = {"variant": variant, "tail_test": tail, "epoch_seconds": times,
              "amortized_step_ms": elapsed / ((epochs - 1) * len(loader)) * 1000,
              "samples_per_second": (epochs - 1) * len(dataset) / elapsed,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "losses": metrics.cpu().tolist(), "sample_sizes": sizes,
              "validation_relative_l2": val_value, "validation_seconds": validation_seconds,
              "final_step": int(fast.counter.item()), "weighted_loss_sum": total.item()}
    state = {"parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
             "orders": orders}
    return result, state


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
    checkpoint = torch.load("/data/experiments/torch_spectral_fast_h100_20260925/"
                            "torch_spectral_fast_h100_20260925_bf16_spectral_d2.pt",
                            map_location="cpu", weights_only=True)
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "batch_size": 256, "results": []}
    reference = None
    tail_reference = None
    for variant, tail in (("serial", False), ("queued", False), ("prefetch", False),
                          ("serial", True), ("prefetch", True)):
        result, state = benchmark(variant, checkpoint, tail)
        previous = tail_reference if tail else reference
        if previous is not None:
            old_result, old_state = previous
            result["max_loss_difference"] = max(abs(a - b) for a, b in zip(result["losses"], old_result["losses"], strict=True))
            result["max_parameter_difference"] = max((p - old_state["parameters"][name]).abs().max().item()
                                                     for name, p in state["parameters"].items())
            assert result["sample_sizes"] == old_result["sample_sizes"]
            assert result["max_loss_difference"] < 1e-6
            if tail:
                for old, new in zip(old_state["orders"], state["orders"], strict=True):
                    for a, b in zip(old, new, strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                result["tensor_order_and_tails_exact"] = True
        elif tail:
            tail_reference = result, state
        else:
            reference = result, state
        report["results"].append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("losses", "sample_sizes")}), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        del state
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=180, retries=0, scaledown_window=2)
def probe_io() -> dict:
    import sys
    from statistics import median
    from time import perf_counter

    import numpy as np

    sys.path.insert(0, "/repo/torch-attention")
    from train import Waves

    started = perf_counter()
    print("Opening production arrays and split metadata", flush=True)
    data = Waves(Path("/data/outputs/paper_dataset_full_equal_20260916/arrays"), "train")
    setup = perf_counter() - started
    print(f"Opened {len(data)} training rows in {setup:.3f}s", flush=True)
    # Sparse samples across the real training split, not a contiguous cached shard.
    indices = np.random.default_rng(0).integers(0, len(data), size=(32, 256)).tolist()
    passes = []
    for repeat in range(2):
        times = []
        for batch_indices in indices:
            tick = perf_counter()
            batch = data[batch_indices]
            times.append(perf_counter() - tick)
            assert all(np.isfinite(value.numpy()).all() for value in batch)
        passes.append({"seconds": sum(times), "median_batch_ms": median(times) * 1000,
                       "samples_per_second": 32 * 256 / sum(times), "batch_seconds": times})
        print(json.dumps({"repeat": repeat, **passes[-1]}), flush=True)
    result = {"training_rows": len(data), "setup_seconds": setup, "passes": passes,
              "scope": "CPU-only direct random gathers: first access and identical-index repeat; 8192 sample presentations per pass, with replacement"}
    output = Path("/data/experiments/torch_spectral_prefetch_io_20260926.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_prefetch_20260926", io_only: bool = False,
         diagnose_io: bool = False) -> None:
    result = diagnose.remote() if diagnose_io else probe_io.remote() if io_only else run.remote(run_name)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=180, retries=0, scaledown_window=2)
def diagnose() -> dict:
    from concurrent.futures import ThreadPoolExecutor
    import mmap
    import shutil
    import sys
    from time import perf_counter

    import numpy as np
    import torch

    sys.path.insert(0, "/repo/torch-attention")
    from train import Waves

    started = perf_counter()
    data = Waves(Path("/data/outputs/paper_dataset_full_equal_20260916/arrays"), "train")
    rng = np.random.default_rng(1234)
    report = {"setup_seconds": perf_counter() - started, "cpu_threads": torch.get_num_threads(),
              "local_disk_free_bytes": shutil.disk_usage("/tmp").free, "results": []}
    print(json.dumps(report), flush=True)
    output = Path("/data/experiments/torch_spectral_io_diagnostic_20260926.json")
    for mode in ("scattered_32", "madvise_random_32", "four_threads_32", "contiguous_8192", "repeat_contiguous_8192"):
        indices = rng.integers(0, len(data), size=32).tolist()
        if mode == "madvise_random_32":
            for array in data.arrays:
                array._mmap.madvise(mmap.MADV_RANDOM)
        print(f"Starting {mode}", flush=True)
        tick = perf_counter()
        if mode == "four_threads_32":
            with ThreadPoolExecutor(max_workers=4) as pool:
                batches = list(pool.map(data.__getitem__, [indices[i:i + 8] for i in range(0, 32, 8)]))
        elif "contiguous" in mode:
            batches = [tuple(torch.from_numpy(np.array(array[1000000:1008192], copy=True)) for array in data.arrays)]
        else:
            batches = [data[indices]]
        elapsed = perf_counter() - tick
        samples = 8192 if "contiguous" in mode else 32
        assert all(torch.isfinite(value).all() for batch in batches for value in batch)
        report["results"].append({"mode": mode, "samples": samples, "seconds": elapsed,
                                   "samples_per_second": samples / elapsed})
        print(json.dumps(report["results"][-1]), flush=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
    return report
