"""CPU-only unseen-row and reader-concurrency checks after production-volume warmup."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention"))
app = modal.App("dno-production-prefetch-cpu")


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384,
              timeout=600, retries=0, scaledown_window=2)
def run() -> dict:
    import hashlib
    import mmap
    import sys
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from statistics import median
    from time import perf_counter, sleep

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/repo/torch-attention")
    from train import Waves

    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    started = perf_counter()
    data = Waves(Path("/data/outputs/paper_dataset_full_equal_20260916/arrays"), "train")
    for array in data.arrays:
        array._mmap.madvise(mmap.MADV_RANDOM)
    permutation = torch.randperm(len(data), generator=torch.Generator().manual_seed(0))[:4 * 65536].reshape(4, 256, 256).tolist()
    output = Path("/data/experiments/torch_production_prefetch_cpu_20260926")
    output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": None, "cpu": 8, "batch_size": 256, "training_rows": len(data),
              "setup_seconds": perf_counter() - started,
              "selection": "Four disjoint65536-row segments of full-split seed0 permutation; same order across warm controls",
              "cache_caveat": "No host-cache eviction; first-access means new worker and unseen-row means new sample indices, not guaranteed cold file chunks",
              "results": []}

    for mode, workers, segment, delay in (("initial_threads32", 32, 0, 0.),
                                          ("warm_threads1", 1, 0, 0.),
                                          ("warm_threads4", 4, 0, 0.),
                                          ("warm_threads16", 16, 0, 0.),
                                          ("warm_threads32", 32, 0, 0.),
                                          ("warm_processes4", 4, 0, 0.),
                                          ("warm_threads1_repeat", 1, 0, 0.),
                                          ("unseen_threads32", 32, 1, 0.),
                                          ("unseen_threads4_paced", 4, 2, .017482),
                                          ("unseen_threads32_paced", 32, 3, .017482)):
        print(f"CPU_PREFETCH_START {mode}", flush=True)
        indices = permutation[segment]
        pool = None
        loader = None
        begin = perf_counter()
        if "processes" in mode:
            loader = DataLoader(data, batch_size=None, sampler=indices, num_workers=workers,
                                prefetch_factor=4, multiprocessing_context="fork")
            iterator = iter(loader)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            pending = deque(pool.submit(data.__getitem__, row) for row in indices[:workers])
        first_last = []
        waits = []
        delay_seconds = 0.
        first_batch_seconds = None
        for index in range(256):
            tick = perf_counter()
            if pool is None:
                batch = next(iterator)
            else:
                batch = pending.popleft().result()
                if index + workers < 256:
                    pending.append(pool.submit(data.__getitem__, indices[index + workers]))
            waits.append(perf_counter() - tick)
            if index in (0, 255):
                first_last.append(batch)
            if first_batch_seconds is None:
                first_batch_seconds = perf_counter() - begin
            if delay:
                tick = perf_counter()
                sleep(delay)
                delay_seconds += perf_counter() - tick
            if (index + 1) % 64 == 0:
                print(f"CPU_PREFETCH_PROGRESS {mode} batches={index + 1} seconds={perf_counter() - begin:.3f}", flush=True)
        elapsed = perf_counter() - begin
        if pool is not None:
            pool.shutdown()
        digest = hashlib.sha256()
        for batch in first_last:
            for tensor in batch:
                assert torch.isfinite(tensor).all()
                digest.update(tensor.numpy().tobytes())
        result = {"mode": mode, "workers": workers, "segment": segment,
                  "seconds": elapsed, "batch_ms": elapsed / 256 * 1000,
                  "samples_per_second": 65536 / elapsed, "first_batch_seconds": first_batch_seconds,
                  "median_wait_ms": median(waits) * 1000, "total_wait_seconds": sum(waits),
                  "simulated_compute_seconds": delay_seconds, "requested_compute_ms": delay * 1000,
                  "first_last_sha256": digest.hexdigest()}
        report["results"].append(result)
        print("CPU_PREFETCH_RESULT " + json.dumps(result), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        if loader is not None:
            del iterator, loader
    assert len({r["first_last_sha256"] for r in report["results"] if r["segment"] == 0}) == 1
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
