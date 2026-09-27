"""CPU-only production-volume concurrency versus sequential local staging."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention"))
app = modal.App("dno-production-io-cpu")


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384,
              timeout=1200, retries=0, scaledown_window=2)
def run(run_name: str) -> dict:
    import hashlib
    import mmap
    import multiprocessing
    import sys
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from statistics import median
    from time import perf_counter

    import numpy as np
    import torch

    sys.path.insert(0, "/repo/torch-attention")
    from train import Waves, cache_data

    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    started = perf_counter()
    source = Path("/data/outputs/paper_dataset_full_equal_20260916/arrays")
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    print("CPU_ONLY Opening full production training split", flush=True)
    tick = perf_counter()
    remote = Waves(source, "train")
    for array in remote.arrays:
        array._mmap.madvise(mmap.MADV_RANDOM)
    selected = torch.randperm(len(remote), generator=torch.Generator().manual_seed(0))[:65536].reshape(256, 256).tolist()
    report = {"source": str(source), "gpu": None, "cpu": 8, "memory_mib": 16384,
              "training_rows": len(remote), "batch_size": 256, "batches": 256,
              "selection": "Same65536 unique examples in seed0 global training permutation, exact order preserved",
              "setup_seconds": perf_counter() - tick, "results": [],
              "cache_caveat": "Filesystem/host caches cannot be flushed; access order and stage phase recorded"}

    def measure(data: Waves, workers: int, pipe: object) -> None:
        first_last = []
        times = []
        waits = []
        start = perf_counter()
        window = start
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque(pool.submit(data.__getitem__, row) for row in selected[:workers])
            for index in range(256):
                tick = perf_counter()
                batch = pending.popleft().result()
                waits.append(perf_counter() - tick)
                if index + workers < 256:
                    pending.append(pool.submit(data.__getitem__, selected[index + workers]))
                if index in (0, 255):
                    first_last.append(batch)
                if (index + 1) % 16 == 0:
                    times.append(perf_counter() - window)
                    window = perf_counter()
                    pipe.send({"completed_batches": index + 1, "elapsed_seconds": perf_counter() - start})
        elapsed = perf_counter() - start
        digest = hashlib.sha256()
        for batch in first_last:
            for tensor in batch:
                assert torch.isfinite(tensor).all()
                digest.update(tensor.numpy().tobytes())
        pipe.send({"complete": True, "seconds": elapsed, "batch_ms": elapsed / 256 * 1000,
                   "samples_per_second": 65536 / elapsed, "window_seconds": times,
                   "median_consumer_wait_ms": median(waits) * 1000,
                   "first_last_batch_sha256": digest.hexdigest()})
        pipe.close()

    # Fork isolates bounded remote-read trials: blocked reads cannot consume the
    # rest of the staging experiment. No CUDA runtime is initialized in this worker.
    context = multiprocessing.get_context("fork")
    local = None
    for mode, workers, limit in (("remote_initial", 32, 60), ("local_serial", 1, 180),
                                  ("local_prefetch4", 4, 180), ("local_prefetch16", 16, 180),
                                  ("remote_after_scan", 32, 60)):
        if mode == "local_serial":
            print("CPU_ONLY Sequentially staging all six arrays to local SSD", flush=True)
            tick = perf_counter()
            cached = cache_data(source, Path("/tmp/dno-production-arrays"))
            staging = perf_counter() - tick
            local = Waves(cached, "train")
            for array in local.arrays:
                array._mmap.madvise(mmap.MADV_RANDOM)
            assert np.array_equal(remote.rows, local.rows)
            assert all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(remote.arrays, local.arrays, strict=True))
            report["staging_seconds"] = staging
            report["staged_bytes"] = sum((cached / name).stat().st_size for name in
                                          ("eta.npy", "xi.npy", "depth.npy", "gxi.npy", "x.npy", "dataset_split.npy"))
            print("CPU_ONLY_STAGING " + json.dumps({"seconds": staging, "bytes": report["staged_bytes"]}), flush=True)
        print(f"CPU_ONLY_PROBE {mode} workers={workers}", flush=True)
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=measure, args=(remote if mode.startswith("remote") else local, workers, child))
        tick = perf_counter()
        process.start()
        child.close()
        result = {"mode": mode, "workers": workers, "completed_batches": 0}
        while perf_counter() - tick < limit and process.is_alive():
            if parent.poll(.2):
                try:
                    result.update(parent.recv())
                except EOFError:
                    break
                if result.get("complete"):
                    break
        if process.is_alive() and not result.get("complete"):
            process.kill()
            result["timed_out"] = True
        process.join(timeout=5)
        while parent.poll():
            try:
                result.update(parent.recv())
            except EOFError:
                break
        result["process_exitcode"] = process.exitcode
        result["probe_wall_seconds"] = perf_counter() - tick
        parent.close()
        report["results"].append(result)
        print("CPU_ONLY_RESULT " + json.dumps(result), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
    complete = [r for r in report["results"] if r.get("complete")]
    assert len({r["first_last_batch_sha256"] for r in complete}) == 1
    # Independent direct-source equivalence on scattered first/last selected rows.
    check = selected[0][:8] + selected[-1][-8:]
    for a, b in zip(remote[check], local[check], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    report["source_local_tensors_exact"] = True
    report["function_seconds"] = perf_counter() - started
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_production_io_cpu_20260926") -> None:
    result = run.remote(run_name)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
