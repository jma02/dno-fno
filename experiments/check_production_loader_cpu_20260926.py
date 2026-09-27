"""Measure the integrated four-thread loader on production data, without a GPU."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention"))
app = modal.App("dno-production-loader-integrated-cpu")


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384,
              timeout=300, retries=0, scaledown_window=2)
def run() -> dict:
    import sys
    from time import perf_counter, sleep

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/repo/torch-attention")
    from train import Waves, cpu_batches

    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    started = perf_counter()
    data = Waves(Path("/data/outputs/paper_dataset_full_equal_20260916/arrays"), "train")
    selected = torch.randperm(len(data), generator=torch.Generator().manual_seed(0))[:2 * 65536].reshape(2, 256, 256).tolist()
    report = {"gpu": None, "cpu": 8, "batch_size": 256, "workers": 4,
              "setup_seconds": perf_counter() - started, "results": []}
    for segment, delay in ((0, 0.), (1, .017482)):
        loader = DataLoader(data, batch_size=None, sampler=selected[segment])
        iterator = iter(cpu_batches(loader, workers=4))
        first = None
        waits = []
        sleep_seconds = 0.
        print(f"INTEGRATED_START segment={segment}", flush=True)
        tick = perf_counter()
        for index in range(256):
            wait = perf_counter()
            batch = next(iterator)
            waits.append(perf_counter() - wait)
            if index == 0:
                first = perf_counter() - tick
            if delay:
                wait = perf_counter()
                sleep(delay)
                sleep_seconds += perf_counter() - wait
            if index in (0, 255):
                for a, b in zip(batch, data[selected[segment][index]], strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            if (index + 1) % 32 == 0:
                print(f"INTEGRATED_PROGRESS segment={segment} batches={index + 1} seconds={perf_counter() - tick:.3f}", flush=True)
        assert next(iterator, None) is None
        elapsed = perf_counter() - tick
        report["results"].append({"segment": segment, "seconds": elapsed,
                                  "batch_ms": elapsed * 1000 / 256,
                                  "first_batch_seconds": first,
                                  "total_wait_seconds": sum(waits),
                                  "requested_compute_ms": delay * 1000,
                                  "simulated_compute_seconds": sleep_seconds,
                                  "samples_per_second": 65536 / elapsed,
                                  "first_last_exact": True})
        print("INTEGRATED_RESULT " + json.dumps(report["results"][-1]), flush=True)
    report["function_seconds"] = perf_counter() - started
    output = Path("/data/experiments/check_production_loader_cpu_20260926.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
