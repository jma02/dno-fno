"""Run the small real-data precision/depth comparison on one exact Modal L4."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dno-torch-attention-l4-pilot")
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_file(ROOT / "experiments/torch_attention_real_pilot_20260925.py",
                         "/repo/experiments/torch_attention_real_pilot_20260925.py")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924",
                        "/local-data/paper_equal_subset_20260924"))


@app.function(image=image, volumes={"/data": volume}, gpu="L4:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def compare(run_name: str, batch_size: int, bf16_only: bool) -> dict:
    import hashlib
    import subprocess
    import sys
    from time import perf_counter

    import torch

    started = perf_counter()
    assert torch.cuda.get_device_name() == "NVIDIA L4", torch.cuda.get_device_name()
    destination = Path("/data/experiments") / run_name
    destination.mkdir(parents=True, exist_ok=False)
    data = Path("/local-data/paper_equal_subset_20260924")
    manifest = json.loads((data / "manifest.json").read_text())
    for name, info in manifest["arrays"].items():
        assert hashlib.sha256((data / f"{name}.npy").read_bytes()).hexdigest() == info["sha256"]
    result = {"run_name": run_name, "data_manifest": manifest, "variants": {}}
    for name, flags in (("fp32_d1", []), ("bf16_d1", ["--bf16"]),
                        ("bf16_d2", ["--bf16", "--depth", "2"])):
        if bf16_only and name == "fp32_d1":
            continue
        tag = f"{run_name}_{name}"
        subprocess.run([
            sys.executable, "/repo/experiments/torch_attention_real_pilot_20260925.py",
            "--device", "cuda", "--tag", tag, "--checkpoint-dir", str(destination),
            "--output-dir", str(destination), "--batch-size", str(batch_size), *flags,
        ], check=True)
        result["variants"][name] = json.loads((destination / f"{tag}.json").read_text())
        volume.commit()
    result["function_seconds"] = perf_counter() - started
    (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_name: str = "torch_attention_l4_20260925", batch_size: int = 64,
         bf16_only: bool = False) -> None:
    result = compare.remote(run_name, batch_size, bf16_only)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    target = ROOT.parent / "pilot-checkpoints" / run_name
    target.mkdir(parents=True, exist_ok=True)
    for name, record in result["variants"].items():
        checkpoint = Path(record["checkpoint"]).name
        path = f"experiments/{run_name}/{checkpoint}"
        with (target / checkpoint).open("wb") as handle:
            for chunk in volume.read_file(path):
                handle.write(chunk)
        print(name, {key: record[key] for key in ("gpu", "steady_median_ms", "training_seconds", "peak_allocated_bytes", "validation", "attention_operators")})
