"""Measure warmed per-family batch rollout time and final surface error on H100."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
NAME = "torch_ema_rollout_family_benchmark_20260926"
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention"))
app = modal.App("dno-ema-family-rollout-benchmark")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(input_bytes: bytes) -> dict:
    import io
    from statistics import median
    import sys
    from time import perf_counter

    import numpy as np
    import torch
    from torch import Tensor

    sys.path.insert(0, "/repo/torch-attention")
    from rollout import rollout
    from spectral import SpectralDNO

    torch.set_num_threads(1)
    output = Path("/data/experiments") / NAME
    output.mkdir(exist_ok=False)
    checkpoint_path = Path("/data/experiments/torch_spectral_p128_epoch4_weightema_20260926/checkpoint_weight_ema.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], **checkpoint["args"],
                        feature_scales=checkpoint["model"]["feature_scales"])
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    with np.load(io.BytesIO(input_bytes)) as source:
        data = dict(source)
    times = torch.as_tensor(data["times"], device="cuda", dtype=torch.float64)
    families = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
    batches, results = {}, {}
    for family in families:
        selected = data["families"] == family
        batches[family] = tuple(torch.as_tensor(data[key][selected], dtype=torch.float64, device="cuda")
                                for key in ("eta", "xi", "depths", "final_truth_eta"))
        results[family] = {"simulation_ids": data["simulation_ids"][selected].tolist(),
                           "batch_size": int(selected.sum()), "seconds": [], "final_surface_relative_l2": []}
        assert int(selected.sum()) == 16
    # Rotate order across repeats to reduce systematic first/last-family bias.
    # Each family first receives one saved interval of untimed warmup.
    orders = [list(families)] + [list(families[r:] + families[:r]) for r in range(3)]
    for repeat, order in enumerate(orders):
        for family in order:
            eta, xi, depth, truth = batches[family]
            model_depths = depth.float().reshape(-1, 1)

            def predict(a: Tensor, b: Tensor) -> Tensor:
                value = model(a.float(), b.float(), model_depths).double()
                return value - value.mean(-1, keepdim=True)

            evaluation_times = times[:2] if repeat == 0 else times
            torch.cuda.synchronize()
            started = perf_counter()
            prediction = rollout(eta, xi, evaluation_times, depth, model.length, predict,
                                 substeps=8, iterations=4, filter_fraction=.25)
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            assert all(torch.isfinite(prediction[key]).all().item() for key in ("eta", "xi", "gxi"))
            if repeat:
                errors = (prediction["eta"][-1] - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1e-12)
                results[family]["seconds"].append(seconds)
                results[family]["final_surface_relative_l2"].append(errors.cpu().tolist())
                print(json.dumps({"family": family, "repeat": repeat, "seconds": seconds,
                                  "mean_final_surface_error_percent": 100 * errors.mean().item()}), flush=True)
            del prediction
    for result in results.values():
        errors = np.asarray(result["final_surface_relative_l2"])
        result.update(median_seconds=median(result["seconds"]),
                      amortized_seconds_per_trajectory=median(result["seconds"]) / result["batch_size"],
                      mean_final_surface_error_percent=float(100 * errors.mean()),
                      median_final_surface_error_percent=float(100 * np.median(errors.mean(axis=0))),
                      worst_final_surface_error_percent=float(100 * errors.max()),
                      max_repeat_error_spread=float(np.ptp(errors, axis=0).max()))
    report = {"checkpoint": str(checkpoint_path), "gpu": torch.cuda.get_device_name(),
              "epoch": checkpoint["epoch"], "weight_ema_decay": checkpoint["weight_ema_decay"],
              "model_bf16": model.bf16, "model_max_mode": model.max_mode,
              "integration_dtype": "float64", "substeps": 8, "implicit_iterations": 4,
              "filter_fraction": .25, "times": data["times"].tolist(),
              "timing_scope": "Synchronized wall time for16 simultaneous trajectories through t=4 including51 saved frames; excludes model load, input transfer, warmup, reference generation, output transfer and plotting",
              "selection": "Same64 held-out snapshots as prior rollout screen,16 distinct simulations per family",
              "repeat_orders": orders[1:], "all_finite": True, "families": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    import io

    import numpy as np

    source = ROOT / "outputs/torch_rollout_screen64_20260926"
    with np.load(source / "initial_conditions.npz") as file:
        data = dict(file)
    with np.load(source / "reference.npz") as file:
        np.testing.assert_array_equal(data["simulation_ids"], file["simulation_ids"])
        np.testing.assert_array_equal(data["times"], file["times"])
        data["final_truth_eta"] = file["truth_eta"][-1]
    data["families"] = np.asarray(json.loads((source / "manifest.json").read_text())["families"])
    buffer = io.BytesIO()
    np.savez(buffer, **data)
    result = run.remote(buffer.getvalue())
    (ROOT / "experiments" / f"{NAME}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({family: {key: value for key, value in row.items()
                             if key not in ("simulation_ids", "final_surface_relative_l2")}
                      for family, row in result["families"].items()}, indent=2))
