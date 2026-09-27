"""Evaluate the spectral DNO checkpoint on the existing test cases, locally or on Modal."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from time import perf_counter

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("DNO_TORCH_WORKTREE", "/tmp/dno-fno-pytorch-spectral-dno-ema-rollouts")) / "torch-attention"
SOURCE = ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NCCL_P2P_LEVEL": "PHB"}))
for name in ("model.py", "spectral.py", "real_fft.py", "rollout.py", "checkpoints/epoch4_ema.pt"):
    image = image.add_local_file(MODEL / name, f"/model/{name}")
app = modal.App("dno-spectral-ema-test", include_source=False)


@app.function(image=image, gpu=os.environ.get("DNO_MODAL_GPU", "A100"),
              cpu=4, memory=16384, timeout=3600, retries=0, max_containers=1,
              scaledown_window=2, serialized=True, include_source=False)
def evaluate(payload: bytes, substeps: int, repeats: int, graph: bool, model_dir: str = "/model") -> bytes:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, model_dir)
    from rollout import prepare_rollout, rollout
    from spectral import SpectralDNO

    torch.set_num_threads(1)
    checkpoint = torch.load(Path(model_dir) / "checkpoints/epoch4_ema.pt", map_location="cpu", weights_only=True)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], **checkpoint["args"],
                        feature_scales=checkpoint["model"]["feature_scales"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    with np.load(io.BytesIO(payload)) as source:
        data = dict(source)
    eta, xi, times, depths = (torch.as_tensor(data[key], dtype=torch.float64, device="cuda")
                              for key in ("eta", "xi", "times", "depths"))
    model_depths = depths.float().reshape(-1, 1)

    def predict(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        value = model(a.float(), b.float(), model_depths).double()
        return value - value.mean(-1, keepdim=True)

    warm_times = times[:min(5, len(times))]
    expected = rollout(eta, xi, warm_times, depths, model.length, predict,
                       substeps=substeps, iterations=4, filter_fraction=.25)
    started = perf_counter()
    runner = prepare_rollout(eta, xi, depths, model.length, predict,
                             substeps=substeps, iterations=4, filter_fraction=.25, cuda_graph=graph)
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    result = runner(eta, xi, warm_times)
    parity = {}
    for key in ("eta", "xi", "gxi"):
        torch.testing.assert_close(result[key], expected[key], rtol=1e-5, atol=1e-9)
        parity[key] = float((result[key] - expected[key]).norm() / expected[key].norm().clamp_min(1e-12))
    print(f"Warm parity: {parity}; setup {setup_seconds:.3f}s", flush=True)
    durations = []
    for repeat in range(repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        result = runner(eta, xi, times)
        torch.cuda.synchronize()
        durations.append(perf_counter() - started)
        print(f"repeat {repeat + 1}: {durations[-1]:.3f}s", flush=True)
    prediction = {f"pred_{key}": result[key].cpu().numpy() for key in ("eta", "xi", "gxi")}
    metadata = {"gpu": torch.cuda.get_device_name(), "batch_size": len(eta),
                "seconds": float(np.median(durations)), "repeat_seconds": durations,
                "epoch": checkpoint["epoch"], "model_bf16": model.bf16,
                "model_max_mode": model.max_mode, "integration_dtype": "float64",
                "substeps": substeps, "implicit_iterations": 4, "filter_fraction": .25,
                "length": model.length, "execution": "PyTorch CUDA-graph GL2 step" if graph else "PyTorch eager",
                "setup_seconds": setup_seconds, "warm_relative_difference": parity,
                "torch_version": torch.__version__, "checkpoint": "epoch4_ema.pt",
                "timing": "Warmed synchronized rollout; excludes input/output transfers and setup"}
    output = io.BytesIO()
    np.savez_compressed(output, allow_pickle=False, **prediction, times=data["times"], depths=data["depths"],
                        simulation_ids=data["simulation_ids"], rollout_metadata_json=json.dumps(metadata))
    return output.getvalue()


@app.local_entrypoint()
def main(output: str, families: str = "stokes,tanaka,benjamin_feir,jonswap_tma",
         batch_size: int = 32, repeats: int = 1, frames: int = 0, graph: bool = True,
         local: bool = False) -> None:
    import numpy as np

    destination = ROOT / output
    destination.mkdir(parents=True, exist_ok=False)
    report = {"batch_size": batch_size, "families": {}}
    for family in families.split(","):
        path = next(SOURCE.glob(f"eval_best_current_test_stratified_n32_fp32net*/{family}_trajs.npz"))
        with np.load(path) as reference:
            times = reference["times"][:frames or None]
            data = {"eta": reference["truth_eta"][0, :batch_size],
                    "xi": reference["truth_xi"][0, :batch_size], "times": times,
                    "depths": reference["depths"][:batch_size],
                    "simulation_ids": reference["simulation_ids"][:batch_size]}
            final_truth = reference["truth_eta"][len(times) - 1, :batch_size]
        payload = io.BytesIO()
        np.savez(payload, **data)
        run = evaluate.local if local else evaluate.remote
        contents = run(payload.getvalue(), 80 if family in ("tanaka", "benjamin_feir") else 8,
                       repeats, graph, str(MODEL) if local else "/model")
        (destination / f"{family}_trajs.npz").write_bytes(contents)
        with np.load(io.BytesIO(contents)) as prediction:
            meta = json.loads(str(prediction["rollout_metadata_json"]))
            finite = all(np.isfinite(prediction[f"pred_{key}"]).all() for key in ("eta", "xi", "gxi"))
            error = np.linalg.norm(prediction["pred_eta"][-1] - final_truth, axis=-1) / np.linalg.norm(final_truth, axis=-1)
        record = {"seconds": meta["repeat_seconds"], "finite": bool(finite),
                  "cached_reference_terminal_eta_error": float(error[0]) if finite else None}
        report.update(device=meta["gpu"], execution=meta["execution"], precision="BF16/FP32 neural inference; FP64 integration")
        report["families"][family] = {"simulation_id": int(data["simulation_ids"][0]),
                                     "horizon": float(times[-1]), "saved_frames": len(times),
                                     "methods": {"spectral_ema": record}}
        (destination / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {family}: {meta['seconds']:.3f}s, all finite={finite}", flush=True)
