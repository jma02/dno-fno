"""Overfit 256 balanced production-training snapshots from the epoch-two checkpoint."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("torch-fixed-set-overfit")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=900, retries=0, scaledown_window=2)
def run(run_name: str, steps: int, lr: float, resume_run: str) -> dict:
    import os
    import sys
    from time import perf_counter

    import numpy as np
    import torch

    sys.path.insert(0, "/repo/torch-attention")
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = perf_counter()
    output = Path("/data/experiments") / run_name
    output.mkdir(parents=True, exist_ok=False)
    source = Path("/subset")
    families = np.load(source / "family_id.npy")
    simulations = np.load(source / "simulation_id.npy")
    source_rows = np.load(source / "source_indices.npy")
    rng = np.random.default_rng(926)
    batches, selections, family_ids = {}, {}, {}
    for split in ("train", "validation"):
        data = Waves(source, split)
        indices = []
        for family in range(1, 5):
            candidates = rng.permutation(np.flatnonzero(families[data.rows] == family))
            _, positions = np.unique(simulations[data.rows[candidates]], return_index=True)
            indices.extend(candidates[np.sort(positions)[:64]].tolist())
        assert len(indices) == 256
        rows = data.rows[indices]
        batches[split] = tuple(v.cuda() for v in data[indices])
        family_ids[split] = torch.as_tensor(families[rows], device="cuda")
        selections[split] = {"local_rows": rows.tolist(), "source_rows": source_rows[rows].tolist(),
                             "simulation_ids": simulations[rows].tolist()}
    assert not set(selections["train"]["simulation_ids"]) & set(selections["validation"]["simulation_ids"])
    reference_path = "/data/experiments/torch_spectral_width256_epoch2_20260926/checkpoint.pt"
    metadata = torch.load(reference_path, map_location="cpu", weights_only=True)
    reference = metadata
    previous = None
    if resume_run:
        reference_path = f"/data/experiments/{resume_run}/final_checkpoint.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        previous = json.loads((Path(reference_path).parent / "results.json").read_text())
        assert previous["selections"] == selections
        assert previous["bf16"] and previous["gradient_ema"] == .8
    model = SpectralDNO(n=metadata["n"], length=metadata["length"], width=256,
                        branches=32, heads=4, depth=2, bf16=True,
                        feature_scales=reference["model"]["feature_scales"]).cuda()
    model.load_state_dict(reference["model"])
    optimizers = build_optimizers(model, "muon-grouped", lr)
    for optimizer, saved in zip(optimizers, reference["optimizers"], strict=True):
        optimizer.load_state_dict(saved)
        for group in optimizer.param_groups:
            group["lr"] = lr
    averages = [v.cuda().clone() for v in reference["gradient_ema"]]
    tick = perf_counter()
    fast = CudaStep(model, optimizers, averages, batches["train"], .8,
                    step=reference["step"], autotune=True)
    compile_seconds = perf_counter() - tick
    assert all(torch.equal(v.cpu(), reference["model"][k]) for k, v in model.state_dict().items())
    assert all(torch.equal(a.cpu(), b) for a, b in zip(averages, reference["gradient_ema"], strict=True))
    for optimizer, saved in zip(optimizers, reference["optimizers"], strict=True):
        current = optimizer.state_dict()["state"]
        for key, state in saved["state"].items():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(current[key][name].cpu(), value.cpu())

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        result = {}
        for split, batch in batches.items():
            prediction = model(*batch[:3])
            error = torch.fft.rfft(prediction - batch[3]).abs().norm(dim=-1)
            error /= torch.fft.rfft(batch[3]).abs().norm(dim=-1).clamp_min(1e-6)
            assert torch.isfinite(error).all()
            result[split] = {"mean": error.mean().item(), "median": error.median().item(),
                             "max": error.max().item(), "per_example": error.cpu().tolist(),
                             "per_family": {str(f): error[family_ids[split] == f].mean().item()
                                            for f in range(1, 5)}}
        model.train()
        return result

    initial = evaluate()
    target = min(initial["train"]["mean"] / 10, .00036036369240484103 / 10)
    if previous:
        target = previous["target"]
        for split in batches:
            np.testing.assert_allclose(initial[split]["per_example"],
                                       previous["progress"][-1][split]["per_example"], rtol=1e-5, atol=1e-9)
    report = {"run_name": run_name, "reference_checkpoint": reference_path,
              "gpu": torch.cuda.get_device_name(), "selection_seed": 926, "selections": selections,
              "parameters": sum(p.numel() for p in model.parameters()), "batch_size": 256,
              "width": 256, "depth": 2, "branches": 32, "bf16": True, "lr": lr, "resume_run": resume_run,
              "gradient_ema": .8, "optimizer": "grouped Muon + AdamW; full state resumed",
              "weight_decay": 1e-4, "initial": initial, "target": target,
              "compile_seconds": compile_seconds, "max_steps": steps, "progress": []}
    print("INITIAL " + json.dumps({k: v for k, v in report.items() if k not in ("selections", "initial")}
          | {"initial": {split: {k: v for k, v in values.items() if k != "per_example"}
                         for split, values in initial.items()}}), flush=True)
    losses = torch.empty(steps, device="cuda")
    completed = 0
    update_seconds = 0.
    best = initial["train"]["mean"]
    best_step = 0
    for endpoint in sorted({min(100, steps), min(500, steps), *range(1000, steps + 1, 1000), steps}):
        tick = perf_counter()
        for step in range(completed, endpoint):
            fast.graph.replay()  # The identical fixed training batch stays resident.
            losses[step].copy_(fast.loss)
        torch.cuda.synchronize()
        update_seconds += perf_counter() - tick
        assert torch.isfinite(losses[completed:endpoint]).all()
        completed = endpoint
        evaluation = evaluate()
        row = {"step": completed, "update_seconds": update_seconds, **evaluation}
        report["progress"].append(row)
        if evaluation["train"]["mean"] < best:
            best, best_step = evaluation["train"]["mean"], completed
            torch.save({"model": model.state_dict(), "diagnostic_steps": completed,
                        "train_relative_l2": best, "reference_checkpoint": reference_path}, output / "best_model.pt")
        print("PROGRESS " + json.dumps({"step": completed, "update_seconds": update_seconds,
              **{split: {k: v for k, v in value.items() if k != "per_example"}
                 for split, value in evaluation.items()}}), flush=True)
        if best <= target:
            break
    assert fast.counter.item() == reference["step"] + completed
    assert all(state["step"].item() == reference["step"] + completed for state in optimizers[-1].state.values())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(a).all() for a in averages)
    torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                "gradient_ema": averages, "step": reference["step"] + completed,
                "diagnostic_steps": completed, "epoch_complete": False,
                "reference_checkpoint": reference_path}, output / "final_checkpoint.pt")
    report.update(completed_steps=completed, best_step=best_step, best_train_relative_l2=best,
                  reduction=initial["train"]["mean"] / best, target_reached=best <= target,
                  update_seconds=update_seconds, losses=losses[:completed].cpu().tolist(),
                  function_seconds=perf_counter() - started)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print("RESULT " + json.dumps({k: report[k] for k in ("completed_steps", "best_step", "best_train_relative_l2",
                                                       "reduction", "target_reached", "update_seconds", "function_seconds")}), flush=True)
    return report


@app.local_entrypoint()
def main(run_name: str = "torch_fixed_set_overfit_20260926", steps: int = 20000,
         lr: float = 1e-5, resume_run: str = "") -> None:
    result = run.remote(run_name, steps, lr, resume_run)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
