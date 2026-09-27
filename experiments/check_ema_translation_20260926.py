"""Measure integer-shift consistency of epoch4 EMA on the64 rollout snapshots."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from model import baseline  # noqa: E402
from spectral import SpectralDNO  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    checkpoint = torch.load(ROOT.parent / "pilot-checkpoints/torch_spectral_p128_epoch4_weightema_20260926_averaged.pt",
                            map_location="cpu", weights_only=True)
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], **checkpoint["args"],
                        feature_scales=checkpoint["model"]["feature_scales"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    folder = ROOT / "outputs/torch_rollout_screen64_20260926"
    with np.load(folder / "initial_conditions.npz") as source:
        eta, xi, depth = (torch.tensor(source[key], dtype=torch.float32) for key in ("eta", "xi", "depths"))
    with np.load(folder / "reference.npz") as source:
        target = torch.tensor(source["truth_gxi"][0], dtype=torch.float32)
    families = np.array(json.loads((folder / "manifest.json").read_text())["families"])
    shifts = (0, 1, 37, 256, 511)
    predictions, baselines = [], []
    started = perf_counter()
    with torch.no_grad():
        for shift in shifts:
            a, b = torch.roll(eta, shift, -1), torch.roll(xi, shift, -1)
            predictions.append(torch.roll(model(a, b, depth), -shift, -1))
            baselines.append(torch.roll(model.project(baseline(a, model.project(b), depth, model.length)), -shift, -1))
        predictions, baselines = torch.stack(predictions), torch.stack(baselines)
        denominator = torch.fft.rfft(target).abs().norm(dim=-1).clamp_min(1e-6)
        errors = torch.fft.rfft(predictions - target).abs().norm(dim=-1) / denominator
        defects = torch.fft.rfft(predictions[1:] - predictions[0]).abs().norm(dim=-1) / denominator
        base_defects = torch.fft.rfft(baselines[1:] - baselines[0]).abs().norm(dim=-1) / denominator
        ensemble_errors = torch.fft.rfft(predictions.mean(0) - target).abs().norm(dim=-1) / denominator
    report = {"scope": "CPU BF16 replay,64 held-out initial snapshots; not full validation or exact CUDA reproduction",
              "shifts": shifts, "seconds": perf_counter() - started,
              "unshifted_mean_relative_l2": errors[0].mean().item(),
              "shifted_mean_relative_l2": errors[1:].mean().item(),
              "mean_shift_defect_relative_to_target": defects.mean().item(),
              "mean_baseline_shift_defect": base_defects.mean().item(),
              "five_shift_ensemble_relative_l2": ensemble_errors.mean().item(),
              "family": {family: {"error": errors[0, families == family].mean().item(),
                                  "shift_defect": defects[:, families == family].mean().item(),
                                  "ensemble_error": ensemble_errors[families == family].mean().item()}
                         for family in sorted(set(families))}}
    assert torch.isfinite(predictions).all() and torch.isfinite(errors).all()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
