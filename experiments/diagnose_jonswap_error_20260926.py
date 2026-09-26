"""Measure family spectra, analytic residuals and learned out-of-band error locally."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT / "torch-attention"))
    from spectral import SpectralDNO
    from train import Waves

    torch.set_num_threads(8)
    source = ROOT.parent / "local-data/paper_equal_subset_20260924"
    data = {name: np.load(source / f"{name}.npy", mmap_mode="r")
            for name in ("eta", "xi", "gxi", "depth", "family_id", "dataset_split")}
    n = data["eta"].shape[-1]
    k = np.arange(n // 2 + 1)
    measurements = {key: [] for key in ("baseline_error", "projected_baseline_error", "baseline_outband_error",
                    "target_outband_fraction", "eta_effective_modes", "eta_k95", "slope_rms", "eta_rms_over_depth")}
    for start in range(0, len(data["eta"]), 256):
        eta, xi, target = (np.array(data[name][start:start + 256], dtype=np.float64) for name in ("eta", "xi", "gxi"))
        h = np.array(data["depth"][start:start + 256], dtype=np.float64).reshape(-1, 1)
        xi -= xi.mean(-1, keepdims=True)
        symbol = k * np.tanh(k * h.clip(max=5))
        xhat = np.fft.rfft(xi)
        g0 = np.fft.irfft(symbol * xhat, n=n)
        dx = np.fft.irfft(1j * k * xhat, n=n)
        base = symbol * xhat - symbol * np.fft.rfft(eta * g0) - 1j * k * np.fft.rfft(eta * dx)
        truth = np.fft.rfft(target)
        error = base - truth
        denominator = np.maximum(np.linalg.norm(truth, axis=-1), 1e-6)
        projected = base.copy()
        projected[:, 129:] = 0
        measurements["baseline_error"].extend(np.linalg.norm(error, axis=-1) / denominator)
        measurements["projected_baseline_error"].extend(np.linalg.norm(projected - truth, axis=-1) / denominator)
        measurements["baseline_outband_error"].extend(np.linalg.norm(error[:, 129:], axis=-1) / denominator)
        measurements["target_outband_fraction"].extend(np.linalg.norm(truth[:, 129:], axis=-1) / denominator)
        ehat = np.fft.rfft(eta)
        energy = abs(ehat)**2
        energy[:, 0] = 0
        fractions = energy / energy.sum(-1, keepdims=True)
        measurements["eta_effective_modes"].extend(1 / (fractions**2).sum(-1))
        measurements["eta_k95"].extend((fractions.cumsum(-1) >= .95).argmax(-1))
        measurements["slope_rms"].extend(np.sqrt(np.mean(np.fft.irfft(1j * k * ehat, n=n)**2, axis=-1)))
        measurements["eta_rms_over_depth"].extend(np.sqrt(np.mean(eta**2, axis=-1)) / h[:, 0])
    measurements = {key: np.asarray(value) for key, value in measurements.items()}
    report = {"source": str(source), "scope": "5120 local production-derived rows; descriptive, not a causal ablation", "families": {}}
    for family, name in enumerate(("stokes", "tanaka", "benjamin_feir", "jonswap_tma"), start=1):
        mask = data["family_id"] == family
        report["families"][name] = {key: {"mean": float(values[mask].mean()), "median": float(np.median(values[mask]))}
                                     for key, values in measurements.items()}
        report["families"][name]["depth_median"] = float(np.median(data["depth"][mask]))
    print("DESCRIPTIVE " + json.dumps(report), flush=True)
    prior = json.loads((ROOT / "experiments/torch_fixed_set_overfit_lr1e6_20260926.json").read_text())
    rows = prior["selections"]["train"]["local_rows"]
    base_checkpoint = torch.load(ROOT.parent / "pilot-checkpoints/torch_spectral_width256_epoch2_20260926.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(ROOT.parent / "pilot-checkpoints/torch_fixed_set_overfit_lr1e6_20260926.pt", map_location="cpu", weights_only=True)
    model = SpectralDNO(n=n, length=base_checkpoint["length"], width=256, branches=32, heads=4, depth=2,
                        bf16=True, feature_scales=checkpoint["model"]["feature_scales"])
    report["learned_diagnostics"] = {}
    cases = (("overfit_train", checkpoint, rows, "train"),
             ("epoch2_validation", base_checkpoint,
              np.flatnonzero(data["dataset_split"] == "validation").tolist(), "validation"))
    for case, saved, rows, split in cases:
        model.load_state_dict(saved["model"])
        model.eval()
        dataset = Waves(source, split)
        indices = np.searchsorted(dataset.rows, rows).tolist()
        results = []
        started = perf_counter()
        with torch.no_grad():
            for start in range(0, len(indices), 8):
                batch = dataset[indices[start:start + 8]]
                prediction = model(*batch[:3]).double()
                target = torch.fft.rfft(batch[3].double())
                predicted = torch.fft.rfft(prediction)
                error = predicted - target
                denominator = target.abs().norm(dim=-1).clamp_min(1e-6)
                projected = predicted.clone()
                projected[:, 129:] = 0
                values = torch.stack((error.abs().norm(dim=-1) / denominator,
                                      (projected - target).abs().norm(dim=-1) / denominator,
                                      error[:, 129:].abs().norm(dim=-1) / denominator), -1)
                results.extend(values.tolist())
                if start % 64 == 0:
                    print(f"{case} {start + len(batch[0])}/{len(indices)}, {perf_counter()-started:.1f}s", flush=True)
        results = np.asarray(results)
        report["learned_diagnostics"][case] = {"device": "cpu", "bf16": True, "checkpoint": case,
                                        "seconds": perf_counter() - started, "rows": rows, "per_example": results.tolist(), "families": {}}
        for family, name in enumerate(("stokes", "tanaka", "benjamin_feir", "jonswap_tma"), start=1):
            values = results[np.asarray(data["family_id"])[rows] == family]
            report["learned_diagnostics"][case]["families"][name] = {
                "mean_error": float(values[:, 0].mean()), "mean_error_after_output_projection": float(values[:, 1].mean()),
                "mean_outband_error": float(values[:, 2].mean()),
                "outband_fraction_of_summed_normalized_squared_error": float((values[:, 2]**2).sum() / (values[:, 0]**2).sum())}
        print("LEARNED " + case + " " + json.dumps(report["learned_diagnostics"][case]["families"]), flush=True)
    (ROOT / "experiments/diagnose_jonswap_error_20260926.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
