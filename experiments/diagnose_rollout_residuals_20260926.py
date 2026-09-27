"""Decompose rollout surface errors into spectral bands, magnitude and phase."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    folder = ROOT / "outputs/torch_p128_epoch3_worst64_rollouts_20260926"
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "comparison_trajs.npz") as data:
        truth, pred, times = (data[key] for key in ("truth_eta", "pred_eta", "times"))
    n = truth.shape[-1]
    k = np.arange(n // 2 + 1) * (2 * np.pi / summary["manifest"]["length"])
    bands = ((1, 16), (17, 32), (33, 64), (65, 96), (97, 128), (129, n // 2))
    weights = np.full(n // 2 + 1, 2.)
    weights[[0, -1]] = 1
    rows = []
    for i, metadata in enumerate(summary["trajectories"]):
        a, b = truth[-1, i], pred[-1, i]
        ahat, bhat = np.fft.rfft(a), np.fft.rfft(b)
        energy = abs(bhat - ahat)**2 * weights
        mag = (abs(bhat) - abs(ahat))**2 * weights
        phase = np.maximum(2 * (abs(bhat) * abs(ahat) - (bhat * ahat.conj()).real), 0) * weights
        np.testing.assert_allclose(energy.sum(), mag.sum() + phase.sum(), rtol=1e-6, atol=1e-15)
        dx = np.fft.irfft(1j * k * ahat, n=n)
        error = b - a
        design = np.stack((a, dx), axis=-1)
        fit, *_ = np.linalg.lstsq(design, error, rcond=None)
        remainder = error - design @ fit
        row = {"simulation_id": metadata["simulation_id"], "family": metadata["family"],
               "depth": metadata["depth"], "relative_l2": metadata["final_relative_l2"]["eta"],
               "phase_fraction_of_squared_error": float(phase.sum() / energy.sum()),
               "frequency_bands_error_fraction": {f"{lo}-{hi}": float(energy[lo:hi+1].sum()/energy.sum()) for lo, hi in bands},
               "frequency_bands_truth_fraction": {f"{lo}-{hi}": float((abs(ahat[lo:hi+1])**2*weights[lo:hi+1]).sum()/(abs(ahat)**2*weights).sum()) for lo, hi in bands},
               "fitted_global_amplitude_bias": float(fit[0]), "fitted_global_shift": float(fit[1]),
               "error_remaining_after_linearized_gain_shift_fit": float(np.linalg.norm(remainder)/np.linalg.norm(error)),
               "relative_l2_by_time": (np.linalg.norm(pred[:, i]-truth[:, i], axis=-1)/(np.linalg.norm(truth[:, i],axis=-1)+1e-12)).tolist()}
        rows.append(row)
    report = {"scope": "Post-hoc diagnostics on64 rollout cases; fits use reference truth and are not deployable corrections",
              "times": times.tolist(), "worst_six": [rows[i] for i in summary["ranked_indices"][:6]], "all_cases": rows}
    (ROOT / "experiments/diagnose_rollout_residuals_20260926.json").write_text(json.dumps(report,indent=2)+'\n')
    for row in report["worst_six"]:
        print(json.dumps({k:v for k,v in row.items() if k not in ('relative_l2_by_time','frequency_bands_truth_fraction')}))


if __name__ == '__main__':
    main()
