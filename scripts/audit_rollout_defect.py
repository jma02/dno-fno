"""Measure same-state DNO defects and signed translation forcing on saved rollouts.

No training or integration. The signed speed proxy concerns eta_t only, not the
full coupled water-wave response. Translation fits select the branch near zero.
Use JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 to compare with the saved Ada
predictions; CPU execution is only an arithmetic-sensitivity diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_dno_pointwise import FAMILIES, ROOT, RUNS, band, chunked, statistics
import jax
import numpy as np

from solver.evals.model_rollout import build_predict_gxi_batched, load_run
from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def translation_fit(reference: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_hat, pred_hat = np.fft.rfft(reference), np.fft.rfft(predicted)
    k = np.arange(ref_hat.shape[-1])
    weights = np.full(len(k), 2.0)
    weights[[0, -1]] = 1.0
    shift = np.zeros(reference.shape[:-1])
    for _ in range(20):
        aligned = pred_hat * np.exp(1j * k * shift[..., None])
        derivative = 1j * k * aligned
        shift -= (np.sum(weights * np.real(np.conj(derivative) * (aligned - ref_hat)), axis=-1)
                  / np.maximum(np.sum(weights * np.abs(derivative)**2, axis=-1), 1e-30))
    aligned = pred_hat * np.exp(1j * k * shift[..., None])
    error = np.sqrt(np.sum(weights * np.abs(aligned - ref_hat)**2, axis=-1)
                    / np.maximum(np.sum(weights * np.abs(ref_hat)**2, axis=-1), 1e-30))
    return shift, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / RUNS["rms_context"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = load_run(args.run_dir, checkpoint="latest")
    predict = build_predict_gxi_batched(run)
    x = np.arange(1024) * (2 * np.pi / 1024)
    synthetic = np.cos(3 * x) + .3 * np.sin(7 * x)
    displaced = np.cos(3 * (x - .001)) + .3 * np.sin(7 * (x - .001))
    synthetic_shift, synthetic_error = translation_fit(synthetic[None], displaced[None])
    np.testing.assert_allclose(synthetic_shift, .001, atol=1e-12)
    assert float(synthetic_error[0]) < 1e-12
    results = {
        "run": str(args.run_dir), "epoch": run.epoch, "checkpoint": "latest",
        "reference": "fresh FP64 M6, pad8, actual depth, actual saved inputs, P128 outputs",
        "device": jax.devices()[0].device_kind,
        "neural_state_network": "actual saved GPU pred_gxi; recomputed predictor used for parity check",
        "speed_proxy": "dc=-<q_method-q6,eta_x>/||eta_x||^2; eta equation only",
        "coherence": "abs(trapezoidal integral dc)/trapezoidal integral abs(dc), per trajectory",
        "translation_sign": "eta_pred(x)=eta_ref(x-s) gives positive fitted displacement s",
        "families": {},
    }
    for family in FAMILIES:
        group = 0 if family in ("stokes", "tanaka") else 1
        folder = args.run_dir / f"eval_epoch{run.epoch}_test_stratified_n32_gpu{group}"
        with np.load(folder / f"{family}_trajs.npz") as archive:
            fields = {name: archive[name] for name in
                      ("truth_eta", "truth_xi", "truth_gxi", "pred_eta", "pred_xi", "pred_gxi")}
            times, depths, ids = (archive[name] for name in ("times", "depths", "simulation_ids"))
        nt, nc, nx = fields["truth_eta"].shape
        k = np.arange(nx // 2 + 1)
        _, solver_k = build_grid(nx, 2 * np.pi)
        time_weights = np.empty(nt)
        time_weights[[0, -1]] = .5 * np.array([times[1] - times[0], times[-1] - times[-2]])
        time_weights[1:-1] = .5 * (times[2:] - times[:-2])
        time_weights /= times[-1] - times[0]
        summary = {"times": times.tolist(), "simulation_ids": ids.tolist(), "states": {}}
        reference_norm = None
        for state, prefix in (("reference", "truth"), ("neural", "pred")):
            eta, xi = fields[f"{prefix}_eta"], fields[f"{prefix}_xi"]
            input_fields = (eta.reshape(-1, nx), xi.reshape(-1, nx), np.tile(depths, nt))
            targets = {}
            for order in (3, 6):
                targets[order] = band(chunked(
                    jax.jit(lambda e, p, h: dno_series_eval(
                        e, p, solver_k, h[:, None], order, pad_factor=8,
                    )), *input_fields,
                )).reshape(nt, nc, nx)
            nn = band(chunked(predict, input_fields[0], input_fields[1], np.log(input_fields[2])))
            nn = nn.reshape(nt, nc, nx)
            target_norm = np.linalg.norm(targets[6], axis=-1)
            if reference_norm is None:
                reference_norm = target_norm
            eta_x = np.fft.irfft(1j * k * np.fft.rfft(eta), n=nx)
            tangent_norm2 = np.sum(eta_x**2, axis=-1)
            state_summary = {
                "saved_q_relative_check": statistics(
                    np.linalg.norm((targets[6] if state == "reference" else nn)
                                   - band(fields[f"{prefix}_gxi"]), axis=-1) / target_norm),
                "methods": {},
            }
            if state == "neural":
                nn = band(fields["pred_gxi"])
            for name, predicted in (("network", nn), ("M3", targets[3])):
                error = predicted - targets[6]
                if not np.isfinite(error).all():
                    raise FloatingPointError(f"Nonfinite defect: {family}/{state}/{name}")
                norm_error = np.linalg.norm(error, axis=-1)
                relative = norm_error / target_norm
                inner = np.sum(error * eta_x, axis=-1)
                dc = -inner / np.maximum(tangent_norm2, 1e-30)
                time_mean = np.sum(time_weights[:, None] * relative, axis=0)
                signed_integral = (times[-1] - times[0]) * np.sum(time_weights[:, None] * dc, axis=0)
                absolute_integral = (times[-1] - times[0]) * np.sum(time_weights[:, None] * np.abs(dc), axis=0)
                coherence = np.abs(signed_integral) / np.maximum(absolute_integral, 1e-30)
                projection_fraction = inner**2 / np.maximum(tangent_norm2 * norm_error**2, 1e-30)
                state_summary["methods"][name] = {
                    "relative_defect": relative.tolist(), "signed_speed_error": dc.tolist(),
                    "time_mean_relative_defect": time_mean.tolist(),
                    "time_mean_common_reference_relative_defect": np.sum(
                        time_weights[:, None] * norm_error / reference_norm, axis=0).tolist(),
                    "time_mean_tangent_fraction_of_defect_energy": np.sum(
                        time_weights[:, None] * projection_fraction, axis=0).tolist(),
                    "integrated_speed_error": signed_integral.tolist(), "coherence": coherence.tolist(),
                }
                print(f"{family}/{state}/{name}: mean_defect={time_mean.mean():.6g} "
                      f"median_case_defect={np.median(time_mean):.6g} "
                      f"median_coherence={np.median(coherence):.4f}", flush=True)
            summary["states"][state] = state_summary
        ref_nn = summary["states"]["reference"]["methods"]["network"]
        own_nn = summary["states"]["neural"]["methods"]["network"]
        growth = np.array(own_nn["time_mean_common_reference_relative_defect"]) / np.maximum(
            ref_nn["time_mean_common_reference_relative_defect"], 1e-30)
        displacement, aligned_error = translation_fit(fields["truth_eta"][-1], fields["pred_eta"][-1])
        final_error = np.linalg.norm(fields["pred_eta"][-1] - fields["truth_eta"][-1], axis=-1)
        final_error /= np.linalg.norm(fields["truth_eta"][-1], axis=-1)
        ref_hat, pred_hat = (np.fft.rfft(fields[f"{prefix}_eta"][-1]) for prefix in ("truth", "pred"))
        weights = np.full(ref_hat.shape[-1], 2.0)
        weights[[0, -1]] = 1.0
        modal_phase_fraction = 1 - np.sum(weights * (np.abs(pred_hat) - np.abs(ref_hat))**2, axis=-1) / np.maximum(
            np.sum(weights * np.abs(pred_hat - ref_hat)**2, axis=-1), 1e-30)
        summary["comparison"] = {
            "neural_over_reference_defect_ratio": growth.tolist(), "defect_growth": statistics(growth),
            "terminal_surface_error": final_error.tolist(), "terminal_displacement": displacement.tolist(),
            "terminal_aligned_surface_error": aligned_error.tolist(),
            "translation_fraction_of_terminal_error_energy": (1 - aligned_error**2 / np.maximum(
                final_error**2, 1e-30)).tolist(),
            "modal_phase_fraction_of_terminal_error_energy": modal_phase_fraction.tolist(),
            "reference_speed_integral_shift_correlation": float(np.corrcoef(
                ref_nn["integrated_speed_error"], displacement)[0, 1]),
            "neural_speed_integral_shift_correlation": float(np.corrcoef(
                own_nn["integrated_speed_error"], displacement)[0, 1]),
        }
        print(f"{family}: defect_growth median={np.median(growth):.4f} "
              f"p95={np.quantile(growth, .95):.4f}; shift_corr="
              f"{summary['comparison']['neural_speed_integral_shift_correlation']:.4f}", flush=True)
        results["families"][family] = summary
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + "\n")
