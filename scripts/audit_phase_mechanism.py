"""Split saved-rollout translation drift into direct forcing and state response.

No training, trajectory integration, or differentiation through a trajectory.
The near-zero surface-alignment phase is meaningful mainly for traveling waves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_rollout_defect import translation_fit
from eval_dno_pointwise import FAMILIES, ROOT, RUNS, band, chunked
import jax
import numpy as np
from scipy.integrate import cumulative_trapezoid

from solver.solvers.dno_series_jax import build_grid, dno_series_eval
from solver.solvers.time_integrator import dealiased_zakharov_xi_rhs


def dx(field: np.ndarray) -> np.ndarray:
    return np.fft.irfft(1j * np.arange(field.shape[-1] // 2 + 1)
                       * np.fft.rfft(field), n=field.shape[-1])


def phase_rate(reference: np.ndarray, predicted: np.ndarray, reference_rhs: np.ndarray,
               predicted_rhs: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Differentiate <T_s predicted-reference, T_s predicted_x>=0 in time."""
    nx = reference.shape[-1]
    k = np.arange(nx // 2 + 1)
    translation = np.exp(1j * k * shift[..., None])
    aligned_hat = np.fft.rfft(predicted) * translation
    residual = np.fft.irfft(aligned_hat, n=nx) - reference
    tangent = np.fft.irfft(1j * k * aligned_hat, n=nx)
    curvature = np.fft.irfft(-k**2 * aligned_hat, n=nx)
    rhs_hat = np.fft.rfft(predicted_rhs) * translation
    aligned_rhs = np.fft.irfft(rhs_hat, n=nx)
    rhs_x = np.fft.irfft(1j * k * rhs_hat, n=nx)
    denominator = np.sum(tangent**2 + residual * curvature, axis=-1)
    if np.any(denominator <= 0):
        raise FloatingPointError("Surface alignment is not at a strict local minimum")
    return -np.sum((aligned_rhs - reference_rhs) * tangent + residual * rhs_x,
                   axis=-1) / denominator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / RUNS["rms_context"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    x = np.arange(1024) * (2 * np.pi / 1024)
    reference = (np.cos(3 * x) + .3 * np.sin(7 * x))[None]
    predicted = (1.003 * np.cos(3 * (x - .001)) + .301 * np.sin(7 * (x - .001)))[None]
    ref_rhs, pred_rhs = -.2 * dx(reference), -.21 * dx(predicted) + .001 * np.cos(4 * x)
    shift, _ = translation_fit(reference, predicted)
    step = 1e-5
    plus, _ = translation_fit(reference + step * ref_rhs, predicted + step * pred_rhs)
    minus, _ = translation_fit(reference - step * ref_rhs, predicted - step * pred_rhs)
    np.testing.assert_allclose(phase_rate(reference, predicted, ref_rhs, pred_rhs, shift),
                               (plus - minus) / (2 * step), atol=1e-10, rtol=1e-7)
    result = {"run": str(args.run_dir), "device": jax.devices()[0].device_kind,
              "definition": "positive s means eta_pred(x)=eta_ref(x-s); eta alignment only",
              "decomposition": "direct: F_network(U_network)-F6(U_network); "
                               "response: F6(U_network)-F6(U_reference), after alignment",
              "reference": "actual saved fields; fresh FP64 M6 pad8 at neural states; P128",
              "families": {}}
    for family in FAMILIES:
        group = 0 if family in ("stokes", "tanaka") else 1
        folder = args.run_dir / f"eval_epoch120_test_stratified_n32_gpu{group}"
        with np.load(folder / f"{family}_trajs.npz") as archive:
            eta, xi, qn, ref_eta, ref_xi, qr = (archive[name] for name in
                ("pred_eta", "pred_xi", "pred_gxi", "truth_eta", "truth_xi", "truth_gxi"))
            times, depths, ids = (archive[name] for name in ("times", "depths", "simulation_ids"))
        nt, nc, nx = eta.shape
        _, solver_k = build_grid(nx, 2 * np.pi)
        q6_full = chunked(jax.jit(lambda e, p, h: dno_series_eval(
            e, p, solver_k, h[:, None], 6, pad_factor=8)),
            eta.reshape(-1, nx), xi.reshape(-1, nx), np.tile(depths, nt)).reshape(nt, nc, nx)
        qn_full = qn
        qn, qr, q6 = band(qn), band(qr), band(q6_full)
        shift, _ = translation_fit(ref_eta, eta)
        direct = phase_rate(ref_eta, eta, np.zeros_like(qr), qn - q6, shift)
        response = phase_rate(ref_eta, eta, qr, q6, shift)
        rates = np.stack((direct, response, direct + response), axis=-1)
        integrals = cumulative_trapezoid(rates, times, axis=0, initial=0)
        finite_difference_checks = []
        for step in (1e-4, 1e-5):
            plus, _ = translation_fit(ref_eta[-1] + step * qr[-1], eta[-1] + step * qn[-1])
            minus, _ = translation_fit(ref_eta[-1] - step * qr[-1], eta[-1] - step * qn[-1])
            finite_difference_checks.append(float(np.max(np.abs(
                (plus - minus) / (2 * step) - rates[-1, :, 2]))))
        eta_x, xi_x, ref_xi_x = dx(eta), dx(xi), dx(ref_xi)
        nonlinear = []
        for q in (qn_full, q6_full):
            nonlinear.append(chunked(jax.jit(dealiased_zakharov_xi_rhs),
                eta_x.reshape(-1, nx), xi_x.reshape(-1, nx), q.reshape(-1, nx)).reshape(nt, nc, nx))
        momentum_forcing = np.mean((qn - q6) * xi_x - eta_x * (nonlinear[0] - nonlinear[1]), axis=-1)
        momentum_scale = np.maximum(np.mean(np.abs(ref_eta[0] * ref_xi_x[0]), axis=-1), 1e-30)
        momentum_drift = np.mean(eta * xi_x - ref_eta * ref_xi_x, axis=-1) / momentum_scale
        momentum_integral = cumulative_trapezoid(momentum_forcing, times, axis=0, initial=0) / momentum_scale
        terminal_scale = np.maximum(np.abs(shift[-1]), 1e-12)
        signed_fractions = integrals[-1, :, :2] / np.where(np.abs(shift[-1, :, None]) > 1e-12,
                                                         shift[-1, :, None], np.nan)
        summary = {
            "times": times.tolist(), "simulation_ids": ids.tolist(), "depths": depths.tolist(),
            "parameter_group_ids": json.loads((folder / f"{family}_summary.json").read_text())[
                "evaluation_source"]["selected_parameter_group_ids"],
            "surface_displacement": shift.tolist(),
            "rate_columns": ["direct", "state_response", "sum"], "phase_rates": rates.tolist(),
            "integrated_phase_rates": integrals.tolist(),
            "terminal_phase_rate_finite_difference_max_abs_error": finite_difference_checks,
            "half_cadence_integrated_phase_rates": np.trapezoid(rates[::2], times[::2], axis=0).tolist(),
            "momentum_drift": momentum_drift.tolist(), "integrated_momentum_forcing": momentum_integral.tolist(),
            "terminal_reconstruction_relative_error_median": float(np.median(
                np.abs(integrals[-1, :, 2] - shift[-1]) / terminal_scale)),
            "terminal_reconstruction_correlation": float(np.corrcoef(integrals[-1, :, 2], shift[-1])[0, 1]),
            "direct_signed_fraction_median": float(np.nanmedian(signed_fractions[:, 0])),
            "response_signed_fraction_median": float(np.nanmedian(signed_fractions[:, 1])),
            "response_dominates_direct_count": int(np.sum(np.abs(integrals[-1, :, 1]) > np.abs(integrals[-1, :, 0]))),
            "momentum_forcing_reconstruction_relative_error_median": float(np.median(
                np.abs(momentum_integral[-1] - momentum_drift[-1]) / np.maximum(np.abs(momentum_drift[-1]), 1e-12))),
            "momentum_displacement_correlation": float(np.corrcoef(momentum_drift[-1], shift[-1])[0, 1]),
            "momentum_response_rate_correlation_median": float(np.median([
                np.corrcoef(momentum_drift[:, case], response[:, case])[0, 1] for case in range(nc)])),
        }
        if not np.isfinite(rates).all():
            raise FloatingPointError(f"Nonfinite phase rate: {family}")
        result["families"][family] = summary
        print(f"{family}: " + json.dumps({key: value for key, value in summary.items()
              if key.endswith(("median", "count", "correlation"))}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
