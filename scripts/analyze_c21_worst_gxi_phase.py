"""Audit G(eta)xi-driven phase propagation in C21's six finite worst cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar


CASES: dict[str, tuple[int, ...]] = {
    "tanaka_g0": (91, 186, 224),
    "tanaka_g1": (88, 202),
    "bf_modal": (22,),
}


def periodic_shift(values: np.ndarray, displacement: float, length: float) -> np.ndarray:
    """Return ``values(x - displacement)`` by Fourier interpolation."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        np.fft.fft(values) * np.exp(-1j * wave_numbers * displacement)
    ).real


def spectral_derivative(values: np.ndarray, length: float) -> np.ndarray:
    """Periodic Fourier derivative along the last axis."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(1j * wave_numbers * np.fft.fft(values)).real


def low_pass(values: np.ndarray, length: float, cutoff: int) -> np.ndarray:
    """Retain periodic modes with absolute wave number at most ``cutoff``."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        np.fft.fft(values, axis=-1)
        * (np.abs(wave_numbers) <= cutoff),
        axis=-1,
    ).real


def optimal_displacement(
    prediction: np.ndarray,
    truth: np.ndarray,
    length: float,
) -> float:
    """Find ``d`` such that prediction is closest to truth(x-d)."""
    nx = truth.size
    dx = length / nx
    correlation = np.fft.ifft(
        np.fft.fft(prediction) * np.conj(np.fft.fft(truth))
    ).real
    peak_indices = np.argpartition(correlation, -8)[-8:]

    def objective(displacement: float) -> float:
        aligned = periodic_shift(prediction, -displacement, length)
        return float(np.vdot(aligned - truth, aligned - truth).real)

    candidates: list[tuple[float, float]] = []
    for peak_index in peak_indices:
        signed_index = int(peak_index)
        if signed_index > nx // 2:
            signed_index -= nx
        coarse = signed_index * dx
        result = minimize_scalar(
            objective,
            bounds=(coarse - dx, coarse + dx),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidates.append((float(result.fun), float(result.x)))
    return min(candidates)[1]


def relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error) / (np.linalg.norm(reference) + 1e-30))


def first_crossing(times: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(np.isfinite(values) & (values > threshold))
    return float(times[indices[0]]) if indices.size else None


def analyze_case(
    archive_path: Path,
    regime: str,
    case_index: int,
    length: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(archive_path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        case_id = int(np.asarray(archive["case_ids"])[case_index])
        depth = float(np.asarray(archive["depths"])[case_index])
        truth_eta = np.asarray(archive["truth_eta"][:, case_index], dtype=np.float64)
        pred_eta = np.asarray(archive["pred_eta"][:, case_index], dtype=np.float64)
        truth_gxi = np.asarray(archive["truth_gxi"][:, case_index], dtype=np.float64)
        pred_gxi = np.asarray(archive["pred_gxi"][:, case_index], dtype=np.float64)

    n_times = times.size
    displacement = np.empty(n_times, dtype=np.float64)
    raw_eta_error = np.empty(n_times, dtype=np.float64)
    aligned_eta_error = np.empty(n_times, dtype=np.float64)
    raw_gxi_error = np.empty(n_times, dtype=np.float64)
    aligned_gxi_error = np.empty(n_times, dtype=np.float64)
    speed_defect = np.empty(n_times, dtype=np.float64)
    gxi_tangent_fraction = np.empty(n_times, dtype=np.float64)
    resolved_speed_defect = np.empty(n_times, dtype=np.float64)

    for frame in range(n_times):
        displacement[frame] = optimal_displacement(
            pred_eta[frame], truth_eta[frame], length
        )
        pred_eta_aligned = periodic_shift(
            pred_eta[frame], -displacement[frame], length
        )
        pred_gxi_aligned = periodic_shift(
            pred_gxi[frame], -displacement[frame], length
        )
        eta_x = spectral_derivative(truth_eta[frame], length)
        gxi_error = pred_gxi_aligned - truth_gxi[frame]
        tangent_norm_sq = float(np.vdot(eta_x, eta_x).real)
        speed_defect[frame] = -float(
            np.vdot(gxi_error, eta_x).real / (tangent_norm_sq + 1e-30)
        )
        tangent_gxi_error = -speed_defect[frame] * eta_x
        gxi_error_norm_sq = float(np.vdot(gxi_error, gxi_error).real)
        gxi_tangent_fraction[frame] = float(
            np.vdot(tangent_gxi_error, tangent_gxi_error).real
            / (gxi_error_norm_sq + 1e-30)
        )

        if regime == "bf_modal":
            # BF's eta has essentially no tangent energy below mode 8, so
            # this low-mode projection is ill-conditioned and meaningless.
            resolved_speed_defect[frame] = np.nan
        else:
            eta_x_resolved = low_pass(eta_x, length, cutoff=8)
            gxi_error_resolved = low_pass(gxi_error, length, cutoff=8)
            resolved_speed_defect[frame] = -float(
                np.vdot(gxi_error_resolved, eta_x_resolved).real
                / (np.vdot(eta_x_resolved, eta_x_resolved).real + 1e-30)
            )

        raw_eta_error[frame] = relative_l2(
            pred_eta[frame] - truth_eta[frame], truth_eta[frame]
        )
        aligned_eta_error[frame] = relative_l2(
            pred_eta_aligned - truth_eta[frame], truth_eta[frame]
        )
        raw_gxi_error[frame] = relative_l2(
            pred_gxi[frame] - truth_gxi[frame], truth_gxi[frame]
        )
        aligned_gxi_error[frame] = relative_l2(
            gxi_error, truth_gxi[frame]
        )

    displacement = np.unwrap(displacement * (2.0 * np.pi / length)) * (
        length / (2.0 * np.pi)
    )
    eta_error = pred_eta - truth_eta
    gxi_error_lab = pred_gxi - truth_gxi
    propagation_by_cutoff: dict[str, dict[str, float]] = {}
    for cutoff in (4, 8, 16, 32):
        eta_error_resolved = low_pass(eta_error, length, cutoff)
        gxi_error_resolved = low_pass(gxi_error_lab, length, cutoff)
        integrated_gxi_error = cumulative_trapezoid(
            gxi_error_resolved,
            times,
            axis=0,
            initial=0.0,
        )
        final_eta_defect = eta_error_resolved[-1]
        final_integrated_defect = integrated_gxi_error[-1]
        propagation_by_cutoff[str(cutoff)] = {
            "relative_residual": relative_l2(
                final_integrated_defect - final_eta_defect,
                final_eta_defect,
            ),
            "cosine": float(
                np.vdot(final_integrated_defect, final_eta_defect).real
                / (
                    np.linalg.norm(final_integrated_defect)
                    * np.linalg.norm(final_eta_defect)
                    + 1e-30
                )
            ),
            "resolved_final_eta_error_over_truth": relative_l2(
                final_eta_defect,
                low_pass(truth_eta[-1], length, cutoff),
            ),
        }
    truth_eta_hat = np.fft.rfft(truth_eta, axis=-1)
    pred_eta_hat = np.fft.rfft(pred_eta, axis=-1)
    truth_gxi_hat = np.fft.rfft(truth_gxi, axis=-1)
    pred_gxi_hat = np.fft.rfft(pred_gxi, axis=-1)
    modal_phase_propagation: dict[str, dict[str, float]] = {}
    modes = (17, 20, 23) if regime == "bf_modal" else tuple(range(1, 9))
    for mode in modes:
        wave_number = 2.0 * np.pi * mode / length
        phase_error = np.unwrap(
            np.angle(
                pred_eta_hat[:, mode] * np.conj(truth_eta_hat[:, mode])
            )
        )
        phase_error -= phase_error[0]
        phase_rate_error = np.imag(
            pred_gxi_hat[:, mode] / (pred_eta_hat[:, mode] + 1e-30)
            - truth_gxi_hat[:, mode] / (truth_eta_hat[:, mode] + 1e-30)
        )
        integrated_phase_rate_error = float(
            np.trapezoid(phase_rate_error, times)
        )
        modal_phase_propagation[str(mode)] = {
            "final_eta_phase_error": float(phase_error[-1]),
            "integrated_gxi_phase_rate_error": integrated_phase_rate_error,
            "phase_identity_residual": float(
                phase_error[-1] - integrated_phase_rate_error
            ),
            "observed_modal_displacement": float(
                -phase_error[-1] / wave_number
            ),
            "gxi_integrated_modal_displacement": float(
                -integrated_phase_rate_error / wave_number
            ),
            "final_eta_amplitude_ratio": float(
                np.abs(pred_eta_hat[-1, mode])
                / (np.abs(truth_eta_hat[-1, mode]) + 1e-30)
            ),
        }
    eta_x_final = spectral_derivative(truth_eta[-1], length)
    kappa_final = float(
        np.linalg.norm(eta_x_final) / (np.linalg.norm(truth_eta[-1]) + 1e-30)
    )
    squared_removed = 1.0 - (
        aligned_eta_error[-1] ** 2 / (raw_eta_error[-1] ** 2 + 1e-30)
    )
    summary: dict[str, object] = {
        "case_index": case_index,
        "case_id": case_id,
        "depth": depth,
        "final_raw_eta_error": float(raw_eta_error[-1]),
        "final_aligned_eta_error": float(aligned_eta_error[-1]),
        "final_eta_squared_error_removed_by_translation": float(squared_removed),
        "final_displacement": float(displacement[-1]),
        "final_displacement_grid_points": float(
            displacement[-1] / (length / truth_eta.shape[-1])
        ),
        "final_kappa": kappa_final,
        "linearized_eta_error_from_final_shift": float(
            abs(displacement[-1]) * kappa_final
        ),
        "final_raw_gxi_error": float(raw_gxi_error[-1]),
        "final_aligned_gxi_error": float(aligned_gxi_error[-1]),
        "median_aligned_gxi_error": float(np.median(aligned_gxi_error)),
        "p95_aligned_gxi_error": float(np.percentile(aligned_gxi_error, 95)),
        "initial_aligned_gxi_error": float(aligned_gxi_error[0]),
        "initial_speed_defect": float(speed_defect[0]),
        "initial_k8_speed_defect": (
            None
            if regime == "bf_modal"
            else float(resolved_speed_defect[0])
        ),
        "lab_frame_eta_t_equals_gxi_error_propagation": propagation_by_cutoff,
        "modal_eta_phase_from_gxi_phase_rate": modal_phase_propagation,
        "median_gxi_error_tangent_fraction": float(
            np.median(gxi_tangent_fraction)
        ),
        "final_gxi_error_tangent_fraction": float(gxi_tangent_fraction[-1]),
        "first_crossing_times": {
            str(threshold): first_crossing(times, raw_eta_error, threshold)
            for threshold in (0.25, 0.5, 0.75, 1.0)
        },
    }
    series = {
        "times": times,
        "displacement": displacement,
        "snapshot_gxi_translation_tangent_projection": speed_defect,
        "snapshot_k8_gxi_translation_tangent_projection": resolved_speed_defect,
        "raw_eta_error": raw_eta_error,
        "aligned_eta_error": aligned_eta_error,
        "raw_gxi_error": raw_gxi_error,
        "aligned_gxi_error": aligned_gxi_error,
        "gxi_tangent_fraction": gxi_tangent_fraction,
    }
    return summary, series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for regime, indices in CASES.items():
        archive_path = eval_dir / regime / f"{regime}_trajs.npz"
        for case_index in indices:
            print(f"analyzing {regime} index {case_index}", flush=True)
            summary, series = analyze_case(
                archive_path, regime, case_index, args.length
            )
            summary["regime"] = regime
            summaries.append(summary)
            np.savez_compressed(
                output_dir / f"{regime}_idx{case_index}_gxi_phase_series.npz",
                **series,
            )

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
