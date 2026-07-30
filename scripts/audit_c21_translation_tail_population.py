"""Population audit of translation diagnostics on the fresh C21 Tanaka panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_c21_worst_gxi_phase import optimal_displacement


REGIMES = ("tanaka_g0", "tanaka_g1")


def aligned_fields(
    prediction: np.ndarray,
    displacement: np.ndarray,
    length: float,
) -> np.ndarray:
    """Shift each predicted row by the negative fitted displacement."""
    nx = prediction.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    phase = np.exp(1j * displacement[:, None] * wave_numbers[None, :])
    return np.fft.ifft(np.fft.fft(prediction, axis=-1) * phase, axis=-1).real


def fitted_displacements(
    prediction: np.ndarray,
    truth: np.ndarray,
    length: float,
) -> np.ndarray:
    """Compute the continuous best periodic displacement for every row."""
    return np.asarray(
        [
            optimal_displacement(prediction_i, truth_i, length)
            for prediction_i, truth_i in zip(prediction, truth)
        ],
        dtype=np.float64,
    )


def horizontal_momentum(
    eta: np.ndarray,
    xi: np.ndarray,
    length: float,
) -> np.ndarray:
    """Return the periodic horizontal momentum density integral per sample."""
    nx = eta.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    xi_x = np.fft.ifft(
        1j * wave_numbers[None, :] * np.fft.fft(xi, axis=-1),
        axis=-1,
    ).real
    return np.mean(eta * xi_x, axis=-1)


def horizontal_momentum_rate(
    eta: np.ndarray,
    xi: np.ndarray,
    q: np.ndarray,
    length: float,
) -> np.ndarray:
    """Return the momentum rate implied by the production Zakharov RHS."""
    nx = eta.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    eta_x = np.fft.ifft(
        1j * wave_numbers[None, :] * np.fft.fft(eta, axis=-1),
        axis=-1,
    ).real
    xi_x = np.fft.ifft(
        1j * wave_numbers[None, :] * np.fft.fft(xi, axis=-1),
        axis=-1,
    ).real
    xi_rhs = (
        -eta
        - 0.5 * xi_x**2
        + (q + eta_x * xi_x) ** 2 / (2.0 * (1.0 + eta_x**2))
    )
    return np.mean(q * xi_x - eta_x * xi_rhs, axis=-1)


def tangent_diagnostics(
    eta: np.ndarray,
    q_prediction: np.ndarray,
    q_target: np.ndarray,
    depth: np.ndarray,
    length: float,
) -> dict[str, np.ndarray]:
    """Reproduce the per-sample localized tangent quantities in NumPy."""
    nx = eta.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    k_rfft = np.abs(2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx))
    eta_hat = np.fft.fft(eta, axis=-1)
    eta_x = np.fft.ifft(
        1j * wave_numbers[None, :] * eta_hat,
        axis=-1,
    ).real
    error = (
        q_prediction
        - np.mean(q_prediction, axis=-1, keepdims=True)
        - q_target
        + np.mean(q_target, axis=-1, keepdims=True)
    )
    fields = np.stack((eta_x * error, eta_x**2), axis=1)
    multiplier = np.exp(-0.5 * (depth[:, None] * k_rfft[None, :]) ** 2)
    smoothed = np.fft.irfft(
        np.fft.rfft(fields, axis=-1) * multiplier[:, None, :],
        n=nx,
        axis=-1,
    ).real
    local_cross = smoothed[:, 0]
    local_energy = np.maximum(smoothed[:, 1], 0.0)
    energy_floor = 1e-3 * np.max(local_energy, axis=-1, keepdims=True)
    local_speed_error = -local_cross / (local_energy + energy_floor + 1e-30)
    physical_speed = np.sqrt(np.maximum(depth, 1e-12))
    local_relative_speed = local_speed_error / physical_speed[:, None]
    local_weight = local_energy / (
        np.sum(local_energy, axis=-1, keepdims=True) + 1e-30
    )
    tangent_loss = np.sum(local_weight * local_relative_speed**2, axis=-1)
    rigid_relative_speed = np.sum(local_weight * local_relative_speed, axis=-1)
    rigid_loss = rigid_relative_speed**2
    differential_loss = np.maximum(tangent_loss - rigid_loss, 0.0)
    kappa_squared = np.sum(eta_x**2, axis=-1) / (
        np.sum(eta**2, axis=-1) + 1e-30
    )
    phase_growth_loss = depth * kappa_squared * tangent_loss

    global_denominator = np.sum(eta_x**2, axis=-1) + 1e-30
    global_speed_error = -np.sum(eta_x * error, axis=-1) / global_denominator
    target_speed = -np.sum(eta_x * q_target, axis=-1) / global_denominator
    error_hat = np.fft.rfft(error, axis=-1)
    target_centered = q_target - np.mean(q_target, axis=-1, keepdims=True)
    target_hat = np.fft.rfft(target_centered, axis=-1)
    h1_weight_squared = 1.0 + k_rfft**2
    relative_q_l2_squared = np.sum(error**2, axis=-1) / (
        np.sum(target_centered**2, axis=-1) + 1e-30
    )
    kinematic_rate_squared = np.sum(error**2, axis=-1) / (
        np.sum(eta**2, axis=-1) + 1e-30
    )
    phase_sensitive_q_l2_squared = (
        depth * kappa_squared * relative_q_l2_squared
    )
    relative_q_h1_squared = np.sum(
        h1_weight_squared[None, :] * np.abs(error_hat) ** 2,
        axis=-1,
    ) / (
        np.sum(
            h1_weight_squared[None, :] * np.abs(target_hat) ** 2,
            axis=-1,
        )
        + 1e-30
    )
    eta_hat_forward = np.fft.rfft(eta, axis=-1, norm="forward")
    prediction_hat_forward = np.fft.rfft(
        q_prediction,
        axis=-1,
        norm="forward",
    )
    target_hat_forward = np.fft.rfft(q_target, axis=-1, norm="forward")
    omega_squared = (
        k_rfft[None, :] * np.tanh(depth[:, None] * k_rfft[None, :])
    )
    effective_frequency_squared = np.sum(
        omega_squared * np.abs(eta_hat_forward) ** 2,
        axis=-1,
    ) / (np.sum(np.abs(eta_hat_forward) ** 2, axis=-1) + 1e-30)
    physical_scale = (
        np.abs(target_hat_forward) ** 2
        + omega_squared * np.abs(eta_hat_forward) ** 2
    )
    mode_band = (k_rfft > 0.0) & (k_rfft <= 128.0)
    sample_scale = np.max(
        np.where(mode_band[None, :], physical_scale, 0.0),
        axis=-1,
        keepdims=True,
    )
    denominator_floor = 1e-6 * sample_scale + 1e-24
    activity_floor = 1e-4 * sample_scale + 1e-24
    mode_weight = np.where(
        mode_band[None, :],
        physical_scale / (physical_scale + activity_floor),
        0.0,
    )
    squared_ratio = (
        np.abs(prediction_hat_forward - target_hat_forward) ** 2
        / (physical_scale + denominator_floor)
    )
    clipped_ratio = np.minimum(squared_ratio, 100.0)
    mode_penalty = 2.0 * clipped_ratio / (
        np.sqrt(1.0 + clipped_ratio) + 1.0
    )
    mode_balanced_loss = np.sum(mode_weight * mode_penalty, axis=-1) / (
        np.sum(mode_weight, axis=-1) + 1e-30
    )
    phase_sensitive_mode_loss = depth * kappa_squared * mode_balanced_loss
    dispersion_sensitive_mode_loss = (
        effective_frequency_squared * mode_balanced_loss
    )
    return {
        "relative_q_l2_squared": relative_q_l2_squared,
        "kinematic_rate_squared": kinematic_rate_squared,
        "phase_sensitive_q_l2_squared": phase_sensitive_q_l2_squared,
        "relative_q_h1_squared": relative_q_h1_squared,
        "mode_balanced_loss": mode_balanced_loss,
        "phase_sensitive_mode_loss": phase_sensitive_mode_loss,
        "dispersion_sensitive_mode_loss": dispersion_sensitive_mode_loss,
        "tangent_loss": tangent_loss,
        "phase_growth_loss": phase_growth_loss,
        "rigid_relative_speed": rigid_relative_speed,
        "differential_fraction": differential_loss / (tangent_loss + 1e-30),
        "global_speed_error": global_speed_error,
        "global_relative_speed_error": global_speed_error
        / (np.abs(target_speed) + 1e-30),
        "kappa": np.sqrt(kappa_squared),
    }


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_x: bool = False,
) -> dict[str, float | None]:
    """Return finite Pearson and Spearman correlations."""
    x_work = np.log10(np.maximum(x, 1e-30)) if log_x else x
    keep = np.isfinite(x_work) & np.isfinite(y)
    if (
        np.count_nonzero(keep) < 3
        or np.ptp(x_work[keep]) == 0.0
        or np.ptp(y[keep]) == 0.0
    ):
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x_work[keep], y[keep]).statistic),
        "spearman": float(spearmanr(x_work[keep], y[keep]).statistic),
    }


def top_fraction_recall(
    score: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> float:
    """Fraction of target cases captured by the largest score fraction."""
    n_selected = max(int(np.ceil(fraction * score.size)), 1)
    selected = np.argpartition(score, -n_selected)[-n_selected:]
    target_count = int(np.count_nonzero(target))
    if target_count == 0:
        return float("nan")
    return float(np.count_nonzero(target[selected]) / target_count)


def score_concentration(score: np.ndarray) -> dict[str, float]:
    """Return upper-tail shares and the participation-ratio sample size."""
    finite_score = np.maximum(score[np.isfinite(score)], 0.0)
    total = float(np.sum(finite_score))
    if total == 0.0:
        return {
            "top_1pct_share": 0.0,
            "top_5pct_share": 0.0,
            "effective_sample_size": 0.0,
        }
    ordered = np.sort(finite_score)[::-1]
    n_top_1 = max(int(np.ceil(0.01 * ordered.size)), 1)
    n_top_5 = max(int(np.ceil(0.05 * ordered.size)), 1)
    return {
        "top_1pct_share": float(np.sum(ordered[:n_top_1]) / total),
        "top_5pct_share": float(np.sum(ordered[:n_top_5]) / total),
        "effective_sample_size": float(
            total**2 / (np.sum(finite_score**2) + 1e-30)
        ),
    }


def audit_regime(
    archive_path: Path,
    regime: str,
    length: float,
    diagnostic_times: tuple[float, ...],
) -> dict[str, Any]:
    """Audit one saved Tanaka panel."""
    with np.load(archive_path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        depth = np.asarray(archive["depths"], dtype=np.float64)
        case_ids = np.asarray(archive["case_ids"], dtype=np.int64)
        truth_valid = np.asarray(archive["truth_valid"], dtype=bool)
        model_nonfinite = np.asarray(archive["model_nonfinite_any"], dtype=bool)
        truth_eta = np.asarray(archive["truth_eta"], dtype=np.float64)
        pred_eta = np.asarray(archive["pred_eta"], dtype=np.float64)
        truth_xi = np.asarray(archive["truth_xi"], dtype=np.float64)
        pred_xi = np.asarray(archive["pred_xi"], dtype=np.float64)
        truth_q = np.asarray(archive["truth_gxi"], dtype=np.float64)
        pred_q = np.asarray(archive["pred_gxi"], dtype=np.float64)

    valid = truth_valid & ~model_nonfinite
    final_displacement = fitted_displacements(pred_eta[-1], truth_eta[-1], length)
    final_eta_aligned = aligned_fields(pred_eta[-1], final_displacement, length)
    truth_norm = np.linalg.norm(truth_eta[-1], axis=-1) + 1e-30
    raw_error = np.linalg.norm(pred_eta[-1] - truth_eta[-1], axis=-1) / truth_norm
    aligned_error = np.linalg.norm(
        final_eta_aligned - truth_eta[-1], axis=-1
    ) / truth_norm
    removed_fraction = 1.0 - aligned_error**2 / (raw_error**2 + 1e-30)
    translation_component = np.sqrt(
        np.maximum(raw_error**2 - aligned_error**2, 0.0)
    )
    translation_tail = valid & (raw_error > 0.25) & (removed_fraction > 0.9)
    initial_momentum = horizontal_momentum(
        truth_eta[0],
        truth_xi[0],
        length,
    )

    frame_results: dict[str, Any] = {}
    frame_arrays: dict[str, dict[str, np.ndarray]] = {}
    for requested_time in diagnostic_times:
        frame = int(np.argmin(np.abs(times - requested_time)))
        displacement = fitted_displacements(
            pred_eta[frame], truth_eta[frame], length
        )
        aligned_q = aligned_fields(pred_q[frame], displacement, length)
        diagnostics = tangent_diagnostics(
            truth_eta[frame], aligned_q, truth_q[frame], depth, length
        )
        truth_momentum = horizontal_momentum(
            truth_eta[frame],
            truth_xi[frame],
            length,
        )
        pred_momentum = horizontal_momentum(
            pred_eta[frame],
            pred_xi[frame],
            length,
        )
        diagnostics["relative_momentum_defect_abs"] = np.abs(
            pred_momentum - truth_momentum
        ) / (np.abs(initial_momentum) + 1e-30)
        truth_momentum_rate = horizontal_momentum_rate(
            truth_eta[frame],
            truth_xi[frame],
            truth_q[frame],
            length,
        )
        model_q_momentum_rate = horizontal_momentum_rate(
            truth_eta[frame],
            truth_xi[frame],
            aligned_q,
            length,
        )
        diagnostics["relative_momentum_rate_defect_abs"] = np.abs(
            model_q_momentum_rate - truth_momentum_rate
        ) / (np.abs(initial_momentum) * np.sqrt(depth) + 1e-30)
        frame_arrays[str(float(times[frame]))] = diagnostics
        metric_results: dict[str, Any] = {}
        for metric_name, values in diagnostics.items():
            values_valid = values[valid]
            is_nonnegative = metric_name in {
                "relative_q_l2_squared",
                "kinematic_rate_squared",
                "phase_sensitive_q_l2_squared",
                "relative_q_h1_squared",
                "mode_balanced_loss",
                "phase_sensitive_mode_loss",
                "dispersion_sensitive_mode_loss",
                "relative_momentum_defect_abs",
                "relative_momentum_rate_defect_abs",
                "tangent_loss",
                "phase_growth_loss",
                "differential_fraction",
                "kappa",
            }
            metric_results[metric_name] = {
                "vs_abs_final_displacement": safe_correlation(
                    values_valid,
                    np.abs(final_displacement[valid]),
                    log_x=is_nonnegative,
                ),
                "vs_final_translation_component": safe_correlation(
                    values_valid,
                    translation_component[valid],
                    log_x=is_nonnegative,
                ),
                "top_10pct_translation_tail_recall": top_fraction_recall(
                    np.abs(values_valid),
                    translation_tail[valid],
                    0.10,
                ),
            }
        signed_speed = diagnostics["global_speed_error"]
        metric_results["signed_global_speed_vs_signed_final_displacement"] = (
            safe_correlation(signed_speed[valid], final_displacement[valid])
        )
        frame_results[str(float(times[frame]))] = metric_results

    initial_key = str(float(times[int(np.argmin(np.abs(times))) ]))
    initial = frame_arrays[initial_key]
    initial_concentration = {
        metric: score_concentration(initial[metric][valid])
        for metric in (
            "relative_q_l2_squared",
            "kinematic_rate_squared",
            "phase_sensitive_q_l2_squared",
            "relative_q_h1_squared",
            "mode_balanced_loss",
            "phase_sensitive_mode_loss",
            "dispersion_sensitive_mode_loss",
            "tangent_loss",
            "phase_growth_loss",
        )
    }
    tail_indices = np.flatnonzero(translation_tail)
    tail_rows = [
        {
            "case_index": int(index),
            "case_id": int(case_ids[index]),
            "depth": float(depth[index]),
            "raw_error": float(raw_error[index]),
            "aligned_error": float(aligned_error[index]),
            "displacement": float(final_displacement[index]),
            "removed_fraction": float(removed_fraction[index]),
            "initial_tangent_loss": float(initial["tangent_loss"][index]),
            "initial_phase_growth_loss": float(
                initial["phase_growth_loss"][index]
            ),
            "initial_rigid_relative_speed": float(
                initial["rigid_relative_speed"][index]
            ),
            "initial_differential_fraction": float(
                initial["differential_fraction"][index]
            ),
        }
        for index in tail_indices
    ]
    return {
        "regime": regime,
        "n_valid": int(np.count_nonzero(valid)),
        "n_raw_above_025": int(np.count_nonzero(valid & (raw_error > 0.25))),
        "n_translation_tails": int(np.count_nonzero(translation_tail)),
        "initial_score_concentration": initial_concentration,
        "frame_diagnostics": frame_results,
        "translation_tail_cases": tail_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument(
        "--diagnostic-times",
        type=float,
        nargs="+",
        default=(0.0, 8.0, 20.0, 40.0),
    )
    args = parser.parse_args()

    results = [
        audit_regime(
            args.eval_dir / regime / f"{regime}_trajs.npz",
            regime,
            args.length,
            tuple(args.diagnostic_times),
        )
        for regime in REGIMES
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
