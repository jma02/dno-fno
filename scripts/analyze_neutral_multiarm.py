"""Analyze saved multi-arm neutral-oracle trajectories on CPU.

The input schema is produced by ``solver/evals/neutral_phase_oracle.py`` when
``--multi-arm-case-id`` is set.  This script performs no model evaluation and
does not import JAX.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.signal import find_peaks


DEFAULT_CHECKPOINT_TIMES = (20.0, 40.0, 80.0, 120.0, 160.0, 200.0)


def periodic_shift(
    values: np.ndarray,
    displacement: float,
    length: float,
) -> np.ndarray:
    """Return ``values(x - displacement)`` by Fourier interpolation."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        np.fft.fft(values) * np.exp(-1j * wave_numbers * displacement)
    ).real


def optimal_displacement(
    prediction: np.ndarray,
    truth: np.ndarray,
    length: float,
) -> float:
    """Find the continuous periodic displacement aligning prediction to truth."""
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        return float("nan")
    nx = truth.size
    dx = length / nx
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    prediction_hat = np.fft.fft(prediction)
    correlation = np.fft.ifft(prediction_hat * np.conj(np.fft.fft(truth))).real
    candidate_count = min(8, nx)
    peak_indices = np.argpartition(correlation, -candidate_count)[-candidate_count:]

    def objective(displacement: float) -> float:
        aligned = np.fft.ifft(
            prediction_hat * np.exp(1j * wave_numbers * displacement)
        ).real
        difference = aligned - truth
        return float(np.vdot(difference, difference).real)

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


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return relative Euclidean error with a finite zero-norm safeguard."""
    return float(np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + 1e-30))


def unwrap_displacements(displacements: np.ndarray, length: float) -> np.ndarray:
    """Unwrap each contiguous finite portion of a periodic displacement series."""
    result = np.asarray(displacements, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(result))
    if not finite_indices.size:
        return result
    split_points = np.flatnonzero(np.diff(finite_indices) > 1) + 1
    for segment in np.split(finite_indices, split_points):
        angles = result[segment] * (2.0 * np.pi / length)
        result[segment] = np.unwrap(angles) * (length / (2.0 * np.pi))
    return result


def _checkpoint_summary(
    times: np.ndarray,
    raw_error: np.ndarray,
    aligned_error: np.ndarray,
    displacement: np.ndarray,
    length: float,
    nx: int,
    checkpoint_times: tuple[float, ...],
) -> dict[str, object]:
    """Sample analyzed series at requested physical times."""
    tolerance = (
        0.5 * float(np.median(np.diff(times))) + 1e-9 if times.size > 1 else 1e-9
    )
    checkpoints: dict[str, object] = {}
    for requested_time in checkpoint_times:
        index = int(np.argmin(np.abs(times - requested_time)))
        if abs(float(times[index]) - requested_time) > tolerance:
            checkpoints[f"{requested_time:g}"] = None
            continue
        checkpoints[f"{requested_time:g}"] = {
            "frame": index,
            "time": float(times[index]),
            "raw_eta_error": float(raw_error[index]),
            "aligned_eta_error": float(aligned_error[index]),
            "displacement": float(displacement[index]),
            "displacement_grid_points": float(displacement[index] / (length / nx)),
        }
    return checkpoints


def _quadratic_peak_position(values: np.ndarray, index: int) -> float:
    """Refine a periodic grid maximum by a three-point quadratic fit."""
    nx = values.size
    left = float(values[(index - 1) % nx])
    center = float(values[index])
    right = float(values[(index + 1) % nx])
    denominator = left - 2.0 * center + right
    if denominator >= 0.0 or abs(denominator) < 1e-30:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float((index + np.clip(offset, -0.5, 0.5)) % nx)


def _detect_periodic_crests(
    values: np.ndarray,
    *,
    min_separation_grid: int,
    height_fraction: float,
    prominence_fraction: float,
) -> np.ndarray:
    """Detect significant positive crests without losing a boundary peak."""
    if not np.isfinite(values).all():
        return np.empty(0, dtype=np.int64)
    value_range = float(np.max(values) - np.min(values))
    if value_range <= 1e-30:
        return np.empty(0, dtype=np.int64)
    cut = int(np.argmin(values))
    shifted = np.roll(values, -cut)
    peaks, _ = find_peaks(
        shifted,
        height=float(np.min(values)) + height_fraction * value_range,
        prominence=prominence_fraction * value_range,
        distance=min_separation_grid,
    )
    return np.sort((peaks + cut) % values.size).astype(np.int64)


def _periodic_grid_delta(
    prediction_position: float,
    truth_position: float,
    nx: int,
) -> float:
    """Return the signed shortest prediction-minus-truth grid displacement."""
    return float((prediction_position - truth_position + 0.5 * nx) % nx - 0.5 * nx)


def terminal_crest_diagnostics(
    prediction: np.ndarray,
    truth: np.ndarray,
    length: float,
    *,
    min_separation_grid: int,
    max_offset_grid: float,
    height_fraction: float,
    prominence_fraction: float,
) -> dict[str, object]:
    """Match well-separated terminal crests and report local aligned errors."""
    truth_peaks = _detect_periodic_crests(
        truth,
        min_separation_grid=min_separation_grid,
        height_fraction=height_fraction,
        prominence_fraction=prominence_fraction,
    )
    prediction_peaks = _detect_periodic_crests(
        prediction,
        min_separation_grid=min_separation_grid,
        height_fraction=height_fraction,
        prominence_fraction=prominence_fraction,
    )
    counts = {
        "truth_peak_count": int(truth_peaks.size),
        "prediction_peak_count": int(prediction_peaks.size),
    }
    if truth_peaks.size < 2:
        return {
            "available": False,
            "reason": "fewer than two significant truth crests",
            **counts,
        }
    if prediction_peaks.size != truth_peaks.size:
        return {
            "available": False,
            "reason": "significant truth/prediction crest counts differ",
            **counts,
        }

    nx = truth.size
    truth_positions = np.asarray(
        [_quadratic_peak_position(truth, int(index)) for index in truth_peaks]
    )
    prediction_positions = np.asarray(
        [_quadratic_peak_position(prediction, int(index)) for index in prediction_peaks]
    )
    signed_cost = np.asarray(
        [
            [
                _periodic_grid_delta(prediction_position, truth_position, nx)
                for prediction_position in prediction_positions
            ]
            for truth_position in truth_positions
        ],
        dtype=np.float64,
    )
    truth_assignment, prediction_assignment = linear_sum_assignment(np.abs(signed_cost))
    matched_offsets = signed_cost[truth_assignment, prediction_assignment]
    if np.any(np.abs(matched_offsets) > max_offset_grid):
        return {
            "available": False,
            "reason": "at least one crest match exceeds max_offset_grid",
            "max_offset_grid": max_offset_grid,
            **counts,
        }

    pairwise_truth_separations = [
        abs(_periodic_grid_delta(truth_positions[first], truth_positions[second], nx))
        for first in range(truth_positions.size)
        for second in range(first + 1, truth_positions.size)
    ]
    minimum_truth_separation = float(min(pairwise_truth_separations))
    if minimum_truth_separation < min_separation_grid:
        return {
            "available": False,
            "reason": "refined truth crests are not sufficiently separated",
            "minimum_truth_separation_grid": minimum_truth_separation,
            **counts,
        }

    grid = np.arange(nx, dtype=np.float64)
    local_radius = 0.45 * minimum_truth_separation
    dx = length / nx
    crest_results: list[dict[str, float | int]] = []
    for truth_index, prediction_index in zip(
        truth_assignment, prediction_assignment, strict=True
    ):
        offset_grid = float(signed_cost[truth_index, prediction_index])
        aligned_prediction = periodic_shift(
            prediction,
            -offset_grid * dx,
            length,
        )
        local_distance = np.abs(
            (grid - truth_positions[truth_index] + 0.5 * nx) % nx - 0.5 * nx
        )
        local_mask = local_distance <= local_radius
        crest_results.append(
            {
                "truth_crest_index": int(truth_index),
                "truth_position_grid": float(truth_positions[truth_index]),
                "prediction_position_grid": float(
                    prediction_positions[prediction_index]
                ),
                "offset_grid_points": offset_grid,
                "offset": offset_grid * dx,
                "truth_peak_amplitude": float(truth[truth_peaks[truth_index]]),
                "prediction_peak_amplitude": float(
                    prediction[prediction_peaks[prediction_index]]
                ),
                "peak_amplitude_ratio": float(
                    prediction[prediction_peaks[prediction_index]]
                    / (truth[truth_peaks[truth_index]] + 1e-30)
                ),
                "local_aligned_eta_error": relative_l2(
                    aligned_prediction[local_mask], truth[local_mask]
                ),
            }
        )
    crest_results.sort(key=lambda item: item["truth_position_grid"])
    return {
        "available": True,
        **counts,
        "minimum_truth_separation_grid": minimum_truth_separation,
        "local_window_radius_grid": local_radius,
        "crests": crest_results,
    }


def analyze_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    length: float,
    checkpoint_times: tuple[float, ...],
    *,
    crest_min_separation_grid: int,
    crest_max_offset_grid: float,
    crest_height_fraction: float,
    crest_prominence_fraction: float,
) -> dict[str, object]:
    """Compute complete framewise and terminal diagnostics for one arm."""
    n_times, nx = truth.shape
    raw_error = np.full(n_times, np.nan, dtype=np.float64)
    aligned_error = np.full(n_times, np.nan, dtype=np.float64)
    displacement = np.full(n_times, np.nan, dtype=np.float64)
    for frame in range(n_times):
        raw_error[frame] = relative_l2(prediction[frame], truth[frame])
        displacement[frame] = optimal_displacement(
            prediction[frame], truth[frame], length
        )
        if np.isfinite(displacement[frame]):
            aligned = periodic_shift(prediction[frame], -displacement[frame], length)
            aligned_error[frame] = relative_l2(aligned, truth[frame])
    displacement = unwrap_displacements(displacement, length)
    terminal_index = n_times - 1
    return {
        "terminal": {
            "time": float(times[terminal_index]),
            "raw_eta_error": float(raw_error[terminal_index]),
            "aligned_eta_error": float(aligned_error[terminal_index]),
            "displacement": float(displacement[terminal_index]),
            "displacement_grid_points": float(
                displacement[terminal_index] / (length / nx)
            ),
            "squared_eta_error_removed_by_alignment": float(
                1.0
                - aligned_error[terminal_index] ** 2
                / (raw_error[terminal_index] ** 2 + 1e-30)
            ),
        },
        "checkpoints": _checkpoint_summary(
            times,
            raw_error,
            aligned_error,
            displacement,
            length,
            nx,
            checkpoint_times,
        ),
        "series": {
            "times": times,
            "raw_eta_error": raw_error,
            "aligned_eta_error": aligned_error,
            "displacement": displacement,
            "displacement_grid_points": displacement / (length / nx),
        },
        "terminal_crest_diagnostics": terminal_crest_diagnostics(
            prediction[terminal_index],
            truth[terminal_index],
            length,
            min_separation_grid=crest_min_separation_grid,
            max_offset_grid=crest_max_offset_grid,
            height_fraction=crest_height_fraction,
            prominence_fraction=crest_prominence_fraction,
        ),
    }


def _decode_arm_names(values: np.ndarray) -> tuple[str, ...]:
    """Decode NumPy unicode or byte-string arm names."""
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def analyze_archive(
    archive_path: Path,
    length: float,
    checkpoint_times: tuple[float, ...],
    *,
    crest_min_separation_grid: int,
    crest_max_offset_grid: float,
    crest_height_fraction: float,
    crest_prominence_fraction: float,
) -> dict[str, object]:
    """Load and analyze one multi-arm trajectory archive."""
    with np.load(archive_path) as archive:
        required = {
            "times",
            "depth",
            "case_id",
            "arm_names",
            "truth_eta",
            "baseline_eta",
            "arm_eta",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{archive_path} is missing fields: {sorted(missing)}")
        times = np.asarray(archive["times"], dtype=np.float64)
        truth_eta = np.asarray(archive["truth_eta"], dtype=np.float64)
        baseline_eta = np.asarray(archive["baseline_eta"], dtype=np.float64)
        arm_eta = np.asarray(archive["arm_eta"], dtype=np.float64)
        arm_names = _decode_arm_names(archive["arm_names"])
        depth = float(np.asarray(archive["depth"]).reshape(-1)[0])
        case_id = int(np.asarray(archive["case_id"]).reshape(-1)[0])

    if times.ndim != 1 or truth_eta.ndim != 2:
        raise ValueError("times must be 1D and truth_eta must have shape (T, N)")
    if baseline_eta.shape != truth_eta.shape:
        raise ValueError("baseline_eta and truth_eta shapes differ")
    if arm_eta.shape != (times.size, len(arm_names), truth_eta.shape[-1]):
        raise ValueError(
            f"arm_eta must have shape (T, len(arm_names), N), got {arm_eta.shape}"
        )
    if truth_eta.shape[0] != times.size:
        raise ValueError("truth_eta time dimension does not match times")
    if times.size < 1 or not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be nonempty and strictly increasing")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("arm_names must be unique")

    analysis_kwargs: dict[str, Any] = {
        "crest_min_separation_grid": crest_min_separation_grid,
        "crest_max_offset_grid": crest_max_offset_grid,
        "crest_height_fraction": crest_height_fraction,
        "crest_prominence_fraction": crest_prominence_fraction,
    }
    arms = {
        arm_name: analyze_prediction(
            arm_eta[:, arm_index],
            truth_eta,
            times,
            length,
            checkpoint_times,
            **analysis_kwargs,
        )
        for arm_index, arm_name in enumerate(arm_names)
    }
    return {
        "archive": str(archive_path.resolve()),
        "case_id": case_id,
        "depth": depth,
        "length": length,
        "nx": int(truth_eta.shape[-1]),
        "n_frames": int(times.size),
        "arm_names": list(arm_names),
        "baseline": analyze_prediction(
            baseline_eta,
            truth_eta,
            times,
            length,
            checkpoint_times,
            **analysis_kwargs,
        ),
        "arms": arms,
    }


def json_safe(value: object) -> object:
    """Convert NumPy containers and nonfinite scalars to strict JSON values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument(
        "--checkpoint-times",
        type=float,
        nargs="+",
        default=DEFAULT_CHECKPOINT_TIMES,
    )
    parser.add_argument("--crest-min-separation-grid", type=int, default=16)
    parser.add_argument("--crest-max-offset-grid", type=float, default=64.0)
    parser.add_argument("--crest-height-fraction", type=float, default=0.15)
    parser.add_argument("--crest-prominence-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length <= 0.0:
        raise ValueError("--length must be positive")
    if args.crest_min_separation_grid < 1:
        raise ValueError("--crest-min-separation-grid must be positive")
    if args.crest_max_offset_grid <= 0.0:
        raise ValueError("--crest-max-offset-grid must be positive")
    checkpoint_times = tuple(float(value) for value in args.checkpoint_times)
    analyses = [
        analyze_archive(
            path.resolve(),
            args.length,
            checkpoint_times,
            crest_min_separation_grid=args.crest_min_separation_grid,
            crest_max_offset_grid=args.crest_max_offset_grid,
            crest_height_fraction=args.crest_height_fraction,
            crest_prominence_fraction=args.crest_prominence_fraction,
        )
        for path in args.input
    ]
    payload = {
        "checkpoint_times": list(checkpoint_times),
        "analyses": analyses,
    }
    serialized = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
