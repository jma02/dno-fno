"""Audit eta-growth guard activity observable in archived rollout frames.

The rollout applies the guard after every internal integrator substep, whereas
trajectory archives contain only the coarser requested output times.  This
script reconstructs the exact detector ratio and sigmoid weight on those saved
states.  It can prove that a trajectory was excluded by the fixed initial-state
selector, but it cannot rule out a trigger between two saved frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

WEIGHT_THRESHOLDS = (1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Evaluation directories, regime directories, or trajectory NPZ files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fft-frame-chunk",
        type=int,
        default=16,
        help="Number of saved frames transformed at once (default: 16).",
    )
    return parser.parse_args()


def discover_archives(inputs: Sequence[Path]) -> list[Path]:
    """Return unique archived trajectories in deterministic regime order."""
    candidates: list[Path] = []
    for path in inputs:
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(path.rglob("*_trajs.npz"))
        else:
            raise FileNotFoundError(path)
    archives = sorted(set(candidate.resolve() for candidate in candidates))
    if not archives:
        raise ValueError("no *_trajs.npz archives found")
    return archives


def require_summary(archive: Path) -> tuple[Path, dict[str, Any]]:
    regime = archive.name.removesuffix("_trajs.npz")
    summary_path = archive.with_name(f"{regime}_summary.json")
    with summary_path.open(encoding="utf-8") as source:
        summary: dict[str, Any] = json.load(source)
    if summary.get("regime") != regime:
        raise ValueError(
            f"summary regime mismatch for {archive}: {summary.get('regime')!r}"
        )
    return summary_path, summary


def band_amplitude(
    eta: NDArray[np.floating[Any]],
    *,
    length: float,
    k_lo: float,
    k_hi: float,
    frame_chunk: int,
) -> FloatArray:
    """Reproduce ``_eta_band_amp`` on each archived eta state."""
    n_frames, n_cases, nx = eta.shape
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    band = (np.abs(wave_numbers) >= k_lo) & (np.abs(wave_numbers) < k_hi)
    if not np.any(band):
        raise ValueError(f"empty detector band [{k_lo}, {k_hi}) for nx={nx}")

    amplitude = np.empty((n_frames, n_cases), dtype=np.float64)
    for start in range(0, n_frames, frame_chunk):
        stop = min(start + frame_chunk, n_frames)
        spectrum = np.fft.fft(eta[start:stop].astype(np.float64), axis=-1)
        band_mean = np.mean(
            np.where(band[None, None, :], np.abs(spectrum) ** 2, 0.0),
            axis=-1,
        )
        amplitude[start:stop] = np.sqrt(band_mean) / float(nx)
    return amplitude


def sigmoid_from_log_ratio(ratio: FloatArray, sharpness: float) -> FloatArray:
    """Evaluate sigmoid(sharpness * log(ratio)) without overflow."""
    logits = sharpness * np.log(ratio + 1e-300)
    return 1.0 / (1.0 + np.exp(np.clip(-logits, -700.0, 700.0)))


def optional_maximum(values: FloatArray, mask: BoolArray) -> float | None:
    selected = values[mask]
    return float(np.max(selected)) if selected.size else None


def audit_archive(archive: Path, frame_chunk: int) -> dict[str, Any]:
    summary_path, summary = require_summary(archive)
    required = (
        "length",
        "nx",
        "eta_growth_guard_enabled",
        "eta_growth_guard_k_lo",
        "eta_growth_guard_k_hi",
        "eta_growth_guard_abs_floor",
        "eta_growth_guard_growth_factor",
        "eta_growth_guard_sharpness",
        "eta_growth_guard_soliton_only",
        "eta_growth_guard_negative_energy_threshold",
        "eta_growth_guard_houli_a",
        "eta_growth_guard_houli_m",
        "eta_growth_guard_k_eff",
        "eta_growth_guard_kh_eff",
        "eta_growth_guard_filter_xi",
        "substeps",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"{summary_path} lacks required fields: {missing}")

    with np.load(archive, allow_pickle=False) as data:
        eta = np.asarray(data["pred_eta"])
        times = np.asarray(data["times"], dtype=np.float64)
        case_ids = np.asarray(data["case_ids"], dtype=np.int64)
        depths = np.asarray(data["depths"], dtype=np.float64)
    if eta.ndim != 3:
        raise ValueError(f"{archive}: expected pred_eta rank 3, got {eta.shape}")
    n_frames, n_cases, nx = eta.shape
    expected_shapes = {
        "times": (n_frames,),
        "case_ids": (n_cases,),
        "depths": (n_cases,),
    }
    actual_shapes = {
        "times": times.shape,
        "case_ids": case_ids.shape,
        "depths": depths.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"{archive}: inconsistent shapes {actual_shapes}")
    if int(summary["nx"]) != nx:
        raise ValueError(f"{archive}: summary nx={summary['nx']} but archive nx={nx}")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError(f"{archive}: saved times must be strictly increasing")

    k_lo = float(summary["eta_growth_guard_k_lo"])
    k_hi = float(summary["eta_growth_guard_k_hi"])
    amplitude = band_amplitude(
        eta,
        length=float(summary["length"]),
        k_lo=k_lo,
        k_hi=k_hi,
        frame_chunk=frame_chunk,
    )
    trigger = np.maximum(
        float(summary["eta_growth_guard_abs_floor"]),
        float(summary["eta_growth_guard_growth_factor"]) * amplitude[0],
    )
    ratio = amplitude / trigger[None, :]
    raw_weight = sigmoid_from_log_ratio(
        ratio, float(summary["eta_growth_guard_sharpness"])
    )

    eta0 = eta[0].astype(np.float64)
    negative_fraction = np.sum(np.minimum(eta0, 0.0) ** 2, axis=-1) / (
        np.sum(eta0**2, axis=-1) + 1e-30
    )
    if bool(summary["eta_growth_guard_soliton_only"]):
        eligible = (negative_fraction < float(
            summary["eta_growth_guard_negative_energy_threshold"]
        )) & (np.mean(eta0, axis=-1) > 0.0)
    else:
        eligible = np.ones(n_cases, dtype=np.bool_)
    guard_applicable = eligible & bool(summary["eta_growth_guard_enabled"])

    effective_weight = raw_weight * guard_applicable[None, :]
    applicable_frames = np.broadcast_to(guard_applicable[None, :], ratio.shape)
    crossing = (ratio >= 1.0) & applicable_frames
    active_case = np.any(crossing, axis=0)

    case_records: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(case_ids):
        maximum_frame = int(np.argmax(ratio[:, case_index]))
        case_records.append(
            {
                "case_index": case_index,
                "case_id": int(case_id),
                "depth": float(depths[case_index]),
                "selector_eligible": bool(eligible[case_index]),
                "initial_negative_eta_energy_fraction": float(
                    negative_fraction[case_index]
                ),
                "initial_band_amplitude": float(amplitude[0, case_index]),
                "trigger_amplitude": float(trigger[case_index]),
                "maximum_saved_amplitude_to_trigger_ratio": float(
                    np.max(ratio[:, case_index])
                ),
                "maximum_saved_effective_weight": float(
                    np.max(effective_weight[:, case_index])
                ),
                "maximum_ratio_saved_frame": maximum_frame,
                "maximum_ratio_saved_time": float(times[maximum_frame]),
                "nominal_trigger_crossed_at_saved_frame": bool(
                    active_case[case_index]
                ),
            }
        )

    if np.any(applicable_frames):
        eligible_weight = np.where(applicable_frames, effective_weight, -1.0)
        max_flat = int(np.argmax(eligible_weight))
        max_frame, max_case = np.unravel_index(max_flat, eligible_weight.shape)
        maximum_location: dict[str, Any] | None = {
            "case_index": int(max_case),
            "case_id": int(case_ids[max_case]),
            "saved_frame": int(max_frame),
            "saved_time": float(times[max_frame]),
        }
    else:
        maximum_location = None

    threshold_counts = {}
    for threshold in WEIGHT_THRESHOLDS:
        above = (effective_weight >= threshold) & applicable_frames
        threshold_counts[f"{threshold:.0e}"] = {
            "saved_case_frames": int(np.count_nonzero(above)),
            "cases": int(np.count_nonzero(np.any(above, axis=0))),
        }

    saved_dt = np.diff(times)
    return {
        "regime": str(summary["regime"]),
        "archive": str(archive),
        "summary": str(summary_path),
        "guard_enabled": bool(summary["eta_growth_guard_enabled"]),
        "selector_soliton_only": bool(summary["eta_growth_guard_soliton_only"]),
        "n_cases": n_cases,
        "n_saved_frames": n_frames,
        "saved_time_start": float(times[0]),
        "saved_time_end": float(times[-1]),
        "saved_dt_min": float(np.min(saved_dt)) if saved_dt.size else None,
        "saved_dt_max": float(np.max(saved_dt)) if saved_dt.size else None,
        "integrator_substeps_per_saved_interval": int(summary["substeps"]),
        "selector_eligible_cases": int(np.count_nonzero(eligible)),
        "selector_eligible_case_ids": [int(value) for value in case_ids[eligible]],
        "selector_ineligible_cases": int(np.count_nonzero(~eligible)),
        "guard_applicable_cases": int(np.count_nonzero(guard_applicable)),
        "nominal_trigger_saved_crossing_count": int(np.count_nonzero(crossing)),
        "nominal_trigger_saved_crossing_case_ids": [
            int(value) for value in case_ids[active_case]
        ],
        "maximum_saved_amplitude_to_trigger_ratio_eligible": optional_maximum(
            ratio, applicable_frames
        ),
        "maximum_saved_effective_weight": optional_maximum(
            effective_weight, applicable_frames
        ),
        "maximum_saved_effective_weight_location": maximum_location,
        "effective_weight_threshold_counts": threshold_counts,
        "guard_parameters": {
            "k_lo": k_lo,
            "k_hi": k_hi,
            "absolute_floor": float(summary["eta_growth_guard_abs_floor"]),
            "growth_factor": float(summary["eta_growth_guard_growth_factor"]),
            "sharpness": float(summary["eta_growth_guard_sharpness"]),
            "negative_energy_threshold": float(
                summary["eta_growth_guard_negative_energy_threshold"]
            ),
            "houli_a": float(summary["eta_growth_guard_houli_a"]),
            "houli_m": float(summary["eta_growth_guard_houli_m"]),
            "k_eff": float(summary["eta_growth_guard_k_eff"]),
            "kh_eff": float(summary["eta_growth_guard_kh_eff"]),
            "filter_xi": bool(summary["eta_growth_guard_filter_xi"]),
        },
        "cases": case_records,
    }


def main() -> None:
    args = parse_args()
    if args.fft_frame_chunk <= 0:
        raise ValueError("--fft-frame-chunk must be positive")
    archives = discover_archives(args.inputs)
    regimes = [audit_archive(path, args.fft_frame_chunk) for path in archives]
    names = [str(item["regime"]) for item in regimes]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate regimes are ambiguous: {names}")

    eligible_results = [
        item
        for item in regimes
        if int(item["selector_eligible_cases"]) > 0
    ]
    maxima = [
        float(item["maximum_saved_amplitude_to_trigger_ratio_eligible"])
        for item in eligible_results
    ]
    result = {
        "schema_version": 1,
        "definition": (
            "For each archived prediction eta(t_j), recompute the guard detector "
            "A_j, trigger=max(abs_floor, growth_factor*A_0), ratio=A_j/trigger, "
            "and effective sigmoid weight selector*sigmoid(sharpness*log(ratio))."
        ),
        "scope": "saved output frames only",
        "limitation": (
            "The guard is applied after every internal integrator substep. Archives "
            "contain only post-guard output frames, so absence of a saved-frame "
            "crossing cannot exclude a crossing or damping at hidden substeps."
        ),
        "aggregate": {
            "n_regimes": len(regimes),
            "n_cases": sum(int(item["n_cases"]) for item in regimes),
            "selector_eligible_cases": sum(
                int(item["selector_eligible_cases"]) for item in regimes
            ),
            "selector_ineligible_cases": sum(
                int(item["selector_ineligible_cases"]) for item in regimes
            ),
            "guard_applicable_cases": sum(
                int(item["guard_applicable_cases"]) for item in regimes
            ),
            "nominal_trigger_saved_crossing_count": sum(
                int(item["nominal_trigger_saved_crossing_count"])
                for item in regimes
            ),
            "nominal_trigger_saved_crossing_regimes": [
                str(item["regime"])
                for item in regimes
                if int(item["nominal_trigger_saved_crossing_count"]) > 0
            ],
            "selector_exactly_inactive_regimes": [
                str(item["regime"])
                for item in regimes
                if int(item["selector_eligible_cases"]) == 0
            ],
            "maximum_saved_amplitude_to_trigger_ratio_eligible": (
                max(maxima) if maxima else None
            ),
        },
        "regimes": regimes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")

    aggregate = result["aggregate"]
    print(
        f"audited {aggregate['n_cases']} cases in {aggregate['n_regimes']} regimes; "
        f"eligible={aggregate['selector_eligible_cases']}, "
        f"saved crossings={aggregate['nominal_trigger_saved_crossing_count']}, "
        f"max ratio={aggregate['maximum_saved_amplitude_to_trigger_ratio_eligible']}"
    )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
