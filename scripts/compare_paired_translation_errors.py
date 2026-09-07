"""Compare final translation and shape errors on paired rollout archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from analyze_rollout_translation_decomposition import (
    optimal_displacement,
    periodic_shift,
)


def archive_metrics(path: Path, length: float) -> dict[str, np.ndarray]:
    """Return final raw, aligned, and translation-component errors."""
    with np.load(path) as archive:
        simulation_ids = np.asarray(archive["simulation_ids"], dtype=np.int64)
        depths = np.asarray(archive["depths"], dtype=np.float64)
        truth = np.asarray(archive["truth_eta"][-1], dtype=np.float64)
        prediction = np.asarray(archive["pred_eta"][-1], dtype=np.float64)
        valid = np.all(np.isfinite(truth) & np.isfinite(prediction), axis=-1)
        valid &= np.asarray(archive["truth_valid"], dtype=bool)
        valid &= ~np.asarray(archive["model_nonfinite_any"], dtype=bool)

    displacement = np.asarray(
        [
            optimal_displacement(prediction_i, truth_i, length)
            for prediction_i, truth_i in zip(prediction, truth)
        ],
        dtype=np.float64,
    )
    aligned_prediction = np.asarray(
        [
            periodic_shift(prediction_i, -shift_i, length)
            for prediction_i, shift_i in zip(prediction, displacement)
        ]
    )
    truth_norm = np.linalg.norm(truth, axis=-1) + 1e-30
    raw = np.linalg.norm(prediction - truth, axis=-1) / truth_norm
    aligned = np.linalg.norm(aligned_prediction - truth, axis=-1) / truth_norm
    translation = np.sqrt(np.maximum(raw**2 - aligned**2, 0.0))
    return {
        "simulation_ids": simulation_ids,
        "depths": depths,
        "valid": valid,
        "truth": truth,
        "raw": raw,
        "aligned": aligned,
        "translation": translation,
        "displacement": displacement,
    }


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize a finite one-dimensional metric."""
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def paired_summary(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize paired changes on a selected subset."""
    result: dict[str, Any] = {"n": int(np.count_nonzero(mask))}
    for metric in ("raw", "aligned", "translation"):
        before = baseline[metric][mask]
        after = candidate[metric][mask]
        differences = after - before
        nonzero = differences != 0.0
        test: Any = wilcoxon(differences[nonzero]) if np.any(nonzero) else None
        result[metric] = {
            "baseline": scalar_summary(before),
            "candidate": scalar_summary(after),
            "median_paired_change": float(np.median(differences)),
            "fraction_improved": float(np.mean(after < before)),
            "wilcoxon_two_sided_p": (None if test is None else float(test.pvalue)),
        }
    result["raw_threshold_counts"] = {
        str(threshold): {
            "baseline": int(np.count_nonzero(baseline["raw"][mask] > threshold)),
            "candidate": int(np.count_nonzero(candidate["raw"][mask] > threshold)),
        }
        for threshold in (0.25, 0.5, 0.75, 1.0)
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--shallow-depth", type=float, default=0.03)
    args = parser.parse_args()

    baseline = archive_metrics(args.baseline, args.length)
    candidate = archive_metrics(args.candidate, args.length)
    if not np.array_equal(baseline["simulation_ids"], candidate["simulation_ids"]):
        raise ValueError("simulation IDs do not match")
    if not np.array_equal(baseline["depths"], candidate["depths"]):
        raise ValueError("depths do not match")
    truth_max_abs_difference = float(
        np.max(np.abs(baseline["truth"] - candidate["truth"]))
    )
    if truth_max_abs_difference > 1e-8:
        raise ValueError(f"final truth states differ by {truth_max_abs_difference:.3e}")
    valid = baseline["valid"] & candidate["valid"]
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "truth_max_abs_difference": truth_max_abs_difference,
        "all_valid": paired_summary(baseline, candidate, valid),
    }
    shallow = valid & (baseline["depths"] < args.shallow_depth)
    if np.any(shallow):
        result["shallow"] = {
            "depth_upper_bound": args.shallow_depth,
            **paired_summary(baseline, candidate, shallow),
        }
    translation_change = candidate["translation"] - baseline["translation"]
    ordered = np.flatnonzero(valid)[np.argsort(translation_change[valid])]
    result["largest_translation_improvements"] = [
        {
            "simulation_index": int(index),
            "simulation_id": int(baseline["simulation_ids"][index]),
            "depth": float(baseline["depths"][index]),
            "baseline": float(baseline["translation"][index]),
            "candidate": float(candidate["translation"][index]),
        }
        for index in ordered[:5]
    ]
    result["largest_translation_regressions"] = [
        {
            "simulation_index": int(index),
            "simulation_id": int(baseline["simulation_ids"][index]),
            "depth": float(baseline["depths"][index]),
            "baseline": float(baseline["translation"][index]),
            "candidate": float(candidate["translation"][index]),
        }
        for index in ordered[-5:][::-1]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
