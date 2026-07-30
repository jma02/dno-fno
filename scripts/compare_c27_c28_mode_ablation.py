"""Validate and compare the fixed C27/C28 epoch-40 rollout panels."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from analyze_c21_worst_gxi_phase import optimal_displacement, periodic_shift


REGIMES = (
    "tanaka_g0",
    "tanaka_g1",
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "random_sea_deep",
    "random_sea_finite",
    "linear",
    "stokes_deep",
    "stokes_finite",
)
GROUPS = {
    "tanaka": REGIMES[:2],
    "benjamin_feir": REGIMES[2:5],
    "non_tanaka": REGIMES[2:],
    "all": REGIMES,
}
IDENTITY_FIELDS = (
    "times",
    "depths",
    "case_ids",
    "truth_eta",
    "truth_xi",
    "truth_gxi",
    "truth_valid",
    "ic_panel_sha256",
    "truth_protocol_json",
    "truth_protocol_sha256",
)
PREDICTION_FIELDS = ("pred_eta", "pred_xi", "pred_gxi")
METRICS = ("raw", "aligned", "translation")
THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
NONINFERIORITY_MARGIN = 0.05

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RegimeArrays:
    """Small per-case arrays retained after validating a rollout archive."""

    truth_valid: BoolArray
    baseline_finite: BoolArray
    candidate_finite: BoolArray
    baseline: dict[str, FloatArray]
    candidate: dict[str, FloatArray]


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _regime_root(
    regime: str,
    tanaka_root: Path,
    non_tanaka_root: Path,
) -> Path:
    """Select the evaluation root containing a regime."""
    root = tanaka_root if regime.startswith("tanaka_") else non_tanaka_root
    return root / regime


def _validate_candidate_summary(
    regime_root: Path,
    regime: str,
    candidate_run: Path,
) -> dict[str, Any]:
    """Require cached truth and the intended final epoch-40 checkpoint."""
    log_text = (regime_root / "eval.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "truth cache HIT:" not in log_text:
        raise ValueError(f"{regime}: evaluation did not report a truth cache HIT")

    summary = _load_json(regime_root / f"{regime}_summary.json")
    if float(summary.get("truth_wall_s", -1.0)) != 0.0:
        raise ValueError(f"{regime}: truth was recomputed")
    if int(summary.get("n_ics_attempted", -1)) != 32:
        raise ValueError(f"{regime}: expected 32 attempted ICs")

    source = summary.get("checkpoint_source")
    if not isinstance(source, dict):
        raise ValueError(f"{regime}: missing checkpoint source")
    expected_path = (candidate_run / "final_ckpt").resolve()
    if (
        source.get("selection") != "final"
        or int(source.get("epoch", -1)) != 40
        or Path(str(source.get("path", ""))).resolve() != expected_path
        or not source.get("sha256")
    ):
        raise ValueError(f"{regime}: unexpected checkpoint source {source}")
    return summary


def _scan_predictions(
    archive: np.lib.npyio.NpzFile,
    n_cases: int,
) -> tuple[BoolArray, FloatArray, dict[str, int]]:
    """Scan every saved prediction and retain final eta."""
    finite = np.ones(n_cases, dtype=bool)
    final_eta: FloatArray | None = None
    nonfinite_case_counts: dict[str, int] = {}
    for field in PREDICTION_FIELDS:
        values = np.asarray(archive[field])
        if values.ndim != 3 or values.shape[1] != n_cases:
            raise ValueError(f"unexpected {field} shape: {values.shape}")
        field_finite = np.all(np.isfinite(values), axis=(0, 2))
        finite &= field_finite
        nonfinite_case_counts[field] = int(np.count_nonzero(~field_finite))
        if field == "pred_eta":
            final_eta = np.asarray(values[-1], dtype=np.float64).copy()
        del values
    if final_eta is None:
        raise AssertionError("pred_eta was not scanned")
    return finite, final_eta, nonfinite_case_counts


def _bitwise_equal(before: NDArray[Any], after: NDArray[Any]) -> bool:
    """Compare dtype, shape, and bytes, including identical NaN payloads."""
    if before.dtype != after.dtype or before.shape != after.shape:
        return False
    before_bytes = np.ascontiguousarray(before).view(np.uint8)
    after_bytes = np.ascontiguousarray(after).view(np.uint8)
    return bool(np.array_equal(before_bytes, after_bytes))


def _final_errors(
    truth: FloatArray,
    prediction: FloatArray,
    valid: BoolArray,
    length: float,
) -> dict[str, FloatArray]:
    """Compute raw, optimally aligned, and translation errors per case."""
    raw = np.full(valid.shape, np.nan, dtype=np.float64)
    aligned = np.full(valid.shape, np.nan, dtype=np.float64)
    translation = np.full(valid.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if not indices.size:
        return {"raw": raw, "aligned": aligned, "translation": translation}

    truth_valid = truth[indices]
    prediction_valid = prediction[indices]
    displacements = np.asarray(
        [
            optimal_displacement(prediction_i, truth_i, length)
            for prediction_i, truth_i in zip(prediction_valid, truth_valid)
        ],
        dtype=np.float64,
    )
    aligned_prediction = np.asarray(
        [
            periodic_shift(prediction_i, -shift_i, length)
            for prediction_i, shift_i in zip(prediction_valid, displacements)
        ],
        dtype=np.float64,
    )
    truth_norm = np.linalg.norm(truth_valid, axis=-1) + 1e-30
    raw_valid = (
        np.linalg.norm(prediction_valid - truth_valid, axis=-1) / truth_norm
    )
    aligned_valid = (
        np.linalg.norm(aligned_prediction - truth_valid, axis=-1) / truth_norm
    )
    raw[indices] = raw_valid
    aligned[indices] = aligned_valid
    translation[indices] = np.sqrt(
        np.maximum(raw_valid**2 - aligned_valid**2, 0.0)
    )
    return {"raw": raw, "aligned": aligned, "translation": translation}


def _validate_regime_archives(
    baseline_path: Path,
    candidate_path: Path,
    length: float,
) -> tuple[RegimeArrays, dict[str, Any]]:
    """Require identical references and scan all baseline/candidate predictions."""
    with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
        truth_valid = np.asarray(candidate["truth_valid"], dtype=bool).copy()
        n_cases = int(truth_valid.size)
        truth_final: FloatArray | None = None
        for field in IDENTITY_FIELDS:
            if field not in baseline.files or field not in candidate.files:
                raise ValueError(f"missing paired identity field: {field}")
            before = np.asarray(baseline[field])
            after = np.asarray(candidate[field])
            if not _bitwise_equal(before, after):
                raise ValueError(f"paired identity field differs: {field}")
            if field == "truth_eta":
                truth_final = np.asarray(after[-1], dtype=np.float64).copy()
            del before, after
        if truth_final is None:
            raise AssertionError("truth_eta was not validated")

        baseline_finite, baseline_eta, baseline_nonfinite = _scan_predictions(
            baseline, n_cases
        )
        candidate_finite, candidate_eta, candidate_nonfinite = _scan_predictions(
            candidate, n_cases
        )

    if not np.all(baseline_finite):
        raise ValueError(f"C27 baseline predictions are nonfinite: {baseline_path}")
    baseline_metrics = _final_errors(
        truth_final,
        baseline_eta,
        truth_valid & baseline_finite,
        length,
    )
    candidate_metrics = _final_errors(
        truth_final,
        candidate_eta,
        truth_valid & candidate_finite,
        length,
    )
    arrays = RegimeArrays(
        truth_valid=truth_valid,
        baseline_finite=baseline_finite,
        candidate_finite=candidate_finite,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
    )
    scan = {
        "n_attempted": n_cases,
        "n_truth_valid": int(np.count_nonzero(truth_valid)),
        "baseline_nonfinite_case_counts": baseline_nonfinite,
        "candidate_nonfinite_case_counts": candidate_nonfinite,
        "candidate_all_predictions_finite": bool(np.all(candidate_finite)),
    }
    return arrays, scan


def _scalar_summary(values: FloatArray) -> dict[str, float | int | None]:
    """Return finite distribution statistics."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "n": 0,
            "median": None,
            "p95": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "mean": float(np.mean(finite)),
        "maximum": float(np.max(finite)),
    }


def _ratio(candidate: float | int | None, baseline: float | int | None) -> float | None:
    """Return a finite candidate/baseline ratio."""
    if candidate is None or baseline is None or float(baseline) == 0.0:
        return None
    return float(candidate) / float(baseline)


def _threshold_counts(
    values: FloatArray,
    truth_valid: BoolArray,
    model_finite: BoolArray,
) -> dict[str, int]:
    """Count nonfinite or over-threshold truth-valid trajectories."""
    return {
        str(threshold): int(
            np.count_nonzero(
                truth_valid & (~model_finite | (values > threshold))
            )
        )
        for threshold in THRESHOLDS
    }


def _group_summary(
    regime_arrays: dict[str, RegimeArrays],
    regimes: tuple[str, ...],
) -> dict[str, Any]:
    """Pool selected regimes while retaining paired truth-valid semantics."""
    truth_valid = np.concatenate(
        [regime_arrays[regime].truth_valid for regime in regimes]
    )
    baseline_finite = np.concatenate(
        [regime_arrays[regime].baseline_finite for regime in regimes]
    )
    candidate_finite = np.concatenate(
        [regime_arrays[regime].candidate_finite for regime in regimes]
    )
    result: dict[str, Any] = {
        "n_attempted": int(truth_valid.size),
        "n_truth_valid": int(np.count_nonzero(truth_valid)),
        "n_candidate_finite_truth_valid": int(
            np.count_nonzero(truth_valid & candidate_finite)
        ),
    }
    pooled: dict[str, tuple[FloatArray, FloatArray]] = {}
    for metric in METRICS:
        before = np.concatenate(
            [regime_arrays[regime].baseline[metric] for regime in regimes]
        )
        after = np.concatenate(
            [regime_arrays[regime].candidate[metric] for regime in regimes]
        )
        pooled[metric] = before, after
        before_summary = _scalar_summary(before[truth_valid & baseline_finite])
        after_summary = _scalar_summary(after[truth_valid & candidate_finite])
        result[metric] = {
            "baseline": before_summary,
            "candidate": after_summary,
            "candidate_over_baseline": {
                statistic: _ratio(
                    after_summary[statistic], before_summary[statistic]
                )
                for statistic in ("median", "p95", "mean", "maximum")
            },
        }
    before_raw, after_raw = pooled["raw"]
    result["raw_threshold_counts"] = {
        "baseline": _threshold_counts(
            before_raw, truth_valid, baseline_finite
        ),
        "candidate": _threshold_counts(
            after_raw, truth_valid, candidate_finite
        ),
    }
    return result


def _ratio_within_margin(
    group: dict[str, Any], metric: str, statistic: str
) -> bool:
    """Test the predeclared multiplicative noninferiority bound."""
    ratio = group[metric]["candidate_over_baseline"][statistic]
    return ratio is not None and float(ratio) <= 1.0 + NONINFERIORITY_MARGIN


def _counts_no_worse(group: dict[str, Any]) -> bool:
    """Require no increase at any registered raw-error threshold."""
    counts = group["raw_threshold_counts"]
    return all(
        int(counts["candidate"][str(threshold)])
        <= int(counts["baseline"][str(threshold)])
        for threshold in THRESHOLDS
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tanaka-dir", type=Path, required=True)
    parser.add_argument("--baseline-non-tanaka-dir", type=Path, required=True)
    parser.add_argument("--candidate-tanaka-dir", type=Path, required=True)
    parser.add_argument("--candidate-non-tanaka-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument(
        "--comparison",
        default="C28 mode-off epoch-40 final versus C27 epoch-40 final",
    )
    parser.add_argument(
        "--decision-label",
        default="C28 mode-loss ablation",
    )
    return parser.parse_args()


def main() -> None:
    """Validate artifacts, compare panels, and write the acceptance decision."""
    args = _parse_args()
    candidate_hashes: set[str] = set()
    regime_arrays: dict[str, RegimeArrays] = {}
    regime_results: dict[str, Any] = {}

    for regime in REGIMES:
        baseline_root = _regime_root(
            regime, args.baseline_tanaka_dir, args.baseline_non_tanaka_dir
        )
        candidate_root = _regime_root(
            regime, args.candidate_tanaka_dir, args.candidate_non_tanaka_dir
        )
        summary = _validate_candidate_summary(
            candidate_root, regime, args.candidate_run_dir
        )
        candidate_hashes.add(str(summary["checkpoint_source"]["sha256"]))
        arrays, scan = _validate_regime_archives(
            baseline_root / f"{regime}_trajs.npz",
            candidate_root / f"{regime}_trajs.npz",
            args.length,
        )
        if int(summary["n_truth_valid"]) != scan["n_truth_valid"]:
            raise ValueError(f"{regime}: summary/archive truth-valid mismatch")
        regime_arrays[regime] = arrays
        regime_results[regime] = {
            **scan,
            **_group_summary(regime_arrays, (regime,)),
        }
        gc.collect()

    if len(candidate_hashes) != 1:
        raise ValueError(f"multiple candidate checkpoint hashes: {candidate_hashes}")

    groups = {
        name: _group_summary(regime_arrays, regimes)
        for name, regimes in GROUPS.items()
    }
    all_group = groups["all"]
    bf_group = groups["benjamin_feir"]
    all_predictions_finite = all(
        result["candidate_all_predictions_finite"]
        for result in regime_results.values()
    )
    checks = {
        "all_320_predictions_finite": all_predictions_finite
        and int(all_group["n_attempted"]) == 320,
        "all_305_truth_valid_retained": int(all_group["n_truth_valid"])
        == 305
        and int(all_group["n_candidate_finite_truth_valid"]) == 305,
        "all_raw_mean_within_5pct": _ratio_within_margin(
            all_group, "raw", "mean"
        ),
        "all_raw_p95_within_5pct": _ratio_within_margin(
            all_group, "raw", "p95"
        ),
        "all_aligned_mean_within_5pct": _ratio_within_margin(
            all_group, "aligned", "mean"
        ),
        "all_aligned_p95_within_5pct": _ratio_within_margin(
            all_group, "aligned", "p95"
        ),
        "all_raw_threshold_counts_no_worse": _counts_no_worse(all_group),
        "bf_raw_p95_within_5pct": _ratio_within_margin(
            bf_group, "raw", "p95"
        ),
        "bf_raw_threshold_counts_no_worse": _counts_no_worse(bf_group),
    }
    result = {
        "comparison": args.comparison,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "checkpoint_sha256": next(iter(candidate_hashes)),
        "baseline": {
            "tanaka": str(args.baseline_tanaka_dir.resolve()),
            "non_tanaka": str(args.baseline_non_tanaka_dir.resolve()),
        },
        "candidate": {
            "run": str(args.candidate_run_dir.resolve()),
            "tanaka": str(args.candidate_tanaka_dir.resolve()),
            "non_tanaka": str(args.candidate_non_tanaka_dir.resolve()),
        },
        "acceptance_definition": {
            "aggregate": "candidate/C27 raw and aligned mean and p95 <= 1.05",
            "thresholds": list(THRESHOLDS),
            "threshold_rule": "no increase in truth-valid raw failures",
            "benjamin_feir": "raw p95 <= 1.05*C27 and no threshold increase",
            "finiteness": "eta, xi, and Gxi finite at every saved frame",
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "groups": groups,
        "regimes": regime_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    decision = "ACCEPT" if result["accepted"] else "REJECT"
    print(f"{args.decision_label} decision: {decision}")
    print(f"Acceptance summary: {args.output}")


if __name__ == "__main__":
    main()
