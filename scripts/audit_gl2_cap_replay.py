#!/usr/bin/env python3
"""Read-only audit of paired paper-corpus GL2-cap replay directories."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    BatchStatus,
    file_sha256,
    inspect_batch,
    write_json_atomic,
)
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityScope,
    reasons_from_bits,
)


DEFAULT_CAP8_ROOT = Path(
    "outputs/noncorpus_batch_canary_revision2_20260728/tanaka_b256"
)
DEFAULT_CAP4_ROOT = Path(
    "outputs/noncorpus_cap4_b256_replay_20260728/tanaka_b256"
)
PAIRED_PROPOSAL_FIELDS = (
    "attempt_index",
    "batch_id",
    "case_id",
    "case_spec_json",
    "cell_id",
    "family_id",
    "revision_id",
    "root_seed",
    "split_id",
    "stream_id",
)
FIELD_NAMES = ("eta", "xi", "gxi")


class AuditError(RuntimeError):
    """Raised when a durable artifact violates the replay contract."""


@dataclass(frozen=True)
class CaseRecord:
    """One attempted case reconstructed from its proposal and result."""

    batch_id: int
    local_index: int
    case_id: int
    cell_id: str
    accepted: bool
    required_bits: int
    evaluated_bits: int
    failed_bits: int
    first_row: int
    row_count: int
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class CompletedRun:
    """Validated fields needed for the paired comparison."""

    root: Path
    summary: Mapping[str, object]
    manifest: Mapping[str, object]
    cases: tuple[CaseRecord, ...]
    source_sha256: Mapping[str, str]
    dependency_environment: Mapping[str, object]
    cap: int
    tolerance: float
    total_seconds: float
    batch_seconds: Mapping[int, float]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise AuditError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise AuditError(f"{path} must contain a JSON object")
    return value


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AuditError(f"{name} must be a JSON object")
    return value


def _as_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise AuditError(f"{name} must be a JSON array")
    return value


def _batch_id(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("batch_"))
    except ValueError as exc:
        raise AuditError(f"invalid batch filename {path.name!r}") from exc


def _proposal_paths(root: Path, family: str, split: str) -> tuple[Path, ...]:
    proposal_dir = root / "proposals" / family / split
    return tuple(sorted(proposal_dir.glob("batch_*.npz")))


def _run_progress(
    root: Path,
    *,
    family: str,
    split: str,
) -> dict[str, object]:
    proposals = _proposal_paths(root, family, split)
    statuses: dict[str, str] = {}
    for proposal in proposals:
        batch_id = _batch_id(proposal)
        paths = BatchPaths.under(
            root,
            family=family,
            split=split,
            batch_id=batch_id,
        )
        statuses[str(batch_id)] = inspect_batch(paths).status.value
    summary = root / f"paper_corpus_{family}_{split}.summary.json"
    return {
        "root": str(root),
        "proposal_count": len(proposals),
        "batch_status": statuses,
        "summary_exists": summary.is_file(),
        "complete": bool(
            proposals
            and summary.is_file()
            and all(status == BatchStatus.COMMITTED.value for status in statuses.values())
        ),
    }


def _load_configuration(
    summary: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    run_spec = _as_mapping(summary.get("run_spec"), "summary.run_spec")
    configuration = _as_mapping(
        run_spec.get("configuration"),
        "summary.run_spec.configuration",
    )
    trajectory = _as_mapping(
        configuration.get("trajectory_execution"),
        "configuration.trajectory_execution",
    )
    numerical = _as_mapping(
        trajectory.get("numerical"),
        "configuration.trajectory_execution.numerical",
    )
    return configuration, numerical


def _case_reason_names(failed_bits: int) -> tuple[str, ...]:
    return tuple(reason.name for reason in reasons_from_bits(failed_bits))


def _validate_case(
    case: CaseRecord,
    *,
    tolerance: float,
    rows_per_case: int,
) -> None:
    decision = QualityDecision.from_bits(
        QualityScope.TRAJECTORY,
        case.required_bits,
        case.evaluated_bits,
        case.failed_bits,
    )
    _require(
        decision.accepted == case.accepted,
        f"case {case.case_id} acceptance disagrees with its quality masks",
    )
    if not case.accepted:
        _require(
            case.first_row == -1 and case.row_count == 0,
            f"rejected case {case.case_id} owns shard rows",
        )
        return

    _require(
        case.first_row >= 0 and case.row_count == rows_per_case,
        f"accepted case {case.case_id} has the wrong row count",
    )
    required_true = (
        "accepted",
        "all_stages_solved",
        "complete_admissible_trajectory",
        "positive_water_column",
        "state_finite",
        "target_finite",
    )
    for name in required_true:
        _require(
            case.metrics.get(name) is True,
            f"accepted case {case.case_id} has {name}={case.metrics.get(name)!r}",
        )
    residual = case.metrics.get("maximum_stage_residual")
    _require(
        isinstance(residual, (int, float))
        and not isinstance(residual, bool)
        and math.isfinite(float(residual))
        and float(residual) <= tolerance,
        f"accepted case {case.case_id} violates the GL2 residual tolerance",
    )


def _manifest_path(
    root: Path,
    summary: Mapping[str, object],
    artifact_name: str,
) -> Path:
    dataset_view = _as_mapping(
        summary.get("dataset_view"),
        "summary.dataset_view",
    )
    artifact = _as_mapping(
        dataset_view.get(artifact_name),
        f"summary.dataset_view.{artifact_name}",
    )
    relative = artifact.get("path")
    _require(isinstance(relative, str), f"{artifact_name}.path must be a string")
    path = root / relative
    _require(path.is_file(), f"missing {artifact_name} artifact {path}")
    _require(
        artifact.get("sha256") == file_sha256(path),
        f"{artifact_name} hash differs from the summary",
    )
    return path


def _audit_completed_run(
    root: Path,
    *,
    family: str,
    split: str,
) -> CompletedRun:
    summary_path = root / f"paper_corpus_{family}_{split}.summary.json"
    summary = _read_json(summary_path)
    _require(summary.get("status") == "complete", f"{root} is not complete")

    configuration, numerical = _load_configuration(summary)
    cap = int(numerical["gl2_iteration_cap"])
    tolerance = float(numerical["gl2_residual_tolerance"])
    stored_policy = _as_mapping(
        _as_mapping(
            configuration["trajectory_execution"],
            "configuration.trajectory_execution",
        ).get("stored_time_policy"),
        "configuration.trajectory_execution.stored_time_policy",
    )
    rows_per_case = int(stored_policy[f"{family}_count"])

    run_spec = _as_mapping(summary["run_spec"], "summary.run_spec")
    cell_codes = _as_mapping(run_spec.get("cell_codes"), "run_spec.cell_codes")
    code_to_cell = {int(code): cell for cell, code in cell_codes.items()}
    expected_fingerprint = summary.get("configuration_fingerprint")
    _require(
        isinstance(expected_fingerprint, str),
        "summary configuration_fingerprint must be a string",
    )

    proposals = _proposal_paths(root, family, split)
    _require(bool(proposals), f"{root} has no proposals")
    cases: list[CaseRecord] = []
    batch_seconds: dict[int, float] = {}
    for proposal_path in proposals:
        batch_id = _batch_id(proposal_path)
        paths = BatchPaths.under(
            root,
            family=family,
            split=split,
            batch_id=batch_id,
        )
        inspection = inspect_batch(
            paths,
            expected_fingerprint=expected_fingerprint,
        )
        _require(
            inspection.status is BatchStatus.COMMITTED,
            f"{root} batch {batch_id} is {inspection.status.value}",
        )
        batch_seconds[batch_id] = (
            paths.result.stat().st_mtime_ns - paths.proposal.stat().st_mtime_ns
        ) / 1.0e9

        with np.load(paths.proposal, allow_pickle=False) as proposal:
            case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
            encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
        result = _read_json(paths.result)
        result_cases = _as_list(result.get("cases"), f"{paths.result}.cases")
        _require(
            len(result_cases) == case_ids.size,
            f"{paths.result} omits proposed cases",
        )
        for local_index, raw_case in enumerate(result_cases):
            case = _as_mapping(raw_case, "result case")
            _require(
                int(case["case_id"]) == int(case_ids[local_index]),
                f"{paths.result} case order differs from its proposal",
            )
            record = CaseRecord(
                batch_id=batch_id,
                local_index=local_index,
                case_id=int(case["case_id"]),
                cell_id=code_to_cell[int(encoded_cells[local_index])],
                accepted=bool(case["accepted"]),
                required_bits=int(case["required_bits"]),
                evaluated_bits=int(case["evaluated_bits"]),
                failed_bits=int(case["failed_bits"]),
                first_row=int(case["first_row"]),
                row_count=int(case["row_count"]),
                metrics=_as_mapping(case.get("metrics"), "case.metrics"),
            )
            _validate_case(
                record,
                tolerance=tolerance,
                rows_per_case=rows_per_case,
            )
            cases.append(record)

    _require(
        len({case.case_id for case in cases}) == len(cases),
        f"{root} repeats a case ID",
    )
    counts = _as_mapping(summary.get("counts"), "summary.counts")
    accepted = sum(case.accepted for case in cases)
    rejected = len(cases) - accepted
    _require(int(counts["attempted"]) == len(cases), "attempt count mismatch")
    _require(int(counts["accepted"]) == accepted, "accepted count mismatch")
    _require(int(counts["rejected"]) == rejected, "rejected count mismatch")

    quotas = _as_list(run_spec.get("quotas"), "run_spec.quotas")
    accepted_by_cell = Counter(
        case.cell_id for case in cases if case.accepted
    )
    attempted_by_cell = Counter(case.cell_id for case in cases)
    for raw_quota in quotas:
        quota = _as_mapping(raw_quota, "run_spec quota")
        cell_id = str(quota["cell_id"])
        _require(
            accepted_by_cell[cell_id] == int(quota["target_accepted"]),
            f"{root} did not meet accepted quota for {cell_id}",
        )
        _require(
            attempted_by_cell[cell_id] >= accepted_by_cell[cell_id],
            f"{root} has impossible counts for {cell_id}",
        )

    manifest_path = _manifest_path(root, summary, "manifest")
    trajectory_map_path = _manifest_path(root, summary, "trajectory_map")
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("configuration_fingerprint") == expected_fingerprint,
        "manifest configuration fingerprint mismatch",
    )
    _require(
        manifest.get("trajectory_map_sha256")
        == file_sha256(trajectory_map_path),
        "trajectory-map hash differs from the manifest",
    )
    for shard_record_raw in _as_list(
        manifest.get("dataset_shards"),
        "manifest.dataset_shards",
    ):
        shard_record = _as_mapping(shard_record_raw, "manifest shard")
        shard_path = root / str(shard_record["path"])
        _require(
            shard_record.get("sha256") == file_sha256(shard_path),
            f"manifest shard hash mismatch for {shard_path}",
        )
    for batch_record_raw in _as_list(
        manifest.get("dataset_batches"),
        "manifest.dataset_batches",
    ):
        batch_record = _as_mapping(batch_record_raw, "manifest batch")
        for prefix in ("proposal", "result"):
            path = root / str(batch_record[f"{prefix}_path"])
            _require(
                batch_record.get(f"{prefix}_sha256") == file_sha256(path),
                f"manifest {prefix} hash mismatch for {path}",
            )

    with np.load(trajectory_map_path, allow_pickle=False) as trajectory_map:
        map_case_ids = np.asarray(
            trajectory_map["trajectory_case_id"],
            dtype=np.int64,
        )
        map_accepted = np.asarray(
            trajectory_map["trajectory_accepted"],
            dtype=np.bool_,
        )
        map_rows = np.asarray(
            trajectory_map["trajectory_row_count"],
            dtype=np.int32,
        )
        map_failed = np.asarray(
            trajectory_map["trajectory_failed_bits"],
            dtype=np.uint32,
        )
    _require(
        np.array_equal(map_case_ids, [case.case_id for case in cases]),
        "trajectory-map case order differs from committed results",
    )
    _require(
        np.array_equal(map_accepted, [case.accepted for case in cases]),
        "trajectory-map acceptance differs from committed results",
    )
    _require(
        np.array_equal(map_rows, [case.row_count for case in cases]),
        "trajectory-map row counts differ from committed results",
    )
    _require(
        np.array_equal(map_failed, [case.failed_bits for case in cases]),
        "trajectory-map failure bits differ from committed results",
    )
    _require(
        not tuple(root.rglob("*.tmp")),
        f"{root} contains an unfinished atomic temporary file",
    )

    source_sha256 = _as_mapping(
        configuration.get("source_sha256"),
        "configuration.source_sha256",
    )
    dependency_environment = _as_mapping(
        configuration.get("dependency_environment"),
        "configuration.dependency_environment",
    )
    timing = _as_mapping(summary.get("timing_seconds"), "summary.timing_seconds")
    return CompletedRun(
        root=root,
        summary=summary,
        manifest=manifest,
        cases=tuple(cases),
        source_sha256={
            str(path): str(digest) for path, digest in source_sha256.items()
        },
        dependency_environment=dependency_environment,
        cap=cap,
        tolerance=tolerance,
        total_seconds=float(timing["total"]),
        batch_seconds=batch_seconds,
    )


def _proposal_comparison(
    cap8: CompletedRun,
    cap4: CompletedRun,
    *,
    family: str,
    split: str,
) -> dict[str, object]:
    paths = [
        run.root / "proposals" / family / split / "batch_000000.npz"
        for run in (cap8, cap4)
    ]
    equal: dict[str, bool] = {}
    with (
        np.load(paths[0], allow_pickle=False) as proposal8,
        np.load(paths[1], allow_pickle=False) as proposal4,
    ):
        for field in PAIRED_PROPOSAL_FIELDS:
            equal[field] = bool(
                np.array_equal(proposal8[field], proposal4[field])
            )
    return {
        "batch_id": 0,
        "compared_fields": list(PAIRED_PROPOSAL_FIELDS),
        "all_input_fields_equal": all(equal.values()),
        "field_equal": equal,
        "cap8_proposal_sha256": file_sha256(paths[0]),
        "cap4_proposal_sha256": file_sha256(paths[1]),
        "proposal_files_expectedly_differ": (
            file_sha256(paths[0]) != file_sha256(paths[1])
        ),
    }


def _outcome_comparison(
    cap8: CompletedRun,
    cap4: CompletedRun,
) -> dict[str, object]:
    outcomes = []
    by_run = [
        {
            case.case_id: case
            for case in run.cases
            if case.batch_id == 0
        }
        for run in (cap8, cap4)
    ]
    shared = sorted(set(by_run[0]).intersection(by_run[1]))
    transitions: Counter[str] = Counter()
    for case_id in shared:
        case8 = by_run[0][case_id]
        case4 = by_run[1][case_id]
        transition = (
            f"{'accepted' if case8.accepted else 'rejected'}_to_"
            f"{'accepted' if case4.accepted else 'rejected'}"
        )
        transitions[transition] += 1
        if (
            case8.accepted != case4.accepted
            or case8.failed_bits != case4.failed_bits
        ):
            outcomes.append(
                {
                    "case_id": case_id,
                    "cell_id": case8.cell_id,
                    "cap8": {
                        "accepted": case8.accepted,
                        "failed_bits": case8.failed_bits,
                        "reasons": _case_reason_names(case8.failed_bits),
                    },
                    "cap4": {
                        "accepted": case4.accepted,
                        "failed_bits": case4.failed_bits,
                        "reasons": _case_reason_names(case4.failed_bits),
                    },
                }
            )
    return {
        "shared_batch0_case_count": len(shared),
        "transitions": dict(sorted(transitions.items())),
        "changed_cases": outcomes,
    }


def _field_comparison(
    cap8: CompletedRun,
    cap4: CompletedRun,
    *,
    family: str,
    split: str,
) -> dict[str, object]:
    cases8 = {
        case.case_id: case
        for case in cap8.cases
        if case.batch_id == 0 and case.accepted
    }
    cases4 = {
        case.case_id: case
        for case in cap4.cases
        if case.batch_id == 0 and case.accepted
    }
    common = sorted(set(cases8).intersection(cases4))
    shard_paths = [
        run.root / "shards" / family / split / "batch_000000.npz"
        for run in (cap8, cap4)
    ]
    selection_mismatches: list[int] = []
    field_metrics: dict[str, object] = {}
    with (
        np.load(shard_paths[0], allow_pickle=False) as shard8,
        np.load(shard_paths[1], allow_pickle=False) as shard4,
    ):
        aligned: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for case_id in common:
            case8, case4 = cases8[case_id], cases4[case_id]
            rows8 = np.arange(
                case8.first_row,
                case8.first_row + case8.row_count,
            )
            rows4 = np.arange(
                case4.first_row,
                case4.first_row + case4.row_count,
            )
            dense8 = np.asarray(shard8["selected_dense_index"])[rows8]
            dense4 = np.asarray(shard4["selected_dense_index"])[rows4]
            common_dense, index8, index4 = np.intersect1d(
                dense8,
                dense4,
                assume_unique=True,
                return_indices=True,
            )
            if not np.array_equal(dense8, dense4):
                selection_mismatches.append(case_id)
            _require(
                common_dense.size > 0,
                f"common accepted case {case_id} has no paired stored times",
            )
            aligned[case_id] = (rows8[index8], rows4[index4])

        for field in FIELD_NAMES:
            values8 = np.asarray(shard8[field])
            values4 = np.asarray(shard4[field])
            maximum = 0.0
            square_sum = 0.0
            value_count = 0
            exact = True
            for case_id in common:
                rows8, rows4 = aligned[case_id]
                left = values8[rows8]
                right = values4[rows4]
                exact &= bool(np.array_equal(left, right))
                difference = left.astype(np.float64) - right.astype(np.float64)
                maximum = max(maximum, float(np.max(np.abs(difference))))
                square_sum += float(np.sum(difference * difference))
                value_count += difference.size
            field_metrics[field] = {
                "exactly_equal": exact,
                "maximum_absolute_difference": maximum,
                "rms_difference": math.sqrt(square_sum / value_count),
                "compared_values": value_count,
            }
    return {
        "common_accepted_batch0_cases": len(common),
        "selection_mismatch_case_ids": selection_mismatches,
        "fields": field_metrics,
    }


def _normalized_configuration(run: CompletedRun) -> dict[str, object]:
    configuration, _ = _load_configuration(run.summary)
    normalized = copy.deepcopy(dict(configuration))
    normalized.pop("source_sha256", None)
    trajectory = normalized.get("trajectory_execution")
    if not isinstance(trajectory, dict):
        raise AuditError("normalized trajectory_execution must be a JSON object")
    numerical = trajectory.get("numerical")
    if not isinstance(numerical, dict):
        raise AuditError(
            "normalized trajectory_execution.numerical must be a JSON object"
        )
    numerical["gl2_iteration_cap"] = "<paired-cap>"
    return normalized


def _paired_report(
    cap8: CompletedRun,
    cap4: CompletedRun,
    *,
    family: str,
    split: str,
) -> dict[str, object]:
    source_paths = sorted(
        set(cap8.source_sha256).union(cap4.source_sha256)
    )
    source_differences = {
        path: {
            "cap8": cap8.source_sha256.get(path),
            "cap4": cap4.source_sha256.get(path),
        }
        for path in source_paths
        if cap8.source_sha256.get(path) != cap4.source_sha256.get(path)
    }
    proposal = _proposal_comparison(
        cap8,
        cap4,
        family=family,
        split=split,
    )
    _require(
        bool(proposal["all_input_fields_equal"]),
        "batch-zero replay inputs are not identical",
    )
    _require(cap8.cap == 8, f"reference run has cap {cap8.cap}, expected 8")
    _require(cap4.cap == 4, f"replay run has cap {cap4.cap}, expected 4")
    _require(
        cap8.tolerance == cap4.tolerance,
        "paired runs use different residual tolerances",
    )
    _require(
        cap8.dependency_environment == cap4.dependency_environment,
        "paired runs use different dependency environments",
    )
    _require(
        _normalized_configuration(cap8) == _normalized_configuration(cap4),
        "paired run configurations differ beyond source hashes and GL2 cap",
    )
    _require(
        set(source_differences)
        == {"solver/gen_data/pipeline/refinement.py"},
        "paired source hashes differ outside the GL2-contract module",
    )

    return {
        "schema": "gl2_cap_b256_replay_audit_v1",
        "status": "pass",
        "cap8": _run_report(cap8),
        "cap4": _run_report(cap4),
        "paired_contract": {
            "residual_tolerance": cap8.tolerance,
            "dependency_environment_equal": True,
            "configuration_equal_after_cap_and_source_normalization": True,
            "source_hash_differences": source_differences,
            "proposal": proposal,
        },
        "outcomes": _outcome_comparison(cap8, cap4),
        "retained_fields": _field_comparison(
            cap8,
            cap4,
            family=family,
            split=split,
        ),
        "performance": {
            "accepted_case_throughput_ratio_cap4_over_cap8": (
                _accepted_throughput(cap4) / _accepted_throughput(cap8)
            ),
            "batch0_wall_time_ratio_cap4_over_cap8": (
                cap4.batch_seconds[0] / cap8.batch_seconds[0]
            ),
        },
    }


def _accepted_throughput(run: CompletedRun) -> float:
    accepted = sum(case.accepted for case in run.cases)
    return accepted * 3600.0 / run.total_seconds


def _run_report(run: CompletedRun) -> dict[str, object]:
    accepted_cases = [case for case in run.cases if case.accepted]
    rejected = [case for case in run.cases if not case.accepted]
    reasons = Counter(
        "+".join(_case_reason_names(case.failed_bits))
        for case in rejected
    )
    accepted_by_cell = Counter(case.cell_id for case in accepted_cases)
    rejected_by_cell = Counter(case.cell_id for case in rejected)
    maximum_accepted_residual = max(
        float(case.metrics["maximum_stage_residual"])
        for case in accepted_cases
    )
    return {
        "root": str(run.root),
        "cap": run.cap,
        "status": "complete_and_validated",
        "artifact_checks": {
            "all_batches_committed": True,
            "all_recorded_hashes_match": True,
            "manifest_and_trajectory_map_match_results": True,
            "no_atomic_temporary_files": True,
            "all_cell_quotas_met": True,
        },
        "attempted": len(run.cases),
        "accepted": len(accepted_cases),
        "rejected": len(rejected),
        "accepted_case_ids": [case.case_id for case in accepted_cases],
        "accepted_by_cell": dict(sorted(accepted_by_cell.items())),
        "rejected_cases": [
            {
                "case_id": case.case_id,
                "cell_id": case.cell_id,
                "failed_bits": case.failed_bits,
                "reasons": _case_reason_names(case.failed_bits),
            }
            for case in rejected
        ],
        "rejected_by_cell": dict(sorted(rejected_by_cell.items())),
        "rejection_reasons": dict(sorted(reasons.items())),
        "numerical_checks": {
            "all_shard_fields_finite": True,
            "all_accepted_stages_solved": True,
            "gl2_residual_tolerance": run.tolerance,
            "maximum_accepted_stage_residual": maximum_accepted_residual,
        },
        "total_seconds": run.total_seconds,
        "accepted_cases_per_hour": _accepted_throughput(run),
        "batch_seconds": {
            str(batch_id): seconds
            for batch_id, seconds in sorted(run.batch_seconds.items())
        },
        "configuration_fingerprint": run.summary[
            "configuration_fingerprint"
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap8-root", type=Path, default=DEFAULT_CAP8_ROOT)
    parser.add_argument("--cap4-root", type=Path, default=DEFAULT_CAP4_ROOT)
    parser.add_argument("--family", default="tanaka")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for an atomic copy of the completed audit report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    progress = {
        "cap8": _run_progress(
            args.cap8_root,
            family=args.family,
            split=args.split,
        ),
        "cap4": _run_progress(
            args.cap4_root,
            family=args.family,
            split=args.split,
        ),
    }
    if not all(bool(run["complete"]) for run in progress.values()):
        print(
            json.dumps(
                {
                    "schema": "gl2_cap_b256_replay_audit_v1",
                    "status": "waiting",
                    "progress": progress,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        raise SystemExit(2)

    cap8 = _audit_completed_run(
        args.cap8_root,
        family=args.family,
        split=args.split,
    )
    cap4 = _audit_completed_run(
        args.cap4_root,
        family=args.family,
        split=args.split,
    )
    report = _paired_report(
        cap8,
        cap4,
        family=args.family,
        split=args.split,
    )
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
