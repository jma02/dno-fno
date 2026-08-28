"""Common proposal and complete-case shard assembly for paper-dataset writers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    CaseCommitRecord,
    commit_batch,
    ensure_proposal,
    ensure_shard,
)
from solver.gen_data.pipeline.case_allocation import (
    AttemptAssignment,
    split_code,
)
from solver.gen_data.pipeline.case_checks import CaseCheckResult


FloatArray: TypeAlias = NDArray[np.floating]
JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class AcceptedCaseRows:
    """Rows retained from one accepted static state or trajectory."""

    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    depth: float
    time: FloatArray
    selected_dense_index: NDArray[np.integer]


@dataclass(frozen=True)
class CaseOutcome:
    """One complete quality decision and its optional retained rows."""

    decision: CaseCheckResult
    rows: AcceptedCaseRows | None
    metrics: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        if self.decision.accepted != (self.rows is not None):
            raise ValueError(
                "accepted decisions must have rows and rejected decisions must not"
            )


def _strict_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_proposal_arrays(
    assignments: Sequence[AttemptAssignment],
    specifications: Sequence[Mapping[str, object]],
    *,
    cell_codes: Mapping[str, int],
    batch_id: int,
    metadata: Mapping[str, object],
) -> dict[str, NDArray[Any]]:
    """Build the exact proposal persisted before numerical construction."""

    if not assignments:
        raise ValueError("assignments must not be empty")
    if len(assignments) != len(specifications):
        raise ValueError("assignments and specifications must have equal lengths")
    if batch_id < 0:
        raise ValueError("batch_id must be nonnegative")

    first_key = assignments[0].case_key
    shared_coordinates = (
        first_key.family_id,
        first_key.revision_id,
        first_key.split_id,
    )
    if any(
        (
            assignment.case_key.family_id,
            assignment.case_key.revision_id,
            assignment.case_key.split_id,
        )
        != shared_coordinates
        for assignment in assignments
    ):
        raise ValueError(
            "one batch must have one family, revision, and preassigned split"
        )

    unknown_cells = {
        assignment.cell_id
        for assignment in assignments
        if assignment.cell_id not in cell_codes
    }
    if unknown_cells:
        raise ValueError(f"missing integer codes for cells {sorted(unknown_cells)}")
    encoded_cell_codes = np.asarray(
        [cell_codes[assignment.cell_id] for assignment in assignments],
        dtype=np.int32,
    )
    if np.any(encoded_cell_codes < 0):
        raise ValueError("cell codes must be nonnegative")

    return {
        "family_id": np.asarray(first_key.family_id, dtype=np.int16),
        "revision_id": np.asarray(first_key.revision_id, dtype=np.int16),
        "split_id": np.asarray(split_code(first_key.split_id), dtype=np.uint8),
        "batch_id": np.asarray(batch_id, dtype=np.int64),
        "case_id": np.asarray(
            [assignment.case_key.case_id for assignment in assignments],
            dtype=np.int64,
        ),
        "cell_id": encoded_cell_codes,
        "root_seed": np.asarray(
            [assignment.case_key.root_seed for assignment in assignments],
            dtype=np.uint64,
        ),
        "stream_id": np.asarray(
            [assignment.case_key.stream_id for assignment in assignments],
            dtype=np.uint32,
        ),
        "attempt_index": np.asarray(
            [assignment.case_key.attempt_index for assignment in assignments],
            dtype=np.uint64,
        ),
        "case_spec_json": np.asarray(
            [_strict_json(dict(specification)) for specification in specifications]
        ),
        "metadata_json": np.asarray(_strict_json(dict(metadata))),
    }


def _validated_rows(
    rows: AcceptedCaseRows,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    eta = np.asarray(rows.eta, dtype=np.float32)
    xi = np.asarray(rows.xi, dtype=np.float32)
    gxi = np.asarray(rows.gxi, dtype=np.float32)
    time = np.asarray(rows.time, dtype=np.float64)
    selected = np.asarray(rows.selected_dense_index, dtype=np.int32)
    if eta.ndim != 2 or eta.shape[0] == 0 or eta.shape[1] == 0:
        raise ValueError("accepted eta must have nonempty shape (time, space)")
    if xi.shape != eta.shape or gxi.shape != eta.shape:
        raise ValueError("accepted eta, xi, and gxi must have identical shapes")
    if time.shape != (eta.shape[0],):
        raise ValueError("accepted time must have one value per field row")
    if selected.shape != time.shape:
        raise ValueError("selected_dense_index must have one value per field row")
    if not np.isfinite(rows.depth) or rows.depth <= 0.0:
        raise ValueError("accepted depth must be finite and positive")
    if not all(np.isfinite(value).all() for value in (eta, xi, gxi, time)):
        raise ValueError("accepted rows must be finite")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("accepted times must increase strictly")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("selected dense indices must increase strictly")
    return eta, xi, gxi, time, selected


def commit_case_outcomes(
    paths: BatchPaths,
    proposal_arrays: Mapping[str, NDArray[Any]],
    outcomes: Sequence[CaseOutcome],
    *,
    metadata: Mapping[str, object],
) -> None:
    """Persist accepted whole-case rows and commit every attempted decision."""

    if len(outcomes) != int(np.asarray(proposal_arrays["case_id"]).size):
        raise ValueError("outcomes must contain every proposed case")
    ensure_proposal(paths, proposal_arrays)

    eta_parts: list[NDArray[np.float32]] = []
    xi_parts: list[NDArray[np.float32]] = []
    gxi_parts: list[NDArray[np.float32]] = []
    depth_parts: list[NDArray[np.float64]] = []
    time_parts: list[NDArray[np.float64]] = []
    case_parts: list[NDArray[np.int32]] = []
    frame_parts: list[NDArray[np.int32]] = []
    selected_parts: list[NDArray[np.int32]] = []
    records: list[CaseCommitRecord] = []
    first_row = 0

    case_ids = np.asarray(proposal_arrays["case_id"], dtype=np.int64)
    for local_index, (case_id, outcome) in enumerate(zip(case_ids, outcomes)):
        decision = outcome.decision
        row_count = 0
        case_first_row = -1
        if outcome.rows is not None:
            eta, xi, gxi, time, selected = _validated_rows(outcome.rows)
            row_count = eta.shape[0]
            case_first_row = first_row
            eta_parts.append(eta)
            xi_parts.append(xi)
            gxi_parts.append(gxi)
            depth_parts.append(np.full(row_count, outcome.rows.depth, dtype=np.float64))
            time_parts.append(time)
            case_parts.append(np.full(row_count, local_index, dtype=np.int32))
            frame_parts.append(np.arange(row_count, dtype=np.int32))
            selected_parts.append(selected)
            first_row += row_count

        records.append(
            CaseCommitRecord(
                case_id=int(case_id),
                accepted=decision.accepted,
                required_bits=int(decision.required),
                evaluated_bits=int(decision.evaluated),
                failed_bits=int(decision.failed),
                first_row=case_first_row,
                row_count=row_count,
                metrics=dict(outcome.metrics),
            )
        )

    if eta_parts:
        ensure_shard(
            paths,
            {
                "eta": np.concatenate(eta_parts),
                "xi": np.concatenate(xi_parts),
                "gxi": np.concatenate(gxi_parts),
                "depth": np.concatenate(depth_parts),
                "time": np.concatenate(time_parts),
                "case_local_index": np.concatenate(case_parts),
                "frame_index": np.concatenate(frame_parts),
                "selected_dense_index": np.concatenate(selected_parts),
            },
        )
    commit_batch(paths, cases=records, metadata=metadata)


def batch_paths_for_assignments(
    root: Path,
    assignments: Sequence[AttemptAssignment],
    *,
    family_name: str,
    batch_id: int,
) -> BatchPaths:
    """Return standard paths after checking one batch's split consistency."""

    if not assignments:
        raise ValueError("assignments must not be empty")
    split_id = assignments[0].case_key.split_id
    if any(assignment.case_key.split_id is not split_id for assignment in assignments):
        raise ValueError("one batch cannot mix data splits")
    return BatchPaths.for_batch(
        root,
        family=family_name,
        split=split_id.value,
        batch_id=batch_id,
    )
