"""Manage the saved proposal, shard, result, and failure files for each batch.

Each attempted batch is proposed before numerical work begins.  A result JSON
is the commit marker; a shard without a result is therefore recoverable, while
a result whose referenced shard is absent is corruption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import (
    ensure_json,
    ensure_npz,
    load_npz,
)
from solver.gen_data.pipeline.batch_format import (
    CaseCommitRecord,
    case_row_blocks as _case_row_blocks,
    parse_case_commit_record as _parse_case_commit_record,
    strict_json_loads as _strict_json_loads,
    validate_proposal_arrays as _validate_proposal_arrays,
    validate_shard_arrays as _validate_shard_arrays,
)

_PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class BatchStatus(str, Enum):
    """State inferred from the files saved for a batch."""

    EMPTY = "empty"
    PROPOSED = "proposed"
    SHARD_WRITTEN = "shard_written"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass(frozen=True)
class BatchPaths:
    """Files belonging to one family, split, and batch."""

    proposal: Path
    shard: Path
    result: Path
    failure: Path

    @classmethod
    def for_batch(
        cls,
        root: Path,
        *,
        family: str,
        split: str,
        batch_id: int,
    ) -> "BatchPaths":
        """Construct the standard paths without creating any files."""

        for value, field_name in ((family, "family"), (split, "split")):
            if _PATH_COMPONENT_PATTERN.fullmatch(value) is None:
                raise ValueError(
                    f"{field_name} must contain only letters, digits, '_' or '-'"
                )
        if batch_id < 0:
            raise ValueError("batch_id must be nonnegative")
        filename = f"batch_{batch_id:06d}"
        return cls(
            proposal=root / "proposals" / family / split / f"{filename}.npz",
            shard=root / "shards" / family / split / f"{filename}.npz",
            result=root / "results" / family / split / f"{filename}.json",
            failure=root / "failures" / family / split / f"{filename}.json",
        )


@dataclass(frozen=True)
class BatchInspection:
    """Validated state reconstructed from one batch's files."""

    status: BatchStatus
    cases: tuple[CaseCommitRecord, ...] = ()


def ensure_proposal(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Create a proposal or verify exact replay of an existing proposal."""

    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a terminal batch proposal")
    ensure_npz(
        paths.proposal,
        arrays,
        validate=_validate_proposal_arrays,
        artifact_name="proposal",
    )


def ensure_shard(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Create a complete-case shard or verify an identical orphaned shard."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before its shard")
    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a shard after a terminal record")
    proposal_arrays = load_npz(paths.proposal)
    _validate_proposal_arrays(proposal_arrays)
    def validate(candidate: Mapping[str, NDArray[Any]]) -> None:
        _validate_shard_arrays(
            candidate,
            proposal_arrays=proposal_arrays,
        )

    ensure_npz(
        paths.shard,
        arrays,
        validate=validate,
        artifact_name="shard",
    )


def commit_batch(
    paths: BatchPaths,
    *,
    cases: Sequence[CaseCommitRecord],
    metadata: Mapping[str, object],
) -> None:
    """Commit the decision for every proposed case."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before its result")
    if paths.failure.exists():
        raise RuntimeError("a failed batch cannot be committed")
    proposal_arrays = load_npz(paths.proposal)
    _validate_proposal_arrays(proposal_arrays)
    proposed_case_ids = proposal_arrays["case_id"]
    if len(cases) != proposed_case_ids.size:
        raise ValueError("the result must contain every proposed case")
    if not np.array_equal(
        np.asarray([case.case_id for case in cases], dtype=np.int64),
        proposed_case_ids,
    ):
        raise ValueError("result cases must preserve proposal order and identity")

    shard_arrays: dict[str, NDArray[Any]] | None = None
    if paths.shard.exists():
        shard_arrays = load_npz(paths.shard)
        _validate_shard_arrays(
            shard_arrays,
            proposal_arrays=proposal_arrays,
        )
    blocks = (
        _case_row_blocks(
            shard_arrays["case_local_index"],
            number_of_cases=int(proposed_case_ids.size),
        )
        if shard_arrays is not None
        else {}
    )
    for local_index, case in enumerate(cases):
        expected = blocks.get(local_index)
        observed = (case.first_row, case.row_count) if case.accepted else None
        if observed != expected:
            raise ValueError(
                f"case {case.case_id} row ownership does not match the shard"
            )

    payload: dict[str, object] = {
        "cases": [asdict(case) for case in cases],
        "metadata": dict(metadata),
    }
    ensure_json(paths.result, payload, artifact_name="result")


def record_fatal_failure(
    paths: BatchPaths,
    *,
    phase: str,
    exception_type: str,
    message: str,
    telemetry: Mapping[str, object],
) -> None:
    """Record a fatal failure of the exact stored proposal."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before a failure record")
    if paths.result.exists():
        raise RuntimeError("a committed batch cannot be marked failed")
    if not phase or not exception_type:
        raise ValueError("phase and exception_type must be nonempty")
    payload: dict[str, object] = {
        "phase": phase,
        "exception_type": exception_type,
        "message": message,
        "telemetry": dict(telemetry),
    }
    ensure_json(paths.failure, payload, artifact_name="failure")


def _load_json_object(path: Path) -> dict[str, object]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def inspect_batch(paths: BatchPaths) -> BatchInspection:
    """Validate the batch transaction and classify its restart state."""

    proposal_exists = paths.proposal.exists()
    shard_exists = paths.shard.exists()
    result_exists = paths.result.exists()
    failure_exists = paths.failure.exists()
    if not any((proposal_exists, shard_exists, result_exists, failure_exists)):
        return BatchInspection(BatchStatus.EMPTY)
    if not proposal_exists:
        raise RuntimeError("orphaned batch artifact exists without its proposal")
    if result_exists and failure_exists:
        raise RuntimeError("a batch cannot have both result and failure records")

    proposal_arrays = load_npz(paths.proposal)
    _validate_proposal_arrays(proposal_arrays)

    shard_arrays: dict[str, NDArray[Any]] | None = None
    if shard_exists:
        shard_arrays = load_npz(paths.shard)
        _validate_shard_arrays(
            shard_arrays,
            proposal_arrays=proposal_arrays,
        )

    if failure_exists:
        _load_json_object(paths.failure)
        return BatchInspection(BatchStatus.FAILED)
    if result_exists:
        result = _load_json_object(paths.result)
        raw_cases = result.get("cases")
        proposed_case_ids = proposal_arrays["case_id"]
        if not isinstance(raw_cases, list) or len(raw_cases) != proposed_case_ids.size:
            raise RuntimeError("result must contain every proposed case")
        blocks = (
            _case_row_blocks(
                shard_arrays["case_local_index"],
                number_of_cases=int(proposed_case_ids.size),
            )
            if shard_arrays is not None
            else {}
        )
        cases: list[CaseCommitRecord] = []
        for local_index, (case_id, raw_case) in enumerate(
            zip(proposed_case_ids, raw_cases)
        ):
            try:
                case = _parse_case_commit_record(
                    raw_case,
                    expected_case_id=int(case_id),
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            declared_block = (case.first_row, case.row_count) if case.accepted else None
            if declared_block != blocks.get(local_index):
                raise RuntimeError("result row ownership differs from its shard")
            cases.append(case)
        return BatchInspection(BatchStatus.COMMITTED, tuple(cases))
    status = BatchStatus.SHARD_WRITTEN if shard_exists else BatchStatus.PROPOSED
    return BatchInspection(status)
