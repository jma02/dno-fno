"""Store generated batches atomically and recover interrupted work.

Each attempted batch is proposed before numerical work begins.  A result JSON
is the commit marker; a shard without a result is therefore recoverable, while
a result whose referenced shard is absent is corruption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_format import (
    BATCH_RESULT_SCHEMA,
    FATAL_FAILURE_SCHEMA,
    CaseCommitRecord,
    case_row_blocks as _case_row_blocks,
    parse_case_commit_record as _parse_case_commit_record,
    strict_json_loads as _strict_json_loads,
    validate_proposal_arrays as _validate_proposal_arrays,
    validate_sha256 as _validate_sha256,
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
    def under(
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
    proposal_sha256: str | None
    cases: tuple[CaseCommitRecord, ...] = ()


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _replace_atomically(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Atomically write a pickle-free NPZ with deterministic member order."""

    ordered = {name: np.asarray(arrays[name]) for name in sorted(arrays)}
    if any(array.dtype.kind == "O" for array in ordered.values()):
        raise TypeError("atomic NPZ arrays cannot use object dtype")

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            # NumPy's stub treats arbitrary NPZ member names as potential
            # ``allow_pickle`` arguments even though object arrays are rejected above.
            np.savez(handle, **ordered)  # pyright: ignore[reportArgumentType]

    _replace_atomically(path, writer)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a strict, key-sorted JSON object."""

    encoded = _strict_json_bytes(payload)

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            handle.write(encoded)

    _replace_atomically(path, writer)


def _write_json_once(
    path: Path,
    payload: Mapping[str, object],
    *,
    artifact_name: str,
) -> str:
    encoded = _strict_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing {artifact_name} differs from replay")
    else:
        write_json_atomic(path, payload)
    return file_sha256(path)


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _arrays_equal(
    left: Mapping[str, NDArray[Any]],
    right: Mapping[str, NDArray[Any]],
) -> bool:
    return set(left) == set(right) and all(
        left[name].dtype == right[name].dtype
        and bool(
            np.array_equal(
                left[name],
                right[name],
                equal_nan=left[name].dtype.kind in {"f", "c"},
            )
        )
        for name in left
    )


def _ensure_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    validate: Callable[[Mapping[str, NDArray[Any]]], object],
    artifact_name: str,
) -> str:
    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    validate(normalized)
    if path.exists():
        existing = _load_npz(path)
        validate(existing)
        if not _arrays_equal(existing, normalized):
            raise RuntimeError(
                f"existing {artifact_name} differs from replayed {artifact_name}"
            )
    else:
        write_npz_atomic(path, normalized)
    return file_sha256(path)


def ensure_proposal(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> str:
    """Create a proposal or verify exact replay of an existing proposal."""

    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a terminal batch proposal")
    return _ensure_npz(
        paths.proposal,
        arrays,
        validate=_validate_proposal_arrays,
        artifact_name="proposal",
    )


def ensure_shard(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> str:
    """Create a complete-case shard or verify an identical orphaned shard."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before its shard")
    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a shard after a terminal record")
    proposal_arrays = _load_npz(paths.proposal)
    _validate_proposal_arrays(proposal_arrays)
    proposal_sha256 = file_sha256(paths.proposal)

    def validate(candidate: Mapping[str, NDArray[Any]]) -> None:
        _validate_shard_arrays(
            candidate,
            proposal_arrays=proposal_arrays,
            proposal_sha256=proposal_sha256,
        )

    return _ensure_npz(
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
) -> str:
    """Commit the decision for every proposed case and return the result hash."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before its result")
    if paths.failure.exists():
        raise RuntimeError("a failed batch cannot be committed")
    proposal_arrays = _load_npz(paths.proposal)
    fingerprint = _validate_proposal_arrays(proposal_arrays)
    proposal_sha256 = file_sha256(paths.proposal)
    proposed_case_ids = proposal_arrays["case_id"]
    if len(cases) != proposed_case_ids.size:
        raise ValueError("the result must contain every proposed case")
    if not np.array_equal(
        np.asarray([case.case_id for case in cases], dtype=np.int64),
        proposed_case_ids,
    ):
        raise ValueError("result cases must preserve proposal order and identity")

    shard_arrays: dict[str, NDArray[Any]] | None = None
    shard_sha256: str | None = None
    if paths.shard.exists():
        shard_arrays = _load_npz(paths.shard)
        _validate_shard_arrays(
            shard_arrays,
            proposal_arrays=proposal_arrays,
            proposal_sha256=proposal_sha256,
        )
        shard_sha256 = file_sha256(paths.shard)
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
        "schema": BATCH_RESULT_SCHEMA,
        "config_fingerprint": fingerprint,
        "proposal_sha256": proposal_sha256,
        "shard_sha256": shard_sha256,
        "cases": [asdict(case) for case in cases],
        "metadata": dict(metadata),
    }
    return _write_json_once(paths.result, payload, artifact_name="result")


def record_fatal_failure(
    paths: BatchPaths,
    *,
    phase: str,
    exception_type: str,
    message: str,
    telemetry: Mapping[str, object],
) -> str:
    """Record a fatal failure of the exact stored proposal."""

    if not paths.proposal.exists():
        raise RuntimeError("a proposal must exist before a failure record")
    if paths.result.exists():
        raise RuntimeError("a committed batch cannot be marked failed")
    if not phase or not exception_type:
        raise ValueError("phase and exception_type must be nonempty")
    payload: dict[str, object] = {
        "schema": FATAL_FAILURE_SCHEMA,
        "proposal_sha256": file_sha256(paths.proposal),
        "phase": phase,
        "exception_type": exception_type,
        "message": message,
        "telemetry": dict(telemetry),
    }
    return _write_json_once(paths.failure, payload, artifact_name="failure")


def _load_json_object(path: Path) -> dict[str, object]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def inspect_batch(
    paths: BatchPaths,
    *,
    expected_fingerprint: str | None = None,
) -> BatchInspection:
    """Validate the batch transaction and classify its restart state."""

    if expected_fingerprint is not None:
        _validate_sha256(expected_fingerprint, "expected_fingerprint")
    proposal_exists = paths.proposal.exists()
    shard_exists = paths.shard.exists()
    result_exists = paths.result.exists()
    failure_exists = paths.failure.exists()
    if not any((proposal_exists, shard_exists, result_exists, failure_exists)):
        return BatchInspection(BatchStatus.EMPTY, None)
    if not proposal_exists:
        raise RuntimeError("orphaned batch artifact exists without its proposal")
    if result_exists and failure_exists:
        raise RuntimeError("a batch cannot have both result and failure records")

    proposal_arrays = _load_npz(paths.proposal)
    fingerprint = _validate_proposal_arrays(proposal_arrays)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RuntimeError("proposal configuration fingerprint does not match")
    proposal_sha256 = file_sha256(paths.proposal)

    shard_arrays: dict[str, NDArray[Any]] | None = None
    shard_sha256: str | None = None
    if shard_exists:
        shard_arrays = _load_npz(paths.shard)
        _validate_shard_arrays(
            shard_arrays,
            proposal_arrays=proposal_arrays,
            proposal_sha256=proposal_sha256,
        )
        shard_sha256 = file_sha256(paths.shard)

    if failure_exists:
        failure = _load_json_object(paths.failure)
        if failure.get("proposal_sha256") != proposal_sha256:
            raise RuntimeError("failure record references a different proposal")
        return BatchInspection(BatchStatus.FAILED, proposal_sha256)
    if result_exists:
        result = _load_json_object(paths.result)
        if result.get("proposal_sha256") != proposal_sha256:
            raise RuntimeError("result references a different proposal")
        recorded_shard = result.get("shard_sha256")
        if recorded_shard is None:
            if shard_exists:
                raise RuntimeError("result omits an existing shard")
        elif not isinstance(recorded_shard, str):
            raise TypeError("result shard_sha256 must be a string or null")
        elif not shard_exists or recorded_shard != shard_sha256:
            raise RuntimeError("result references a missing or different shard")
        if result.get("config_fingerprint") != fingerprint:
            raise RuntimeError("result configuration fingerprint does not match")
        if result.get("schema") != BATCH_RESULT_SCHEMA:
            raise RuntimeError("result has an unknown schema")
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
        return BatchInspection(
            BatchStatus.COMMITTED,
            proposal_sha256,
            tuple(cases),
        )
    status = BatchStatus.SHARD_WRITTEN if shard_exists else BatchStatus.PROPOSED
    return BatchInspection(status, proposal_sha256)
