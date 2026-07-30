"""Transactional batch storage for the paper corpus.

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


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

_PROPOSAL_DTYPES = {
    "family_id": np.dtype(np.int16),
    "revision_id": np.dtype(np.int16),
    "split_id": np.dtype(np.uint8),
    "batch_id": np.dtype(np.int64),
    "case_id": np.dtype(np.int64),
    "cell_id": np.dtype(np.int32),
}
_PROPOSAL_STRING_FIELDS = {
    "config_fingerprint",
    "case_spec_json",
    "metadata_json",
}
_SHARD_DTYPES = {
    "eta": np.dtype(np.float32),
    "xi": np.dtype(np.float32),
    "gxi": np.dtype(np.float32),
    "depth": np.dtype(np.float64),
    "time": np.dtype(np.float64),
    "case_local_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
    "selected_dense_index": np.dtype(np.int32),
}
_SHARD_STRING_FIELDS = {
    "config_fingerprint",
    "proposal_sha256",
}


class BatchStatus(str, Enum):
    """State inferred from durable batch files."""

    EMPTY = "empty"
    PROPOSED = "proposed"
    SHARD_WRITTEN = "shard_written"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass(frozen=True)
class BatchPaths:
    """All durable paths belonging to one family/split/batch transaction."""

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
class CaseCommitRecord:
    """Row ownership and numerical decision for one attempted case."""

    case_id: int
    accepted: bool
    required_bits: int
    evaluated_bits: int
    failed_bits: int
    first_row: int
    row_count: int
    metrics: Mapping[str, float | int | bool | None]

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.required_bits, "required_bits"),
            (self.evaluated_bits, "evaluated_bits"),
            (self.failed_bits, "failed_bits"),
        ):
            if value < 0 or value >= 1 << 32:
                raise ValueError(f"{field_name} must fit in uint32")
        if self.failed_bits & ~self.evaluated_bits:
            raise ValueError("failed_bits must be a subset of evaluated_bits")
        missing_bits = self.required_bits & ~self.evaluated_bits
        required_failures = self.required_bits & self.failed_bits
        if self.accepted != (missing_bits == 0 and required_failures == 0):
            raise ValueError(
                "accepted must agree with the required, evaluated, and failed masks"
            )
        if self.accepted:
            if self.first_row < 0 or self.row_count <= 0:
                raise ValueError("accepted cases must own at least one row")
        elif self.first_row != -1 or self.row_count != 0:
            raise ValueError("rejected cases cannot own shard rows")


@dataclass(frozen=True)
class BatchInspection:
    """Validated durable state of one batch."""

    status: BatchStatus
    proposal_sha256: str | None
    shard_sha256: str | None


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


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


def _write_npz_atomic(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    ordered = {name: np.asarray(arrays[name]) for name in sorted(arrays)}

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez(handle, **ordered)

    _replace_atomically(path, writer)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _strict_json_bytes(payload)

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            handle.write(encoded)

    _replace_atomically(path, writer)


def write_npz_atomic(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Atomically write a pickle-free NPZ with deterministic member order."""

    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    if any(array.dtype.kind == "O" for array in normalized.values()):
        raise TypeError("atomic NPZ arrays cannot use object dtype")
    _write_npz_atomic(path, normalized)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a strict, key-sorted JSON object."""

    _write_json_atomic(path, payload)


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _arrays_equal(
    left: Mapping[str, NDArray[Any]],
    right: Mapping[str, NDArray[Any]],
) -> bool:
    def equal(name: str) -> bool:
        left_array = left[name]
        right_array = right[name]
        if left_array.dtype.kind in {"f", "c"}:
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        return bool(np.array_equal(left_array, right_array))

    return set(left) == set(right) and all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and equal(name)
        for name in left
    )


def _scalar_string(array: NDArray[Any], field_name: str) -> str:
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{field_name} must be a scalar string array")
    return str(array.item())


def _require_exact_dtype(
    arrays: Mapping[str, NDArray[Any]],
    expected: Mapping[str, np.dtype[Any]],
) -> None:
    for name, dtype in expected.items():
        if name not in arrays:
            raise ValueError(f"archive is missing required array {name!r}")
        if arrays[name].dtype != dtype:
            raise TypeError(
                f"array {name!r} has dtype {arrays[name].dtype}; expected {dtype}"
            )


def validate_proposal_arrays(
    arrays: Mapping[str, NDArray[Any]],
) -> tuple[int, str]:
    """Validate a proposal and return ``(number_of_cases, fingerprint)``."""

    _require_exact_dtype(arrays, _PROPOSAL_DTYPES)
    missing_strings = _PROPOSAL_STRING_FIELDS.difference(arrays)
    if missing_strings:
        raise ValueError(
            f"proposal is missing required arrays {sorted(missing_strings)}"
        )
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise TypeError("proposal arrays cannot use object dtype")

    for name in ("family_id", "revision_id", "split_id", "batch_id"):
        if arrays[name].ndim != 0:
            raise ValueError(f"proposal field {name!r} must be scalar")

    case_ids = arrays["case_id"]
    cell_ids = arrays["cell_id"]
    specifications = arrays["case_spec_json"]
    if case_ids.ndim != 1 or case_ids.size == 0:
        raise ValueError("case_id must be a nonempty one-dimensional array")
    if cell_ids.shape != case_ids.shape:
        raise ValueError("cell_id must have the same shape as case_id")
    if specifications.ndim != 1 or specifications.shape != case_ids.shape:
        raise ValueError("case_spec_json must have one string per case")
    if specifications.dtype.kind not in {"U", "S"}:
        raise TypeError("case_spec_json must have a string dtype")
    if np.unique(case_ids).size != case_ids.size:
        raise ValueError("case_id values must be unique within a proposal")

    fingerprint = _scalar_string(
        arrays["config_fingerprint"],
        "config_fingerprint",
    )
    _validate_sha256(fingerprint, "config_fingerprint")
    metadata = _scalar_string(arrays["metadata_json"], "metadata_json")
    if not isinstance(_strict_json_loads(metadata), dict):
        raise ValueError("metadata_json must encode a JSON object")
    for specification in specifications:
        if not isinstance(_strict_json_loads(str(specification)), dict):
            raise ValueError("every case_spec_json entry must encode a JSON object")

    scalar_or_case_length = {
        name
        for name, array in arrays.items()
        if array.ndim == 0 or array.shape[0] == case_ids.size
    }
    if scalar_or_case_length != set(arrays):
        raise ValueError(
            "each optional proposal array must be scalar or have one entry per case"
        )
    for name, array in arrays.items():
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"proposal array {name!r} contains nonfinite values")
    return int(case_ids.size), fingerprint


def ensure_proposal(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> str:
    """Create a proposal or verify exact replay of an existing proposal."""

    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    validate_proposal_arrays(normalized)
    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a terminal batch proposal")
    if paths.proposal.exists():
        existing = _load_npz(paths.proposal)
        validate_proposal_arrays(existing)
        if not _arrays_equal(existing, normalized):
            raise RuntimeError("existing proposal differs from replayed proposal")
        return file_sha256(paths.proposal)
    _write_npz_atomic(paths.proposal, normalized)
    return file_sha256(paths.proposal)


def _validate_shard_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    proposal_arrays: Mapping[str, NDArray[Any]],
    proposal_sha256: str,
) -> tuple[int, str]:
    _require_exact_dtype(arrays, _SHARD_DTYPES)
    missing_strings = _SHARD_STRING_FIELDS.difference(arrays)
    if missing_strings:
        raise ValueError(f"shard is missing required arrays {sorted(missing_strings)}")
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise TypeError("shard arrays cannot use object dtype")

    eta = arrays["eta"]
    if eta.ndim != 2 or eta.shape[0] == 0 or eta.shape[1] == 0:
        raise ValueError("eta must have nonempty shape (row, space)")
    for name in ("xi", "gxi"):
        if arrays[name].shape != eta.shape:
            raise ValueError(f"{name} must have the same shape as eta")
    row_count = eta.shape[0]
    for name in (
        "depth",
        "time",
        "case_local_index",
        "frame_index",
        "selected_dense_index",
    ):
        if arrays[name].shape != (row_count,):
            raise ValueError(f"{name} must have one entry per shard row")
    for name in ("eta", "xi", "gxi", "depth", "time"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"shard array {name!r} contains nonfinite values")
    if np.any(arrays["depth"] <= 0.0):
        raise ValueError("every stored depth must be positive")

    fingerprint = _scalar_string(
        arrays["config_fingerprint"],
        "config_fingerprint",
    )
    proposal_fingerprint = _scalar_string(
        proposal_arrays["config_fingerprint"],
        "proposal config_fingerprint",
    )
    if fingerprint != proposal_fingerprint:
        raise ValueError("shard and proposal configuration fingerprints differ")
    stored_proposal_sha256 = _scalar_string(
        arrays["proposal_sha256"],
        "proposal_sha256",
    )
    if stored_proposal_sha256 != proposal_sha256:
        raise ValueError("shard references a different proposal hash")

    case_local_index = arrays["case_local_index"]
    if np.any(case_local_index < 0):
        raise ValueError("case_local_index must be nonnegative")
    if int(np.max(case_local_index)) >= proposal_arrays["case_id"].size:
        raise ValueError("case_local_index references a missing proposal case")
    if np.any(np.diff(case_local_index) < 0):
        raise ValueError("rows from each case must form one ordered block")

    for case_index in np.unique(case_local_index):
        selected = np.flatnonzero(case_local_index == case_index)
        frame_index = arrays["frame_index"][selected]
        dense_index = arrays["selected_dense_index"][selected]
        times = arrays["time"][selected]
        expected_frames = np.arange(selected.size, dtype=np.int32)
        if not np.array_equal(frame_index, expected_frames):
            raise ValueError("frame_index must start at zero within every case")
        if np.any(np.diff(dense_index) <= 0):
            raise ValueError(
                "selected_dense_index must increase strictly within every case"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("stored times must increase strictly within every case")
        if not np.all(arrays["depth"][selected] == arrays["depth"][selected[0]]):
            raise ValueError("depth must remain constant within each case")
    return row_count, fingerprint


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
    validate_proposal_arrays(proposal_arrays)
    proposal_sha256 = file_sha256(paths.proposal)
    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    _validate_shard_arrays(
        normalized,
        proposal_arrays=proposal_arrays,
        proposal_sha256=proposal_sha256,
    )
    if paths.shard.exists():
        existing = _load_npz(paths.shard)
        _validate_shard_arrays(
            existing,
            proposal_arrays=proposal_arrays,
            proposal_sha256=proposal_sha256,
        )
        if not _arrays_equal(existing, normalized):
            raise RuntimeError("existing shard differs from reproduced shard")
        return file_sha256(paths.shard)
    _write_npz_atomic(paths.shard, normalized)
    return file_sha256(paths.shard)


def _case_row_blocks(
    shard_arrays: Mapping[str, NDArray[Any]] | None,
) -> dict[int, tuple[int, int]]:
    if shard_arrays is None:
        return {}
    case_local_index = shard_arrays["case_local_index"]
    return {
        int(case_index): (
            int(np.flatnonzero(case_local_index == case_index)[0]),
            int(np.count_nonzero(case_local_index == case_index)),
        )
        for case_index in np.unique(case_local_index)
    }


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
    _, fingerprint = validate_proposal_arrays(proposal_arrays)
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
    blocks = _case_row_blocks(shard_arrays)
    for local_index, case in enumerate(cases):
        expected = blocks.get(local_index)
        observed = (case.first_row, case.row_count) if case.accepted else None
        if observed != expected:
            raise ValueError(
                f"case {case.case_id} row ownership does not match the shard"
            )

    payload: dict[str, object] = {
        "schema": "paper_corpus_batch_result_v1",
        "config_fingerprint": fingerprint,
        "proposal_sha256": proposal_sha256,
        "shard_sha256": shard_sha256,
        "cases": [asdict(case) for case in cases],
        "metadata": dict(metadata),
    }
    encoded = _strict_json_bytes(payload)
    if paths.result.exists():
        if paths.result.read_bytes() != encoded:
            raise RuntimeError("existing result differs from replayed result")
        return file_sha256(paths.result)
    _write_json_atomic(paths.result, payload)
    return file_sha256(paths.result)


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
        "schema": "paper_corpus_fatal_failure_v1",
        "proposal_sha256": file_sha256(paths.proposal),
        "phase": phase,
        "exception_type": exception_type,
        "message": message,
        "telemetry": dict(telemetry),
    }
    encoded = _strict_json_bytes(payload)
    if paths.failure.exists():
        if paths.failure.read_bytes() != encoded:
            raise RuntimeError("existing failure differs from replayed failure")
        return file_sha256(paths.failure)
    _write_json_atomic(paths.failure, payload)
    return file_sha256(paths.failure)


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
    exists = {
        "proposal": paths.proposal.exists(),
        "shard": paths.shard.exists(),
        "result": paths.result.exists(),
        "failure": paths.failure.exists(),
    }
    if not any(exists.values()):
        return BatchInspection(BatchStatus.EMPTY, None, None)
    if not exists["proposal"]:
        raise RuntimeError("orphaned batch artifact exists without its proposal")
    if exists["result"] and exists["failure"]:
        raise RuntimeError("a batch cannot have both result and failure records")

    proposal_arrays = _load_npz(paths.proposal)
    _, fingerprint = validate_proposal_arrays(proposal_arrays)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RuntimeError("proposal configuration fingerprint does not match")
    proposal_sha256 = file_sha256(paths.proposal)

    shard_sha256: str | None = None
    if exists["shard"]:
        shard_arrays = _load_npz(paths.shard)
        _validate_shard_arrays(
            shard_arrays,
            proposal_arrays=proposal_arrays,
            proposal_sha256=proposal_sha256,
        )
        shard_sha256 = file_sha256(paths.shard)

    if exists["failure"]:
        failure = _load_json_object(paths.failure)
        if failure.get("proposal_sha256") != proposal_sha256:
            raise RuntimeError("failure record references a different proposal")
        return BatchInspection(
            BatchStatus.FAILED,
            proposal_sha256,
            shard_sha256,
        )
    if exists["result"]:
        result = _load_json_object(paths.result)
        if result.get("proposal_sha256") != proposal_sha256:
            raise RuntimeError("result references a different proposal")
        recorded_shard = result.get("shard_sha256")
        if recorded_shard is None:
            if exists["shard"]:
                raise RuntimeError("result omits an existing shard")
        elif not isinstance(recorded_shard, str):
            raise TypeError("result shard_sha256 must be a string or null")
        elif not exists["shard"] or recorded_shard != shard_sha256:
            raise RuntimeError("result references a missing or different shard")
        if result.get("config_fingerprint") != fingerprint:
            raise RuntimeError("result configuration fingerprint does not match")
        return BatchInspection(
            BatchStatus.COMMITTED,
            proposal_sha256,
            shard_sha256,
        )
    status = BatchStatus.SHARD_WRITTEN if exists["shard"] else BatchStatus.PROPOSED
    return BatchInspection(status, proposal_sha256, shard_sha256)
