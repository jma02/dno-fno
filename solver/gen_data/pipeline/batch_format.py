"""Validate the proposal, shard, and result records stored for one batch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


_PROPOSAL_DTYPES = {
    "family_id": np.dtype(np.int16),
    "revision_id": np.dtype(np.int16),
    "split_id": np.dtype(np.uint8),
    "batch_id": np.dtype(np.int64),
    "case_id": np.dtype(np.int64),
    "cell_id": np.dtype(np.int32),
}
_PROPOSAL_STRING_FIELDS = {
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
    metrics: Mapping[str, str | float | int | bool | None]

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
            raise ValueError("accepted must agree with the quality masks")
        if self.accepted:
            if self.first_row < 0 or self.row_count <= 0:
                raise ValueError("accepted cases must own at least one row")
        elif self.first_row != -1 or self.row_count != 0:
            raise ValueError("rejected cases cannot own shard rows")


def parse_case_commit_record(
    value: object,
    *,
    expected_case_id: int,
) -> CaseCommitRecord:
    """Parse and validate one case entry from a batch result."""

    if not isinstance(value, dict):
        raise TypeError("every result case must be a JSON object")

    def integer(name: str) -> int:
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int):
            raise TypeError(f"result {name} field must be an integer")
        return field

    case_id = integer("case_id")
    if case_id != expected_case_id:
        raise ValueError("result case identity differs from its proposal")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("result accepted field must be boolean")
    raw_metrics = value.get("metrics", {})
    if not isinstance(raw_metrics, dict) or any(
        not isinstance(name, str)
        or metric is not None
        and not isinstance(metric, (str, float, int, bool))
        for name, metric in raw_metrics.items()
    ):
        raise TypeError("result metrics must map strings to JSON scalar values")
    return CaseCommitRecord(
        case_id=case_id,
        accepted=accepted,
        required_bits=integer("required_bits"),
        evaluated_bits=integer("evaluated_bits"),
        failed_bits=integer("failed_bits"),
        first_row=integer("first_row"),
        row_count=integer("row_count"),
        metrics=raw_metrics,
    )


def strict_json_loads(text: str) -> object:
    """Load JSON while rejecting nonfinite numeric constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


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


def case_row_blocks(
    case_local_index: NDArray[Any],
    *,
    number_of_cases: int,
) -> dict[int, tuple[int, int]]:
    """Return each case's contiguous ``(first_row, row_count)`` block."""

    indices = np.asarray(case_local_index)
    if indices.ndim != 1:
        raise ValueError("case_local_index must be one-dimensional")
    if indices.size == 0:
        return {}
    if np.any(indices < 0) or int(np.max(indices)) >= number_of_cases:
        raise ValueError("case_local_index references a missing proposal case")
    if np.any(np.diff(indices) < 0):
        raise ValueError("rows from each case must form one ordered block")
    starts = np.concatenate((np.asarray([0]), np.flatnonzero(np.diff(indices)) + 1))
    case_ids = indices[starts]
    if np.unique(case_ids).size != case_ids.size:
        raise ValueError("rows from each case must form one ordered block")
    ends = np.concatenate((starts[1:], np.asarray([indices.size])))
    return {
        int(case_id): (int(first_row), int(end - first_row))
        for case_id, first_row, end in zip(case_ids, starts, ends)
    }


def validate_proposal_arrays(arrays: Mapping[str, NDArray[Any]]) -> None:
    """Validate the arrays stored in one batch proposal."""

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

    metadata = _scalar_string(arrays["metadata_json"], "metadata_json")
    if not isinstance(strict_json_loads(metadata), dict):
        raise ValueError("metadata_json must encode a JSON object")
    for specification in specifications:
        if not isinstance(strict_json_loads(str(specification)), dict):
            raise ValueError("every case_spec_json entry must encode a JSON object")

    if any(
        array.ndim != 0 and array.shape[0] != case_ids.size for array in arrays.values()
    ):
        raise ValueError(
            "each optional proposal array must be scalar or have one entry per case"
        )
    for name, array in arrays.items():
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"proposal array {name!r} contains nonfinite values")
def validate_shard_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    proposal_arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Validate all stored rows and their link to a proposal."""

    _require_exact_dtype(arrays, _SHARD_DTYPES)
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

    blocks = case_row_blocks(
        arrays["case_local_index"],
        number_of_cases=int(proposal_arrays["case_id"].size),
    )
    for first_row, case_row_count in blocks.values():
        selected = slice(first_row, first_row + case_row_count)
        frame_index = arrays["frame_index"][selected]
        dense_index = arrays["selected_dense_index"][selected]
        times = arrays["time"][selected]
        if not np.array_equal(
            frame_index,
            np.arange(case_row_count, dtype=np.int32),
        ):
            raise ValueError("frame_index must start at zero within every case")
        if np.any(np.diff(dense_index) <= 0):
            raise ValueError(
                "selected_dense_index must increase strictly within every case"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("stored times must increase strictly within every case")
        if not np.all(arrays["depth"][selected] == arrays["depth"][first_row]):
            raise ValueError("depth must remain constant within each case")
