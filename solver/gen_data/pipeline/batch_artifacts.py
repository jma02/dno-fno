"""Define and validate one batch's saved plan, simulation rows, and results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import parse_json


_BATCH_PLAN_DTYPES = {
    "family_id": np.dtype(np.int16),
    "revision_id": np.dtype(np.int16),
    "split_id": np.dtype(np.uint8),
    "batch_id": np.dtype(np.int64),
    "simulation_id": np.dtype(np.int64),
    "cell_id": np.dtype(np.int32),
    "root_seed": np.dtype(np.uint64),
    "stream_id": np.dtype(np.uint32),
    "attempt_index": np.dtype(np.uint64),
}
_BATCH_PLAN_STRING_FIELDS = {
    "simulation_spec_json",
    "metadata_json",
}
_SHARD_DTYPES = {
    "eta": np.dtype(np.float32),
    "xi": np.dtype(np.float32),
    "gxi": np.dtype(np.float32),
    "depth": np.dtype(np.float64),
    "time": np.dtype(np.float64),
    "simulation_local_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
    "selected_dense_index": np.dtype(np.int32),
}


@dataclass(frozen=True)
class SimulationCommitRecord:
    """Row ownership and numerical decision for one attempted simulation."""

    simulation_id: int
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
            if value < 0 or value > np.iinfo(np.uint32).max:
                raise ValueError(f"{field_name} must fit in uint32")
        if self.failed_bits & ~self.evaluated_bits:
            raise ValueError("failed_bits must be a subset of evaluated_bits")
        missing_bits = self.required_bits & ~self.evaluated_bits
        required_failures = self.required_bits & self.failed_bits
        if self.accepted != (missing_bits == 0 and required_failures == 0):
            raise ValueError("accepted must agree with the quality masks")
        if self.accepted:
            if self.first_row < 0 or self.row_count <= 0:
                raise ValueError("accepted simulations must own at least one row")
        elif self.first_row != -1 or self.row_count != 0:
            raise ValueError("rejected simulations cannot own stored rows")

    def to_json_record(self) -> dict[str, object]:
        """Return the existing on-disk result representation."""

        return {
            "simulation_id": self.simulation_id,
            "accepted": self.accepted,
            "required_bits": self.required_bits,
            "evaluated_bits": self.evaluated_bits,
            "failed_bits": self.failed_bits,
            "first_row": self.first_row,
            "row_count": self.row_count,
            "metrics": dict(self.metrics),
        }


def parse_simulation_result(value: object) -> SimulationCommitRecord:
    """Parse and validate one simulation entry from a batch result."""

    if not isinstance(value, dict):
        raise TypeError("every result simulation must be a JSON object")

    def required_integer(name: str) -> int:
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int):
            raise TypeError(f"result {name} field must be an integer")
        return field

    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("result accepted field must be boolean")
    metrics = value.get("metrics", {})
    if not isinstance(metrics, dict) or any(
        not isinstance(name, str)
        or metric is not None
        and not isinstance(metric, (str, float, int, bool))
        for name, metric in metrics.items()
    ):
        raise TypeError("result metrics must map strings to JSON scalar values")
    return SimulationCommitRecord(
        simulation_id=required_integer("simulation_id"),
        accepted=accepted,
        required_bits=required_integer("required_bits"),
        evaluated_bits=required_integer("evaluated_bits"),
        failed_bits=required_integer("failed_bits"),
        first_row=required_integer("first_row"),
        row_count=required_integer("row_count"),
        metrics=metrics,
    )


def parse_batch_plan_metadata(metadata: NDArray[Any]) -> dict[str, object]:
    """Parse the scalar JSON metadata stored with a batch plan."""

    if metadata.ndim != 0 or metadata.dtype.kind not in {"U", "S"}:
        raise TypeError("metadata_json must be a scalar string array")
    value = parse_json(str(metadata.item()))
    if not isinstance(value, dict):
        raise ValueError("metadata_json must encode a JSON object")
    return value


def _validate_required_array_dtypes(
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


def compute_simulation_row_blocks(
    simulation_local_index: NDArray[Any],
    *,
    number_of_simulations: int,
) -> dict[int, tuple[int, int]]:
    """Return each simulation's contiguous ``(first_row, row_count)`` block."""

    indices = np.asarray(simulation_local_index)
    if indices.ndim != 1:
        raise ValueError("simulation_local_index must be one-dimensional")
    if indices.size == 0:
        return {}
    if np.any(indices < 0) or int(np.max(indices)) >= number_of_simulations:
        raise ValueError(
            "simulation_local_index references a simulation absent from the plan"
        )
    if np.any(np.diff(indices) < 0):
        raise ValueError("rows from each simulation must form one ordered block")
    starts = np.concatenate((np.asarray([0]), np.flatnonzero(np.diff(indices)) + 1))
    simulation_ids = indices[starts]
    if np.unique(simulation_ids).size != simulation_ids.size:
        raise ValueError("rows from each simulation must form one ordered block")
    ends = np.concatenate((starts[1:], np.asarray([indices.size])))
    return {
        int(simulation_id): (int(first_row), int(end - first_row))
        for simulation_id, first_row, end in zip(simulation_ids, starts, ends)
    }


def validate_batch_plan(batch_plan: Mapping[str, NDArray[Any]]) -> None:
    """Validate the simulations that one batch plans to run."""

    _validate_required_array_dtypes(batch_plan, _BATCH_PLAN_DTYPES)
    missing_strings = _BATCH_PLAN_STRING_FIELDS.difference(batch_plan)
    if missing_strings:
        raise ValueError(
            f"batch plan is missing required arrays {sorted(missing_strings)}"
        )
    if any(array.dtype.kind == "O" for array in batch_plan.values()):
        raise TypeError("batch-plan arrays cannot use object dtype")

    for name in ("family_id", "revision_id", "split_id", "batch_id"):
        if batch_plan[name].ndim != 0:
            raise ValueError(f"batch-plan field {name!r} must be scalar")

    simulation_ids = batch_plan["simulation_id"]
    specifications = batch_plan["simulation_spec_json"]
    if simulation_ids.ndim != 1 or simulation_ids.size == 0:
        raise ValueError("simulation_id must be a nonempty one-dimensional array")
    for name in ("cell_id", "root_seed", "stream_id", "attempt_index"):
        if batch_plan[name].shape != simulation_ids.shape:
            raise ValueError(f"{name} must have the same shape as simulation_id")
    if specifications.shape != simulation_ids.shape:
        raise ValueError("simulation_spec_json must have one string per simulation")
    if specifications.dtype.kind not in {"U", "S"}:
        raise TypeError("simulation_spec_json must have a string dtype")
    if np.unique(simulation_ids).size != simulation_ids.size:
        raise ValueError("simulation_id values must be unique within a batch plan")

    parse_batch_plan_metadata(batch_plan["metadata_json"])
    for specification in specifications:
        if not isinstance(parse_json(str(specification)), dict):
            raise ValueError(
                "every simulation_spec_json entry must encode a JSON object"
            )

    if any(
        array.ndim != 0 and array.shape[0] != simulation_ids.size
        for array in batch_plan.values()
    ):
        raise ValueError(
            "each optional batch-plan array must be scalar or have one entry per simulation"
        )
    for name, array in batch_plan.items():
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"batch-plan array {name!r} contains nonfinite values")


def validate_shard(
    shard: Mapping[str, NDArray[Any]],
    *,
    batch_plan: Mapping[str, NDArray[Any]],
) -> None:
    """Validate one batch's shard against its saved plan."""

    _validate_required_array_dtypes(
        shard,
        _SHARD_DTYPES,
    )
    if any(array.dtype.kind == "O" for array in shard.values()):
        raise TypeError("shard arrays cannot use object dtype")

    eta = shard["eta"]
    if eta.ndim != 2 or eta.shape[0] == 0 or eta.shape[1] == 0:
        raise ValueError("eta must have nonempty shape (row, space)")
    for name in ("xi", "gxi"):
        if shard[name].shape != eta.shape:
            raise ValueError(f"{name} must have the same shape as eta")
    stored_row_count = eta.shape[0]
    for name in (
        "depth",
        "time",
        "simulation_local_index",
        "frame_index",
        "selected_dense_index",
    ):
        if shard[name].shape != (stored_row_count,):
            raise ValueError(f"{name} must have one entry per stored row")
    for name in ("eta", "xi", "gxi", "depth", "time"):
        if not np.isfinite(shard[name]).all():
            raise ValueError(f"shard array {name!r} contains nonfinite values")
    if np.any(shard["depth"] <= 0.0):
        raise ValueError("every stored depth must be positive")

    row_blocks_by_simulation = compute_simulation_row_blocks(
        shard["simulation_local_index"],
        number_of_simulations=int(batch_plan["simulation_id"].size),
    )
    for first_row, frame_count in row_blocks_by_simulation.values():
        simulation_slice = slice(first_row, first_row + frame_count)
        stored_frame_indices = shard["frame_index"][simulation_slice]
        source_frame_indices = shard["selected_dense_index"][simulation_slice]
        simulation_times = shard["time"][simulation_slice]
        simulation_depths = shard["depth"][simulation_slice]
        if not np.array_equal(
            stored_frame_indices,
            np.arange(frame_count, dtype=np.int32),
        ):
            raise ValueError("frame_index must be 0, 1, ... within every simulation")
        if np.any(np.diff(source_frame_indices) <= 0):
            raise ValueError(
                "selected_dense_index must increase strictly within every simulation"
            )
        if np.any(np.diff(simulation_times) <= 0.0):
            raise ValueError(
                "stored times must increase strictly within every simulation"
            )
        if not np.all(simulation_depths == simulation_depths[0]):
            raise ValueError("depth must remain constant within each simulation")
