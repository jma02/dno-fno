"""Define and validate one batch's saved plan, simulation rows, and results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import parse_json
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit
from solver.gen_data.pipeline.types import JsonObject, JsonScalar


_BATCH_PLAN_STRING_FIELDS = {
    "dataset_split",
    "parameter_group_id",
    "simulation_spec_json",
}
_SHARD_DTYPES = {
    "eta": np.dtype(np.float32),
    "xi": np.dtype(np.float32),
    "gxi": np.dtype(np.float32),
    "depth": np.dtype(np.float64),
    "time": np.dtype(np.float64),
    "simulation_local_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
}
BATCH_PLAN_FIELDS = frozenset(("family_id", *_BATCH_PLAN_STRING_FIELDS))
SHARD_FIELDS = frozenset(_SHARD_DTYPES)

_FAILED_CHECK_NAMES = (
    "nonfinite_state",
    "nonfinite_target",
    "nonpositive_water_height",
    "hamiltonian_drift",
    "integration_failure",
    "outside_support",
    "incomplete_trajectory",
)


def _canonical_failed_checks(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(check, str) for check in value
    ):
        raise TypeError("failed_checks must be a list or tuple of strings")
    if len(value) != len(set(value)):
        raise ValueError("failed_checks contains duplicates")
    unknown = set(value).difference(_FAILED_CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown failed checks: {sorted(unknown)}")
    selected = set(value)
    return tuple(name for name in _FAILED_CHECK_NAMES if name in selected)


@dataclass(frozen=True)
class SimulationResult:
    """Saved outcome and diagnostics for one attempted simulation."""

    accepted: bool
    failed_checks: tuple[str, ...]
    metrics: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failed_checks",
            _canonical_failed_checks(self.failed_checks),
        )

    def to_json_record(self) -> JsonObject:
        """Return the result representation stored in a completed batch."""

        return {
            "accepted": self.accepted,
            "failed_checks": list(self.failed_checks),
            "metrics": dict(self.metrics),
        }


def parse_simulation_result(value: object) -> SimulationResult:
    """Parse and validate one simulation entry from a batch result."""

    if not isinstance(value, dict):
        raise TypeError("every result simulation must be a JSON object")
    unexpected = value.keys() - {"accepted", "failed_checks", "metrics"}
    if unexpected:
        raise ValueError(
            f"result simulation contains unexpected fields {sorted(unexpected)}"
        )

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
    return SimulationResult(
        accepted=accepted,
        failed_checks=_canonical_failed_checks(value.get("failed_checks")),
        metrics=metrics,
    )


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
    local_indices = indices[starts]
    ends = np.concatenate((starts[1:], np.asarray([indices.size])))
    return {
        int(local_index): (int(first_row), int(end - first_row))
        for local_index, first_row, end in zip(local_indices, starts, ends)
    }


def validate_batch_plan(batch_plan: Mapping[str, Any]) -> None:
    """Validate the simulations that one batch plans to run."""

    _validate_required_array_dtypes(
        batch_plan,
        {"family_id": np.dtype(np.int16)},
    )
    missing_strings = _BATCH_PLAN_STRING_FIELDS.difference(batch_plan)
    if missing_strings:
        raise ValueError(
            f"batch plan is missing required arrays {sorted(missing_strings)}"
        )
    unexpected = batch_plan.keys() - BATCH_PLAN_FIELDS
    if unexpected:
        raise ValueError(f"batch plan contains unexpected arrays {sorted(unexpected)}")
    for name in _BATCH_PLAN_STRING_FIELDS:
        if batch_plan[name].dtype.kind not in {"U", "S"}:
            raise TypeError(f"batch-plan field {name!r} must contain strings")

    for name in ("family_id", "dataset_split"):
        if batch_plan[name].ndim != 0:
            raise ValueError(f"batch-plan field {name!r} must be scalar")
    try:
        DatasetSplit(str(batch_plan["dataset_split"].item()))
    except ValueError as error:
        raise ValueError(
            "dataset_split must be 'train', 'validation', or 'test'"
        ) from error

    specifications = batch_plan["simulation_spec_json"]
    if specifications.ndim != 1 or specifications.size == 0:
        raise ValueError(
            "simulation_spec_json must be a nonempty one-dimensional array"
        )
    if batch_plan["parameter_group_id"].shape != specifications.shape:
        raise ValueError("parameter_group_id must have one entry per simulation")

    for specification in specifications:
        if not isinstance(parse_json(str(specification)), dict):
            raise ValueError(
                "every simulation_spec_json entry must encode a JSON object"
            )


def validate_shard(
    shard: Mapping[str, Any],
    *,
    batch_plan: Mapping[str, Any],
) -> None:
    """Validate one batch's shard against its saved plan."""

    _validate_required_array_dtypes(
        shard,
        _SHARD_DTYPES,
    )
    unexpected = shard.keys() - SHARD_FIELDS
    if unexpected:
        raise ValueError(f"shard contains unexpected arrays {sorted(unexpected)}")

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
        number_of_simulations=int(batch_plan["simulation_spec_json"].size),
    )
    for first_row, frame_count in row_blocks_by_simulation.values():
        simulation_slice = slice(first_row, first_row + frame_count)
        stored_frame_indices = shard["frame_index"][simulation_slice]
        simulation_times = shard["time"][simulation_slice]
        simulation_depths = shard["depth"][simulation_slice]
        if not np.array_equal(
            stored_frame_indices,
            np.arange(frame_count, dtype=np.int32),
        ):
            raise ValueError("frame_index must be 0, 1, ... within every simulation")
        if np.any(np.diff(simulation_times) <= 0.0):
            raise ValueError(
                "stored times must increase strictly within every simulation"
            )
        if not np.all(simulation_depths == simulation_depths[0]):
            raise ValueError("depth must remain constant within each simulation")
