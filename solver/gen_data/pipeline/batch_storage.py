"""Manage the saved plan, shard, result, and failure for each batch.

The plan is saved before numerical work begins. A result JSON commits the batch;
a shard without a result can be resumed, while a result that refers to an absent
shard is corrupt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import (
    ensure_npz,
    load_npz,
    read_json_object,
    write_json_atomic,
)
from solver.gen_data.pipeline.batch_artifacts import (
    SimulationCommitRecord,
    compute_simulation_row_blocks,
    parse_simulation_result,
    validate_batch_plan,
    validate_shard,
)

_PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class BatchStatus(str, Enum):
    """State inferred from the files saved for a batch."""

    EMPTY = "empty"
    PLAN_SAVED = "plan_saved"
    SHARD_WRITTEN = "shard_written"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass(frozen=True)
class BatchPaths:
    """Files belonging to one family, split, and batch."""

    batch_plan: Path
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
    ) -> BatchPaths:
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
            batch_plan=root / "batch_plans" / family / split / f"{filename}.npz",
            shard=root / "shards" / family / split / f"{filename}.npz",
            result=root / "results" / family / split / f"{filename}.json",
            failure=root / "failures" / family / split / f"{filename}.json",
        )


@dataclass(frozen=True)
class BatchInspection:
    """Validated state reconstructed from one batch's files."""

    status: BatchStatus
    simulations: tuple[SimulationCommitRecord, ...] = ()


def _load_batch_arrays(
    paths: BatchPaths,
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]] | None]:
    """Load and validate a batch plan and its optional shard."""

    batch_plan = load_npz(paths.batch_plan)
    validate_batch_plan(batch_plan)
    if not paths.shard.exists():
        return batch_plan, None
    shard = load_npz(paths.shard)
    validate_shard(shard, batch_plan=batch_plan)
    return batch_plan, shard


def save_batch_plan(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Save a batch plan or verify that the existing plan is identical."""

    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a terminal batch plan")
    ensure_npz(
        paths.batch_plan,
        arrays,
        validate=validate_batch_plan,
    )


def save_shard(
    paths: BatchPaths,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Save a shard or verify that the existing shard is identical."""

    if not paths.batch_plan.exists():
        raise RuntimeError("a batch plan must exist before its shard")
    if paths.result.exists() or paths.failure.exists():
        raise RuntimeError("cannot replace a shard after a terminal record")
    batch_plan = load_npz(paths.batch_plan)
    validate_batch_plan(batch_plan)

    def validate(candidate: Mapping[str, NDArray[Any]]) -> None:
        validate_shard(
            candidate,
            batch_plan=batch_plan,
        )

    ensure_npz(
        paths.shard,
        arrays,
        validate=validate,
    )


def commit_batch(
    paths: BatchPaths,
    *,
    simulations: Sequence[SimulationCommitRecord],
    metadata: Mapping[str, object],
) -> None:
    """Commit the decision for every planned simulation."""

    if not paths.batch_plan.exists():
        raise RuntimeError("a batch plan must exist before its result")
    if paths.failure.exists():
        raise RuntimeError("a failed batch cannot be committed")
    if paths.result.exists():
        raise RuntimeError("batch is already committed")
    batch_plan, shard = _load_batch_arrays(paths)
    planned_simulation_ids = batch_plan["simulation_id"]
    if len(simulations) != planned_simulation_ids.size:
        raise ValueError("the result must contain every planned simulation")
    if not np.array_equal(
        np.asarray(
            [simulation.simulation_id for simulation in simulations], dtype=np.int64
        ),
        planned_simulation_ids,
    ):
        raise ValueError(
            "result simulations must preserve batch-plan order and identity"
        )

    blocks = (
        compute_simulation_row_blocks(
            shard["simulation_local_index"],
            number_of_simulations=int(planned_simulation_ids.size),
        )
        if shard is not None
        else {}
    )
    for local_index, simulation in enumerate(simulations):
        expected = blocks.get(local_index)
        observed = (
            (simulation.first_row, simulation.row_count)
            if simulation.accepted
            else None
        )
        if observed != expected:
            raise ValueError(
                f"simulation {simulation.simulation_id} row ownership does not match the shard"
            )

    payload: dict[str, object] = {
        "simulations": [simulation.to_json_record() for simulation in simulations],
        "metadata": dict(metadata),
    }
    write_json_atomic(paths.result, payload)


def record_fatal_failure(
    paths: BatchPaths,
    *,
    phase: str,
    exception_type: str,
    message: str,
    telemetry: Mapping[str, object],
) -> None:
    """Record a fatal failure of the exact stored batch plan."""

    if not paths.batch_plan.exists():
        raise RuntimeError("a batch plan must exist before a failure record")
    if paths.result.exists():
        raise RuntimeError("a committed batch cannot be marked failed")
    if paths.failure.exists():
        raise RuntimeError("batch is already marked failed")
    if not phase or not exception_type:
        raise ValueError("phase and exception_type must be nonempty")
    payload: dict[str, object] = {
        "phase": phase,
        "exception_type": exception_type,
        "message": message,
        "telemetry": dict(telemetry),
    }
    write_json_atomic(paths.failure, payload)


def inspect_batch(paths: BatchPaths) -> BatchInspection:
    """Validate the batch transaction and classify its restart state."""

    plan_exists = paths.batch_plan.exists()
    shard_exists = paths.shard.exists()
    result_exists = paths.result.exists()
    failure_exists = paths.failure.exists()
    if not any((plan_exists, shard_exists, result_exists, failure_exists)):
        return BatchInspection(BatchStatus.EMPTY)
    if not plan_exists:
        raise RuntimeError("orphaned batch artifact exists without its batch plan")
    if result_exists and failure_exists:
        raise RuntimeError("a batch cannot have both result and failure records")

    batch_plan, shard = _load_batch_arrays(paths)

    if failure_exists:
        read_json_object(paths.failure)
        return BatchInspection(BatchStatus.FAILED)
    if result_exists:
        result = read_json_object(paths.result)
        raw_simulations = result.get("simulations")
        planned_simulation_ids = batch_plan["simulation_id"]
        if (
            not isinstance(raw_simulations, list)
            or len(raw_simulations) != planned_simulation_ids.size
        ):
            raise RuntimeError("result must contain every planned simulation")
        blocks = (
            compute_simulation_row_blocks(
                shard["simulation_local_index"],
                number_of_simulations=int(planned_simulation_ids.size),
            )
            if shard is not None
            else {}
        )
        simulations: list[SimulationCommitRecord] = []
        for local_index, (simulation_id, raw_simulation) in enumerate(
            zip(planned_simulation_ids, raw_simulations)
        ):
            try:
                simulation = parse_simulation_result(raw_simulation)
                if simulation.simulation_id != int(simulation_id):
                    raise ValueError(
                        "result simulation identity differs from its batch plan"
                    )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            declared_block = (
                (simulation.first_row, simulation.row_count)
                if simulation.accepted
                else None
            )
            if declared_block != blocks.get(local_index):
                raise RuntimeError("result row ownership differs from its shard")
            simulations.append(simulation)
        return BatchInspection(BatchStatus.COMMITTED, tuple(simulations))
    status = BatchStatus.SHARD_WRITTEN if shard_exists else BatchStatus.PLAN_SAVED
    return BatchInspection(status)
