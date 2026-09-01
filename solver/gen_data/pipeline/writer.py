"""Save simulation assignments, accepted rows, and quality decisions by batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import json_text
from solver.gen_data.pipeline.batch_artifacts import (
    SimulationCommitRecord,
    compute_batch_simulation_ids,
)
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
)
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.types import (
    BatchPlanArrays,
    DatasetShardArrays,
    JsonScalar,
)


@dataclass(frozen=True)
class AcceptedSimulationRows:
    """Rows retained from one accepted static state or trajectory."""

    eta: NDArray[np.floating]
    xi: NDArray[np.floating]
    gxi: NDArray[np.floating]
    depth: float
    time: NDArray[np.floating]


@dataclass(frozen=True)
class SimulationOutcome:
    """One complete quality decision and its optional retained rows."""

    decision: SimulationCheckResult
    rows: AcceptedSimulationRows | None
    metrics: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        if self.decision.accepted != (self.rows is not None):
            raise ValueError(
                "accepted decisions must have rows and rejected decisions must not"
            )


def build_batch_plan(
    assignments: Sequence[AttemptAssignment],
    specifications: Sequence[Mapping[str, object]],
    *,
    metadata: Mapping[str, object],
) -> BatchPlanArrays:
    """Describe the sampled simulations in one batch."""

    if not assignments:
        raise ValueError("assignments must not be empty")
    if len(assignments) != len(specifications):
        raise ValueError("assignments and specifications must have equal lengths")
    first_key = assignments[0].simulation_key
    shared_coordinates = (
        first_key.family_id,
        first_key.dataset_split,
    )
    if any(
        (
            assignment.simulation_key.family_id,
            assignment.simulation_key.dataset_split,
        )
        != shared_coordinates
        for assignment in assignments
    ):
        raise ValueError("one batch must have one family and preassigned split")

    return {
        "family_id": np.asarray(first_key.family_id, dtype=np.int16),
        "dataset_split": np.asarray(first_key.dataset_split.value),
        "parameter_group_id": np.asarray(
            [assignment.parameter_group_id for assignment in assignments]
        ),
        "worker_stream_id": np.asarray(
            [assignment.simulation_key.worker_stream_id for assignment in assignments],
            dtype=np.uint32,
        ),
        "attempt_index": np.asarray(
            [assignment.simulation_key.attempt_index for assignment in assignments],
            dtype=np.uint64,
        ),
        "simulation_spec_json": np.asarray(
            [json_text(specification) for specification in specifications]
        ),
        "metadata_json": np.asarray(json_text(metadata)),
    }


def commit_simulation_outcomes(
    path: Path,
    batch_plan: BatchPlanArrays,
    outcomes: Sequence[SimulationOutcome],
    *,
    metadata: Mapping[str, object],
) -> None:
    """Persist accepted whole-simulation rows and every attempted decision."""

    simulation_ids = compute_batch_simulation_ids(batch_plan)
    if len(outcomes) != simulation_ids.size:
        raise ValueError("outcomes must contain every planned simulation")

    eta_parts: list[NDArray[np.float32]] = []
    xi_parts: list[NDArray[np.float32]] = []
    gxi_parts: list[NDArray[np.float32]] = []
    depth_parts: list[NDArray[np.float64]] = []
    time_parts: list[NDArray[np.float64]] = []
    simulation_parts: list[NDArray[np.int32]] = []
    frame_parts: list[NDArray[np.int32]] = []
    records: list[SimulationCommitRecord] = []
    first_row = 0

    for local_index, (simulation_id, outcome) in enumerate(
        zip(simulation_ids, outcomes, strict=True)
    ):
        decision = outcome.decision
        row_count = 0
        simulation_first_row = -1
        rows = outcome.rows
        if rows is not None:
            eta = np.asarray(rows.eta, dtype=np.float32)
            if eta.ndim != 2:
                raise ValueError("accepted eta must have shape (time, space)")
            xi = np.asarray(rows.xi, dtype=np.float32)
            gxi = np.asarray(rows.gxi, dtype=np.float32)
            time = np.asarray(rows.time, dtype=np.float64)
            row_count = eta.shape[0]
            simulation_first_row = first_row
            eta_parts.append(eta)
            xi_parts.append(xi)
            gxi_parts.append(gxi)
            depth_parts.append(np.full(row_count, rows.depth, dtype=np.float64))
            time_parts.append(time)
            simulation_parts.append(np.full(row_count, local_index, dtype=np.int32))
            frame_parts.append(np.arange(row_count, dtype=np.int32))
            first_row += row_count

        records.append(
            SimulationCommitRecord(
                simulation_id=int(simulation_id),
                accepted=decision.accepted,
                required_bits=int(decision.required),
                evaluated_bits=int(decision.evaluated),
                failed_bits=int(decision.failed),
                first_row=simulation_first_row,
                row_count=row_count,
                metrics=dict(outcome.metrics),
            )
        )

    shard = (
        DatasetShardArrays(
            eta=np.concatenate(eta_parts),
            xi=np.concatenate(xi_parts),
            gxi=np.concatenate(gxi_parts),
            depth=np.concatenate(depth_parts),
            time=np.concatenate(time_parts),
            simulation_local_index=np.concatenate(simulation_parts),
            frame_index=np.concatenate(frame_parts),
        )
        if eta_parts
        else None
    )
    save_completed_batch(
        path,
        plan=batch_plan,
        shard=shard,
        simulations=records,
        metadata=metadata,
    )
