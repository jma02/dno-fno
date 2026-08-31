"""Save simulation assignments, accepted rows, and quality decisions by batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import json_text
from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    SimulationCommitRecord,
    commit_batch,
    save_batch_plan,
    save_shard,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    SPLIT_CODE_BY_ID,
)
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult


FloatArray: TypeAlias = NDArray[np.floating]
JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class AcceptedSimulationRows:
    """Rows retained from one accepted static state or trajectory."""

    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    depth: float
    time: FloatArray
    selected_dense_index: NDArray[np.integer]


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
    cell_codes: Mapping[str, int],
    batch_id: int,
    metadata: Mapping[str, object],
) -> dict[str, NDArray[Any]]:
    """Build the exact batch plan saved before numerical construction."""

    if not assignments:
        raise ValueError("assignments must not be empty")
    if len(assignments) != len(specifications):
        raise ValueError("assignments and specifications must have equal lengths")
    if batch_id < 0:
        raise ValueError("batch_id must be nonnegative")

    first_key = assignments[0].simulation_key
    shared_coordinates = (
        first_key.family_id,
        first_key.revision_id,
        first_key.split_id,
    )
    if any(
        (
            assignment.simulation_key.family_id,
            assignment.simulation_key.revision_id,
            assignment.simulation_key.split_id,
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
        "split_id": np.asarray(
            SPLIT_CODE_BY_ID[first_key.split_id], dtype=np.uint8
        ),
        "batch_id": np.asarray(batch_id, dtype=np.int64),
        "simulation_id": np.asarray(
            [assignment.simulation_key.simulation_id for assignment in assignments],
            dtype=np.int64,
        ),
        "cell_id": encoded_cell_codes,
        "root_seed": np.asarray(
            [assignment.simulation_key.root_seed for assignment in assignments],
            dtype=np.uint64,
        ),
        "stream_id": np.asarray(
            [assignment.simulation_key.stream_id for assignment in assignments],
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
    paths: BatchPaths,
    batch_plan: Mapping[str, NDArray[Any]],
    outcomes: Sequence[SimulationOutcome],
    *,
    metadata: Mapping[str, object],
) -> None:
    """Persist accepted whole-simulation rows and every attempted decision."""

    simulation_ids = np.asarray(batch_plan["simulation_id"], dtype=np.int64)
    if len(outcomes) != simulation_ids.size:
        raise ValueError("outcomes must contain every planned simulation")
    save_batch_plan(paths, batch_plan)

    eta_parts: list[NDArray[np.float32]] = []
    xi_parts: list[NDArray[np.float32]] = []
    gxi_parts: list[NDArray[np.float32]] = []
    depth_parts: list[NDArray[np.float64]] = []
    time_parts: list[NDArray[np.float64]] = []
    simulation_parts: list[NDArray[np.int32]] = []
    frame_parts: list[NDArray[np.int32]] = []
    selected_parts: list[NDArray[np.int32]] = []
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
            selected = np.asarray(rows.selected_dense_index, dtype=np.int32)
            row_count = eta.shape[0]
            simulation_first_row = first_row
            eta_parts.append(eta)
            xi_parts.append(xi)
            gxi_parts.append(gxi)
            depth_parts.append(np.full(row_count, rows.depth, dtype=np.float64))
            time_parts.append(time)
            simulation_parts.append(np.full(row_count, local_index, dtype=np.int32))
            frame_parts.append(np.arange(row_count, dtype=np.int32))
            selected_parts.append(selected)
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

    if eta_parts:
        save_shard(
            paths,
            {
                "eta": np.concatenate(eta_parts),
                "xi": np.concatenate(xi_parts),
                "gxi": np.concatenate(gxi_parts),
                "depth": np.concatenate(depth_parts),
                "time": np.concatenate(time_parts),
                "simulation_local_index": np.concatenate(simulation_parts),
                "frame_index": np.concatenate(frame_parts),
                "selected_dense_index": np.concatenate(selected_parts),
            },
        )
    commit_batch(paths, simulations=records, metadata=metadata)
