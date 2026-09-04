"""Save simulation specifications, accepted rows, and decisions by batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import json_text
from solver.gen_data.pipeline.batch_artifacts import SimulationResult
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.types import (
    BatchPlanArrays,
    DatasetSplit,
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
    parameter_group_ids: Sequence[str],
    specifications: Sequence[Mapping[str, object]],
    *,
    family_id: int,
    dataset_split: DatasetSplit,
) -> BatchPlanArrays:
    """Describe the sampled simulations in one batch."""

    if not parameter_group_ids:
        raise ValueError("parameter_group_ids must not be empty")
    if len(parameter_group_ids) != len(specifications):
        raise ValueError(
            "parameter_group_ids and specifications must have equal lengths"
        )

    return {
        "family_id": np.asarray(family_id, dtype=np.int16),
        "dataset_split": np.asarray(dataset_split.value),
        "parameter_group_id": np.asarray(parameter_group_ids),
        "simulation_spec_json": np.asarray(
            [json_text(specification) for specification in specifications]
        ),
    }


def commit_simulation_outcomes(
    path: Path,
    batch_plan: BatchPlanArrays,
    outcomes: Sequence[SimulationOutcome],
) -> None:
    """Persist accepted whole-simulation rows and every attempted decision."""

    if len(outcomes) != int(batch_plan["simulation_spec_json"].size):
        raise ValueError("outcomes must contain every planned simulation")

    eta_parts: list[NDArray[np.float32]] = []
    xi_parts: list[NDArray[np.float32]] = []
    gxi_parts: list[NDArray[np.float32]] = []
    depth_parts: list[NDArray[np.float64]] = []
    time_parts: list[NDArray[np.float64]] = []
    simulation_parts: list[NDArray[np.int32]] = []
    frame_parts: list[NDArray[np.int32]] = []
    records: list[SimulationResult] = []

    for local_index, outcome in enumerate(outcomes):
        decision = outcome.decision
        rows = outcome.rows
        if rows is not None:
            eta = np.asarray(rows.eta, dtype=np.float32)
            if eta.ndim != 2:
                raise ValueError("accepted eta must have shape (time, space)")
            xi = np.asarray(rows.xi, dtype=np.float32)
            gxi = np.asarray(rows.gxi, dtype=np.float32)
            time = np.asarray(rows.time, dtype=np.float64)
            row_count = eta.shape[0]
            eta_parts.append(eta)
            xi_parts.append(xi)
            gxi_parts.append(gxi)
            depth_parts.append(np.full(row_count, rows.depth, dtype=np.float64))
            time_parts.append(time)
            simulation_parts.append(np.full(row_count, local_index, dtype=np.int32))
            frame_parts.append(np.arange(row_count, dtype=np.int32))

        records.append(
            SimulationResult(
                accepted=decision.accepted,
                failed_checks=decision.failed_checks,
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
    )
