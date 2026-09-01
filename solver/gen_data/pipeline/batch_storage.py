"""Read and write one complete artifact per simulation batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import (
    json_text,
    load_npz,
    parse_json,
    write_npz_atomic,
)
from solver.gen_data.pipeline.batch_artifacts import (
    BATCH_PLAN_FIELDS,
    SHARD_FIELDS,
    SimulationCommitRecord,
    compute_batch_simulation_ids,
    compute_simulation_row_blocks,
    parse_simulation_result,
    validate_batch_plan,
    validate_shard,
)
from solver.gen_data.pipeline.types import (
    BatchPlanArrays,
    DatasetShardArrays,
    JsonObject,
)

_PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_RESULTS_JSON = "simulation_results_json"
_BATCH_METADATA_JSON = "batch_metadata_json"
_COMPLETED_BATCH_FIELDS = (
    BATCH_PLAN_FIELDS
    | SHARD_FIELDS
    | {
        _RESULTS_JSON,
        _BATCH_METADATA_JSON,
    }
)


@dataclass(frozen=True)
class CompletedBatch:
    """Validated contents of one finished simulation batch."""

    plan: BatchPlanArrays
    shard: DatasetShardArrays | None
    simulations: tuple[SimulationCommitRecord, ...]
    metadata: JsonObject


def batch_path(
    root: Path,
    *,
    family: str,
    split: str,
    batch_id: int,
) -> Path:
    """Return the standard path for one completed batch."""

    for value, field_name in ((family, "family"), (split, "split")):
        if _PATH_COMPONENT_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"{field_name} must contain only letters, digits, '_' or '-'"
            )
    if batch_id < 0:
        raise ValueError("batch_id must be nonnegative")
    return root / "batches" / family / split / f"batch_{batch_id:06d}.npz"


def _validate_simulation_results(
    plan: BatchPlanArrays,
    shard: DatasetShardArrays | None,
    simulations: Sequence[SimulationCommitRecord],
) -> None:
    simulation_ids = compute_batch_simulation_ids(plan)
    if len(simulations) != simulation_ids.size:
        raise ValueError("results must contain every planned simulation")
    if not np.array_equal(
        np.asarray(
            [simulation.simulation_id for simulation in simulations],
            dtype=np.int64,
        ),
        simulation_ids,
    ):
        raise ValueError("results must preserve planned simulation order")

    row_blocks = (
        compute_simulation_row_blocks(
            shard["simulation_local_index"],
            number_of_simulations=simulation_ids.size,
        )
        if shard is not None
        else {}
    )
    for local_index, simulation in enumerate(simulations):
        declared_rows = (
            (simulation.first_row, simulation.row_count)
            if simulation.accepted
            else None
        )
        if declared_rows != row_blocks.get(local_index):
            raise ValueError(
                f"simulation {simulation.simulation_id} rows do not match the shard"
            )


def save_completed_batch(
    path: Path,
    *,
    plan: BatchPlanArrays,
    shard: DatasetShardArrays | None,
    simulations: Sequence[SimulationCommitRecord],
    metadata: Mapping[str, object],
) -> None:
    """Atomically save a finished batch; interrupted batches leave no artifact."""

    if path.exists():
        raise FileExistsError(f"completed batch already exists: {path}")
    validate_batch_plan(plan)
    if shard is not None:
        validate_shard(shard, batch_plan=plan)
    _validate_simulation_results(plan, shard, simulations)

    arrays: dict[str, NDArray[Any]] = {
        name: np.asarray(value) for name, value in plan.items()
    }
    if shard is not None:
        arrays.update({name: np.asarray(value) for name, value in shard.items()})
    arrays[_RESULTS_JSON] = np.asarray(
        json_text([simulation.to_json_record() for simulation in simulations])
    )
    arrays[_BATCH_METADATA_JSON] = np.asarray(json_text(metadata))
    write_npz_atomic(path, arrays)


def load_completed_batch(path: Path) -> CompletedBatch:
    """Load and validate one completed batch artifact."""

    arrays = load_npz(path)
    unexpected = arrays.keys() - _COMPLETED_BATCH_FIELDS
    if unexpected:
        raise ValueError(
            f"completed batch contains unexpected arrays {sorted(unexpected)}"
        )

    plan = cast(
        BatchPlanArrays,
        {name: arrays[name] for name in BATCH_PLAN_FIELDS if name in arrays},
    )
    validate_batch_plan(plan)

    present_shard_fields = arrays.keys() & SHARD_FIELDS
    if present_shard_fields and present_shard_fields != SHARD_FIELDS:
        missing = SHARD_FIELDS - present_shard_fields
        raise ValueError(f"completed batch is missing shard arrays {sorted(missing)}")
    shard = (
        cast(DatasetShardArrays, {name: arrays[name] for name in SHARD_FIELDS})
        if present_shard_fields
        else None
    )
    if shard is not None:
        validate_shard(shard, batch_plan=plan)

    for name in (_RESULTS_JSON, _BATCH_METADATA_JSON):
        if name not in arrays or arrays[name].ndim != 0:
            raise ValueError(f"completed batch requires scalar {name}")
    raw_results = parse_json(str(arrays[_RESULTS_JSON].item()))
    if not isinstance(raw_results, list):
        raise ValueError("simulation_results_json must encode a list")
    simulations = tuple(parse_simulation_result(value) for value in raw_results)
    _validate_simulation_results(plan, shard, simulations)

    metadata = parse_json(str(arrays[_BATCH_METADATA_JSON].item()))
    if not isinstance(metadata, dict):
        raise ValueError("batch_metadata_json must encode an object")
    return CompletedBatch(
        plan=plan,
        shard=shard,
        simulations=simulations,
        metadata=cast(JsonObject, metadata),
    )
