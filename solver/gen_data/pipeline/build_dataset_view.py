"""Build a loader-facing view from completed paper-dataset batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_storage import (
    load_completed_batch,
    simulation_row_blocks,
)
from solver.gen_data.pipeline.artifact_io import (
    write_json_atomic,
    write_npz_atomic,
)
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    JsonObject,
    JsonValue,
    SimulationIndexArrays,
)


DATASET_VIEW_SCHEMA_VERSION = 2
TRAJECTORY_MAP_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DatasetViewPaths:
    """Files created by :func:`build_dataset_view`."""

    # JSON listing the dataset shards, grid settings, and split sizes.
    manifest: Path
    # NPZ mapping stored rows to simulations and recording each simulation's result.
    trajectory_map: Path


def build_dataset_view(
    root: Path,
    batches: Sequence[Path],
    *,
    name: str = "paper_dataset",
    length: float = 2.0 * math.pi,
) -> DatasetViewPaths:
    """Write a training manifest and attempted-simulation trajectory map.

    ``batches`` defines the immutable shard order. A batch with no accepted
    simulations legitimately contains no row arrays.
    """

    if not name or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in name
    ):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not batches:
        raise ValueError("at least one completed batch is required")
    resolved_batches = tuple(batch.resolve() for batch in batches)
    if len(set(resolved_batches)) != len(resolved_batches):
        raise ValueError("completed batch paths must be unique")
    output_paths = DatasetViewPaths(
        manifest=root / f"{name}.dataset.json",
        trajectory_map=root / f"{name}.trajectory_map.npz",
    )
    row_trajectory_parts: list[NDArray[np.int32]] = []
    row_frame_parts: list[NDArray[np.int32]] = []
    row_shard_parts: list[NDArray[np.int32]] = []
    row_shard_row_parts: list[NDArray[np.int64]] = []
    family_ids: list[int] = []
    dataset_splits: list[str] = []
    simulation_ids: list[int] = []
    parameter_group_ids: list[str] = []
    accepted_simulations: list[bool] = []
    first_rows: list[int] = []
    row_counts: list[int] = []
    shard_records: list[JsonValue] = []
    global_row_count = 0
    trajectory_offset = 0
    spatial_size: int | None = None
    next_simulation_id: dict[tuple[int, DatasetSplit], int] = {}

    for batch_path in batches:
        batch = load_completed_batch(batch_path)
        family_id = int(batch.family_id)
        dataset_split = batch.dataset_split
        number_of_simulations = len(batch.parameter_group_ids)

        shard_index: int | None = None
        shard_row_count = 0
        row_blocks: dict[int, tuple[int, int]] = {}
        if batch.shard is not None:
            shard = batch.shard
            current_spatial_size = int(shard["eta"].shape[1])
            if spatial_size is None:
                spatial_size = current_spatial_size
            elif current_spatial_size != spatial_size:
                raise ValueError("all dataset shards must use the same spatial grid")
            shard_index = len(shard_records)
            shard_row_count = int(shard["eta"].shape[0])
            shard_records.append(
                {
                    "path": os.path.relpath(
                        batch_path.resolve(),
                        start=output_paths.manifest.parent.resolve(),
                    ),
                    "n_rows": shard_row_count,
                }
            )
            local_index = shard["simulation_local_index"]
            row_blocks = simulation_row_blocks(
                local_index,
                number_of_simulations=number_of_simulations,
            )
            row_trajectory_parts.append(
                np.asarray(local_index + trajectory_offset, dtype=np.int32)
            )
            row_frame_parts.append(np.asarray(shard["frame_index"], dtype=np.int32))
            row_shard_parts.append(
                np.full(shard_row_count, shard_index, dtype=np.int32)
            )
            row_shard_row_parts.append(np.arange(shard_row_count, dtype=np.int64))
        family_ids.extend([family_id] * number_of_simulations)
        dataset_splits.extend([dataset_split.value] * number_of_simulations)
        simulation_group = (family_id, dataset_split)
        first_simulation_id = next_simulation_id.get(simulation_group, 0)
        simulation_ids.extend(
            range(first_simulation_id, first_simulation_id + number_of_simulations)
        )
        next_simulation_id[simulation_group] = (
            first_simulation_id + number_of_simulations
        )
        parameter_group_ids.extend(batch.parameter_group_ids)
        accepted_simulations.extend(map(bool, batch.accepted_simulations))
        for local_index in range(number_of_simulations):
            first_row, row_count = row_blocks.get(local_index, (-1, 0))
            first_rows.append(global_row_count + first_row if first_row >= 0 else -1)
            row_counts.append(row_count)

        global_row_count += shard_row_count
        trajectory_offset += number_of_simulations

    if spatial_size is None:
        raise ValueError("a dataset view must contain at least one accepted row")
    if trajectory_offset > np.iinfo(np.int32).max:
        raise ValueError("trajectory count exceeds the int32 map capacity")
    if (
        len(set(zip(family_ids, dataset_splits, simulation_ids, strict=True)))
        != trajectory_offset
    ):
        raise ValueError(
            "compound simulation IDs (family, split, simulation) must be unique"
        )

    dataset_split_array = np.asarray(dataset_splits)
    accepted_array = np.asarray(accepted_simulations, dtype=np.bool_)
    trajectory_map: SimulationIndexArrays = {
        "schema_version": np.asarray(
            TRAJECTORY_MAP_SCHEMA_VERSION,
            dtype=np.int16,
        ),
        "trajectory_index": np.concatenate(row_trajectory_parts),
        "frame_index": np.concatenate(row_frame_parts),
        "shard_index": np.concatenate(row_shard_parts),
        "shard_row": np.concatenate(row_shard_row_parts),
        "trajectory_family_id": np.asarray(family_ids, dtype=np.int16),
        "trajectory_dataset_split": dataset_split_array,
        "trajectory_simulation_id": np.asarray(simulation_ids, dtype=np.int64),
        "trajectory_parameter_group_id": np.asarray(parameter_group_ids),
        "trajectory_accepted": accepted_array,
        "trajectory_first_row": np.asarray(first_rows, dtype=np.int64),
        "trajectory_row_count": np.asarray(
            row_counts,
            dtype=np.int32,
        ),
    }
    write_npz_atomic(
        output_paths.trajectory_map,
        cast(Mapping[str, NDArray[Any]], trajectory_map),
    )

    split_counts: JsonObject = {
        dataset_split.value: {
            "attempted": int(
                np.count_nonzero(dataset_split_array == dataset_split.value)
            ),
            "accepted": int(
                np.count_nonzero(
                    (dataset_split_array == dataset_split.value) & accepted_array
                )
            ),
        }
        for dataset_split in DatasetSplit
    }
    manifest: JsonObject = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "dataset_shards": shard_records,
        "trajectory_map_npz": output_paths.trajectory_map.name,
        "requires_trajectory_map": True,
        "n_rows": global_row_count,
        "n_trajectories": trajectory_offset,
        "n_accepted_trajectories": int(np.count_nonzero(accepted_array)),
        "n_accepted_rows": global_row_count,
        "grid": {
            "length": float(length),
            "nx": spatial_size,
        },
        "split_counts": split_counts,
    }
    write_json_atomic(output_paths.manifest, manifest)
    return output_paths
