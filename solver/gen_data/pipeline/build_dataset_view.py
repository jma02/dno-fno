"""Use completed batches as one dataset without merging their data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_storage import load_completed_batch
from solver.gen_data.pipeline.artifact_io import write_json_atomic, write_npz_atomic
from solver.gen_data.pipeline.types import DatasetSplit


DatasetViewPaths = NamedTuple(
    "DatasetViewPaths", [("manifest", Path), ("trajectory_map", Path)]
)


def build_dataset_view(
    root: Path,
    batches: Sequence[Path],
    *,
    name: str = "paper_dataset",
    length: float = 2.0 * math.pi,
) -> DatasetViewPaths:
    """Write the dataset's JSON file list and NPZ row-to-simulation map.

    Batches keep the supplied order. Rejected simulations are recorded
    but contribute no data rows.
    """

    if not name or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in name
    ):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    resolved_batches = tuple(batch.resolve() for batch in batches)
    if len(set(resolved_batches)) != len(resolved_batches):
        raise ValueError("completed batch paths must be unique")
    output_paths = DatasetViewPaths(
        manifest=root / f"{name}.dataset.json",
        trajectory_map=root / f"{name}.trajectory_map.npz",
    )
    output_directory = root.resolve()
    map_parts: defaultdict[str, list[NDArray[Any]]] = defaultdict(list)
    shard_records: list[dict[str, object]] = []
    total_rows = 0
    total_simulations = 0
    spatial_size: int | None = None
    next_simulation_id: defaultdict[tuple[int, DatasetSplit], int] = defaultdict(int)

    for batch_path in resolved_batches:
        batch = load_completed_batch(batch_path)
        family_id = int(batch.family_id)
        dataset_split = batch.dataset_split
        number_of_simulations = len(batch.parameter_group_ids)

        if total_simulations + number_of_simulations > np.iinfo(np.int32).max:
            raise ValueError("trajectory count exceeds the int32 map capacity")
        first_rows = np.full(number_of_simulations, -1, dtype=np.int64)
        row_counts = np.zeros(number_of_simulations, dtype=np.int32)
        if batch.shard is not None:
            shard = batch.shard
            shard_row_count, current_spatial_size = shard["eta"].shape
            if spatial_size is not None and current_spatial_size != spatial_size:
                raise ValueError("all dataset shards must use the same spatial grid")
            spatial_size = current_spatial_size
            shard_index = len(shard_records)
            shard_records.append(
                {
                    "path": os.path.relpath(batch_path, start=output_directory),
                    "n_rows": shard_row_count,
                }
            )
            local_index = shard["simulation_local_index"]
            accepted_indices, first_local_rows, rows_per_simulation = np.unique(
                local_index, return_index=True, return_counts=True
            )
            first_rows[accepted_indices] = total_rows + first_local_rows
            row_counts[accepted_indices] = rows_per_simulation
            # Each saved row points to its simulation in the list of all attempts.
            map_parts["trajectory_index"].append(local_index + total_simulations)
            map_parts["frame_index"].append(shard["frame_index"])
            map_parts["shard_index"].append(
                np.full(shard_row_count, shard_index, dtype=np.int32)
            )
            map_parts["shard_row"].append(np.arange(shard_row_count, dtype=np.int64))
            total_rows += shard_row_count

        # Continue simulation IDs across batches of the same family and split.
        family_split = (family_id, dataset_split)
        first_simulation_id = next_simulation_id[family_split]
        next_simulation_id[family_split] += number_of_simulations
        for field, values in {
            "trajectory_family_id": np.full(
                number_of_simulations, family_id, dtype=np.int16
            ),
            "trajectory_dataset_split": np.full(
                number_of_simulations, dataset_split.value
            ),
            "trajectory_simulation_id": np.arange(
                first_simulation_id,
                next_simulation_id[family_split],
                dtype=np.int64,
            ),
            "trajectory_parameter_group_id": np.asarray(batch.parameter_group_ids),
            "trajectory_accepted": batch.accepted_simulations,
            "trajectory_first_row": first_rows,
            "trajectory_row_count": row_counts,
        }.items():
            map_parts[field].append(values)

        total_simulations += number_of_simulations

    if spatial_size is None:
        raise ValueError("a dataset view must contain at least one accepted row")
    trajectory_map = {
        field: np.concatenate(parts) for field, parts in map_parts.items()
    }
    write_npz_atomic(output_paths.trajectory_map, trajectory_map)
    dataset_split_array = trajectory_map["trajectory_dataset_split"]
    accepted_array = trajectory_map["trajectory_accepted"]
    manifest = {
        "dataset_shards": shard_records,
        "trajectory_map_npz": output_paths.trajectory_map.name,
        "n_rows": total_rows,
        "n_trajectories": total_simulations,
        "n_accepted_trajectories": int(np.count_nonzero(accepted_array)),
        "n_accepted_rows": total_rows,
        "grid": {"length": float(length), "nx": spatial_size},
        "split_counts": {
            split.value: {
                "attempted": int(np.count_nonzero(dataset_split_array == split.value)),
                "accepted": int(
                    np.count_nonzero(
                        (dataset_split_array == split.value) & accepted_array
                    )
                ),
            }
            for split in DatasetSplit
        },
    }
    write_json_atomic(output_paths.manifest, manifest)
    return output_paths
