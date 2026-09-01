"""Structured records shared by the dataset-generation pipeline."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class BatchPlanArrays(TypedDict):
    """Arrays stored before a batch's simulations are executed."""

    family_id: NDArray[np.int16]
    dataset_split: NDArray[np.str_]
    parameter_group_id: NDArray[np.int32]
    # Distinguishes independently generated chunks within one split.
    worker_stream_id: NDArray[np.uint32]
    # Candidate position in that worker stream, including rejected candidates.
    attempt_index: NDArray[np.uint64]
    simulation_spec_json: NDArray[np.str_]
    metadata_json: NDArray[np.str_]


class DatasetShardArrays(TypedDict):
    """Training rows stored for the accepted simulations in one batch."""

    eta: NDArray[np.float32]
    xi: NDArray[np.float32]
    gxi: NDArray[np.float32]
    depth: NDArray[np.float64]
    time: NDArray[np.float64]
    simulation_local_index: NDArray[np.int32]
    frame_index: NDArray[np.int32]


class SimulationIndexArrays(TypedDict):
    """Row ownership and acceptance data written beside the dataset manifest."""

    schema_version: NDArray[np.int16]
    trajectory_index: NDArray[np.int32]
    frame_index: NDArray[np.int32]
    shard_index: NDArray[np.int32]
    shard_row: NDArray[np.int64]
    trajectory_family_id: NDArray[np.int16]
    trajectory_dataset_split: NDArray[np.str_]
    trajectory_simulation_id: NDArray[np.int64]
    trajectory_parameter_group_id: NDArray[np.int32]
    trajectory_accepted: NDArray[np.bool_]
    trajectory_required_bits: NDArray[np.uint32]
    trajectory_evaluated_bits: NDArray[np.uint32]
    trajectory_failed_bits: NDArray[np.uint32]
    trajectory_first_row: NDArray[np.int64]
    trajectory_row_count: NDArray[np.int32]
