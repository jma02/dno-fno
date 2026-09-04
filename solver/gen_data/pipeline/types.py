"""Structured records shared by the dataset-generation pipeline."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Final, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PhysicalFamilyId(IntEnum):
    STOKES = 1
    TANAKA = 2
    BENJAMIN_FEIR = 3
    JONSWAP_TMA = 4


ROOT_SEED_BY_DATASET_SPLIT: Final = {
    DatasetSplit.TRAIN: 2026072210,
    DatasetSplit.VALIDATION: 2026072204,
    DatasetSplit.TEST: 2026072205,
}


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class BatchPlanArrays(TypedDict):
    """Arrays describing the simulations assigned to one batch."""

    family_id: NDArray[np.int16]
    dataset_split: NDArray[np.str_]
    parameter_group_id: NDArray[np.str_]
    simulation_spec_json: NDArray[np.str_]


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
    trajectory_parameter_group_id: NDArray[np.str_]
    trajectory_accepted: NDArray[np.bool_]
    trajectory_first_row: NDArray[np.int64]
    trajectory_row_count: NDArray[np.int32]
