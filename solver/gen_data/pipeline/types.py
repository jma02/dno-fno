"""Structured records shared by the dataset-generation pipeline."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Final, NamedTuple, TypeAlias, TypedDict

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


FloatArray: TypeAlias = NDArray[np.floating[Any]]
SimulationRows = NamedTuple(
    "SimulationRows",
    [
        ("eta", FloatArray),
        ("xi", FloatArray),
        ("gxi", FloatArray),
        ("depth", float),
        ("time", FloatArray),
    ],
)


class DatasetShardArrays(TypedDict):
    """Training rows stored for the accepted simulations in one batch."""

    eta: NDArray[np.float32]
    xi: NDArray[np.float32]
    gxi: NDArray[np.float32]
    depth: NDArray[np.float64]
    time: NDArray[np.float64]
    simulation_local_index: NDArray[np.int32]
    frame_index: NDArray[np.int32]
