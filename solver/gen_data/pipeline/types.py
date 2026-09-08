"""Structured records shared by the dataset-generation pipeline."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, NamedTuple, TypeAlias, TypedDict

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


# Parameter-group name -> requested successful simulations; names come from the sampler.
RequestedSimulationsPerGroup: TypeAlias = dict[str, int]


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
