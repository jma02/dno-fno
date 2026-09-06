"""Structured records shared by the dataset-generation pipeline."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Final, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict


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


# Values are requested successful simulations; omitted parameter groups are not requested.
StokesRequestedSimulationsPerGroup = TypedDict(
    "StokesRequestedSimulationsPerGroup",
    {
        "finite_low": int,
        "finite_moderate": int,
        "deep_low": int,
        "deep_moderate": int,
    },
    total=False,
    closed=True,
)

TanakaRequestedSimulationsPerGroup = TypedDict(
    "TanakaRequestedSimulationsPerGroup",
    {
        "main_m1_q0": int,
        "main_m1_q1": int,
        "main_m2_q0": int,
        "main_m2_q1": int,
        "main_m2_q2": int,
        "main_m3_q0": int,
        "main_m3_q1": int,
        "main_m3_q2": int,
        "main_m3_q3": int,
        "steep_m1_q0": int,
        "steep_m1_q1": int,
    },
    total=False,
    closed=True,
)

BenjaminFeirRequestedSimulationsPerGroup = TypedDict(
    "BenjaminFeirRequestedSimulationsPerGroup",
    {
        "n_c_04__delta_n_01": int,
        "n_c_05__delta_n_01": int,
        "n_c_06__delta_n_01": int,
        "n_c_06__delta_n_02": int,
        "n_c_07__delta_n_01": int,
        "n_c_07__delta_n_02": int,
        "n_c_08__delta_n_01": int,
        "n_c_08__delta_n_02": int,
        "n_c_09__delta_n_01": int,
        "n_c_09__delta_n_02": int,
        "n_c_09__delta_n_03": int,
        "n_c_10__delta_n_01": int,
        "n_c_10__delta_n_02": int,
        "n_c_10__delta_n_03": int,
        "n_c_11__delta_n_01": int,
        "n_c_11__delta_n_02": int,
        "n_c_11__delta_n_03": int,
        "n_c_11__delta_n_04": int,
        "n_c_12__delta_n_01": int,
        "n_c_12__delta_n_02": int,
        "n_c_12__delta_n_03": int,
        "n_c_12__delta_n_04": int,
        "n_c_13__delta_n_01": int,
        "n_c_13__delta_n_02": int,
        "n_c_13__delta_n_03": int,
        "n_c_13__delta_n_04": int,
        "n_c_14__delta_n_01": int,
        "n_c_14__delta_n_02": int,
        "n_c_14__delta_n_03": int,
        "n_c_14__delta_n_04": int,
        "n_c_14__delta_n_05": int,
        "n_c_15__delta_n_01": int,
        "n_c_15__delta_n_02": int,
        "n_c_15__delta_n_03": int,
        "n_c_15__delta_n_04": int,
        "n_c_15__delta_n_05": int,
        "n_c_16__delta_n_01": int,
        "n_c_16__delta_n_02": int,
        "n_c_16__delta_n_03": int,
        "n_c_16__delta_n_04": int,
        "n_c_16__delta_n_05": int,
        "n_c_17__delta_n_01": int,
        "n_c_17__delta_n_02": int,
        "n_c_17__delta_n_03": int,
        "n_c_17__delta_n_04": int,
        "n_c_17__delta_n_05": int,
        "n_c_17__delta_n_06": int,
        "n_c_18__delta_n_01": int,
        "n_c_18__delta_n_02": int,
        "n_c_18__delta_n_03": int,
        "n_c_18__delta_n_04": int,
        "n_c_18__delta_n_05": int,
        "n_c_18__delta_n_06": int,
        "n_c_19__delta_n_01": int,
        "n_c_19__delta_n_02": int,
        "n_c_19__delta_n_03": int,
        "n_c_19__delta_n_04": int,
        "n_c_19__delta_n_05": int,
        "n_c_19__delta_n_06": int,
        "n_c_20__delta_n_01": int,
        "n_c_20__delta_n_02": int,
        "n_c_20__delta_n_03": int,
        "n_c_20__delta_n_04": int,
        "n_c_20__delta_n_05": int,
        "n_c_20__delta_n_06": int,
        "n_c_20__delta_n_07": int,
    },
    total=False,
    closed=True,
)

JonswapTmaRequestedSimulationsPerGroup = TypedDict(
    "JonswapTmaRequestedSimulationsPerGroup",
    {
        "shallow__gamma_1__right_0": int,
        "shallow__gamma_1__right_0p5": int,
        "shallow__gamma_1__right_1": int,
        "shallow__gamma_3p3__right_0": int,
        "shallow__gamma_3p3__right_0p5": int,
        "shallow__gamma_3p3__right_1": int,
        "shallow__gamma_5__right_0": int,
        "shallow__gamma_5__right_0p5": int,
        "shallow__gamma_5__right_1": int,
        "finite__gamma_1__right_0": int,
        "finite__gamma_1__right_0p5": int,
        "finite__gamma_1__right_1": int,
        "finite__gamma_3p3__right_0": int,
        "finite__gamma_3p3__right_0p5": int,
        "finite__gamma_3p3__right_1": int,
        "finite__gamma_5__right_0": int,
        "finite__gamma_5__right_0p5": int,
        "finite__gamma_5__right_1": int,
        "deep__gamma_1__right_0": int,
        "deep__gamma_1__right_0p5": int,
        "deep__gamma_1__right_1": int,
        "deep__gamma_3p3__right_0": int,
        "deep__gamma_3p3__right_0p5": int,
        "deep__gamma_3p3__right_1": int,
        "deep__gamma_5__right_0": int,
        "deep__gamma_5__right_0p5": int,
        "deep__gamma_5__right_1": int,
    },
    total=False,
    closed=True,
)

RequestedSimulationsPerGroup: TypeAlias = (
    StokesRequestedSimulationsPerGroup
    | TanakaRequestedSimulationsPerGroup
    | BenjaminFeirRequestedSimulationsPerGroup
    | JonswapTmaRequestedSimulationsPerGroup
)


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
