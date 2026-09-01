"""Create reproducible simulation IDs and assign attempts across parameter groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import islice
from typing import Final, NamedTuple

import numpy as np


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
DATASET_SPLIT_INDEX: Final = {
    DatasetSplit.TRAIN: 0,
    DatasetSplit.VALIDATION: 1,
    DatasetSplit.TEST: 2,
}

_WORKER_STREAMS_PER_SPLIT = 1_048_576
_ATTEMPTS_PER_WORKER_STREAM = 1_099_511_627_776
_SIMULATION_IDS_PER_SPLIT = _WORKER_STREAMS_PER_SPLIT * _ATTEMPTS_PER_WORKER_STREAM
_MAX_STORED_ID = 32_767
_MAX_WORKER_STREAM_ID = _WORKER_STREAMS_PER_SPLIT - 1
_MAX_ATTEMPT_INDEX = _ATTEMPTS_PER_WORKER_STREAM - 1


@dataclass(frozen=True, order=True)
class SimulationKey:
    """Everything needed to reproduce one simulation's ID and random draws."""

    family_id: int
    dataset_split: DatasetSplit
    # Gives each concurrently generated dataset chunk a separate ID/RNG sequence.
    worker_stream_id: int
    # Candidate position within that sequence; rejected attempts still consume one.
    attempt_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.family_id <= _MAX_STORED_ID:
            raise ValueError(f"family_id must be between 0 and {_MAX_STORED_ID}")
        if not 0 <= self.worker_stream_id <= _MAX_WORKER_STREAM_ID:
            raise ValueError(
                f"worker_stream_id must be between 0 and {_MAX_WORKER_STREAM_ID}"
            )
        if not 0 <= self.attempt_index <= _MAX_ATTEMPT_INDEX:
            raise ValueError(
                f"attempt_index must be between 0 and {_MAX_ATTEMPT_INDEX}"
            )

    @property
    def seed_words(self) -> tuple[int, int, int, int]:
        return (
            ROOT_SEED_BY_DATASET_SPLIT[self.dataset_split],
            self.family_id,
            self.worker_stream_id,
            self.attempt_index,
        )

    @property
    def simulation_id(self) -> int:
        # Each split owns a range of IDs, subdivided by generation worker.
        return (
            DATASET_SPLIT_INDEX[self.dataset_split] * _SIMULATION_IDS_PER_SPLIT
            + self.worker_stream_id * _ATTEMPTS_PER_WORKER_STREAM
            + self.attempt_index
        )


def random_generator_for_simulation(
    simulation_key: SimulationKey,
) -> np.random.Generator:
    # PCG64 is NumPy's 64-bit Permuted Congruential Generator. Keeping the
    # algorithm fixed makes every simulation's random draws exactly reproducible.
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(simulation_key.seed_words))
    )


class ParameterGroupTarget(NamedTuple):
    """How many successful simulations one parameter group needs."""

    parameter_group_id: str
    simulation_count: int


class AttemptAssignment(NamedTuple):
    """The identity and parameter group for one solver attempt."""

    simulation_key: SimulationKey
    parameter_group_id: str

    def to_json_record(self) -> dict[str, object]:
        key = self.simulation_key
        return {
            "simulation_id": key.simulation_id,
            "family_id": key.family_id,
            "dataset_split": key.dataset_split.value,
            "worker_stream_id": key.worker_stream_id,
            "attempt_index": key.attempt_index,
            "parameter_group_id": self.parameter_group_id,
        }


def balanced_simulation_targets(
    parameter_group_ids: Sequence[str],
    *,
    simulation_count: int,
) -> tuple[ParameterGroupTarget, ...]:
    """Divide successful simulations evenly across parameter groups."""

    if simulation_count < 0:
        raise ValueError("simulation_count must be nonnegative")
    if not parameter_group_ids:
        raise ValueError("parameter_group_ids must not be empty")
    if any(not parameter_group_id for parameter_group_id in parameter_group_ids):
        raise ValueError("parameter_group_ids must not contain empty values")
    if len(set(parameter_group_ids)) != len(parameter_group_ids):
        raise ValueError("parameter_group_ids must be unique")

    quotient, remainder = divmod(simulation_count, len(parameter_group_ids))
    return tuple(
        ParameterGroupTarget(
            parameter_group_id=parameter_group_id,
            simulation_count=quotient + int(index < remainder),
        )
        for index, parameter_group_id in enumerate(parameter_group_ids)
    )


def build_next_attempt_batch(
    targets: Sequence[ParameterGroupTarget],
    accepted_simulation_counts: Mapping[str, int],
    simulation_attempt_counts: Mapping[str, int],
    attempt_limits_by_parameter_group: Mapping[str, int],
    *,
    family_id: int,
    dataset_split: DatasetSplit,
    worker_stream_id: int,
    first_attempt_index: int,
    batch_size: int,
) -> tuple[AttemptAssignment, ...]:
    """Build a balanced attempt batch without exceeding retry limits."""

    schedulable_target_by_parameter_group = {
        target.parameter_group_id: accepted_simulation_counts.get(
            target.parameter_group_id, 0
        )
        + min(
            target.simulation_count
            - accepted_simulation_counts.get(target.parameter_group_id, 0),
            attempt_limits_by_parameter_group[target.parameter_group_id]
            - simulation_attempt_counts.get(target.parameter_group_id, 0),
        )
        for target in targets
    }
    scheduled_parameter_groups = islice(
        (
            target.parameter_group_id
            for level in range(
                max(schedulable_target_by_parameter_group.values(), default=0)
            )
            for target in targets
            if (
                accepted_simulation_counts.get(target.parameter_group_id, 0)
                <= level
                < schedulable_target_by_parameter_group[target.parameter_group_id]
            )
        ),
        batch_size,
    )

    return tuple(
        AttemptAssignment(
            simulation_key=SimulationKey(
                family_id=family_id,
                dataset_split=dataset_split,
                worker_stream_id=worker_stream_id,
                attempt_index=first_attempt_index + offset,
            ),
            parameter_group_id=parameter_group_id,
        )
        for offset, parameter_group_id in enumerate(scheduled_parameter_groups)
    )
