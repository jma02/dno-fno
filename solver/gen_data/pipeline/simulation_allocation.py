"""Create reproducible simulation IDs and assign attempts across parameter groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import islice
from typing import Final, NamedTuple

import numpy as np


class SplitId(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PhysicalFamilyId(IntEnum):
    STOKES = 1
    TANAKA = 2
    BENJAMIN_FEIR = 3
    JONSWAP_TMA = 4


SPLIT_ROOT_SEED_BY_ID: Final = {
    SplitId.TRAIN: 2026072210,
    SplitId.VALIDATION: 2026072204,
    SplitId.TEST: 2026072205,
}
SPLIT_CODE_BY_ID: Final = {
    SplitId.TRAIN: 0,
    SplitId.VALIDATION: 1,
    SplitId.TEST: 2,
}


DATASET_REVISION_BY_FAMILY: Final = {
    PhysicalFamilyId.STOKES: 2,
    PhysicalFamilyId.TANAKA: 3,
    PhysicalFamilyId.BENJAMIN_FEIR: 4,
    PhysicalFamilyId.JONSWAP_TMA: 4,
}


_STREAMS_PER_SPLIT = 1_048_576
_ATTEMPTS_PER_STREAM = 1_099_511_627_776
_SIMULATION_IDS_PER_SPLIT = _STREAMS_PER_SPLIT * _ATTEMPTS_PER_STREAM
_MAX_STORED_ID = 32_767
_MAX_STREAM_ID = _STREAMS_PER_SPLIT - 1
_MAX_ATTEMPT_INDEX = _ATTEMPTS_PER_STREAM - 1


@dataclass(frozen=True, order=True)
class SimulationKey:
    """Everything needed to reproduce one simulation's ID and random draws."""

    family_id: int
    revision_id: int
    split_id: SplitId
    stream_id: int
    attempt_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.family_id <= _MAX_STORED_ID:
            raise ValueError(f"family_id must be between 0 and {_MAX_STORED_ID}")
        if not 0 <= self.revision_id <= _MAX_STORED_ID:
            raise ValueError(f"revision_id must be between 0 and {_MAX_STORED_ID}")
        if not 0 <= self.stream_id <= _MAX_STREAM_ID:
            raise ValueError(f"stream_id must be between 0 and {_MAX_STREAM_ID}")
        if not 0 <= self.attempt_index <= _MAX_ATTEMPT_INDEX:
            raise ValueError(
                f"attempt_index must be between 0 and {_MAX_ATTEMPT_INDEX}"
            )

    @property
    def root_seed(self) -> int:
        return SPLIT_ROOT_SEED_BY_ID[self.split_id]

    @property
    def seed_words(self) -> tuple[int, int, int, int, int]:
        return (
            self.root_seed,
            self.family_id,
            self.revision_id,
            self.stream_id,
            self.attempt_index,
        )

    @property
    def simulation_id(self) -> int:
        # Each split owns a range of IDs, subdivided into one range per stream.
        return (
            SPLIT_CODE_BY_ID[self.split_id] * _SIMULATION_IDS_PER_SPLIT
            + self.stream_id * _ATTEMPTS_PER_STREAM
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


class SampleCellTarget(NamedTuple):
    """How many successful simulations one parameter cell needs."""

    cell_id: str
    simulation_count: int


class AttemptAssignment(NamedTuple):
    """The identity and parameter cell for one solver attempt."""

    simulation_key: SimulationKey
    cell_id: str

    def to_json_record(self) -> dict[str, object]:
        key = self.simulation_key
        return {
            "simulation_id": key.simulation_id,
            "family_id": key.family_id,
            "revision_id": key.revision_id,
            "split_id": key.split_id.value,
            "root_seed": key.root_seed,
            "stream_id": key.stream_id,
            "attempt_index": key.attempt_index,
            "seed_words": list(key.seed_words),
            "cell_id": self.cell_id,
        }


def balanced_simulation_targets(
    cell_ids: Sequence[str],
    *,
    simulation_count: int,
) -> tuple[SampleCellTarget, ...]:
    """Divide successful simulations as evenly as possible across the cells."""

    if simulation_count < 0:
        raise ValueError("simulation_count must be nonnegative")
    if not cell_ids:
        raise ValueError("cell_ids must not be empty")
    if any(not cell_id for cell_id in cell_ids):
        raise ValueError("cell_ids must not contain empty values")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")

    quotient, remainder = divmod(simulation_count, len(cell_ids))
    return tuple(
        SampleCellTarget(
            cell_id=cell_id,
            simulation_count=quotient + int(index < remainder),
        )
        for index, cell_id in enumerate(cell_ids)
    )


def build_next_attempt_batch(
    targets: Sequence[SampleCellTarget],
    accepted_simulation_counts: Mapping[str, int],
    simulation_attempt_counts: Mapping[str, int],
    attempt_limits_by_parameter_group: Mapping[str, int],
    *,
    family_id: int,
    revision_id: int,
    split_id: SplitId,
    stream_id: int,
    first_attempt_index: int,
    batch_size: int,
) -> tuple[AttemptAssignment, ...]:
    """Build a balanced attempt batch without exceeding retry limits."""

    schedulable_target_by_parameter_group = {
        target.cell_id: accepted_simulation_counts.get(target.cell_id, 0)
        + min(
            target.simulation_count - accepted_simulation_counts.get(target.cell_id, 0),
            attempt_limits_by_parameter_group[target.cell_id]
            - simulation_attempt_counts.get(target.cell_id, 0),
        )
        for target in targets
    }
    scheduled_parameter_groups = islice(
        (
            target.cell_id
            for level in range(
                max(schedulable_target_by_parameter_group.values(), default=0)
            )
            for target in targets
            if (
                accepted_simulation_counts.get(target.cell_id, 0)
                <= level
                < schedulable_target_by_parameter_group[target.cell_id]
            )
        ),
        batch_size,
    )

    return tuple(
        AttemptAssignment(
            simulation_key=SimulationKey(
                family_id=family_id,
                revision_id=revision_id,
                split_id=split_id,
                stream_id=stream_id,
                attempt_index=first_attempt_index + offset,
            ),
            cell_id=parameter_group_id,
        )
        for offset, parameter_group_id in enumerate(scheduled_parameter_groups)
    )
