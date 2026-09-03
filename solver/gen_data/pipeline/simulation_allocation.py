"""Assign simulation attempts across parameter groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum, IntEnum
from typing import Final

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


def random_generator_for_attempt(
    *,
    family_id: int,
    dataset_split: DatasetSplit,
    attempt_number: int,
) -> np.random.Generator:
    """Create the random-number generator for one solver attempt."""

    seed_words = (
        ROOT_SEED_BY_DATASET_SPLIT[dataset_split],
        family_id,
        attempt_number,
    )
    # A separate seed for each split, family, and attempt means rerunning an
    # interrupted batch draws the same parameters.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_words)))


def balanced_simulation_targets(
    parameter_group_ids: Sequence[str],
    *,
    simulation_count: int,
) -> dict[str, int]:
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
    return {
        parameter_group_id: quotient + int(index < remainder)
        for index, parameter_group_id in enumerate(parameter_group_ids)
    }


def select_next_parameter_groups(
    simulation_targets: Mapping[str, int],
    accepted_simulation_counts: Mapping[str, int],
    simulation_attempt_counts: Mapping[str, int],
    attempt_limits_by_parameter_group: Mapping[str, int],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    """Select a balanced batch of parameter groups within the retry limits."""

    attempts_to_schedule = {
        parameter_group_id: min(
            target_count - accepted_simulation_counts.get(parameter_group_id, 0),
            attempt_limits_by_parameter_group[parameter_group_id]
            - simulation_attempt_counts.get(parameter_group_id, 0),
        )
        for parameter_group_id, target_count in simulation_targets.items()
    }
    accepted_after_scheduling = {
        parameter_group_id: accepted_simulation_counts.get(parameter_group_id, 0)
        for parameter_group_id in simulation_targets
    }
    scheduled_parameter_groups: list[str] = []
    while len(scheduled_parameter_groups) < batch_size:
        available = tuple(
            parameter_group_id
            for parameter_group_id, count in attempts_to_schedule.items()
            if count > 0
        )
        if not available:
            break
        lowest_accepted_count = min(
            accepted_after_scheduling[parameter_group_id]
            for parameter_group_id in available
        )
        for parameter_group_id in simulation_targets:
            if (
                attempts_to_schedule[parameter_group_id] == 0
                or accepted_after_scheduling[parameter_group_id]
                != lowest_accepted_count
            ):
                continue
            scheduled_parameter_groups.append(parameter_group_id)
            attempts_to_schedule[parameter_group_id] -= 1
            accepted_after_scheduling[parameter_group_id] += 1
            if len(scheduled_parameter_groups) == batch_size:
                break

    return tuple(scheduled_parameter_groups)
