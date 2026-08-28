"""Create reproducible case IDs and assign attempts across parameter groups."""

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


DATASET_REVISION_BY_FAMILY: Final = {
    PhysicalFamilyId.STOKES: 2,
    PhysicalFamilyId.TANAKA: 3,
    PhysicalFamilyId.BENJAMIN_FEIR: 4,
    PhysicalFamilyId.JONSWAP_TMA: 4,
}


_ATTEMPT_INDEX_BITS = 40
_STREAM_ID_BITS = 20


def split_root(split_id: SplitId) -> int:
    if split_id is SplitId.TRAIN:
        return 2026072210
    if split_id is SplitId.VALIDATION:
        return 2026072204
    if split_id is SplitId.TEST:
        return 2026072205
    raise ValueError(f"unknown split: {split_id!r}")


def split_code(split_id: SplitId) -> int:
    if split_id is SplitId.TRAIN:
        return 0
    if split_id is SplitId.VALIDATION:
        return 1
    if split_id is SplitId.TEST:
        return 2
    raise ValueError(f"unknown split: {split_id!r}")


@dataclass(frozen=True, order=True)
class CaseKey:
    """Everything needed to reproduce one case's ID and random draws."""

    family_id: int
    revision_id: int
    split_id: SplitId
    stream_id: int
    attempt_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.family_id < 1 << 15:
            raise ValueError("family_id must fit in nonnegative int16")
        if not 0 <= self.revision_id < 1 << 15:
            raise ValueError("revision_id must fit in nonnegative int16")
        if not 0 <= self.stream_id < 1 << _STREAM_ID_BITS:
            raise ValueError(f"stream_id must fit in {_STREAM_ID_BITS} bits")
        if not 0 <= self.attempt_index < 1 << _ATTEMPT_INDEX_BITS:
            raise ValueError(f"attempt_index must fit in {_ATTEMPT_INDEX_BITS} bits")

    @property
    def root_seed(self) -> int:
        return split_root(self.split_id)

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
    def case_id(self) -> int:
        # This layout lets parallel workers make IDs without sharing a counter.
        # Family and revision are stored separately in each batch.
        return (
            (split_code(self.split_id) << (_STREAM_ID_BITS + _ATTEMPT_INDEX_BITS))
            | (self.stream_id << _ATTEMPT_INDEX_BITS)
            | self.attempt_index
        )


def random_generator_for_case(case_key: CaseKey) -> np.random.Generator:
    # PCG64 is NumPy's 64-bit Permuted Congruential Generator. Keeping the
    # algorithm fixed makes every case's random draws exactly reproducible.
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(case_key.seed_words))
    )


class SampleCellTarget(NamedTuple):
    """How many successful cases one parameter cell needs."""

    cell_id: str
    case_count: int


class AttemptAssignment(NamedTuple):
    """The identity and parameter cell for one solver attempt."""

    case_key: CaseKey
    cell_id: str

    def to_json_record(self) -> dict[str, object]:
        key = self.case_key
        return {
            "case_id": key.case_id,
            "family_id": key.family_id,
            "revision_id": key.revision_id,
            "split_id": key.split_id.value,
            "root_seed": key.root_seed,
            "stream_id": key.stream_id,
            "attempt_index": key.attempt_index,
            "seed_words": list(key.seed_words),
            "cell_id": self.cell_id,
        }


def balanced_valid_case_targets(
    cell_ids: Sequence[str],
    *,
    case_count: int,
) -> tuple[SampleCellTarget, ...]:
    """Divide successful cases as evenly as possible across the cells."""

    if case_count < 0:
        raise ValueError("case_count must be nonnegative")
    if not cell_ids:
        raise ValueError("cell_ids must not be empty")
    if any(not cell_id for cell_id in cell_ids):
        raise ValueError("cell_ids must not contain empty values")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")

    quotient, remainder = divmod(case_count, len(cell_ids))
    return tuple(
        SampleCellTarget(
            cell_id=cell_id,
            case_count=quotient + int(index < remainder),
        )
        for index, cell_id in enumerate(cell_ids)
    )


def assign_next_cases(
    targets: Sequence[SampleCellTarget],
    accepted_by_cell: Mapping[str, int],
    *,
    family_id: int,
    revision_id: int,
    split_id: SplitId,
    stream_id: int,
    first_attempt_index: int,
    batch_size: int,
) -> tuple[AttemptAssignment, ...]:
    """Fill the next batch with cells that still need successful cases."""

    # Scan quota levels instead of filling one cell at a time. This preserves
    # the caller's cell order and spreads every batch across the cells.
    scheduled_cells = islice(
        (
            target.cell_id
            for level in range(max(target.case_count for target in targets))
            for target in targets
            if accepted_by_cell.get(target.cell_id, 0) <= level < target.case_count
        ),
        batch_size,
    )

    return tuple(
        AttemptAssignment(
            case_key=CaseKey(
                family_id=family_id,
                revision_id=revision_id,
                split_id=split_id,
                stream_id=stream_id,
                attempt_index=first_attempt_index + offset,
            ),
            cell_id=cell_id,
        )
        for offset, cell_id in enumerate(scheduled_cells)
    )
