"""Deterministic case allocation for the paper-dataset generators.

This module deliberately contains no numerical solver or archive logic.  It
only fixes a split, a parameter cell, and a reproducible random-stream key for
each attempted case before construction or integration begins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Final

import numpy as np


class SplitId(str, Enum):
    """A case-level dataset split."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PhysicalFamilyId(IntEnum):
    """Stable integer identifiers for the four paper-dataset families."""

    STOKES = 1
    TANAKA = 2
    BENJAMIN_FEIR = 3
    JONSWAP_TMA = 4


PAPER_DATASET_REVISION_BY_FAMILY: Final[Mapping[PhysicalFamilyId, int]] = (
    MappingProxyType(
        {
            PhysicalFamilyId.STOKES: 2,
            PhysicalFamilyId.TANAKA: 3,
            PhysicalFamilyId.BENJAMIN_FEIR: 4,
            PhysicalFamilyId.JONSWAP_TMA: 4,
        }
    )
)


def paper_dataset_revision_id(family_id: PhysicalFamilyId) -> int:
    """Return the current generator revision for one physical family."""

    if not isinstance(family_id, PhysicalFamilyId):
        raise TypeError("family_id must be a PhysicalFamilyId")
    return PAPER_DATASET_REVISION_BY_FAMILY[family_id]


_SPLIT_ROOTS = {
    SplitId.TRAIN: 2026072210,
    SplitId.VALIDATION: 2026072204,
    SplitId.TEST: 2026072205,
}
_SPLIT_CODES = {
    SplitId.TRAIN: 0,
    SplitId.VALIDATION: 1,
    SplitId.TEST: 2,
}
_ATTEMPT_INDEX_BITS = 40
_STREAM_ID_BITS = 20


def split_root(split_id: SplitId) -> int:
    """Return the predeclared PCG64 root seed for ``split_id``."""

    return _SPLIT_ROOTS[split_id]


def split_code(split_id: SplitId) -> int:
    """Return the persisted integer code for ``split_id``."""

    return _SPLIT_CODES[split_id]


@dataclass(frozen=True, order=True)
class CaseKey:
    """Stable identity and random-stream coordinates of one attempted case."""

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
            raise ValueError(
                f"attempt_index must fit in {_ATTEMPT_INDEX_BITS} bits"
            )

    @property
    def root_seed(self) -> int:
        """Root seed fixed by the case's split."""

        return split_root(self.split_id)

    @property
    def seed_words(self) -> tuple[int, int, int, int, int]:
        """Seed-sequence words for deterministic reconstruction of the case."""

        return (
            self.root_seed,
            self.family_id,
            self.revision_id,
            self.stream_id,
            self.attempt_index,
        )

    @property
    def case_id(self) -> int:
        """Return a collision-free int64 ID for split, stream, and attempt.

        Family and generator revision remain separate identity fields in the
        archive.  The bit layout is
        ``[split:2][stream:20][attempt:40]``.
        """

        return (
            (split_code(self.split_id) << (_STREAM_ID_BITS + _ATTEMPT_INDEX_BITS))
            | (self.stream_id << _ATTEMPT_INDEX_BITS)
            | self.attempt_index
        )


def random_generator_for_case(case_key: CaseKey) -> np.random.Generator:
    """Create the reproducible NumPy PCG64 generator for one attempted case."""

    seed_sequence = np.random.SeedSequence(case_key.seed_words)
    # PCG64 is NumPy's 64-bit Permuted Congruential Generator. Keeping the
    # algorithm fixed makes every case's random draws exactly reproducible.
    return np.random.Generator(np.random.PCG64(seed_sequence))


@dataclass(frozen=True)
class CellQuota:
    """Required accepted-case count for one declared parameter cell."""

    cell_id: str
    target_accepted: int

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must not be empty")
        if self.target_accepted < 0:
            raise ValueError("target_accepted must be nonnegative")


@dataclass(frozen=True)
class AttemptAssignment:
    """Cell and split assignment made before an attempt's outcome is known."""

    case_key: CaseKey
    cell_id: str

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must not be empty")


def balanced_cell_quotas(
    cell_ids: Sequence[str],
    *,
    accepted_case_count: int,
) -> tuple[CellQuota, ...]:
    """Divide an accepted-case total across cells by quotient and remainder.

    If ``accepted_case_count = q * len(cell_ids) + r``, the first ``r`` cells
    receive ``q + 1`` cases and every other cell receives ``q``.  The caller's
    cell order is therefore part of the reproducible dataset specification.
    """

    if accepted_case_count < 0:
        raise ValueError("accepted_case_count must be nonnegative")
    if not cell_ids:
        raise ValueError("cell_ids must not be empty")
    if any(not cell_id for cell_id in cell_ids):
        raise ValueError("cell_ids must not contain empty values")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")

    quotient, remainder = divmod(accepted_case_count, len(cell_ids))
    return tuple(
        CellQuota(
            cell_id=cell_id,
            target_accepted=quotient + int(index < remainder),
        )
        for index, cell_id in enumerate(cell_ids)
    )


def schedule_attempt_batch(
    quotas: Sequence[CellQuota],
    accepted_by_cell: Mapping[str, int],
    *,
    family_id: int,
    revision_id: int,
    split_id: SplitId,
    stream_id: int,
    first_attempt_index: int,
    batch_size: int,
) -> tuple[AttemptAssignment, ...]:
    """Schedule at most ``batch_size`` whole cases against remaining quotas.

    The function is called only after the preceding batch has resolved.  A
    rejected attempt does not increase ``accepted_by_cell``; consequently its
    original cell retains that place in the next batch.  Planned assignments
    never exceed a cell's remaining accepted quota, so the last batch can be
    shorter than ``batch_size`` without truncating a case.
    """

    if not quotas:
        raise ValueError("quotas must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if stream_id < 0:
        raise ValueError("stream_id must be nonnegative")
    if first_attempt_index < 0:
        raise ValueError("first_attempt_index must be nonnegative")

    cell_ids = tuple(quota.cell_id for quota in quotas)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("quota cell_ids must be unique")

    unknown_cells = set(accepted_by_cell).difference(cell_ids)
    if unknown_cells:
        names = ", ".join(sorted(unknown_cells))
        raise ValueError(f"accepted_by_cell contains unknown cells: {names}")

    remaining_by_cell: dict[str, int] = {}
    for quota in quotas:
        accepted = accepted_by_cell.get(quota.cell_id, 0)
        if accepted < 0:
            raise ValueError("accepted case counts must be nonnegative")
        if accepted > quota.target_accepted:
            raise ValueError(f"accepted count exceeds quota for cell {quota.cell_id!r}")
        remaining_by_cell[quota.cell_id] = quota.target_accepted - accepted

    accepted_to_skip = {
        quota.cell_id: quota.target_accepted - remaining_by_cell[quota.cell_id]
        for quota in quotas
    }
    scheduled_cells: list[str] = []
    for quota_level in range(max(quota.target_accepted for quota in quotas)):
        for quota in quotas:
            if quota.target_accepted <= quota_level:
                continue
            if accepted_to_skip[quota.cell_id] > 0:
                accepted_to_skip[quota.cell_id] -= 1
                continue
            scheduled_cells.append(quota.cell_id)
            if len(scheduled_cells) == batch_size:
                break
        if len(scheduled_cells) == batch_size:
            break

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
