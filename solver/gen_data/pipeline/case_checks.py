"""Record which checks ran and failed for a generated case."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from operator import index
from typing import SupportsIndex


class CaseCheck(IntFlag):
    # idk wtf the ai is doing but ok
    NONE = 0
    NONFINITE_STATE = 1
    NONFINITE_TARGET = 2
    BOTTOM_CLEARANCE = 4
    HAMILTONIAN_DRIFT = 8
    GL2_STAGE_RESIDUAL = 16
    TEMPORAL_DEFECT = 32
    IC_GRID_DEFECT = 64
    LABEL_GRID_DEFECT = 128
    OUTSIDE_SUPPORT = 256
    INCOMPLETE_TRAJECTORY = 512


def checks_from_bits(bits: SupportsIndex) -> tuple[CaseCheck, ...]:
    """Return the individual failure reasons stored in a bit mask."""

    mask = CaseCheck(index(bits))
    return tuple(reason for reason in CaseCheck if reason & mask)


@dataclass(frozen=True)
class CaseCheckResult:
    """Checks required, checks run, and checks failed for one case."""

    required: CaseCheck
    evaluated: CaseCheck
    failed: CaseCheck

    def __post_init__(self) -> None:
        if self.failed & ~self.evaluated:
            raise ValueError("failed checks must also be marked as evaluated")

    @classmethod
    def from_bits(
        cls,
        required_bits: SupportsIndex,
        evaluated_bits: SupportsIndex,
        failed_bits: SupportsIndex,
    ) -> CaseCheckResult:
        return cls(
            required=CaseCheck(index(required_bits)),
            evaluated=CaseCheck(index(evaluated_bits)),
            failed=CaseCheck(index(failed_bits)),
        )

    @property
    def missing(self) -> CaseCheck:
        return self.required & ~self.evaluated

    @property
    def accepted(self) -> bool:
        return not self.missing and not (self.failed & self.required)
