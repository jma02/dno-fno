"""Record which checks ran and failed for a generated simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from operator import index
from typing import SupportsIndex


class SimulationCheck(IntFlag):
    """Reasons a generated simulation can be rejected."""

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


def checks_from_bits(bits: SupportsIndex) -> tuple[SimulationCheck, ...]:
    """Return the individual failure reasons stored in a bit mask."""

    mask = SimulationCheck(index(bits))
    return tuple(reason for reason in SimulationCheck if reason & mask)


@dataclass(frozen=True)
class SimulationCheckResult:
    """Checks required, checks run, and checks failed for one simulation."""

    required: SimulationCheck
    evaluated: SimulationCheck
    failed: SimulationCheck

    def __post_init__(self) -> None:
        if self.failed & ~self.evaluated:
            raise ValueError("failed checks must also be marked as evaluated")

    @classmethod
    def from_bits(
        cls,
        required_bits: SupportsIndex,
        evaluated_bits: SupportsIndex,
        failed_bits: SupportsIndex,
    ) -> SimulationCheckResult:
        return cls(
            required=SimulationCheck(index(required_bits)),
            evaluated=SimulationCheck(index(evaluated_bits)),
            failed=SimulationCheck(index(failed_bits)),
        )

    @property
    def missing(self) -> SimulationCheck:
        return self.required & ~self.evaluated

    @property
    def accepted(self) -> bool:
        return not self.missing and not (self.failed & self.required)
