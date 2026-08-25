"""Quality-check vocabulary and acceptance semantics for generated data."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from operator import index
from typing import SupportsIndex, Tuple


class QualityScope(str, Enum):
    """Level at which a quality decision applies."""

    SAMPLE = "sample"
    TRAJECTORY = "trajectory"
    GENERATOR_REVISION = "generator_revision"


class QualityReason(IntFlag):
    """Stable uint32 bits shared by evaluated and failed quality masks.

    A bit in ``evaluated`` means that the corresponding check was run. The
    same bit in ``failed`` records that the evaluated check failed. Explicit
    values are part of the persisted-data contract and must not be reordered.
    """

    NONE = 0
    NONFINITE_STATE = 1 << 0
    NONFINITE_TARGET = 1 << 1
    BOTTOM_CLEARANCE = 1 << 2
    HAMILTONIAN_DRIFT = 1 << 3
    GL2_STAGE_RESIDUAL = 1 << 4
    TEMPORAL_DEFECT = 1 << 5
    IC_GRID_DEFECT = 1 << 6
    LABEL_GRID_DEFECT = 1 << 7
    OUTSIDE_SUPPORT = 1 << 8
    INCOMPLETE_TRAJECTORY = 1 << 9


_KNOWN_REASON_BITS = sum(int(reason) for reason in QualityReason)
_UINT32_MAX = (1 << 32) - 1


def _reason_mask_from_bits(
    bits: SupportsIndex, field_name: str
) -> QualityReason:
    if isinstance(bits, bool):
        raise TypeError(f"{field_name} must be integer-like")
    try:
        value = index(bits)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be integer-like") from exc
    if value < 0 or value > _UINT32_MAX:
        raise ValueError(f"{field_name} must fit in uint32")
    unknown = value & ~_KNOWN_REASON_BITS
    if unknown:
        raise ValueError(f"{field_name} contains unknown quality bits: {unknown:#x}")
    return QualityReason(value)


def reasons_from_bits(bits: SupportsIndex) -> Tuple[QualityReason, ...]:
    """Decode a persisted uint32 value into its individual reason flags."""

    mask = _reason_mask_from_bits(bits, "bits")
    return tuple(
        reason
        for reason in QualityReason
        if reason is not QualityReason.NONE and reason & mask
    )


@dataclass(frozen=True)
class QualityDecision:
    """Result of applying a declared set of checks at one quality scope."""

    scope: QualityScope
    required: QualityReason
    evaluated: QualityReason
    failed: QualityReason

    def __post_init__(self) -> None:
        if not isinstance(self.scope, QualityScope):
            raise TypeError("scope must be a QualityScope")
        for field_name, mask in (
            ("required", self.required),
            ("evaluated", self.evaluated),
            ("failed", self.failed),
        ):
            if not isinstance(mask, QualityReason):
                raise TypeError(f"{field_name} must be a QualityReason mask")
            _reason_mask_from_bits(int(mask), field_name)
        if (self.failed & self.evaluated) != self.failed:
            raise ValueError("failed checks must also be marked as evaluated")

    @classmethod
    def from_bits(
        cls,
        scope: QualityScope,
        required_bits: SupportsIndex,
        evaluated_bits: SupportsIndex,
        failed_bits: SupportsIndex,
    ) -> "QualityDecision":
        """Build a decision from masks read from a persisted sidecar."""

        return cls(
            scope=scope,
            required=_reason_mask_from_bits(required_bits, "required_bits"),
            evaluated=_reason_mask_from_bits(evaluated_bits, "evaluated_bits"),
            failed=_reason_mask_from_bits(failed_bits, "failed_bits"),
        )

    @property
    def missing(self) -> QualityReason:
        """Required checks for which no result was recorded."""

        return QualityReason(int(self.required) & ~int(self.evaluated))

    @property
    def accepted(self) -> bool:
        """Whether every required check ran and every required check passed."""

        required_failures = self.failed & self.required
        return (
            self.missing == QualityReason.NONE
            and required_failures == QualityReason.NONE
        )

    @property
    def required_bits(self) -> int:
        return int(self.required)

    @property
    def evaluated_bits(self) -> int:
        return int(self.evaluated)

    @property
    def failed_bits(self) -> int:
        return int(self.failed)
