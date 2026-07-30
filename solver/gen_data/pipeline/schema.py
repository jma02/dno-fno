"""Stable source and trajectory identifiers for generated data."""
from __future__ import annotations

from dataclasses import dataclass


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class GeneratorRevisionRecord:
    """Identity of the generator revision used for one physical family."""

    family_id: str
    generator_revision_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "family_id")
        _require_nonempty(self.generator_revision_id, "generator_revision_id")


@dataclass(frozen=True)
class TrajectoryRecord:
    """Identity of one candidate trajectory within a generator revision."""

    generator_revision: GeneratorRevisionRecord
    trajectory_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.trajectory_id, "trajectory_id")


@dataclass(frozen=True)
class SampleRecord:
    """Identity of one stored frame from a candidate trajectory."""

    trajectory: TrajectoryRecord
    frame_index: int

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
