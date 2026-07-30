"""Dataset-aware source IDs for source-conditioned training losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from solver.gen_data.pipeline.production import PhysicalFamilyId


LEGACY_V9_TANAKA_SOURCE_IDS = (5, 6, 14)
LEGACY_V9_FINITE_TIME_SOURCE_IDS = (5, 6, 14, 7, 8, 9)


@dataclass(frozen=True)
class SourceConditioning:
    """Source IDs with the semantics of one dataset representation."""

    dataset_schema: str
    tanaka_source_ids: tuple[int, ...]
    finite_time_source_ids: tuple[int, ...]


def source_conditioning_for_dataset(
    dataset: Mapping[str, object],
) -> SourceConditioning:
    """Return source IDs without mixing legacy labels and physical-family IDs.

    Schema-v2 views expose the stable physical family ID as ``source`` and
    always carry the case-level ``split_id`` field.  Flat legacy datasets do
    not carry that field; in v9, integer source 2 means linear data rather than
    Tanaka data.
    """
    if "split_id" in dataset:
        return SourceConditioning(
            dataset_schema="paper_corpus_v2",
            tanaka_source_ids=(int(PhysicalFamilyId.TANAKA),),
            finite_time_source_ids=(
                int(PhysicalFamilyId.TANAKA),
                int(PhysicalFamilyId.BENJAMIN_FEIR),
            ),
        )
    return SourceConditioning(
        dataset_schema="legacy",
        tanaka_source_ids=LEGACY_V9_TANAKA_SOURCE_IDS,
        finite_time_source_ids=LEGACY_V9_FINITE_TIME_SOURCE_IDS,
    )


def parse_source_ids(
    value: str | None,
    *,
    automatic: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve an optional comma-separated override against dataset defaults."""
    if value is None:
        return automatic
    source_ids = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )
    if not source_ids:
        raise ValueError("source IDs must not be empty")
    return source_ids
