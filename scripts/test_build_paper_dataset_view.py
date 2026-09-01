"""Tests for ordering completed chunks into one training view."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.build_paper_dataset_view import (
    FAMILY_ORDER,
    CompletedChunk,
    preflight,
    validate_combined_plan,
)
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit


def _chunk(
    root: Path,
    *,
    family: str,
    split: DatasetSplit,
    accepted_before: int = 0,
    accepted_count: int = 8,
    worker_stream_id: int = 0,
) -> CompletedChunk:
    return CompletedChunk(
        summary_path=(root / f"{split.value}_{family}_{accepted_before}.summary.json"),
        root=(root / f"{split.value}_{family}_{accepted_before}"),
        family=family,
        split=split,
        worker_stream_id=worker_stream_id,
        accepted_before=accepted_before,
        accepted_count=accepted_count,
        accepted_after=accepted_before + accepted_count,
        attempted_count=accepted_count + 1,
        batches=(),
    )


def _one_split(root: Path, *, count: int = 8) -> tuple[CompletedChunk, ...]:
    return tuple(
        _chunk(root, family=family, split=DatasetSplit.TRAIN, accepted_count=count)
        for family in FAMILY_ORDER
    )


class CombinedPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_equal_family_chunks_form_one_ordered_plan(self) -> None:
        chunks = tuple(reversed(_one_split(self.root)))
        plan = validate_combined_plan(chunks)

        self.assertEqual(
            tuple(chunk.family for chunk in plan.chunks),
            FAMILY_ORDER,
        )
        self.assertEqual(plan.splits, (DatasetSplit.TRAIN,))
        self.assertEqual(plan.accepted_simulations_per_family_by_split, {"train": 8})
        self.assertEqual(plan.accepted_simulations, 32)
        self.assertEqual(plan.attempted_simulations, 36)
        self.assertEqual(plan.expected_rows, 8 * (1 + 200 + 200 + 16))

    def test_additive_chunks_must_be_contiguous(self) -> None:
        chunks = list(_one_split(self.root))
        tanaka = chunks[1]
        chunks[1] = replace(tanaka, accepted_before=2, accepted_after=10)
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            validate_combined_plan(chunks)

    def test_each_split_requires_all_families_with_equal_totals(self) -> None:
        with self.assertRaisesRegex(ValueError, "all four"):
            validate_combined_plan(_one_split(self.root)[:-1])

        chunks = list(_one_split(self.root))
        chunks[-1] = replace(
            chunks[-1],
            accepted_count=7,
            accepted_after=7,
            attempted_count=8,
        )
        with self.assertRaisesRegex(ValueError, "equal accepted counts"):
            validate_combined_plan(chunks)

    def test_additive_chunks_need_distinct_streams_and_roots(self) -> None:
        chunks = list(_one_split(self.root))
        first = chunks[0]
        second = replace(
            first,
            summary_path=self.root / "second.summary.json",
            accepted_before=8,
            accepted_after=16,
        )
        with self.assertRaisesRegex(ValueError, "distinct worker streams and roots"):
            validate_combined_plan((*chunks, second))

    def test_preflight_is_write_free_and_has_plain_chunk_records(self) -> None:
        chunks = _one_split(self.root)
        output = self.root / "view"
        with mock.patch(
            "scripts.build_paper_dataset_view.load_completed_chunk",
            side_effect=chunks,
        ):
            plan, resolved, name, record = preflight(
                [chunk.summary_path for chunk in chunks],
                output_root=output,
                name="combined",
            )

        self.assertEqual(plan.accepted_simulations, 32)
        self.assertEqual(resolved, output.resolve())
        self.assertEqual(name, "combined")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
