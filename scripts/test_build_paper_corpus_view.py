"""No-I/O structural tests for combined paper-corpus chunk planning."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from scripts.build_paper_corpus_view import (
    CompletedChunk,
    FAMILY_ORDER,
    _configuration_record,
    _ordered_cell_ids,
    validate_combined_plan,
)
from scripts.run_paper_corpus_quota import (
    GenerationRequest,
    build_run_spec,
)
from solver.gen_data.pipeline.production import SplitId
from solver.gen_data.pipeline.quota_driver import canonical_json_sha256


def _chunk(
    family: str,
    *,
    before: int,
    count: int,
    stream_id: int,
    suffix: str,
    split: SplitId = SplitId.TRAIN,
) -> CompletedChunk:
    root = Path(f"/tmp/{split.value}_{family}_{suffix}")
    identity = f"{split.value}:{family}:{before}:{count}:{stream_id}:{suffix}"
    return CompletedChunk(
        summary_path=root / f"{family}.summary.json",
        summary_sha256=hashlib.sha256(
            f"summary:{identity}".encode()
        ).hexdigest(),
        root=root,
        family=family,
        split=split,
        stream_id=stream_id,
        accepted_before=before,
        accepted_count=count,
        accepted_after=before + count,
        attempted_count=count + 1,
        fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
        dependency_fingerprint="d" * 64,
        source_fingerprint=f"{FAMILY_ORDER.index(family) + 1}" * 64,
        source_sha256={"shared.py": "e" * 64},
        execution_platform="cpu",
        batches=(),
    )


class CombinedPaperCorpusViewTests(unittest.TestCase):
    def test_completed_chunk_accepts_frozen_ordered_cell_sequence(self) -> None:
        request = GenerationRequest(
            output_root=Path("/tmp/frozen_ordered_cells"),
            family="stokes",
            split=SplitId.VALIDATION,
            accepted_cases=1,
            batch_size=1,
            accepted_cases_before=0,
            platform="cpu",
            stream_id=0,
            first_attempt_index=0,
        )
        spec = build_run_spec(request)

        self.assertIsInstance(
            spec.configuration["ordered_cell_ids"],
            tuple,
        )
        self.assertEqual(
            _ordered_cell_ids(spec.configuration),
            (
                "finite_low",
                "finite_moderate",
                "deep_low",
                "deep_moderate",
            ),
        )
        configuration = _configuration_record(spec)
        self.assertIsInstance(configuration["ordered_cell_ids"], list)
        self.assertEqual(
            len(canonical_json_sha256(configuration)),
            64,
        )

    def test_equal_nested_chunks_have_expected_case_and_row_counts(self) -> None:
        chunks = tuple(
            chunk
            for family in FAMILY_ORDER
            for chunk in (
                _chunk(
                    family,
                    before=0,
                    count=2_048,
                    stream_id=0,
                    suffix="a",
                ),
                _chunk(
                    family,
                    before=2_048,
                    count=2_048,
                    stream_id=1,
                    suffix="b",
                ),
            )
        )
        plan = validate_combined_plan(chunks)

        self.assertEqual(
            plan.accepted_cases_per_family_by_split,
            {"train": 4_096},
        )
        self.assertEqual(plan.accepted_cases, 4 * 4_096)
        self.assertEqual(plan.expected_rows, 417 * 4_096)
        self.assertEqual(plan.attempted_cases, 4 * (2_049 + 2_049))
        self.assertEqual(
            tuple(chunk.family for chunk in plan.chunks[:2]),
            ("stokes", "stokes"),
        )

    def test_train_validation_and_test_combine_in_one_view(self) -> None:
        chunks = tuple(
            _chunk(
                family,
                before=0,
                count=count,
                stream_id=0,
                suffix=f"{split.value[0]}{index}",
                split=split,
            )
            for split, count in (
                (SplitId.TRAIN, 32),
                (SplitId.VALIDATION, 8),
                (SplitId.TEST, 4),
            )
            for index, family in enumerate(FAMILY_ORDER)
        )
        plan = validate_combined_plan(chunks)

        self.assertEqual(
            plan.splits,
            (SplitId.TRAIN, SplitId.VALIDATION, SplitId.TEST),
        )
        self.assertEqual(
            plan.accepted_cases_per_family_by_split,
            {"train": 32, "validation": 8, "test": 4},
        )
        self.assertEqual(plan.accepted_cases, 4 * 44)
        self.assertEqual(plan.expected_rows, 417 * 44)

    def test_gap_duplicate_stream_and_unequal_family_total_fail(self) -> None:
        baseline = [
            _chunk(
                family,
                before=0,
                count=16,
                stream_id=0,
                suffix=chr(ord("a") + index),
            )
            for index, family in enumerate(FAMILY_ORDER)
        ]

        gap = [
            *baseline,
            _chunk(
                "stokes",
                before=17,
                count=1,
                stream_id=1,
                suffix="z",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            validate_combined_plan(gap)

        repeated_stream = [
            *baseline,
            _chunk(
                "stokes",
                before=16,
                count=1,
                stream_id=0,
                suffix="y",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "repeat a stream"):
            validate_combined_plan(repeated_stream)

        unequal = [
            chunk
            for chunk in baseline
            if chunk.family != "jonswap_tma"
        ]
        unequal.append(
            _chunk(
                "jonswap_tma",
                before=0,
                count=15,
                stream_id=0,
                suffix="x",
            )
        )
        with self.assertRaisesRegex(ValueError, "equal accepted-case counts"):
            validate_combined_plan(unequal)

        missing_test_family = [
            _chunk(
                family,
                before=0,
                count=4,
                stream_id=0,
                suffix=f"m{index}",
                split=SplitId.TEST,
            )
            for index, family in enumerate(FAMILY_ORDER[:-1])
        ]
        with self.assertRaisesRegex(
            ValueError,
            "test view requires all four",
        ):
            validate_combined_plan([*baseline, *missing_test_family])


if __name__ == "__main__":
    unittest.main()
