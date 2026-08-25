"""Tests for deterministic case-level production allocation."""

from __future__ import annotations

import unittest

from solver.gen_data.pipeline.production import (
    PAPER_DATASET_REVISION_BY_FAMILY,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
    balanced_valid_case_targets,
    paper_dataset_revision_id,
    schedule_attempt_batch,
    split_code,
    split_root,
)


class FamilyIdentityTest(unittest.TestCase):
    def test_paper_family_ids_and_revisions_are_stable(self) -> None:
        self.assertEqual(
            tuple((family.name, int(family)) for family in PhysicalFamilyId),
            (
                ("STOKES", 1),
                ("TANAKA", 2),
                ("BENJAMIN_FEIR", 3),
                ("JONSWAP_TMA", 4),
            ),
        )
        self.assertEqual(
            dict(PAPER_DATASET_REVISION_BY_FAMILY),
            {
                PhysicalFamilyId.STOKES: 2,
                PhysicalFamilyId.TANAKA: 3,
                PhysicalFamilyId.BENJAMIN_FEIR: 4,
                PhysicalFamilyId.JONSWAP_TMA: 4,
            },
        )
        self.assertEqual(
            {
                family: paper_dataset_revision_id(family)
                for family in PhysicalFamilyId
            },
            dict(PAPER_DATASET_REVISION_BY_FAMILY),
        )

    def test_revision_accessor_requires_a_typed_family(self) -> None:
        with self.assertRaisesRegex(TypeError, "PhysicalFamilyId"):
            paper_dataset_revision_id(1)  # type: ignore[arg-type]


class BalancedValidCaseTargetsTest(unittest.TestCase):
    def test_uses_exact_quotient_and_remainder_balance(self) -> None:
        targets = balanced_valid_case_targets(
            ("shallow", "finite", "deep"),
            case_count=8,
        )

        self.assertEqual(
            [(target.cell_id, target.case_count) for target in targets],
            [("shallow", 3), ("finite", 3), ("deep", 2)],
        )
        self.assertEqual(sum(target.case_count for target in targets), 8)

    def test_total_smaller_than_cell_count_is_deterministic(self) -> None:
        targets = balanced_valid_case_targets(
            ("a", "b", "c", "d"),
            case_count=2,
        )

        self.assertEqual(
            [target.case_count for target in targets],
            [1, 1, 0, 0],
        )

    def test_rejects_duplicate_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            balanced_valid_case_targets(("same", "same"), case_count=2)


class AttemptScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = balanced_valid_case_targets(
            ("low", "moderate"),
            case_count=4,
        )

    def test_replay_has_identical_assignments_and_seed_words(self) -> None:
        arguments = {
            "family_id": 2,
            "revision_id": 1,
            "split_id": SplitId.VALIDATION,
            "stream_id": 9,
            "first_attempt_index": 17,
            "batch_size": 3,
        }

        first = schedule_attempt_batch(self.targets, {}, **arguments)
        replay = schedule_attempt_batch(self.targets, {}, **arguments)

        self.assertEqual(first, replay)
        self.assertEqual(
            [assignment.cell_id for assignment in first],
            ["low", "moderate", "low"],
        )
        self.assertEqual(
            [assignment.case_key.attempt_index for assignment in first],
            [17, 18, 19],
        )
        self.assertTrue(
            all(
                (assignment.case_key.family_id, assignment.case_key.revision_id)
                == (2, 1)
                for assignment in first
            )
        )
        self.assertEqual(
            first[0].case_key.seed_words,
            (2026072204, 2, 1, 9, 17),
        )
        self.assertEqual(
            first[0].case_key.case_id,
            (split_code(SplitId.VALIDATION) << 60) | (9 << 40) | 17,
        )

    def test_case_id_separates_splits_streams_and_attempts(self) -> None:
        keys = (
            CaseKey(2, 1, SplitId.TRAIN, 0, 0),
            CaseKey(2, 1, SplitId.VALIDATION, 0, 0),
            CaseKey(2, 1, SplitId.TEST, 0, 0),
            CaseKey(2, 1, SplitId.TRAIN, 1, 0),
            CaseKey(2, 1, SplitId.TRAIN, 0, 1),
        )

        self.assertEqual(len({key.case_id for key in keys}), len(keys))
        self.assertTrue(all(0 <= key.case_id < 1 << 63 for key in keys))

    def test_case_key_rejects_ids_that_do_not_fit_the_archive_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream_id"):
            CaseKey(2, 1, SplitId.TRAIN, 1 << 20, 0)
        with self.assertRaisesRegex(ValueError, "attempt_index"):
            CaseKey(2, 1, SplitId.TRAIN, 0, 1 << 40)

    def test_rejection_leaves_the_target_unmet_in_its_original_cell(self) -> None:
        first = schedule_attempt_batch(
            self.targets,
            {},
            family_id=2,
            revision_id=1,
            split_id=SplitId.TRAIN,
            stream_id=3,
            first_attempt_index=0,
            batch_size=2,
        )
        self.assertEqual(
            [assignment.cell_id for assignment in first],
            ["low", "moderate"],
        )

        # The low-cell attempt passed; the moderate-cell attempt failed.
        replacement = schedule_attempt_batch(
            self.targets,
            {"low": 1, "moderate": 0},
            family_id=2,
            revision_id=1,
            split_id=SplitId.TRAIN,
            stream_id=3,
            first_attempt_index=2,
            batch_size=1,
        )

        self.assertEqual(replacement[0].cell_id, "moderate")
        self.assertEqual(replacement[0].case_key.attempt_index, 2)

    def test_split_is_fixed_on_every_pre_outcome_assignment(self) -> None:
        assignments = schedule_attempt_batch(
            self.targets,
            {},
            family_id=2,
            revision_id=1,
            split_id=SplitId.TEST,
            stream_id=5,
            first_attempt_index=10,
            batch_size=2,
        )

        self.assertTrue(
            all(
                assignment.case_key.split_id is SplitId.TEST
                for assignment in assignments
            )
        )
        self.assertTrue(
            all(
                assignment.case_key.root_seed == split_root(SplitId.TEST)
                for assignment in assignments
            )
        )

    def test_final_batch_is_partial_and_contains_only_whole_cases(self) -> None:
        assignments = schedule_attempt_batch(
            self.targets,
            {"low": 2, "moderate": 1},
            family_id=2,
            revision_id=1,
            split_id=SplitId.TRAIN,
            stream_id=0,
            first_attempt_index=11,
            batch_size=8,
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].cell_id, "moderate")
        self.assertEqual(
            assignments[0].case_key,
            CaseKey(
                family_id=2,
                revision_id=1,
                split_id=SplitId.TRAIN,
                stream_id=0,
                attempt_index=11,
            ),
        )

    def test_met_targets_schedule_no_more_attempts(self) -> None:
        assignments = schedule_attempt_batch(
            self.targets,
            {"low": 2, "moderate": 2},
            family_id=2,
            revision_id=1,
            split_id=SplitId.TRAIN,
            stream_id=0,
            first_attempt_index=4,
            batch_size=4,
        )

        self.assertEqual(assignments, ())

    def test_rejects_unknown_or_overfilled_cell_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            schedule_attempt_batch(
                self.targets,
                {"other": 1},
                family_id=2,
                revision_id=1,
                split_id=SplitId.TRAIN,
                stream_id=0,
                first_attempt_index=0,
                batch_size=1,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            schedule_attempt_batch(
                self.targets,
                {"low": 3},
                family_id=2,
                revision_id=1,
                split_id=SplitId.TRAIN,
                stream_id=0,
                first_attempt_index=0,
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
