"""Tests for deterministic simulation allocation."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.simulation_allocation import (
    ROOT_SEED_BY_DATASET_SPLIT,
    PhysicalFamilyId,
    SimulationKey,
    DatasetSplit,
    balanced_simulation_targets,
    build_next_attempt_batch,
)


class FamilyIdentityTest(unittest.TestCase):
    def test_paper_family_ids_are_stable(self) -> None:
        self.assertEqual(
            tuple((family.name, int(family)) for family in PhysicalFamilyId),
            (
                ("STOKES", 1),
                ("TANAKA", 2),
                ("BENJAMIN_FEIR", 3),
                ("JONSWAP_TMA", 4),
            ),
        )


class BalancedParameterGroupTargetsTest(unittest.TestCase):
    def test_uses_exact_quotient_and_remainder_balance(self) -> None:
        targets = balanced_simulation_targets(
            ("shallow", "finite", "deep"),
            simulation_count=8,
        )

        self.assertEqual(
            [
                (target.parameter_group_id, target.simulation_count)
                for target in targets
            ],
            [("shallow", 3), ("finite", 3), ("deep", 2)],
        )
        self.assertEqual(sum(target.simulation_count for target in targets), 8)

    def test_total_smaller_than_group_count_is_deterministic(self) -> None:
        targets = balanced_simulation_targets(
            ("a", "b", "c", "d"),
            simulation_count=2,
        )

        self.assertEqual(
            [target.simulation_count for target in targets],
            [1, 1, 0, 0],
        )

    def test_rejects_duplicate_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            balanced_simulation_targets(("same", "same"), simulation_count=2)


class AttemptScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = balanced_simulation_targets(
            ("low", "moderate"),
            simulation_count=4,
        )
        self.attempt_limits = {target.parameter_group_id: 34 for target in self.targets}

    def test_replay_has_identical_assignments_and_seed_words(self) -> None:
        arguments = {
            "family_id": 2,
            "dataset_split": DatasetSplit.VALIDATION,
            "worker_stream_id": 9,
            "first_attempt_index": 17,
            "batch_size": 3,
        }

        first = build_next_attempt_batch(
            self.targets,
            {},
            {},
            self.attempt_limits,
            **arguments,
        )
        replay = build_next_attempt_batch(
            self.targets,
            {},
            {},
            self.attempt_limits,
            **arguments,
        )

        self.assertEqual(first, replay)
        self.assertEqual(
            [assignment.parameter_group_id for assignment in first],
            ["low", "moderate", "low"],
        )
        self.assertEqual(
            [assignment.simulation_key.attempt_index for assignment in first],
            [17, 18, 19],
        )
        self.assertTrue(
            all(assignment.simulation_key.family_id == 2 for assignment in first)
        )
        self.assertEqual(
            first[0].simulation_key.seed_words,
            (2026072204, 2, 9, 17),
        )
        self.assertEqual(
            first[0].simulation_key.simulation_id,
            1_152_931_400_211_496_977,
        )

    def test_simulation_id_separates_splits_worker_streams_and_attempts(self) -> None:
        keys = (
            SimulationKey(2, DatasetSplit.TRAIN, 0, 0),
            SimulationKey(2, DatasetSplit.VALIDATION, 0, 0),
            SimulationKey(2, DatasetSplit.TEST, 0, 0),
            SimulationKey(2, DatasetSplit.TRAIN, 1, 0),
            SimulationKey(2, DatasetSplit.TRAIN, 0, 1),
        )

        self.assertEqual(len({key.simulation_id for key in keys}), len(keys))
        self.assertTrue(
            all(key.simulation_id <= np.iinfo(np.int64).max for key in keys)
        )

    def test_simulation_key_rejects_ids_that_do_not_fit_the_archive_layout(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "worker_stream_id"):
            SimulationKey(2, DatasetSplit.TRAIN, 1_048_576, 0)
        with self.assertRaisesRegex(ValueError, "attempt_index"):
            SimulationKey(2, DatasetSplit.TRAIN, 0, 1_099_511_627_776)

    def test_rejection_leaves_the_target_unmet_in_its_original_group(self) -> None:
        first = build_next_attempt_batch(
            self.targets,
            {},
            {},
            self.attempt_limits,
            family_id=2,
            dataset_split=DatasetSplit.TRAIN,
            worker_stream_id=3,
            first_attempt_index=0,
            batch_size=2,
        )
        self.assertEqual(
            [assignment.parameter_group_id for assignment in first],
            ["low", "moderate"],
        )

        # The low-group attempt passed; the moderate-group attempt failed.
        replacement = build_next_attempt_batch(
            self.targets,
            {"low": 1, "moderate": 0},
            {"low": 1, "moderate": 1},
            self.attempt_limits,
            family_id=2,
            dataset_split=DatasetSplit.TRAIN,
            worker_stream_id=3,
            first_attempt_index=2,
            batch_size=1,
        )

        self.assertEqual(replacement[0].parameter_group_id, "moderate")
        self.assertEqual(replacement[0].simulation_key.attempt_index, 2)

    def test_split_is_fixed_on_every_pre_outcome_assignment(self) -> None:
        assignments = build_next_attempt_batch(
            self.targets,
            {},
            {},
            self.attempt_limits,
            family_id=2,
            dataset_split=DatasetSplit.TEST,
            worker_stream_id=5,
            first_attempt_index=10,
            batch_size=2,
        )

        self.assertTrue(
            all(
                assignment.simulation_key.dataset_split is DatasetSplit.TEST
                for assignment in assignments
            )
        )
        self.assertTrue(
            all(
                assignment.simulation_key.seed_words[0]
                == ROOT_SEED_BY_DATASET_SPLIT[DatasetSplit.TEST]
                for assignment in assignments
            )
        )

    def test_final_batch_is_partial_and_contains_only_whole_simulations(self) -> None:
        assignments = build_next_attempt_batch(
            self.targets,
            {"low": 2, "moderate": 1},
            {"low": 2, "moderate": 1},
            self.attempt_limits,
            family_id=2,
            dataset_split=DatasetSplit.TRAIN,
            worker_stream_id=0,
            first_attempt_index=11,
            batch_size=8,
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].parameter_group_id, "moderate")
        self.assertEqual(
            assignments[0].simulation_key,
            SimulationKey(
                family_id=2,
                dataset_split=DatasetSplit.TRAIN,
                worker_stream_id=0,
                attempt_index=11,
            ),
        )

    def test_met_targets_schedule_no_more_attempts(self) -> None:
        assignments = build_next_attempt_batch(
            self.targets,
            {"low": 2, "moderate": 2},
            {"low": 2, "moderate": 2},
            self.attempt_limits,
            family_id=2,
            dataset_split=DatasetSplit.TRAIN,
            worker_stream_id=0,
            first_attempt_index=4,
            batch_size=4,
        )

        self.assertEqual(assignments, ())


if __name__ == "__main__":
    unittest.main()
