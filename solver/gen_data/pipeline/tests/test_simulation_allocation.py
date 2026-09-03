"""Tests for deterministic solver-attempt allocation."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.simulation_allocation import (
    ROOT_SEED_BY_DATASET_SPLIT,
    DatasetSplit,
    PhysicalFamilyId,
    balanced_simulation_targets,
    random_generator_for_attempt,
    select_next_parameter_groups,
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


class BalancedSimulationTargetsTest(unittest.TestCase):
    def test_uses_exact_quotient_and_remainder_balance(self) -> None:
        targets = balanced_simulation_targets(
            ("shallow", "finite", "deep"),
            simulation_count=8,
        )

        self.assertEqual(
            targets,
            {"shallow": 3, "finite": 3, "deep": 2},
        )
        self.assertEqual(sum(targets.values()), 8)

    def test_total_smaller_than_group_count_is_deterministic(self) -> None:
        targets = balanced_simulation_targets(
            ("a", "b", "c", "d"),
            simulation_count=2,
        )

        self.assertEqual(
            list(targets.values()),
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
        self.attempt_limits = dict.fromkeys(self.targets, 34)

    def test_replay_has_identical_parameter_groups_and_random_draws(self) -> None:
        first = select_next_parameter_groups(
            self.targets,
            {},
            {},
            self.attempt_limits,
            batch_size=3,
        )
        replay = select_next_parameter_groups(
            self.targets,
            {},
            {},
            self.attempt_limits,
            batch_size=3,
        )

        self.assertEqual(first, replay)
        self.assertEqual(first, ("low", "moderate", "low"))
        np.testing.assert_array_equal(
            random_generator_for_attempt(
                family_id=2,
                dataset_split=DatasetSplit.VALIDATION,
                attempt_number=0,
            ).integers(0, 2**32, size=8),
            random_generator_for_attempt(
                family_id=2,
                dataset_split=DatasetSplit.VALIDATION,
                attempt_number=0,
            ).integers(0, 2**32, size=8),
        )

    def test_rejection_leaves_the_target_unmet_in_its_original_group(self) -> None:
        first = select_next_parameter_groups(
            self.targets,
            {},
            {},
            self.attempt_limits,
            batch_size=2,
        )
        self.assertEqual(first, ("low", "moderate"))

        # The low-group attempt passed; the moderate-group attempt failed.
        replacement = select_next_parameter_groups(
            self.targets,
            {"low": 1, "moderate": 0},
            {"low": 1, "moderate": 1},
            self.attempt_limits,
            batch_size=1,
        )

        self.assertEqual(replacement, ("moderate",))

    def test_random_generator_uses_family_split_and_attempt_number(self) -> None:
        expected_rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    (ROOT_SEED_BY_DATASET_SPLIT[DatasetSplit.TEST], 2, 0)
                )
            )
        )
        np.testing.assert_array_equal(
            random_generator_for_attempt(
                family_id=2,
                dataset_split=DatasetSplit.TEST,
                attempt_number=0,
            ).integers(0, 2**32, size=8),
            expected_rng.integers(0, 2**32, size=8),
        )

    def test_final_batch_is_partial_and_contains_only_whole_simulations(self) -> None:
        parameter_group_ids = select_next_parameter_groups(
            self.targets,
            {"low": 2, "moderate": 1},
            {"low": 2, "moderate": 1},
            self.attempt_limits,
            batch_size=8,
        )

        self.assertEqual(parameter_group_ids, ("moderate",))

    def test_met_targets_schedule_no_more_attempts(self) -> None:
        parameter_group_ids = select_next_parameter_groups(
            self.targets,
            {"low": 2, "moderate": 2},
            {"low": 2, "moderate": 2},
            self.attempt_limits,
            batch_size=4,
        )

        self.assertEqual(parameter_group_ids, ())


if __name__ == "__main__":
    unittest.main()
