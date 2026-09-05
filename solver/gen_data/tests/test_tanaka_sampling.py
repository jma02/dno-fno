"""Scientific support tests for the paper-dataset Tanaka sampler."""

from __future__ import annotations

import math
import unittest

from solver.gen_data.pipeline.types import DatasetSplit
from solver.gen_data.tanaka_sampling import (
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
    TANAKA_DELIVERED_MAXIMUM_WAVENUMBER,
    TANAKA_MINIMUM_RESOLUTION_RATIO,
    TANAKA_PARAMETER_GROUPS,
    sample_tanaka_simulation,
)


DOMAIN_LENGTH = 2.0 * math.pi


class TanakaSamplingTest(unittest.TestCase):
    def test_parameter_groups_are_the_exact_regime_direction_product(self) -> None:
        self.assertEqual(
            TANAKA_PARAMETER_GROUPS,
            {
                "main_m1_q0": ("main", 1, 0),
                "main_m1_q1": ("main", 1, 1),
                "main_m2_q0": ("main", 2, 0),
                "main_m2_q1": ("main", 2, 1),
                "main_m2_q2": ("main", 2, 2),
                "main_m3_q0": ("main", 3, 0),
                "main_m3_q1": ("main", 3, 1),
                "main_m3_q2": ("main", 3, 2),
                "main_m3_q3": ("main", 3, 3),
                "steep_m1_q0": ("steep", 1, 0),
                "steep_m1_q1": ("steep", 1, 1),
            },
        )

    def test_sampling_is_deterministic_for_split_group_and_attempt(self) -> None:
        for attempt_number, parameter_group_id in enumerate(
            TANAKA_PARAMETER_GROUPS, start=90
        ):
            with self.subTest(parameter_group=parameter_group_id):
                first = sample_tanaka_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                second = sample_tanaka_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                validation = sample_tanaka_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.VALIDATION,
                    attempt_number=attempt_number,
                )
                self.assertEqual(first, second)
                self.assertNotEqual(first, validation)

    def test_many_draws_obey_amplitude_depth_direction_and_separation_support(
        self,
    ) -> None:
        seen_main_m3_q1_directions: set[tuple[int, ...]] = set()
        for parameter_group_id, (
            regime,
            crest_count,
            right_moving_count,
        ) in TANAKA_PARAMETER_GROUPS.items():
            for attempt_number in range(512):
                sample = sample_tanaka_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                alphas = tuple(crest.alpha for crest in sample.crests)
                centers = sorted(crest.center for crest in sample.crests)
                directions = tuple(crest.direction for crest in sample.crests)

                self.assertEqual(len(sample.crests), crest_count)
                self.assertTrue(
                    all(math.isfinite(alpha) and alpha > 0 for alpha in alphas)
                )
                self.assertTrue(
                    all(
                        math.isfinite(center) and 0 <= center < DOMAIN_LENGTH
                        for center in centers
                    )
                )
                self.assertTrue(all(direction in (-1, 1) for direction in directions))
                self.assertEqual(
                    sum(direction == 1 for direction in directions), right_moving_count
                )

                base_depth_bounds = (
                    MAIN_DEPTH_BOUNDS if regime == "main" else STEEP_DEPTH_BOUNDS
                )
                alpha_maximum = max(alphas)
                depth_lower = max(
                    base_depth_bounds[0],
                    TANAKA_MINIMUM_RESOLUTION_RATIO
                    * math.sqrt(3.0 * alpha_maximum)
                    / (2.0 * TANAKA_DELIVERED_MAXIMUM_WAVENUMBER),
                )
                self.assertTrue(depth_lower <= sample.depth <= base_depth_bounds[1])
                self.assertGreaterEqual(
                    TANAKA_DELIVERED_MAXIMUM_WAVENUMBER
                    / (math.sqrt(3.0 * alpha_maximum) / (2.0 * sample.depth)),
                    TANAKA_MINIMUM_RESOLUTION_RATIO,
                )

                if regime == "main":
                    self.assertTrue(
                        MAIN_TOTAL_ALPHA_BOUNDS[0]
                        <= sum(alphas)
                        <= MAIN_TOTAL_ALPHA_BOUNDS[1]
                    )
                else:
                    self.assertTrue(
                        STEEP_ALPHA_BOUNDS[0] <= alphas[0] <= STEEP_ALPHA_BOUNDS[1]
                    )

                if crest_count > 1:
                    cyclic_gaps = tuple(
                        (centers[(index + 1) % crest_count] - center) % DOMAIN_LENGTH
                        for index, center in enumerate(centers)
                    )
                    self.assertGreaterEqual(min(cyclic_gaps), 3.0 * sample.depth)
                if parameter_group_id == "main_m3_q1":
                    seen_main_m3_q1_directions.add(directions)

        self.assertEqual(
            seen_main_m3_q1_directions,
            {(1, -1, -1), (-1, 1, -1), (-1, -1, 1)},
        )


if __name__ == "__main__":
    unittest.main()
