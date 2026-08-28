"""CPU tests for the paper-dataset Tanaka sampler."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

import numpy as np

from solver.gen_data.pipeline.case_allocation import (
    AttemptAssignment,
    CaseKey,
    SplitId,
    random_generator_for_case,
)
from solver.gen_data.tanaka_sampling import (
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
    TANAKA_DELIVERED_MAXIMUM_WAVENUMBER,
    TANAKA_MINIMUM_RESOLUTION_RATIO,
    TANAKA_SAMPLE_CELL_IDS,
    TANAKA_SAMPLE_CELLS,
    TANAKA_SAMPLING_REVISION_V3,
    TanakaCrest,
    sample_tanaka_case,
    tanaka_conditional_depth_bounds,
    tanaka_inverse_width,
    tanaka_resolution_ratio,
    tanaka_support_violations,
)


DOMAIN_LENGTH = 2.0 * np.pi


def assignment(
    cell_index: int,
    *,
    family_id: int = 2,
    revision_id: int = TANAKA_SAMPLING_REVISION_V3,
    split_id: SplitId = SplitId.TRAIN,
    stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a Tanaka sample cell."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        case_key=CaseKey(
            family_id=family_id,
            revision_id=revision_id,
            split_id=split_id,
            stream_id=stream_id,
            attempt_index=attempt,
        ),
        cell_id=TANAKA_SAMPLE_CELL_IDS[cell_index],
    )


class TanakaSamplingTest(unittest.TestCase):
    def test_cells_are_exactly_the_declared_eleven(self) -> None:
        coordinates = tuple(
            (
                cell_id,
                regime,
                crest_count,
                right_moving_count,
            )
            for cell_id, (regime, crest_count, right_moving_count) in (
                TANAKA_SAMPLE_CELLS.items()
            )
        )
        self.assertEqual(
            coordinates,
            (
                ("main_m1_q0", "main", 1, 0),
                ("main_m1_q1", "main", 1, 1),
                ("main_m2_q0", "main", 2, 0),
                ("main_m2_q1", "main", 2, 1),
                ("main_m2_q2", "main", 2, 2),
                ("main_m3_q0", "main", 3, 0),
                ("main_m3_q1", "main", 3, 1),
                ("main_m3_q2", "main", 3, 2),
                ("main_m3_q3", "main", 3, 3),
                ("steep_m1_q0", "steep", 1, 0),
                ("steep_m1_q1", "steep", 1, 1),
            ),
        )

    def test_revision_3_replay_is_bitwise_deterministic(self) -> None:
        for cell_index, cell_id in enumerate(TANAKA_SAMPLE_CELL_IDS):
            with self.subTest(cell=cell_id):
                attempted = assignment(
                    cell_index,
                    revision_id=TANAKA_SAMPLING_REVISION_V3,
                    attempt_index=91,
                )
                first = sample_tanaka_case(attempted)
                second = sample_tanaka_case(attempted)
                self.assertEqual(first, second)
                self.assertEqual(first.to_json_record(), second.to_json_record())
                self.assertNotIn("schema", first.to_json_record())

    def test_pcg64_uses_all_case_key_seed_words(self) -> None:
        base = assignment(0, attempt_index=41).case_key
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(base.seed_words))
        ).random(8)
        np.testing.assert_array_equal(
            random_generator_for_case(base).random(8), expected
        )

        keys = (
            base,
            CaseKey(1, 3, SplitId.TRAIN, 0, 41),
            CaseKey(2, 4, SplitId.TRAIN, 0, 41),
            CaseKey(2, 3, SplitId.VALIDATION, 0, 41),
            CaseKey(2, 3, SplitId.TRAIN, 1, 41),
            CaseKey(2, 3, SplitId.TRAIN, 0, 42),
        )
        first_draws = {tuple(random_generator_for_case(key).random(8)) for key in keys}
        self.assertEqual(len(first_draws), len(keys))

    def test_support_sums_and_separation_over_many_attempts(self) -> None:
        for cell_index, cell_id in enumerate(TANAKA_SAMPLE_CELL_IDS):
            regime, crest_count, right_moving_count = TANAKA_SAMPLE_CELLS[cell_id]
            for attempt_index in range(512):
                sample = sample_tanaka_case(
                    assignment(cell_index, attempt_index=attempt_index)
                )
                self.assertEqual(tanaka_support_violations(sample), ())
                self.assertEqual(len(sample.crests), crest_count)
                self.assertEqual(
                    sum(crest.direction == 1 for crest in sample.crests),
                    right_moving_count,
                )
                self.assertEqual(
                    sample.total_dimensionless_amplitude,
                    sum(crest.alpha for crest in sample.crests),
                )
                self.assertTrue(all(crest.alpha > 0.0 for crest in sample.crests))
                self.assertTrue(
                    all(0.0 <= crest.center < DOMAIN_LENGTH for crest in sample.crests)
                )
                if crest_count > 1:
                    self.assertGreaterEqual(
                        sample.achieved_minimum_separation,
                        sample.required_minimum_separation,
                    )

                if regime == "main":
                    self.assertTrue(
                        MAIN_DEPTH_BOUNDS[0] <= sample.depth <= MAIN_DEPTH_BOUNDS[1]
                    )
                    self.assertTrue(
                        MAIN_TOTAL_ALPHA_BOUNDS[0]
                        <= sample.total_dimensionless_amplitude
                        <= MAIN_TOTAL_ALPHA_BOUNDS[1]
                    )
                else:
                    self.assertTrue(
                        STEEP_DEPTH_BOUNDS[0] <= sample.depth <= STEEP_DEPTH_BOUNDS[1]
                    )
                    self.assertTrue(
                        STEEP_ALPHA_BOUNDS[0]
                        <= sample.crests[0].alpha
                        <= STEEP_ALPHA_BOUNDS[1]
                    )

    def test_revision_3_direct_sampling_satisfies_conditional_support(
        self,
    ) -> None:
        for cell_index, cell_id in enumerate(TANAKA_SAMPLE_CELL_IDS):
            regime, _, _ = TANAKA_SAMPLE_CELLS[cell_id]
            for attempt_index in range(512):
                sample = sample_tanaka_case(
                    assignment(
                        cell_index,
                        revision_id=TANAKA_SAMPLING_REVISION_V3,
                        attempt_index=attempt_index,
                    )
                )
                alpha_max = max(crest.alpha for crest in sample.crests)
                inverse_width = math.sqrt(3.0 * alpha_max) / (2.0 * sample.depth)
                resolution_ratio = TANAKA_DELIVERED_MAXIMUM_WAVENUMBER / inverse_width
                base_bounds = (
                    MAIN_DEPTH_BOUNDS if regime == "main" else STEEP_DEPTH_BOUNDS
                )
                expected_lower = max(
                    base_bounds[0],
                    TANAKA_MINIMUM_RESOLUTION_RATIO
                    * math.sqrt(3.0 * alpha_max)
                    / (2.0 * TANAKA_DELIVERED_MAXIMUM_WAVENUMBER),
                )

                self.assertEqual(tanaka_support_violations(sample), ())
                self.assertEqual(
                    sample.maximum_dimensionless_crest_amplitude,
                    alpha_max,
                )
                self.assertEqual(sample.maximum_inverse_width, inverse_width)
                self.assertEqual(sample.resolution_ratio, resolution_ratio)
                self.assertEqual(
                    sample.conditional_depth_lower_bound,
                    expected_lower,
                )
                self.assertGreaterEqual(sample.depth, expected_lower)
                self.assertGreaterEqual(
                    sample.resolution_ratio,
                    TANAKA_MINIMUM_RESOLUTION_RATIO,
                )

                record = sample.to_json_record()
                profile = record["profile_resolution"]
                self.assertIsInstance(profile, dict)
                assert isinstance(profile, dict)
                self.assertEqual(
                    profile,
                    {
                        "delivered_maximum_wavenumber": 128.0,
                        "maximum_dimensionless_crest_amplitude": alpha_max,
                        "maximum_inverse_width": inverse_width,
                        "wavenumbers_per_inverse_width": resolution_ratio,
                        "required_minimum_wavenumbers_per_inverse_width": 10.0,
                        "conditional_depth_lower_bound": expected_lower,
                    },
                )

    def test_revision_3_draw_order_is_amplitudes_then_conditional_depth(
        self,
    ) -> None:
        cell_index = 6
        attempted = assignment(
            cell_index,
            revision_id=TANAKA_SAMPLING_REVISION_V3,
            attempt_index=314,
        )
        regime, crest_count, right_moving_count = TANAKA_SAMPLE_CELLS[
            TANAKA_SAMPLE_CELL_IDS[cell_index]
        ]
        rng = random_generator_for_case(attempted.case_key)

        total_alpha = float(rng.uniform(*MAIN_TOTAL_ALPHA_BOUNDS))
        weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
        alpha_values = [float(total_alpha * weight) for weight in weights]
        alpha_values[-1] = total_alpha - sum(alpha_values[:-1])
        alphas = tuple(alpha_values)
        depth_bounds = tanaka_conditional_depth_bounds(regime, alphas)
        depth = float(
            np.exp(
                rng.uniform(
                    np.log(depth_bounds[0]),
                    np.log(depth_bounds[1]),
                )
            )
        )

        minimum_separation = 3.0 * depth
        slack = DOMAIN_LENGTH - crest_count * minimum_separation
        gap_weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
        gaps = minimum_separation + slack * gap_weights
        origin = float(rng.uniform(0.0, DOMAIN_LENGTH))
        offsets = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(gaps[:-1])))
        centers = tuple(
            float(center) for center in np.mod(origin + offsets, DOMAIN_LENGTH)
        )
        direction_multiset = np.concatenate(
            (
                -np.ones(
                    crest_count - right_moving_count,
                    dtype=np.int8,
                ),
                np.ones(right_moving_count, dtype=np.int8),
            )
        )
        directions = tuple(
            int(direction) for direction in rng.permutation(direction_multiset)
        )

        sample = sample_tanaka_case(attempted)
        self.assertEqual(sample.depth, depth)
        self.assertEqual(tuple(crest.alpha for crest in sample.crests), alphas)
        self.assertEqual(
            tuple(crest.center for crest in sample.crests),
            centers,
        )
        self.assertEqual(
            tuple(crest.direction for crest in sample.crests),
            directions,
        )

    def test_direction_compositions_and_json_are_exact(self) -> None:
        directions: set[int] = set()
        for cell_index in range(len(TANAKA_SAMPLE_CELL_IDS)):
            sample = sample_tanaka_case(
                assignment(cell_index, attempt_index=700 + cell_index)
            )
            directions.update(crest.direction for crest in sample.crests)
            record = sample.to_json_record()
            self.assertEqual(
                record["seed_words"],
                list(sample.assignment.case_key.seed_words),
            )
            self.assertEqual(record["crest_count"], len(sample.crests))
            self.assertEqual(
                record["right_moving_count"],
                sample.right_moving_count,
            )
            json.dumps(record, sort_keys=True, allow_nan=False)

        mixed_assignments: set[tuple[int, ...]] = set()
        for attempt_index in range(256):
            sample = sample_tanaka_case(
                assignment(6, attempt_index=1000 + attempt_index)
            )
            directions.update(crest.direction for crest in sample.crests)
            mixed_assignments.add(tuple(crest.direction for crest in sample.crests))
        self.assertEqual(directions, {-1, 1})
        self.assertEqual(
            mixed_assignments,
            {
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
            },
        )

    def test_revision_3_boundary_corruption_fails_below_and_passes_above(
        self,
    ) -> None:
        sample = sample_tanaka_case(
            assignment(
                0,
                revision_id=TANAKA_SAMPLING_REVISION_V3,
                attempt_index=812,
            )
        )
        lower = sample.conditional_depth_lower_bound
        below = replace(sample, depth=lower * (1.0 - 1.0e-8))
        boundary = replace(sample, depth=lower)
        above = replace(sample, depth=lower * (1.0 + 1.0e-8))

        self.assertLess(
            tanaka_resolution_ratio(
                alpha=below.maximum_dimensionless_crest_amplitude,
                depth=below.depth,
            ),
            TANAKA_MINIMUM_RESOLUTION_RATIO,
        )
        self.assertIn(
            "depth is below the profile-resolution support",
            tanaka_support_violations(below),
        )
        self.assertIn(
            "profile resolution ratio is below the required minimum",
            tanaka_support_violations(below),
        )
        self.assertEqual(tanaka_support_violations(boundary), ())
        self.assertEqual(tanaka_support_violations(above), ())
        self.assertGreater(
            tanaka_resolution_ratio(
                alpha=above.maximum_dimensionless_crest_amplitude,
                depth=above.depth,
            ),
            TANAKA_MINIMUM_RESOLUTION_RATIO,
        )

    def test_unknown_revision_fails_closed(self) -> None:
        for revision_id in (1, 4):
            with self.subTest(revision_id=revision_id):
                attempted = assignment(0, revision_id=revision_id)
                with self.assertRaisesRegex(ValueError, "unsupported.*revision"):
                    sample_tanaka_case(attempted)

                valid = sample_tanaka_case(assignment(0))
                corrupt = replace(valid, assignment=attempted)
                self.assertIn(
                    "unsupported Tanaka sampling revision",
                    tanaka_support_violations(corrupt),
                )
                with self.assertRaisesRegex(ValueError, "unsupported.*revision"):
                    corrupt.to_json_record()

    def test_resolution_helpers_fail_closed_on_invalid_inputs(self) -> None:
        self.assertEqual(
            tanaka_inverse_width(alpha=0.25, depth=0.5),
            math.sqrt(0.75),
        )
        for alpha, depth in (
            (0.0, 0.2),
            (-0.1, 0.2),
            (math.nan, 0.2),
            (0.2, 0.0),
            (0.2, math.inf),
        ):
            with self.subTest(alpha=alpha, depth=depth):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    tanaka_inverse_width(alpha=alpha, depth=depth)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tanaka_conditional_depth_bounds("main", ())
        with self.assertRaisesRegex(ValueError, "no admissible depth"):
            tanaka_conditional_depth_bounds("main", (100.0,))

    def test_invalid_records_and_impossible_geometry_fail_closed(self) -> None:
        sample = sample_tanaka_case(assignment(3))
        mismatched_cell = replace(sample, right_moving_count=0)
        self.assertIn(
            "sample parameters do not match the assigned cell",
            tanaka_support_violations(mismatched_cell),
        )
        corrupt = replace(
            sample,
            crests=(
                TanakaCrest(
                    alpha=sample.crests[0].alpha,
                    center=sample.crests[0].center,
                    direction=0,
                ),
                sample.crests[1],
            ),
        )
        self.assertIn(
            "every crest direction must equal -1 or +1",
            tanaka_support_violations(corrupt),
        )
        wrong_composition = replace(
            sample,
            crests=tuple(replace(crest, direction=-1) for crest in sample.crests),
        )
        self.assertIn(
            "number of right-moving crests does not match the parameter category",
            tanaka_support_violations(wrong_composition),
        )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            sample_tanaka_case(assignment(0), domain_length=np.nan)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            sample_tanaka_case(assignment(5), domain_length=0.01)


if __name__ == "__main__":
    unittest.main()
