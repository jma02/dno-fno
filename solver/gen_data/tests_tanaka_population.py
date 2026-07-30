"""CPU tests for the paper-corpus Tanaka population sampler."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.tanaka_population import (
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
    TANAKA_POPULATION_CELLS,
    TanakaCrest,
    TanakaPopulationCell,
    pcg64_for_case,
    sample_tanaka_population,
    tanaka_support_violations,
)


DOMAIN_LENGTH = 2.0 * np.pi


def assignment(
    cell_index: int,
    *,
    family_id: int = 1,
    revision_id: int = 1,
    split_id: SplitId = SplitId.TRAIN,
    stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a Tanaka population cell."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        case_key=CaseKey(
            family_id=family_id,
            revision_id=revision_id,
            split_id=split_id,
            stream_id=stream_id,
            attempt_index=attempt,
        ),
        cell_id=TANAKA_POPULATION_CELLS[cell_index].cell_id,
    )


class TanakaPopulationTest(unittest.TestCase):
    def test_cells_are_exactly_the_declared_eleven(self) -> None:
        coordinates = tuple(
            (
                cell.cell_id,
                cell.regime,
                cell.crest_count,
                cell.right_moving_count,
            )
            for cell in TANAKA_POPULATION_CELLS
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

    def test_replay_is_bitwise_deterministic(self) -> None:
        for cell_index in range(len(TANAKA_POPULATION_CELLS)):
            with self.subTest(cell=TANAKA_POPULATION_CELLS[cell_index].cell_id):
                first = sample_tanaka_population(
                    assignment(cell_index, attempt_index=91)
                )
                second = sample_tanaka_population(
                    assignment(cell_index, attempt_index=91)
                )
                self.assertEqual(first, second)
                self.assertEqual(first.to_json_record(), second.to_json_record())
                self.assertEqual(
                    first.to_json_record()["schema"],
                    "tanaka_population_spec_v2",
                )

    def test_pcg64_uses_all_case_key_seed_words(self) -> None:
        base = assignment(0, attempt_index=41).case_key
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(base.seed_words))
        ).random(8)
        np.testing.assert_array_equal(pcg64_for_case(base).random(8), expected)

        keys = (
            base,
            CaseKey(2, 1, SplitId.TRAIN, 0, 41),
            CaseKey(1, 2, SplitId.TRAIN, 0, 41),
            CaseKey(1, 1, SplitId.VALIDATION, 0, 41),
            CaseKey(1, 1, SplitId.TRAIN, 1, 41),
            CaseKey(1, 1, SplitId.TRAIN, 0, 42),
        )
        first_draws = {tuple(pcg64_for_case(key).random(8)) for key in keys}
        self.assertEqual(len(first_draws), len(keys))

    def test_support_sums_and_separation_over_many_attempts(self) -> None:
        for cell_index, cell in enumerate(TANAKA_POPULATION_CELLS):
            for attempt_index in range(512):
                sample = sample_tanaka_population(
                    assignment(cell_index, attempt_index=attempt_index)
                )
                self.assertEqual(tanaka_support_violations(sample), ())
                self.assertEqual(len(sample.crests), cell.crest_count)
                self.assertEqual(
                    sum(crest.direction == 1 for crest in sample.crests),
                    cell.right_moving_count,
                )
                self.assertEqual(
                    sample.total_dimensionless_amplitude,
                    sum(crest.alpha for crest in sample.crests),
                )
                self.assertTrue(
                    all(crest.alpha > 0.0 for crest in sample.crests)
                )
                self.assertTrue(
                    all(
                        0.0 <= crest.center < DOMAIN_LENGTH
                        for crest in sample.crests
                    )
                )
                if cell.crest_count > 1:
                    self.assertGreaterEqual(
                        sample.achieved_minimum_separation,
                        sample.required_minimum_separation,
                    )

                if cell.regime == "main":
                    self.assertTrue(
                        MAIN_DEPTH_BOUNDS[0]
                        <= sample.depth
                        <= MAIN_DEPTH_BOUNDS[1]
                    )
                    self.assertTrue(
                        MAIN_TOTAL_ALPHA_BOUNDS[0]
                        <= sample.total_dimensionless_amplitude
                        <= MAIN_TOTAL_ALPHA_BOUNDS[1]
                    )
                else:
                    self.assertTrue(
                        STEEP_DEPTH_BOUNDS[0]
                        <= sample.depth
                        <= STEEP_DEPTH_BOUNDS[1]
                    )
                    self.assertTrue(
                        STEEP_ALPHA_BOUNDS[0]
                        <= sample.crests[0].alpha
                        <= STEEP_ALPHA_BOUNDS[1]
                    )

    def test_direction_compositions_and_json_are_exact(self) -> None:
        directions: set[int] = set()
        for cell_index in range(len(TANAKA_POPULATION_CELLS)):
            sample = sample_tanaka_population(
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
                sample.cell.right_moving_count,
            )
            json.dumps(record, sort_keys=True, allow_nan=False)

        mixed_assignments: set[tuple[int, ...]] = set()
        for attempt_index in range(256):
            sample = sample_tanaka_population(
                assignment(6, attempt_index=1000 + attempt_index)
            )
            directions.update(crest.direction for crest in sample.crests)
            mixed_assignments.add(
                tuple(crest.direction for crest in sample.crests)
            )
        self.assertEqual(directions, {-1, 1})
        self.assertEqual(
            mixed_assignments,
            {
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
            },
        )

    def test_invalid_records_and_impossible_geometry_fail_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            TanakaPopulationCell(
                "bad_m",
                "main",
                1.0,  # type: ignore[arg-type]
                1,
            )
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            TanakaPopulationCell("bad_m", "main", True, 1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            TanakaPopulationCell("bad_q", "main", 2, 1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            TanakaPopulationCell("bad_q", "main", 2, True)
        with self.assertRaisesRegex(ValueError, "between zero and crest_count"):
            TanakaPopulationCell("bad_q", "main", 2, 3)

        sample = sample_tanaka_population(assignment(3))
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
            crests=tuple(
                replace(crest, direction=-1) for crest in sample.crests
            ),
        )
        self.assertIn(
            "number of right-moving crests does not match the parameter category",
            tanaka_support_violations(wrong_composition),
        )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            sample_tanaka_population(assignment(0), domain_length=np.nan)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            sample_tanaka_population(assignment(5), domain_length=0.01)


if __name__ == "__main__":
    unittest.main()
