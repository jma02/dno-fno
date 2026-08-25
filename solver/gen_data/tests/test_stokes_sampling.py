"""CPU tests for the paper-dataset Stokes sampler."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import jax  # noqa: E402

from solver.reference_solutions.stokes_wave import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
    random_generator_for_case,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEEP_CARRIER_MODE_BOUNDS,
    DEEP_DEPTH_BOUNDS,
    DEEP_DEPTH_WAVENUMBER_MINIMUM,
    FINITE_CARRIER_MODE_BOUNDS,
    FINITE_DEPTH_BOUNDS,
    FINITE_DEPTH_WAVENUMBER_BOUNDS,
    PAPER_AMPLITUDE_BOUNDS,
    PAPER_STOKES_STEEPNESS_CELLS,
    STOKES_SAMPLE_CELL_IDS,
    STOKES_SAMPLE_CELLS,
    UrsellRedrawLimitReached,
    effective_amplitude_bounds,
    effective_depth_bounds,
    feasible_carrier_modes,
    sample_stokes_case,
    stokes_support_violations,
)


def assignment(
    cell_index: int,
    *,
    family_id: int = 1,
    revision_id: int = 2,
    split_id: SplitId = SplitId.TRAIN,
    stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a Stokes sample cell."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        case_key=CaseKey(
            family_id=family_id,
            revision_id=revision_id,
            split_id=split_id,
            stream_id=stream_id,
            attempt_index=attempt,
        ),
        cell_id=STOKES_SAMPLE_CELL_IDS[cell_index],
    )


class StokesSamplingTest(unittest.TestCase):
    def test_cells_are_exactly_branch_by_steepness_product(self) -> None:
        coordinates = tuple(
            (
                cell_id,
                branch,
                steepness_cell_index,
                tuple(PAPER_STOKES_STEEPNESS_CELLS[steepness_cell_index]),
                steepness_cell_index == 1,
            )
            for cell_id, (branch, steepness_cell_index) in (STOKES_SAMPLE_CELLS.items())
        )
        self.assertEqual(
            coordinates,
            (
                ("finite_low", "finite", 0, (0.005, 0.03), False),
                ("finite_moderate", "finite", 1, (0.03, 0.15), True),
                ("deep_low", "deep", 0, (0.005, 0.03), False),
                ("deep_moderate", "deep", 1, (0.03, 0.15), True),
            ),
        )

    def test_replay_is_bitwise_deterministic(self) -> None:
        for cell_index, cell_id in enumerate(STOKES_SAMPLE_CELL_IDS):
            with self.subTest(cell=cell_id):
                first = sample_stokes_case(
                    assignment(cell_index, attempt_index=90 + cell_index)
                )
                second = sample_stokes_case(
                    assignment(cell_index, attempt_index=90 + cell_index)
                )
                self.assertEqual(first, second)
                self.assertEqual(first.to_json_record(), second.to_json_record())

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
            CaseKey(2, 2, SplitId.TRAIN, 0, 41),
            CaseKey(1, 3, SplitId.TRAIN, 0, 41),
            CaseKey(1, 2, SplitId.VALIDATION, 0, 41),
            CaseKey(1, 2, SplitId.TRAIN, 1, 41),
            CaseKey(1, 2, SplitId.TRAIN, 0, 42),
        )
        first_draws = {tuple(random_generator_for_case(key).random(8)) for key in keys}
        self.assertEqual(len(first_draws), len(keys))

    def test_feasible_modes_and_conditional_bounds_are_exact(self) -> None:
        self.assertEqual(
            feasible_carrier_modes("finite", 0),
            tuple(range(FINITE_CARRIER_MODE_BOUNDS[0], 27)),
        )
        self.assertEqual(
            feasible_carrier_modes("finite", 1),
            tuple(range(FINITE_CARRIER_MODE_BOUNDS[0], 27)),
        )
        self.assertEqual(
            feasible_carrier_modes("deep", 0),
            tuple(range(DEEP_CARRIER_MODE_BOUNDS[0], 21)),
        )
        self.assertEqual(
            feasible_carrier_modes("deep", 1),
            tuple(range(3, 21)),
        )

        self.assertEqual(
            effective_depth_bounds("finite", wavenumber=14.0),
            (0.5 / 14.0, 5.0 / 14.0),
        )
        self.assertEqual(
            effective_depth_bounds("deep", wavenumber=1.0),
            (5.0, DEEP_DEPTH_BOUNDS[1]),
        )
        self.assertEqual(
            effective_amplitude_bounds(1, wavenumber=3.0),
            (0.01, PAPER_AMPLITUDE_BOUNDS[1]),
        )

    def test_many_draws_obey_branch_cell_and_ursell_support(self) -> None:
        accepted_by_cell = [0] * len(STOKES_SAMPLE_CELL_IDS)
        failures_by_cell = [0] * len(STOKES_SAMPLE_CELL_IDS)
        for cell_index, cell_id in enumerate(STOKES_SAMPLE_CELL_IDS):
            branch, steepness_cell_index = STOKES_SAMPLE_CELLS[cell_id]
            for attempt_index in range(96):
                try:
                    sample = sample_stokes_case(
                        assignment(cell_index, attempt_index=attempt_index),
                    )
                except UrsellRedrawLimitReached as error:
                    failures_by_cell[cell_index] += 1
                    json.dumps(
                        error.failure_record,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    continue

                accepted_by_cell[cell_index] += 1
                self.assertEqual(stokes_support_violations(sample), ())
                self.assertTrue(0.0 <= sample.phase < 2.0 * np.pi)
                self.assertTrue(
                    PAPER_AMPLITUDE_BOUNDS[0]
                    <= sample.amplitude
                    <= PAPER_AMPLITUDE_BOUNDS[1]
                )
                lower_ka, upper_ka = sample.steepness_bounds
                self.assertGreaterEqual(sample.steepness, lower_ka)
                if steepness_cell_index == 1:
                    self.assertLessEqual(sample.steepness, upper_ka)
                else:
                    self.assertLess(sample.steepness, upper_ka)

                depth_wavenumber = sample.depth * sample.wavenumber
                if branch == "finite":
                    self.assertTrue(
                        FINITE_DEPTH_BOUNDS[0] <= sample.depth <= FINITE_DEPTH_BOUNDS[1]
                    )
                    self.assertTrue(
                        FINITE_DEPTH_WAVENUMBER_BOUNDS[0]
                        <= depth_wavenumber
                        <= FINITE_DEPTH_WAVENUMBER_BOUNDS[1]
                    )
                    with jax.enable_x64():
                        recomputed = float(
                            np.asarray(
                                finite_depth_stokes_ursell_upper_bound(
                                    sample.wavenumber,
                                    sample.depth,
                                    sample.gravity,
                                    sample.amplitude,
                                )
                            )
                        )
                    self.assertAlmostEqual(
                        recomputed,
                        sample.ursell_upper_bound,
                        places=11,
                    )
                    self.assertLessEqual(
                        recomputed,
                        FINITE_DEPTH_STOKES_URSELL_LIMIT,
                    )
                else:
                    self.assertTrue(
                        DEEP_DEPTH_BOUNDS[0] <= sample.depth <= DEEP_DEPTH_BOUNDS[1]
                    )
                    self.assertGreaterEqual(
                        depth_wavenumber,
                        DEEP_DEPTH_WAVENUMBER_MINIMUM,
                    )
                    self.assertIsNone(sample.ursell_upper_bound)

        self.assertTrue(all(count > 0 for count in accepted_by_cell))
        self.assertEqual(failures_by_cell[2:], [0, 0])

    def test_cell_and_depth_boundaries_fail_closed(self) -> None:
        deep_low = sample_stokes_case(assignment(2, attempt_index=400))
        low_upper = deep_low.steepness_bounds[1] / deep_low.wavenumber
        outside_low = replace(
            deep_low,
            amplitude_attempts=((low_upper, None),),
        )
        self.assertIn(
            "amplitude attempt lies outside its assigned cell",
            stokes_support_violations(outside_low),
        )

        deep_moderate = sample_stokes_case(assignment(3, attempt_index=401))
        lower_moderate = deep_moderate.steepness_bounds[0] / deep_moderate.wavenumber
        at_lower_moderate = replace(
            deep_moderate,
            amplitude_attempts=((lower_moderate, None),),
        )
        self.assertEqual(stokes_support_violations(at_lower_moderate), ())

        mode_one = next(
            sample
            for index in range(700, 800)
            if (
                sample := sample_stokes_case(assignment(2, attempt_index=index))
            ).carrier_mode
            == 1
        )
        at_deep_boundary = replace(
            mode_one,
            depth=DEEP_DEPTH_WAVENUMBER_MINIMUM / mode_one.wavenumber,
        )
        self.assertEqual(stokes_support_violations(at_deep_boundary), ())
        below_deep_boundary = replace(
            at_deep_boundary,
            depth=(DEEP_DEPTH_WAVENUMBER_MINIMUM - 1e-12) / mode_one.wavenumber,
        )
        self.assertIn(
            "depth lies outside the branch support",
            stokes_support_violations(below_deep_boundary),
        )

    def test_success_record_is_strict_and_contains_every_attempt(self) -> None:
        values = iter((27.0, 25.0))
        with patch(
            "solver.gen_data.stokes_sampling._evaluate_finite_depth_ursell",
            side_effect=lambda **_: next(values),
        ):
            sample = sample_stokes_case(
                assignment(0, attempt_index=500),
                maximum_ursell_redraws=1,
            )

        record = sample.to_json_record()
        self.assertEqual(record["support_resampling_count"], 1)
        self.assertEqual(len(record["amplitude_attempts"]), 2)
        self.assertEqual(
            [attempt["ursell_upper_bound"] for attempt in record["amplitude_attempts"]],
            [27.0, 25.0],
        )
        self.assertEqual(
            record["seed_words"],
            list(sample.assignment.case_key.seed_words),
        )
        json.dumps(record, sort_keys=True, allow_nan=False)

    def test_exhaustion_record_retains_nonfinite_and_finite_failures(self) -> None:
        values = iter((np.inf, 28.0, 27.0))
        with (
            patch(
                "solver.gen_data.stokes_sampling._evaluate_finite_depth_ursell",
                side_effect=lambda **_: next(values),
            ),
            self.assertRaises(UrsellRedrawLimitReached) as caught,
        ):
            sample_stokes_case(
                assignment(1, attempt_index=501),
                maximum_ursell_redraws=2,
            )

        record = caught.exception.failure_record
        attempts = record["amplitude_attempts"]
        self.assertEqual(record["status"], "failed_ursell_redraw_limit")
        self.assertEqual(record["support_resampling_count"], 2)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            [attempt["ursell_upper_bound"] for attempt in attempts],
            [None, 28.0, 27.0],
        )
        self.assertEqual(
            [attempt["ursell_upper_bound_was_finite"] for attempt in attempts],
            [False, True, True],
        )
        self.assertTrue(
            all(
                record["amplitude_draw_lower"]
                <= attempt["amplitude"]
                < record["amplitude_draw_upper"]
                for attempt in attempts
            )
        )
        json.dumps(record, sort_keys=True, allow_nan=False)

    def test_invalid_inputs_and_corrupt_histories_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Stokes"):
            sample_stokes_case(replace(assignment(0), cell_id="unknown"))
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            sample_stokes_case(assignment(0), domain_length=np.nan)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sample_stokes_case(
                assignment(0),
                maximum_ursell_redraws=-1,
            )

        sample = sample_stokes_case(assignment(2, attempt_index=600))
        corrupt = replace(
            sample,
            amplitude_attempts=((np.nan, None),),
        )
        self.assertIn(
            "amplitude attempt lies outside its assigned cell",
            stokes_support_violations(corrupt),
        )

    def test_forged_cell_reusing_a_canonical_id_fails_closed(self) -> None:
        sample = sample_stokes_case(assignment(2, attempt_index=601))
        forged = replace(
            sample,
            branch="finite",
        )
        self.assertIn(
            "sample parameters do not match the assigned cell",
            stokes_support_violations(forged),
        )


if __name__ == "__main__":
    unittest.main()
