"""CPU tests for the paper-dataset Stokes sampler."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from typing import cast
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
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    DatasetSplit,
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
    STOKES_PARAMETER_GROUP_IDS,
    STOKES_PARAMETER_GROUPS,
    StokesSample,
    UrsellRedrawLimitReached,
    effective_amplitude_bounds,
    effective_depth_bounds,
    feasible_carrier_modes,
    sample_stokes_simulation,
    stokes_support_violations,
)


def sample_stokes(
    cell_index: int,
    *,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    attempt_number: int | None = None,
    domain_length: float = 2.0 * np.pi,
    maximum_ursell_redraws: int = 1000,
) -> StokesSample:
    """Sample one Stokes parameter group for a test."""

    identifier = cell_index if attempt_number is None else attempt_number
    return sample_stokes_simulation(
        STOKES_PARAMETER_GROUP_IDS[cell_index],
        dataset_split=dataset_split,
        attempt_number=identifier,
        domain_length=domain_length,
        maximum_ursell_redraws=maximum_ursell_redraws,
    )


class StokesSamplingTest(unittest.TestCase):
    def test_cells_are_exactly_branch_by_steepness_product(self) -> None:
        coordinates = tuple(
            (
                parameter_group_id,
                branch,
                steepness_cell_index,
                tuple(PAPER_STOKES_STEEPNESS_CELLS[steepness_cell_index]),
                steepness_cell_index == 1,
            )
            for parameter_group_id, (branch, steepness_cell_index) in (
                STOKES_PARAMETER_GROUPS.items()
            )
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
        for cell_index, parameter_group_id in enumerate(STOKES_PARAMETER_GROUP_IDS):
            with self.subTest(parameter_group=parameter_group_id):
                first = sample_stokes(
                    cell_index,
                    attempt_number=90 + cell_index,
                )
                second = sample_stokes(
                    cell_index,
                    attempt_number=90 + cell_index,
                )
                self.assertEqual(first, second)
                self.assertEqual(first.to_json_record(), second.to_json_record())

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
        accepted_by_cell = [0] * len(STOKES_PARAMETER_GROUP_IDS)
        failures_by_cell = [0] * len(STOKES_PARAMETER_GROUP_IDS)
        for cell_index, parameter_group_id in enumerate(STOKES_PARAMETER_GROUP_IDS):
            branch, steepness_cell_index = STOKES_PARAMETER_GROUPS[parameter_group_id]
            for attempt_number in range(96):
                try:
                    sample = sample_stokes(
                        cell_index,
                        attempt_number=attempt_number,
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
                    accepted_ursell = sample.ursell_upper_bound
                    assert accepted_ursell is not None
                    self.assertAlmostEqual(recomputed, accepted_ursell, places=11)
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
        deep_low = sample_stokes(2, attempt_number=400)
        low_upper = deep_low.steepness_bounds[1] / deep_low.wavenumber
        outside_low = replace(
            deep_low,
            amplitude_attempts=((low_upper, None),),
        )
        self.assertIn(
            "amplitude attempt lies outside its steepness interval",
            stokes_support_violations(outside_low),
        )

        deep_moderate = sample_stokes(3, attempt_number=401)
        lower_moderate = deep_moderate.steepness_bounds[0] / deep_moderate.wavenumber
        at_lower_moderate = replace(
            deep_moderate,
            amplitude_attempts=((lower_moderate, None),),
        )
        self.assertEqual(stokes_support_violations(at_lower_moderate), ())

        mode_one = next(
            sample
            for index in range(700, 800)
            if (sample := sample_stokes(2, attempt_number=index)).carrier_mode == 1
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
            sample = sample_stokes(
                0,
                attempt_number=500,
                maximum_ursell_redraws=1,
            )

        record = sample.to_json_record()
        attempts = cast(list[dict[str, object]], record["amplitude_attempts"])
        self.assertEqual(record["support_resampling_count"], 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            [attempt["ursell_upper_bound"] for attempt in attempts],
            [27.0, 25.0],
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
            sample_stokes(
                1,
                attempt_number=501,
                maximum_ursell_redraws=2,
            )

        record = caught.exception.failure_record
        attempts = cast(list[dict[str, object]], record["amplitude_attempts"])
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
                cast(float, record["amplitude_draw_lower"])
                <= cast(float, attempt["amplitude"])
                < cast(float, record["amplitude_draw_upper"])
                for attempt in attempts
            )
        )
        json.dumps(record, sort_keys=True, allow_nan=False)

    def test_invalid_inputs_and_corrupt_histories_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Stokes"):
            sample_stokes_simulation(
                "unknown",
                dataset_split=DatasetSplit.TRAIN,
                attempt_number=0,
            )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            sample_stokes(0, domain_length=np.nan)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sample_stokes(0, maximum_ursell_redraws=-1)

        sample = sample_stokes(2, attempt_number=600)
        corrupt = replace(
            sample,
            amplitude_attempts=((np.nan, None),),
        )
        self.assertIn(
            "amplitude attempt lies outside its steepness interval",
            stokes_support_violations(corrupt),
        )

    def test_forged_cell_reusing_a_canonical_id_fails_closed(self) -> None:
        sample = sample_stokes(2, attempt_number=601)
        forged = replace(
            sample,
            branch="finite",
        )
        self.assertIn(
            "sample parameters do not match the assigned parameter group",
            stokes_support_violations(forged),
        )


if __name__ == "__main__":
    unittest.main()
