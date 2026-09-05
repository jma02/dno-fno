"""CPU tests for the paper-dataset JONSWAP/TMA sampler."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.jonswap_tma import (
    PAPER_PEAK_ENHANCEMENTS,
    PAPER_PEAK_STEEPNESS_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RIGHT_MOVING_FRACTIONS,
    PAPER_SHALLOW_PEAK_MODES,
    ResolvedBand,
    finite_depth_angular_frequency,
    positive_mode_wavenumbers,
)
from solver.gen_data.jonswap_tma_sampling import (
    DEEP_DEPTH_BOUNDS,
    FINITE_DEPTH_BOUNDS,
    JONSWAP_TMA_PARAMETER_GROUPS,
    PEAK_WAVENUMBER_BOUNDS,
    SHALLOW_DEPTH_WAVENUMBER_BOUNDS,
    SHALLOW_RELATIVE_HEIGHT_BOUNDS,
    SIGNIFICANT_HEIGHT_BOUNDS,
    JonswapTmaSample,
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.types import DatasetSplit


BAND = ResolvedBand(2.0 * np.pi, 128.0)


def sample_parameter_group(
    parameter_group_index: int,
    *,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    attempt_number: int | None = None,
) -> JonswapTmaSample:
    return sample_jonswap_tma_simulation(
        tuple(JONSWAP_TMA_PARAMETER_GROUPS)[parameter_group_index],
        dataset_split=dataset_split,
        attempt_number=(
            parameter_group_index if attempt_number is None else attempt_number
        ),
        band=BAND,
    )


class JonswapTmaSamplingTest(unittest.TestCase):
    def test_parameter_groups_are_the_exact_cartesian_product(self) -> None:
        expected = {
            (stratum, gamma, right_moving_fraction)
            for stratum in ("shallow", "finite", "deep")
            for gamma in PAPER_PEAK_ENHANCEMENTS
            for right_moving_fraction in PAPER_RIGHT_MOVING_FRACTIONS
        }
        self.assertEqual(len(JONSWAP_TMA_PARAMETER_GROUPS), 27)
        self.assertEqual(set(JONSWAP_TMA_PARAMETER_GROUPS.values()), expected)

    def test_every_parameter_group_samples_inside_its_declared_range(self) -> None:
        for index, (group_id, group) in enumerate(JONSWAP_TMA_PARAMETER_GROUPS.items()):
            stratum, peak_enhancement, right_moving_fraction = group
            parameters = sample_parameter_group(index).parameters
            with self.subTest(parameter_group=group_id):
                self.assertTrue(np.isfinite(parameters).all())
                self.assertEqual(parameters.peak_enhancement, peak_enhancement)
                self.assertEqual(
                    parameters.right_moving_fraction, right_moving_fraction
                )
                self.assertLessEqual(
                    parameters.peak_wavenumber * parameters.significant_height / 2.0,
                    PAPER_PEAK_STEEPNESS_MAXIMUM,
                )
                self.assertLessEqual(
                    PEAK_WAVENUMBER_BOUNDS[0], parameters.peak_wavenumber
                )
                if stratum == "shallow":
                    peak_mode = parameters.peak_wavenumber * BAND.length / (2.0 * np.pi)
                    self.assertIn(round(peak_mode), PAPER_SHALLOW_PEAK_MODES)
                    self.assertAlmostEqual(peak_mode, round(peak_mode), places=12)
                    self.assertTrue(
                        SHALLOW_DEPTH_WAVENUMBER_BOUNDS[0]
                        <= parameters.peak_wavenumber * parameters.depth
                        <= SHALLOW_DEPTH_WAVENUMBER_BOUNDS[1]
                    )
                    self.assertTrue(
                        SHALLOW_RELATIVE_HEIGHT_BOUNDS[0]
                        <= parameters.significant_height / (2.0 * parameters.depth)
                        <= SHALLOW_RELATIVE_HEIGHT_BOUNDS[1]
                    )
                else:
                    depth_bounds = (
                        FINITE_DEPTH_BOUNDS
                        if stratum == "finite"
                        else DEEP_DEPTH_BOUNDS
                    )
                    self.assertLessEqual(
                        parameters.peak_wavenumber, PEAK_WAVENUMBER_BOUNDS[1]
                    )
                    self.assertTrue(
                        depth_bounds[0] <= parameters.depth <= depth_bounds[1]
                    )
                    self.assertTrue(
                        SIGNIFICANT_HEIGHT_BOUNDS[0]
                        <= parameters.significant_height
                        <= SIGNIFICANT_HEIGHT_BOUNDS[1]
                    )

    def test_replay_is_bitwise_deterministic(self) -> None:
        first = sample_parameter_group(13)
        second = sample_parameter_group(13)

        self.assertEqual(first.parameters, second.parameters)
        np.testing.assert_array_equal(first.phase_right, second.phase_right)
        np.testing.assert_array_equal(first.phase_left, second.phase_left)

    def test_phases_are_explicit_independent_and_cover_every_mode(self) -> None:
        sample = sample_parameter_group(8)
        expected_shape = positive_mode_wavenumbers(band=BAND).shape
        self.assertEqual(sample.phase_right.shape, expected_shape)
        self.assertEqual(sample.phase_left.shape, expected_shape)
        self.assertTrue(
            np.all((0.0 <= sample.phase_right) & (sample.phase_right < 2.0 * np.pi))
        )
        self.assertTrue(
            np.all((0.0 <= sample.phase_left) & (sample.phase_left < 2.0 * np.pi))
        )
        self.assertFalse(np.array_equal(sample.phase_right, sample.phase_left))

    def test_sampled_spectra_fit_the_paper_relative_frequency_band(self) -> None:
        for attempt_number in range(512):
            sample = sample_parameter_group(
                attempt_number % len(JONSWAP_TMA_PARAMETER_GROUPS),
                attempt_number=attempt_number,
            )
            frequencies = finite_depth_angular_frequency(
                np.asarray(
                    [sample.parameters.peak_wavenumber, BAND.maximum_wavenumber]
                ),
                depth=sample.parameters.depth,
                gravity=1.0,
            )
            self.assertGreaterEqual(
                frequencies[1],
                PAPER_RELATIVE_FREQUENCY_MAXIMUM * frequencies[0],
            )


if __name__ == "__main__":
    unittest.main()
