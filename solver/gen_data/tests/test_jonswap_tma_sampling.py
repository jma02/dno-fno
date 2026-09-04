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
    find_jonswap_parameter_violations,
    positive_mode_wavenumbers,
    relative_frequency_interval_fits,
)
from solver.gen_data.jonswap_tma_sampling import (
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
    JONSWAP_TMA_PARAMETER_GROUPS,
    JonswapTmaSample,
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.types import (
    DatasetSplit,
)


BAND = ResolvedBand(
    length=2.0 * np.pi,
    maximum_wavenumber=128.0,
    transition_wavenumber=96.0,
)


def sample_parameter_group(
    parameter_group_index: int,
    *,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    attempt_number: int | None = None,
) -> JonswapTmaSample:
    """Sample one JONSWAP/TMA parameter group for a test."""

    identifier = parameter_group_index if attempt_number is None else attempt_number
    return sample_jonswap_tma_simulation(
        JONSWAP_TMA_PARAMETER_GROUP_IDS[parameter_group_index],
        dataset_split=dataset_split,
        attempt_number=identifier,
        band=BAND,
    )


class JonswapTmaSamplingTest(unittest.TestCase):
    def test_parameter_groups_are_exact_cartesian_product(self) -> None:
        coordinates = set(JONSWAP_TMA_PARAMETER_GROUPS.values())
        expected = {
            (stratum, gamma, right_moving_fraction)
            for stratum in ("shallow", "finite", "deep")
            for gamma in PAPER_PEAK_ENHANCEMENTS
            for right_moving_fraction in PAPER_RIGHT_MOVING_FRACTIONS
        }
        self.assertEqual(len(JONSWAP_TMA_PARAMETER_GROUP_IDS), 27)
        self.assertEqual(coordinates, expected)

    def test_every_cell_samples_inside_declared_support(self) -> None:
        for index, (parameter_group_id, parameter_group) in enumerate(
            JONSWAP_TMA_PARAMETER_GROUPS.items()
        ):
            stratum, peak_enhancement, right_moving_fraction = parameter_group
            with self.subTest(parameter_group=parameter_group_id):
                sample = sample_parameter_group(index)
                self.assertEqual(
                    find_jonswap_parameter_violations(
                        sample.parameters,
                        stratum=stratum,
                        length=BAND.length,
                    ),
                    (),
                )
                self.assertEqual(
                    sample.parameters.peak_enhancement,
                    peak_enhancement,
                )
                self.assertEqual(
                    sample.parameters.right_moving_fraction,
                    right_moving_fraction,
                )

    def test_replay_is_bitwise_deterministic(self) -> None:
        first = sample_parameter_group(13)
        second = sample_parameter_group(13)

        self.assertEqual(first.parameters, second.parameters)
        np.testing.assert_array_equal(first.phase_right, second.phase_right)
        np.testing.assert_array_equal(first.phase_left, second.phase_left)

    def test_phases_are_explicit_and_independent(self) -> None:
        sample = sample_parameter_group(8)
        expected_shape = positive_mode_wavenumbers(band=BAND).shape
        self.assertEqual(sample.phase_right.shape, expected_shape)
        self.assertEqual(sample.phase_left.shape, expected_shape)
        self.assertTrue(np.all((0.0 <= sample.phase_right)))
        self.assertTrue(np.all(sample.phase_right < 2.0 * np.pi))
        self.assertTrue(np.all((0.0 <= sample.phase_left)))
        self.assertTrue(np.all(sample.phase_left < 2.0 * np.pi))
        self.assertFalse(np.array_equal(sample.phase_right, sample.phase_left))

    def test_shallow_draws_obey_parameterization_and_constraint(self) -> None:
        shallow_parameter_group_index = 4
        for attempt_number in range(256):
            sample = sample_parameter_group(
                shallow_parameter_group_index,
                attempt_number=attempt_number,
            )
            parameters = sample.parameters
            peak_mode = parameters.peak_wavenumber * BAND.length / (2.0 * np.pi)
            depth_wavenumber = parameters.peak_wavenumber * parameters.depth
            relative_height = parameters.significant_height / (2.0 * parameters.depth)
            self.assertIn(round(peak_mode), PAPER_SHALLOW_PEAK_MODES)
            self.assertAlmostEqual(peak_mode, round(peak_mode), places=12)
            self.assertLessEqual(
                depth_wavenumber * relative_height,
                PAPER_PEAK_STEEPNESS_MAXIMUM,
            )
            self.assertAlmostEqual(
                parameters.significant_height,
                2.0 * parameters.depth * relative_height,
                places=15,
            )

    def test_draws_fit_the_published_relative_frequency_interval(
        self,
    ) -> None:
        for attempt_number in range(512):
            parameter_group_index = attempt_number % len(
                JONSWAP_TMA_PARAMETER_GROUP_IDS
            )
            sample = sample_parameter_group(
                parameter_group_index,
                attempt_number=attempt_number,
            )
            self.assertTrue(
                relative_frequency_interval_fits(
                    sample.parameters,
                    band=BAND,
                    relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
                )
            )


if __name__ == "__main__":
    unittest.main()
