"""CPU tests for the paper-dataset JONSWAP/TMA sampler."""

from __future__ import annotations

import json
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
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    SimulationKey,
    DatasetSplit,
    random_generator_for_simulation,
)


BAND = ResolvedBand(
    length=2.0 * np.pi,
    maximum_wavenumber=128.0,
    transition_wavenumber=96.0,
)


def assignment(
    cell_index: int,
    *,
    family_id: int = 4,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    worker_stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a declared parameter group."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        simulation_key=SimulationKey(
            family_id=family_id,
            dataset_split=dataset_split,
            worker_stream_id=worker_stream_id,
            attempt_index=attempt,
        ),
        parameter_group_id=JONSWAP_TMA_PARAMETER_GROUP_IDS[cell_index],
    )


class JonswapTmaSamplingTest(unittest.TestCase):
    def test_cells_are_exact_cartesian_product(self) -> None:
        coordinates = set(JONSWAP_TMA_PARAMETER_GROUPS.values())
        expected = {
            (stratum, gamma, direction)
            for stratum in ("shallow", "finite", "deep")
            for gamma in PAPER_PEAK_ENHANCEMENTS
            for direction in PAPER_RIGHT_MOVING_FRACTIONS
        }
        self.assertEqual(len(JONSWAP_TMA_PARAMETER_GROUP_IDS), 27)
        self.assertEqual(coordinates, expected)
        self.assertEqual(
            len(JONSWAP_TMA_PARAMETER_GROUP_IDS),
            27,
        )

    def test_every_cell_samples_inside_declared_support(self) -> None:
        for index, (parameter_group_id, parameter_group) in enumerate(
            JONSWAP_TMA_PARAMETER_GROUPS.items()
        ):
            stratum, peak_enhancement, right_moving_fraction = parameter_group
            with self.subTest(parameter_group=parameter_group_id):
                sample = sample_jonswap_tma_simulation(
                    assignment(index),
                    band=BAND,
                )
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
        first = sample_jonswap_tma_simulation(assignment(13), band=BAND)
        second = sample_jonswap_tma_simulation(assignment(13), band=BAND)

        self.assertEqual(first.parameters, second.parameters)
        np.testing.assert_array_equal(first.phase_right, second.phase_right)
        np.testing.assert_array_equal(first.phase_left, second.phase_left)
        self.assertEqual(first.to_json_record(), second.to_json_record())

    def test_pcg64_uses_all_simulation_key_seed_words(self) -> None:
        base = assignment(0, attempt_index=41).simulation_key
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(base.seed_words))
        ).random(8)
        np.testing.assert_array_equal(
            random_generator_for_simulation(base).random(8), expected
        )

        keys = (
            base,
            SimulationKey(5, DatasetSplit.TRAIN, 0, 41),
            SimulationKey(4, DatasetSplit.VALIDATION, 0, 41),
            SimulationKey(4, DatasetSplit.TRAIN, 1, 41),
            SimulationKey(4, DatasetSplit.TRAIN, 0, 42),
        )
        first_draws = {
            tuple(random_generator_for_simulation(key).random(8)) for key in keys
        }
        self.assertEqual(len(first_draws), len(keys))

    def test_phases_are_explicit_independent_and_json_ready(self) -> None:
        sample = sample_jonswap_tma_simulation(assignment(8), band=BAND)
        expected_shape = positive_mode_wavenumbers(band=BAND).shape
        self.assertEqual(sample.phase_right.shape, expected_shape)
        self.assertEqual(sample.phase_left.shape, expected_shape)
        self.assertTrue(np.all((0.0 <= sample.phase_right)))
        self.assertTrue(np.all(sample.phase_right < 2.0 * np.pi))
        self.assertTrue(np.all((0.0 <= sample.phase_left)))
        self.assertTrue(np.all(sample.phase_left < 2.0 * np.pi))
        self.assertFalse(np.array_equal(sample.phase_right, sample.phase_left))

        record = sample.to_json_record()
        self.assertIsInstance(record["phase_right"], list)
        self.assertIsInstance(record["phase_left"], list)
        json.dumps(record, sort_keys=True, allow_nan=False)

    def test_shallow_draws_obey_parameterization_and_constraint(self) -> None:
        shallow_cell_index = 4
        for attempt_index in range(256):
            sample = sample_jonswap_tma_simulation(
                assignment(shallow_cell_index, attempt_index=attempt_index),
                band=BAND,
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
        for attempt_index in range(512):
            cell_index = attempt_index % len(JONSWAP_TMA_PARAMETER_GROUP_IDS)
            sample = sample_jonswap_tma_simulation(
                assignment(
                    cell_index,
                    attempt_index=attempt_index,
                ),
                band=BAND,
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
