"""Scientific support tests for the paper-dataset Benjamin--Feir sampler."""

from __future__ import annotations

import math
import unittest

import numpy as np

from solver.gen_data.benjamin_feir_jcp09 import (
    CARRIER_MODE_MAX,
    CARRIER_MODE_MIN,
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    PERTURBATION_RATIO_MIN,
    focused_steepness_proxy,
    instability_band_fraction,
)
from solver.gen_data.benjamin_feir_sampling import (
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    BENJAMIN_FEIR_PARAMETER_GROUPS,
    PAPER_DOMAIN_LENGTH,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    PAPER_PERTURBATION_RATIO_MAX,
    sample_benjamin_feir_simulation,
)
from solver.gen_data.pipeline.types import DatasetSplit


class BenjaminFeirSamplingTest(unittest.TestCase):
    def test_parameter_groups_are_the_66_feasible_mode_pairs(self) -> None:
        expected = tuple(
            (carrier_mode, sideband_offset)
            for carrier_mode in range(CARRIER_MODE_MIN, CARRIER_MODE_MAX + 1)
            for sideband_offset in range(1, carrier_mode)
            if sideband_offset / carrier_mode
            < 2.0 * math.sqrt(2.0) * CARRIER_STEEPNESS_MAX
        )
        self.assertEqual(tuple(BENJAMIN_FEIR_PARAMETER_GROUPS.values()), expected)
        self.assertEqual(len(BENJAMIN_FEIR_PARAMETER_GROUP_IDS), 66)

    def test_sampling_is_deterministic_for_split_group_and_attempt(self) -> None:
        for attempt_number, parameter_group_id in enumerate(
            BENJAMIN_FEIR_PARAMETER_GROUP_IDS, start=90
        ):
            with self.subTest(parameter_group=parameter_group_id):
                first = sample_benjamin_feir_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                second = sample_benjamin_feir_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                validation = sample_benjamin_feir_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.VALIDATION,
                    attempt_number=attempt_number,
                )
                self.assertEqual(first, second)
                self.assertNotEqual(first, validation)

    def test_many_draws_obey_mode_steepness_and_sideband_support(self) -> None:
        for parameter_group_id, (
            carrier_mode,
            sideband_offset,
        ) in BENJAMIN_FEIR_PARAMETER_GROUPS.items():
            for attempt_number in range(128):
                sample = sample_benjamin_feir_simulation(
                    parameter_group_id,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_number=attempt_number,
                )
                self.assertEqual(sample.carrier_mode, carrier_mode)
                self.assertEqual(sample.sideband_offset, sideband_offset)
                self.assertGreaterEqual(carrier_mode - sideband_offset, 1)

                steepness_lower = max(
                    CARRIER_STEEPNESS_MIN,
                    sideband_offset / (2.0 * math.sqrt(2.0) * carrier_mode),
                )
                self.assertGreater(sample.carrier_steepness, steepness_lower)
                self.assertLessEqual(sample.carrier_steepness, CARRIER_STEEPNESS_MAX)
                self.assertTrue(
                    PERTURBATION_RATIO_MIN
                    <= sample.perturbation_ratio
                    < PAPER_PERTURBATION_RATIO_MAX
                )
                self.assertTrue(0.0 <= sample.translation < PAPER_DOMAIN_LENGTH)

                band_fraction = float(
                    instability_band_fraction(
                        carrier_mode,
                        sideband_offset,
                        sample.carrier_steepness,
                    )
                )
                focused_steepness = float(
                    focused_steepness_proxy(
                        carrier_mode,
                        sideband_offset,
                        sample.carrier_steepness,
                    )
                )
                self.assertTrue(0.0 < band_fraction < 1.0)
                self.assertLessEqual(
                    focused_steepness,
                    np.nextafter(PAPER_FOCUSED_STEEPNESS_LIMIT, np.inf),
                )


if __name__ == "__main__":
    unittest.main()
