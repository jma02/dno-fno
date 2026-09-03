"""CPU tests for the paper-dataset Benjamin--Feir sampler."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    CARRIER_MODE_MAX,
    CARRIER_MODE_MIN,
    CARRIER_STEEPNESS_MAX,
    JCP09_RELATIVE_SIDEBAND_PHASE,
    PERTURBATION_RATIO_MIN,
)
from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    BENJAMIN_FEIR_PARAMETER_GROUPS,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    PAPER_PERTURBATION_RATIO_MAX,
    BenjaminFeirSample,
    find_benjamin_feir_sample_violations,
    sample_benjamin_feir_simulation,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    DatasetSplit,
)


DOMAIN_LENGTH = 2.0 * math.pi


def sample_benjamin_feir(
    cell_index: int,
    *,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
    attempt_number: int | None = None,
    domain_length: float = DOMAIN_LENGTH,
) -> BenjaminFeirSample:
    """Sample one Benjamin--Feir parameter group for a test."""

    identifier = cell_index if attempt_number is None else attempt_number
    return sample_benjamin_feir_simulation(
        BENJAMIN_FEIR_PARAMETER_GROUP_IDS[cell_index],
        dataset_split=dataset_split,
        attempt_number=identifier,
        domain_length=domain_length,
    )


class BenjaminFeirSamplingTest(unittest.TestCase):
    def test_cells_are_exactly_all_66_feasible_integer_pairs(self) -> None:
        expected = tuple(
            (carrier_mode, sideband_offset)
            for carrier_mode in range(CARRIER_MODE_MIN, CARRIER_MODE_MAX + 1)
            for sideband_offset in range(1, carrier_mode)
            if sideband_offset / carrier_mode
            < 2.0 * math.sqrt(2.0) * CARRIER_STEEPNESS_MAX
        )
        realized = tuple(BENJAMIN_FEIR_PARAMETER_GROUPS.values())
        self.assertEqual(len(realized), 66)
        self.assertEqual(realized, expected)
        self.assertEqual(
            len(BENJAMIN_FEIR_PARAMETER_GROUP_IDS),
            66,
        )

    def test_replay_is_bitwise_deterministic_and_constructor_ready(self) -> None:
        first = sample_benjamin_feir(37, attempt_number=91)
        second = sample_benjamin_feir(37, attempt_number=91)
        self.assertEqual(first, second)
        self.assertEqual(first.to_json_record(), second.to_json_record())
        first_arrays = first.to_parameter_arrays()
        second_arrays = second.to_parameter_arrays()
        self.assertEqual(set(first_arrays), set(second_arrays))
        for name in first_arrays:
            np.testing.assert_array_equal(first_arrays[name], second_arrays[name])

    def test_all_cells_remain_in_support_under_many_attempts(self) -> None:
        for cell_index, (carrier_mode, sideband_offset) in enumerate(
            BENJAMIN_FEIR_PARAMETER_GROUPS.values()
        ):
            for local_attempt in range(128):
                sample = sample_benjamin_feir(
                    cell_index,
                    attempt_number=10_000 * cell_index + local_attempt,
                )
                self.assertEqual(find_benjamin_feir_sample_violations(sample), ())
                self.assertEqual(sample.carrier_mode, carrier_mode)
                self.assertEqual(sample.sideband_offset, sideband_offset)
                steepness_lower, steepness_upper = sample.conditional_steepness_bounds
                self.assertGreater(
                    sample.carrier_steepness,
                    steepness_lower,
                )
                self.assertLessEqual(
                    sample.carrier_steepness,
                    steepness_upper,
                )
                self.assertGreaterEqual(
                    sample.perturbation_ratio,
                    PERTURBATION_RATIO_MIN,
                )
                self.assertLessEqual(
                    sample.perturbation_ratio,
                    PAPER_PERTURBATION_RATIO_MAX,
                )
                self.assertTrue(0.0 <= sample.translation < 2.0 * math.pi)
                self.assertTrue(0.0 < sample.band_fraction < 1.0)
                self.assertLessEqual(
                    sample.focused_steepness,
                    np.nextafter(PAPER_FOCUSED_STEEPNESS_LIMIT, np.inf),
                )
                self.assertGreaterEqual(sample.left_mode, 1)

    def test_every_cell_has_a_nonempty_closed_form_focused_interval(
        self,
    ) -> None:
        for cell_index in range(len(BENJAMIN_FEIR_PARAMETER_GROUP_IDS)):
            sample = sample_benjamin_feir(cell_index)
            lower, upper = sample.conditional_steepness_bounds
            self.assertLess(
                lower,
                upper,
            )
            self.assertLessEqual(
                upper,
                CARRIER_STEEPNESS_MAX,
            )

    def test_json_persists_sampled_parameters_and_phases(self) -> None:
        sample = sample_benjamin_feir(
            65,
            dataset_split=DatasetSplit.TEST,
            attempt_number=123_456,
        )
        record = sample.to_json_record()
        self.assertNotIn("schema", record)
        self.assertEqual(record["carrier_mode"], sample.carrier_mode)
        self.assertEqual(
            record["sideband_offset"],
            sample.sideband_offset,
        )
        self.assertEqual(record["translation"], sample.translation)
        self.assertEqual(record["carrier_phase_in_translated_frame"], 0.0)
        self.assertEqual(
            record["relative_sideband_phase"],
            JCP09_RELATIVE_SIDEBAND_PHASE,
        )
        self.assertEqual(
            record["first_harmonic_carrier_steepness"],
            sample.carrier_steepness,
        )
        self.assertEqual(
            record["first_harmonic_sideband_ratio"],
            sample.perturbation_ratio,
        )
        self.assertEqual(record["focused_steepness"], sample.focused_steepness)
        self.assertEqual(
            record["focused_steepness_limit"],
            PAPER_FOCUSED_STEEPNESS_LIMIT,
        )
        self.assertNotIn("sideband_phase", record)
        self.assertNotIn("carrier_steepness", record)
        self.assertNotIn("perturbation_ratio", record)
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def test_invalid_cells_parameters_and_lengths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Benjamin--Feir"):
            sample_benjamin_feir_simulation(
                "not_a_pair",
                dataset_split=DatasetSplit.TRAIN,
                attempt_number=0,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            sample_benjamin_feir(0, domain_length=math.nan)
        sample = sample_benjamin_feir(0)
        wrong_cell = replace(sample, carrier_mode=21)
        self.assertIn(
            "sample parameters do not match the assigned parameter group",
            find_benjamin_feir_sample_violations(wrong_cell),
        )
        corrupt = replace(sample, carrier_steepness=math.nan)
        self.assertIn(
            "carrier_steepness is outside its conditional support",
            find_benjamin_feir_sample_violations(corrupt),
        )
        with self.assertRaisesRegex(ValueError, "cannot serialize unsupported"):
            corrupt.to_json_record()


if __name__ == "__main__":
    unittest.main()
