"""CPU tests for the paper-corpus Benjamin--Feir population sampler."""

from __future__ import annotations

from collections import Counter
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
    PERTURBATION_RATIO_MAX,
    PERTURBATION_RATIO_MIN,
)
from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BENJAMIN_FEIR_POPULATION_CELLS,
    BenjaminFeirPopulationCell,
    benjamin_feir_support_violations,
    pcg64_for_case,
    sample_benjamin_feir_population,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
    balanced_cell_quotas,
    schedule_attempt_batch,
)


DOMAIN_LENGTH = 2.0 * math.pi


def assignment(
    cell_index: int,
    *,
    family_id: int = 3,
    revision_id: int = 1,
    split_id: SplitId = SplitId.TRAIN,
    stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a declared population cell."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        case_key=CaseKey(
            family_id=family_id,
            revision_id=revision_id,
            split_id=split_id,
            stream_id=stream_id,
            attempt_index=attempt,
        ),
        cell_id=BENJAMIN_FEIR_POPULATION_CELLS[cell_index].cell_id,
    )


class BenjaminFeirPopulationTest(unittest.TestCase):
    def test_cells_are_exactly_all_66_feasible_integer_pairs(self) -> None:
        expected = tuple(
            (carrier_mode, sideband_offset)
            for carrier_mode in range(CARRIER_MODE_MIN, CARRIER_MODE_MAX + 1)
            for sideband_offset in range(1, carrier_mode)
            if sideband_offset / carrier_mode
            < 2.0 * math.sqrt(2.0) * CARRIER_STEEPNESS_MAX
        )
        realized = tuple(
            (cell.carrier_mode, cell.sideband_offset)
            for cell in BENJAMIN_FEIR_POPULATION_CELLS
        )
        self.assertEqual(len(realized), 66)
        self.assertEqual(realized, expected)
        self.assertEqual(
            len({cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS}),
            66,
        )

    def test_common_scheduler_balances_the_pair_cells_exactly(self) -> None:
        cell_ids = tuple(
            cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS
        )
        quotas = balanced_cell_quotas(cell_ids, accepted_case_count=20_000)
        scheduled = schedule_attempt_batch(
            quotas,
            {},
            family_id=3,
            revision_id=1,
            split_id=SplitId.TRAIN,
            stream_id=0,
            first_attempt_index=0,
            batch_size=20_000,
        )
        counts = Counter(item.cell_id for item in scheduled)
        self.assertEqual(len(scheduled), 20_000)
        self.assertEqual(set(counts), set(cell_ids))
        self.assertEqual(set(counts.values()), {303, 304})
        self.assertEqual(
            sum(count == 304 for count in counts.values()),
            2,
        )

    def test_replay_is_bitwise_deterministic_and_constructor_ready(self) -> None:
        first = sample_benjamin_feir_population(
            assignment(37, attempt_index=91)
        )
        second = sample_benjamin_feir_population(
            assignment(37, attempt_index=91)
        )
        self.assertEqual(first, second)
        self.assertEqual(first.to_json_record(), second.to_json_record())
        first_arrays = first.to_parameter_arrays()
        second_arrays = second.to_parameter_arrays()
        self.assertEqual(set(first_arrays), set(second_arrays))
        for name in first_arrays:
            np.testing.assert_array_equal(first_arrays[name], second_arrays[name])

    def test_every_key_coordinate_enters_the_pcg64_seed(self) -> None:
        base = assignment(0, attempt_index=41).case_key
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(base.seed_words))
        ).random(8)
        np.testing.assert_array_equal(pcg64_for_case(base).random(8), expected)

        keys = (
            base,
            CaseKey(4, 1, SplitId.TRAIN, 0, 41),
            CaseKey(3, 2, SplitId.TRAIN, 0, 41),
            CaseKey(3, 1, SplitId.VALIDATION, 0, 41),
            CaseKey(3, 1, SplitId.TRAIN, 1, 41),
            CaseKey(3, 1, SplitId.TRAIN, 0, 42),
        )
        first_draws = {tuple(pcg64_for_case(key).random(8)) for key in keys}
        self.assertEqual(len(first_draws), len(keys))

    def test_all_cells_remain_in_support_under_many_attempts(self) -> None:
        for cell_index, cell in enumerate(BENJAMIN_FEIR_POPULATION_CELLS):
            for local_attempt in range(128):
                sample = sample_benjamin_feir_population(
                    assignment(
                        cell_index,
                        attempt_index=10_000 * cell_index + local_attempt,
                    )
                )
                self.assertEqual(benjamin_feir_support_violations(sample), ())
                self.assertEqual(sample.cell, cell)
                self.assertGreater(
                    sample.carrier_steepness,
                    cell.conditional_steepness_lower_bound,
                )
                self.assertLessEqual(
                    sample.carrier_steepness,
                    CARRIER_STEEPNESS_MAX,
                )
                self.assertGreaterEqual(
                    sample.perturbation_ratio,
                    PERTURBATION_RATIO_MIN,
                )
                self.assertLessEqual(
                    sample.perturbation_ratio,
                    PERTURBATION_RATIO_MAX,
                )
                self.assertTrue(0.0 <= sample.sideband_phase < 2.0 * math.pi)
                self.assertTrue(0.0 < sample.band_fraction < 1.0)
                self.assertGreaterEqual(sample.left_mode, 1)

    def test_json_persists_parameters_phases_and_case_coordinates(self) -> None:
        sample = sample_benjamin_feir_population(
            assignment(
                65,
                family_id=17,
                revision_id=6,
                split_id=SplitId.TEST,
                stream_id=29,
                attempt_index=123_456,
            )
        )
        record = sample.to_json_record()
        key = sample.assignment.case_key
        self.assertEqual(record["case_id"], key.case_id)
        self.assertEqual(record["family_id"], 17)
        self.assertEqual(record["revision_id"], 6)
        self.assertEqual(record["split_id"], "test")
        self.assertEqual(record["root_seed"], key.root_seed)
        self.assertEqual(record["stream_id"], 29)
        self.assertEqual(record["attempt_index"], 123_456)
        self.assertEqual(record["seed_words"], list(key.seed_words))
        self.assertEqual(record["carrier_mode"], sample.cell.carrier_mode)
        self.assertEqual(
            record["sideband_offset"],
            sample.cell.sideband_offset,
        )
        self.assertEqual(record["carrier_phase"], 0.0)
        self.assertEqual(record["sideband_phase"], sample.sideband_phase)
        self.assertEqual(
            record["carrier_steepness"],
            sample.carrier_steepness,
        )
        self.assertEqual(
            record["perturbation_ratio"],
            sample.perturbation_ratio,
        )
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def test_invalid_cells_parameters_and_lengths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Benjamin--Feir"):
            sample_benjamin_feir_population(
                AttemptAssignment(
                    case_key=assignment(0).case_key,
                    cell_id="not_a_pair",
                )
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            sample_benjamin_feir_population(
                assignment(0),
                domain_length=math.nan,
            )
        with self.assertRaisesRegex(ValueError, "instability band"):
            BenjaminFeirPopulationCell(
                carrier_mode=4,
                sideband_offset=2,
            )
        with self.assertRaisesRegex(ValueError, "declared support"):
            BenjaminFeirPopulationCell(
                carrier_mode=21,
                sideband_offset=1,
            )

        sample = sample_benjamin_feir_population(assignment(0))
        corrupt = replace(sample, carrier_steepness=math.nan)
        self.assertIn(
            "carrier_steepness is outside its conditional support",
            benjamin_feir_support_violations(corrupt),
        )
        with self.assertRaisesRegex(ValueError, "cannot serialize unsupported"):
            corrupt.to_json_record()


if __name__ == "__main__":
    unittest.main()
