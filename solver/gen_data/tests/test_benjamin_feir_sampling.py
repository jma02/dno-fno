"""CPU tests for the paper-dataset Benjamin--Feir sampler."""

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
    JCP09_RELATIVE_SIDEBAND_PHASE,
    PERTURBATION_RATIO_MIN,
)
from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_SAMPLE_CELL_IDS,
    BENJAMIN_FEIR_SAMPLE_CELLS,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    PAPER_PERTURBATION_RATIO_MAX,
    find_benjamin_feir_sample_violations,
    sample_benjamin_feir_simulation,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    AttemptAssignment,
    SimulationKey,
    SplitId,
    balanced_simulation_targets,
    random_generator_for_simulation,
    build_next_attempt_batch,
)


DOMAIN_LENGTH = 2.0 * math.pi


def assignment(
    cell_index: int,
    *,
    family_id: int = 3,
    revision_id: int = 4,
    split_id: SplitId = SplitId.TRAIN,
    stream_id: int = 0,
    attempt_index: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic assignment for a declared sample cell."""

    attempt = cell_index if attempt_index is None else attempt_index
    return AttemptAssignment(
        simulation_key=SimulationKey(
            family_id=family_id,
            revision_id=revision_id,
            split_id=split_id,
            stream_id=stream_id,
            attempt_index=attempt,
        ),
        cell_id=BENJAMIN_FEIR_SAMPLE_CELL_IDS[cell_index],
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
        realized = tuple(BENJAMIN_FEIR_SAMPLE_CELLS.values())
        self.assertEqual(len(realized), 66)
        self.assertEqual(realized, expected)
        self.assertEqual(
            len(BENJAMIN_FEIR_SAMPLE_CELL_IDS),
            66,
        )

    def test_common_scheduler_balances_the_pair_cells_exactly(self) -> None:
        cell_ids = BENJAMIN_FEIR_SAMPLE_CELL_IDS
        targets = balanced_simulation_targets(cell_ids, simulation_count=20_000)
        scheduled = build_next_attempt_batch(
            targets,
            {},
            {},
            {target.cell_id: target.simulation_count for target in targets},
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
        first = sample_benjamin_feir_simulation(assignment(37, attempt_index=91))
        second = sample_benjamin_feir_simulation(assignment(37, attempt_index=91))
        self.assertEqual(first, second)
        self.assertEqual(first.to_json_record(), second.to_json_record())
        first_arrays = first.to_parameter_arrays()
        second_arrays = second.to_parameter_arrays()
        self.assertEqual(set(first_arrays), set(second_arrays))
        for name in first_arrays:
            np.testing.assert_array_equal(first_arrays[name], second_arrays[name])

    def test_every_key_coordinate_enters_the_pcg64_seed(self) -> None:
        base = assignment(0, attempt_index=41).simulation_key
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(base.seed_words))
        ).random(8)
        np.testing.assert_array_equal(
            random_generator_for_simulation(base).random(8), expected
        )

        keys = (
            base,
            SimulationKey(4, 4, SplitId.TRAIN, 0, 41),
            SimulationKey(3, 5, SplitId.TRAIN, 0, 41),
            SimulationKey(3, 4, SplitId.VALIDATION, 0, 41),
            SimulationKey(3, 4, SplitId.TRAIN, 1, 41),
            SimulationKey(3, 4, SplitId.TRAIN, 0, 42),
        )
        first_draws = {
            tuple(random_generator_for_simulation(key).random(8)) for key in keys
        }
        self.assertEqual(len(first_draws), len(keys))

    def test_all_cells_remain_in_support_under_many_attempts(self) -> None:
        for cell_index, (carrier_mode, sideband_offset) in enumerate(
            BENJAMIN_FEIR_SAMPLE_CELLS.values()
        ):
            for local_attempt in range(128):
                sample = sample_benjamin_feir_simulation(
                    assignment(
                        cell_index,
                        attempt_index=10_000 * cell_index + local_attempt,
                    )
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
        for cell_index in range(len(BENJAMIN_FEIR_SAMPLE_CELL_IDS)):
            sample = sample_benjamin_feir_simulation(assignment(cell_index))
            lower, upper = sample.conditional_steepness_bounds
            self.assertLess(
                lower,
                upper,
            )
            self.assertLessEqual(
                upper,
                CARRIER_STEEPNESS_MAX,
            )

    def test_json_persists_parameters_phases_and_simulation_coordinates(self) -> None:
        sample = sample_benjamin_feir_simulation(
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
        key = sample.assignment.simulation_key
        self.assertNotIn("schema", record)
        self.assertEqual(record["simulation_id"], key.simulation_id)
        self.assertEqual(record["family_id"], 17)
        self.assertEqual(record["revision_id"], 6)
        self.assertEqual(record["split_id"], "test")
        self.assertEqual(record["root_seed"], key.root_seed)
        self.assertEqual(record["stream_id"], 29)
        self.assertEqual(record["attempt_index"], 123_456)
        self.assertEqual(record["seed_words"], list(key.seed_words))
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
                AttemptAssignment(
                    simulation_key=assignment(0).simulation_key,
                    cell_id="not_a_pair",
                )
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            sample_benjamin_feir_simulation(
                assignment(0),
                domain_length=math.nan,
            )
        sample = sample_benjamin_feir_simulation(assignment(0))
        wrong_cell = replace(sample, carrier_mode=21)
        self.assertIn(
            "sample parameters do not match the assigned cell",
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
