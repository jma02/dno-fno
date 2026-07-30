"""Focused CPU tests for the paper-corpus population smoke."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_population_500k_smoke import (
    FAMILY_ORDER,
    run_smoke,
    uniform_ecdf_distance,
)
from solver.gen_data.pipeline.archive import file_sha256


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


class PopulationSmokeTests(unittest.TestCase):
    def test_uniform_ecdf_distance(self) -> None:
        self.assertAlmostEqual(
            uniform_ecdf_distance(np.asarray([0.25, 0.75])),
            0.25,
        )
        self.assertAlmostEqual(
            uniform_ecdf_distance(np.asarray([0.0, 1.0])),
            0.5,
        )
        for invalid in (
            np.asarray([]),
            np.asarray([-0.01, 0.5]),
            np.asarray([0.5, 1.01]),
            np.asarray([0.5, np.nan]),
        ):
            with self.assertRaises(ValueError):
                uniform_ecdf_distance(invalid)

    def test_small_complete_run_is_strict_and_replayable(self) -> None:
        # Sixty-six cases per family exercises every allocation cell, including
        # all 66 Benjamin--Feir mode/offset cells.
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "smoke"
            summary = run_smoke(
                total_samples=264,
                batch_size=31,
                output_dir=output,
            )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                summary["counts"],
                {
                    "accepted_specifications": 264,
                    "attempted_specifications": 264,
                    "families": 4,
                    "accepted_per_family": 66,
                },
            )
            self.assertTrue(summary["runtime"]["cpu_only_verified"])
            self.assertTrue(summary["sentinel_validation"]["replay_verified"])

            strict = json.loads(
                (output / "summary.json").read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite,
            )
            for family in FAMILY_ORDER:
                family_summary = strict["families"][family]
                self.assertEqual(family_summary["accepted_count"], 66)
                self.assertEqual(family_summary["support_violation_count"], 0)
                archive = strict["artifacts"][family]
                path = output / archive["path"]
                self.assertEqual(file_sha256(path), archive["sha256"])
                with np.load(path, allow_pickle=False) as arrays:
                    self.assertEqual(arrays["case_id"].shape, (66,))
                    self.assertEqual(np.unique(arrays["case_id"]).size, 66)

            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                run_smoke(
                    total_samples=264,
                    batch_size=31,
                    output_dir=output,
                )

    def test_total_must_divide_evenly_across_families(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            with self.assertRaisesRegex(ValueError, "divisible by four"):
                run_smoke(
                    total_samples=263,
                    batch_size=31,
                    output_dir=Path(raw_directory) / "smoke",
                )


if __name__ == "__main__":
    unittest.main()
