"""CPU tests for finite-Stokes Ursell-support sampling."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.data.stokes_truth_jax import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
)
from solver.gen_data.generate_stokes_dataset import (  # noqa: E402
    StokesSupportSamplingError,
    sample_stokes_case_params,
    serialize_specs,
    validate_paper_target_configuration,
)

jax.config.update("jax_enable_x64", True)


class StokesUrsellSamplingTest(unittest.TestCase):
    def test_noncanonical_paper_target_configuration_is_rejected(self) -> None:
        canonical = {
            "nx": 1024,
            "length": 2.0 * np.pi,
            "gravity": 1.0,
            "dno_order": 6,
            "pad_factor": 8,
            "rollout_dtype": "float64",
        }
        validate_paper_target_configuration(**canonical)
        for name, value in (
            ("nx", 512),
            ("length", 10.0),
            ("gravity", 9.81),
            ("dno_order", 5),
            ("pad_factor", 4),
            ("rollout_dtype", "float32"),
        ):
            changed = {**canonical, name: value}
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "requires"):
                    validate_paper_target_configuration(**changed)

    def test_unsupported_amplitudes_are_redrawn_in_the_same_cell(self) -> None:
        params = sample_stokes_case_params(
            np.random.default_rng(0),
            batch_size=64,
            length=2.0 * np.pi,
            gravity=1.0,
            n0_min=14,
            n0_max=26,
            a0_min=0.000766,
            a0_max=0.011494,
            steepness_max=0.15,
            depth_min=0.02,
            depth_max=1.5,
            kh_min=0.5,
            kh_max=5.0,
            rejection_attempts=1000,
            finite_depth=True,
        )

        self.assertLessEqual(
            float(np.max(params["ursell_upper_bound"])),
            FINITE_DEPTH_STOKES_URSELL_LIMIT,
        )
        self.assertGreater(
            int(np.sum(params["support_resampling_count"])),
            0,
        )
        rejection_count = params["support_resampling_count"]
        self.assertEqual(
            params["rejected_a0"].shape,
            params["rejected_ursell_upper_bound"].shape,
        )
        self.assertEqual(
            params["rejected_a0"].shape[1],
            int(np.max(rejection_count)),
        )
        for case_index, count in enumerate(rejection_count):
            rejected_ursell = params["rejected_ursell_upper_bound"][
                case_index, :count
            ]
            self.assertTrue(
                np.all(
                    ~np.isfinite(rejected_ursell)
                    | (
                        rejected_ursell
                        > FINITE_DEPTH_STOKES_URSELL_LIMIT
                    )
                )
            )
        serialized = serialize_specs(params, ichoi=1)
        self.assertTrue(
            all(
                len(specification["rejected_a0"])
                == specification["support_resampling_count"]
                for specification in serialized
            )
        )
        wavenumber = params["n0"].astype(np.float64)
        self.assertEqual(
            set(params["steepness_cell_index"].tolist()),
            {0, 1},
        )
        cell_counts = np.bincount(
            params["steepness_cell_index"],
            minlength=2,
        )
        self.assertLessEqual(abs(int(cell_counts[0] - cell_counts[1])), 1)
        self.assertTrue(np.all(params["phase"] >= 0.0))
        self.assertTrue(np.all(params["phase"] < 2.0 * np.pi))
        np.testing.assert_array_less(
            params["steepness_cell_lower"] - 1e-14,
            wavenumber * params["a0"],
        )
        np.testing.assert_array_less(
            wavenumber * params["a0"],
            params["steepness_cell_upper"] + 1e-14,
        )
        for case_index, count in enumerate(rejection_count):
            rejected_amplitudes = params["rejected_a0"][
                case_index, :count
            ]
            self.assertTrue(
                np.all(
                    rejected_amplitudes
                    >= params["amplitude_cell_lower"][case_index]
                )
            )
            self.assertTrue(
                np.all(
                    rejected_amplitudes
                    < params["amplitude_cell_upper"][case_index]
                )
            )
        np.testing.assert_array_less(
            np.full(64, 0.5 - 1e-14),
            wavenumber * params["depth"],
        )
        np.testing.assert_array_less(
            wavenumber * params["depth"],
            np.full(64, 5.0 + 1e-14),
        )

    def test_valid_last_allowed_redraw_is_retained(self) -> None:
        amplitude_draws = (
            np.asarray([0.005], dtype=np.float64),
            np.asarray([0.003], dtype=np.float64),
        )
        ursell_values = (
            np.asarray([27.0], dtype=np.float64),
            np.asarray([25.0], dtype=np.float64),
        )
        with (
            patch(
                "solver.gen_data.generate_stokes_dataset."
                "_sample_amplitudes_in_bounds",
                side_effect=amplitude_draws,
            ),
            patch(
                "solver.gen_data.generate_stokes_dataset."
                "_finite_depth_ursell_batch",
                side_effect=ursell_values,
            ),
        ):
            params = sample_stokes_case_params(
                np.random.default_rng(5),
                batch_size=1,
                length=2.0 * np.pi,
                gravity=1.0,
                n0_min=14,
                n0_max=14,
                a0_min=0.0005,
                a0_max=0.012,
                steepness_max=0.2,
                depth_min=0.05,
                depth_max=0.05,
                kh_min=0.0,
                kh_max=np.inf,
                rejection_attempts=1,
                finite_depth=True,
            )

        self.assertEqual(int(params["support_resampling_count"][0]), 1)
        self.assertAlmostEqual(float(params["a0"][0]), 0.003)
        self.assertEqual(int(params["steepness_cell_index"][0]), 1)
        self.assertGreaterEqual(
            float(params["rejected_a0"][0, 0]),
            float(params["amplitude_cell_lower"][0]),
        )
        self.assertLess(
            float(params["rejected_a0"][0, 0]),
            float(params["amplitude_cell_upper"][0]),
        )
        self.assertLessEqual(
            float(params["ursell_upper_bound"][0]),
            FINITE_DEPTH_STOKES_URSELL_LIMIT,
        )

    def test_exhausted_redraws_retain_the_complete_failed_history(self) -> None:
        amplitude_draws = (
            np.asarray([0.005], dtype=np.float64),
            np.asarray([0.004], dtype=np.float64),
        )
        ursell_values = (
            np.asarray([np.inf], dtype=np.float64),
            np.asarray([27.0], dtype=np.float64),
        )
        with (
            patch(
                "solver.gen_data.generate_stokes_dataset."
                "_sample_amplitudes_in_bounds",
                side_effect=amplitude_draws,
            ),
            patch(
                "solver.gen_data.generate_stokes_dataset."
                "_finite_depth_ursell_batch",
                side_effect=ursell_values,
            ),
            self.assertRaises(StokesSupportSamplingError) as caught,
        ):
            sample_stokes_case_params(
                np.random.default_rng(5),
                batch_size=1,
                length=2.0 * np.pi,
                gravity=1.0,
                n0_min=14,
                n0_max=14,
                a0_min=0.0005,
                a0_max=0.012,
                steepness_max=0.2,
                depth_min=0.05,
                depth_max=0.05,
                kh_min=0.0,
                kh_max=np.inf,
                rejection_attempts=1,
                finite_depth=True,
            )

        record = caught.exception.case_records[0]
        self.assertEqual(record["status"], "failed_ursell_redraw_limit")
        self.assertEqual(record["support_resampling_count"], 1)
        self.assertEqual(record["rejected_a0"], [0.005, 0.004])
        self.assertEqual(
            record["rejected_ursell_upper_bound"],
            [None, 27.0],
        )


if __name__ == "__main__":
    unittest.main()
