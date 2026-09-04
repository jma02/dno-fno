"""CPU tests for the paper-dataset Stokes sampler."""

from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEEP_DEPTH_BOUNDS,
    DEEP_DEPTH_WAVENUMBER_MINIMUM,
    FINITE_DEPTH_BOUNDS,
    FINITE_DEPTH_WAVENUMBER_BOUNDS,
    PAPER_AMPLITUDE_BOUNDS,
    PAPER_DOMAIN_LENGTH,
    PAPER_GRAVITY,
    STOKES_PARAMETER_GROUP_IDS,
    STOKES_PARAMETER_GROUPS,
    StokesSample,
    sample_stokes_simulation,
)
from solver.reference_solutions.stokes_wave import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
)

jax.config.update("jax_enable_x64", True)


def sample_stokes(
    parameter_group_id: str,
    *,
    attempt_number: int,
    dataset_split: DatasetSplit = DatasetSplit.TRAIN,
) -> StokesSample | None:
    return sample_stokes_simulation(
        parameter_group_id,
        dataset_split=dataset_split,
        attempt_number=attempt_number,
    )


class StokesSamplingTest(unittest.TestCase):
    def test_parameter_groups_are_the_exact_branch_steepness_product(self) -> None:
        self.assertEqual(
            STOKES_PARAMETER_GROUPS,
            {
                "finite_low": ("finite", (0.005, 0.03), tuple(range(14, 27))),
                "finite_moderate": (
                    "finite",
                    (0.03, 0.15),
                    tuple(range(14, 27)),
                ),
                "deep_low": ("deep", (0.005, 0.03), tuple(range(1, 21))),
                "deep_moderate": ("deep", (0.03, 0.15), tuple(range(3, 21))),
            },
        )

    def test_sampling_is_deterministic_for_split_group_and_attempt(self) -> None:
        for attempt_number, parameter_group_id in enumerate(
            STOKES_PARAMETER_GROUP_IDS, start=90
        ):
            with self.subTest(parameter_group=parameter_group_id):
                first = sample_stokes(
                    parameter_group_id,
                    attempt_number=attempt_number,
                )
                second = sample_stokes(
                    parameter_group_id,
                    attempt_number=attempt_number,
                )
                validation = sample_stokes(
                    parameter_group_id,
                    attempt_number=attempt_number,
                    dataset_split=DatasetSplit.VALIDATION,
                )
                self.assertEqual(first, second)
                self.assertNotEqual(first, validation)

    def test_many_draws_cover_modes_and_obey_support(self) -> None:
        for parameter_group_id, (
            branch,
            steepness_bounds,
            carrier_modes,
        ) in STOKES_PARAMETER_GROUPS.items():
            seen_modes: set[int] = set()
            for attempt_number in range(512):
                sample = sample_stokes(
                    parameter_group_id,
                    attempt_number=attempt_number,
                )
                self.assertIsNotNone(sample)
                assert sample is not None
                seen_modes.add(sample.carrier_mode)
                self.assertEqual(sample.branch, branch)
                self.assertIn(sample.carrier_mode, carrier_modes)
                self.assertTrue(0.0 <= sample.phase < 2.0 * math.pi)

                wavenumber = 2.0 * math.pi * sample.carrier_mode / PAPER_DOMAIN_LENGTH
                amplitude_lower = max(
                    PAPER_AMPLITUDE_BOUNDS[0], steepness_bounds[0] / wavenumber
                )
                amplitude_upper = min(
                    PAPER_AMPLITUDE_BOUNDS[1], steepness_bounds[1] / wavenumber
                )
                self.assertTrue(amplitude_lower <= sample.amplitude < amplitude_upper)

                if branch == "finite":
                    depth_lower = max(
                        FINITE_DEPTH_BOUNDS[0],
                        FINITE_DEPTH_WAVENUMBER_BOUNDS[0] / wavenumber,
                    )
                    depth_upper = min(
                        FINITE_DEPTH_BOUNDS[1],
                        FINITE_DEPTH_WAVENUMBER_BOUNDS[1] / wavenumber,
                    )
                    self.assertTrue(depth_lower <= sample.depth <= depth_upper)
                    with jax.enable_x64():
                        ursell = finite_depth_stokes_ursell_upper_bound(
                            jnp.asarray(wavenumber, dtype=jnp.float64),
                            jnp.asarray(sample.depth, dtype=jnp.float64),
                            jnp.asarray(PAPER_GRAVITY, dtype=jnp.float64),
                            jnp.asarray(sample.amplitude, dtype=jnp.float64),
                        )
                    self.assertLessEqual(
                        float(np.asarray(jax.device_get(ursell))),
                        FINITE_DEPTH_STOKES_URSELL_LIMIT,
                    )
                else:
                    depth_lower = max(
                        DEEP_DEPTH_BOUNDS[0],
                        DEEP_DEPTH_WAVENUMBER_MINIMUM / wavenumber,
                    )
                    self.assertTrue(depth_lower <= sample.depth <= DEEP_DEPTH_BOUNDS[1])

            self.assertEqual(seen_modes, set(carrier_modes))

    def test_finite_depth_ursell_failure_redraws_only_amplitude(self) -> None:
        calls: list[tuple[float, float, float, float]] = []
        ursell_values = iter((27.0, 25.0))

        def controlled_ursell(
            wavenumber: jax.Array,
            depth: jax.Array,
            gravity: jax.Array,
            amplitude: jax.Array,
        ) -> jax.Array:
            calls.append(
                (
                    float(np.asarray(wavenumber)),
                    float(np.asarray(depth)),
                    float(np.asarray(gravity)),
                    float(np.asarray(amplitude)),
                )
            )
            return jnp.asarray(next(ursell_values), dtype=jnp.float64)

        with (
            patch(
                "solver.gen_data.stokes_sampling._compiled_finite_depth_ursell",
                side_effect=controlled_ursell,
            ),
            patch("solver.gen_data.stokes_sampling.MAXIMUM_URSELL_REDRAWS", 1),
        ):
            sample = sample_stokes(
                "finite_low",
                attempt_number=500,
            )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:3], calls[1][:3])
        self.assertNotEqual(calls[0][3], calls[1][3])
        self.assertEqual(sample.amplitude, calls[1][3])

    def test_deep_water_skips_ursell_and_exhaustion_returns_none(self) -> None:
        with patch(
            "solver.gen_data.stokes_sampling._compiled_finite_depth_ursell"
        ) as ursell:
            deep = sample_stokes("deep_low", attempt_number=501)
        self.assertIsNotNone(deep)
        ursell.assert_not_called()

        with (
            patch(
                "solver.gen_data.stokes_sampling._compiled_finite_depth_ursell",
                return_value=jnp.asarray(math.inf, dtype=jnp.float64),
            ) as ursell,
            patch("solver.gen_data.stokes_sampling.MAXIMUM_URSELL_REDRAWS", 2),
        ):
            exhausted = sample_stokes(
                "finite_moderate",
                attempt_number=502,
            )
        self.assertIsNone(exhausted)
        self.assertEqual(ursell.call_count, 3)


if __name__ == "__main__":
    unittest.main()
