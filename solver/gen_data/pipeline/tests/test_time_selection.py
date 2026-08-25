"""CPU tests for deterministic post-acceptance time selection."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.time_selection import (
    _midpoint_quantile_indices,
    select_tanaka_times,
    select_uniform_times,
)


class TemporalSelectionTest(unittest.TestCase):
    def test_tanaka_rule_is_translation_invariant_and_endpoint_pinned(self) -> None:
        nx = 64
        times = np.linspace(0.0, 1.0, 301)
        x = 2.0 * np.pi * np.arange(nx) / nx
        amplitude = 0.01 + 0.02 * np.exp(-((times - 0.63) / 0.08) ** 2)
        eta = amplitude[:, None] * np.cos(5.0 * x)[None, :]
        shifted = np.roll(eta, 17, axis=-1)

        selection = select_tanaka_times(
            eta,
            length=2.0 * np.pi,
            keep_samples=40,
            sigma_steps=5.0,
        )
        translated = select_tanaka_times(
            shifted,
            length=2.0 * np.pi,
            keep_samples=40,
            sigma_steps=5.0,
        )

        np.testing.assert_array_equal(selection.indices, translated.indices)
        np.testing.assert_allclose(
            selection.activity,
            translated.activity,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        self.assertEqual((selection.indices[0], selection.indices[-1]), (0, 300))
        self.assertTrue(np.all(np.diff(selection.indices) > 0))
        self.assertAlmostEqual(float(np.sum(selection.density)), 1.0)

    def test_concentrated_density_still_returns_unique_indices(self) -> None:
        density = np.zeros(251, dtype=np.float64)
        density[-2] = 1.0
        indices = _midpoint_quantile_indices(density, keep_samples=200)
        self.assertEqual((indices[0], indices[-1]), (0, 250))
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_uniform_rule_is_reproducible_and_nearest_to_exact_grid(self) -> None:
        indices = select_uniform_times(359, keep_samples=16)
        expected_real = np.linspace(0.0, 358.0, 16)
        np.testing.assert_array_less(np.abs(indices - expected_real), 0.5000001)
        np.testing.assert_array_equal(indices, select_uniform_times(359))
        self.assertEqual((indices[0], indices[-1]), (0, 358))

    def test_nonfinite_surface_fails_closed(self) -> None:
        eta = np.ones((20, 16), dtype=np.float64)
        eta[4, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            select_tanaka_times(
                eta,
                length=2.0 * np.pi,
                keep_samples=10,
            )


if __name__ == "__main__":
    unittest.main()
