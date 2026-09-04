"""CPU tests for deterministic post-acceptance time selection."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.time_selection import (
    select_tanaka_times,
    select_uniform_times,
)


class TemporalSelectionTest(unittest.TestCase):
    def test_tanaka_selection_is_translation_invariant_endpoint_pinned_and_unique(
        self,
    ) -> None:
        nx = 64
        times = np.linspace(0.0, 1.0, 301)
        x = 2.0 * np.pi * np.arange(nx) / nx
        amplitude = 0.01 + 0.02 * np.exp(-(((times - 0.63) / 0.08) ** 2))
        eta = amplitude[:, None] * np.cos(5.0 * x)[None, :]

        indices = select_tanaka_times(eta, length=2.0 * np.pi)
        translated = select_tanaka_times(np.roll(eta, 17, axis=-1), length=2.0 * np.pi)

        np.testing.assert_array_equal(indices, translated)
        self.assertEqual(indices.size, 200)
        self.assertEqual((indices[0], indices[-1]), (0, 300))
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_uniform_selection_is_exact_endpoint_pinned_and_unique(self) -> None:
        indices = select_uniform_times(359, keep_samples=16)
        expected = np.floor(np.arange(16) * 358 / 15 + 0.5).astype(np.int32)

        np.testing.assert_array_equal(indices, expected)
        self.assertEqual((indices[0], indices[-1]), (0, 358))
        self.assertTrue(np.all(np.diff(indices) > 0))


if __name__ == "__main__":
    unittest.main()
