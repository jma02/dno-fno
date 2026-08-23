"""Regression tests for collision-free adaptive time indices."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.adaptive_sampling import adaptive_indices_from_signal


class AdaptiveIndicesFromSignalTests(unittest.TestCase):
    def test_terminal_spike_produces_unique_indices(self) -> None:
        signal = np.zeros((1, 16), dtype=np.float64)
        signal[0, -1] = 1.0

        indices = adaptive_indices_from_signal(
            signal,
            keep_samples=8,
            alpha=1.0,
            smooth_sigma_steps=0.0,
            pin_endpoints=False,
            density_mode="magnitude",
        )

        self.assertTrue(np.all(np.diff(indices, axis=1) > 0))
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), signal.shape[1])

    def test_terminal_spike_preserves_pinned_endpoints(self) -> None:
        signal = np.zeros((2, 16), dtype=np.float64)
        signal[0, -1] = 1.0
        signal[1, 0] = 1.0

        indices = adaptive_indices_from_signal(
            signal,
            keep_samples=8,
            alpha=1.0,
            smooth_sigma_steps=0.0,
            pin_endpoints=True,
            density_mode="magnitude",
        )

        self.assertTrue(np.all(np.diff(indices, axis=1) > 0))
        np.testing.assert_array_equal(indices[:, 0], 0)
        np.testing.assert_array_equal(indices[:, -1], signal.shape[1] - 1)


if __name__ == "__main__":
    unittest.main()
