"""Tests for shared stored-trajectory health checks."""
from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.health import (
    StoredFrameStatistics,
    StoredTrajectoryThresholds,
    evaluate_stored_frame_statistics,
    evaluate_stored_trajectory,
)
from solver.gen_data.pipeline.quality import QualityReason, QualityScope


class StoredTrajectoryHealthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eta = np.full((3, 8), 0.1, dtype=np.float64)
        self.xi = np.zeros_like(self.eta)
        self.gxi = np.zeros_like(self.eta)

    def test_stationary_finite_trajectory_passes(self) -> None:
        metrics, decision = evaluate_stored_trajectory(
            self.eta,
            self.xi,
            self.gxi,
            depth=1.0,
            gravity=1.0,
            dx=0.25,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(metrics.maximum_hamiltonian_drift, 0.0)
        self.assertAlmostEqual(metrics.minimum_water_column_fraction, 1.1)

    def test_nonfinite_target_is_rejected_without_energy_claim(self) -> None:
        gxi = self.gxi.copy()
        gxi[-1, 0] = np.nan
        metrics, decision = evaluate_stored_trajectory(
            self.eta,
            self.xi,
            gxi,
            depth=1.0,
            gravity=1.0,
            dx=0.25,
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(decision.failed & QualityReason.NONFINITE_TARGET)
        self.assertFalse(decision.evaluated & QualityReason.HAMILTONIAN_DRIFT)
        self.assertIsNone(metrics.maximum_hamiltonian_drift)

    def test_bottom_collision_is_rejected(self) -> None:
        eta = self.eta.copy()
        eta[1, 2] = -1.0
        _, decision = evaluate_stored_trajectory(
            eta,
            self.xi,
            self.gxi,
            depth=1.0,
            gravity=1.0,
            dx=0.25,
        )

        self.assertTrue(decision.failed & QualityReason.BOTTOM_CLEARANCE)
        self.assertFalse(decision.accepted)

    def test_positive_water_column_passes(self) -> None:
        eta = self.eta.copy()
        eta[1, 2] = -0.5
        _, decision = evaluate_stored_trajectory(
            eta,
            self.xi,
            self.gxi,
            depth=1.0,
            gravity=1.0,
            dx=0.25,
        )

        self.assertFalse(decision.failed & QualityReason.BOTTOM_CLEARANCE)

    def test_hamiltonian_drift_is_rejected(self) -> None:
        eta = self.eta.copy()
        eta[-1] *= 1.1
        metrics, decision = evaluate_stored_trajectory(
            eta,
            self.xi,
            self.gxi,
            depth=1.0,
            gravity=1.0,
            dx=0.25,
            thresholds=StoredTrajectoryThresholds(hamiltonian_drift=1e-3),
        )

        self.assertIsNotNone(metrics.maximum_hamiltonian_drift)
        self.assertGreater(metrics.maximum_hamiltonian_drift, 1e-3)
        self.assertTrue(decision.failed & QualityReason.HAMILTONIAN_DRIFT)
        self.assertTrue(decision.accepted)

    def test_static_sample_does_not_claim_hamiltonian_drift(self) -> None:
        metrics, decision = evaluate_stored_frame_statistics(
            StoredFrameStatistics(
                state_finite=True,
                target_finite=True,
                minimum_water_column_fraction=1.1,
                initial_hamiltonian=12.0,
                minimum_hamiltonian=1.0,
                maximum_hamiltonian=20.0,
                hamiltonian_scale=None,
            ),
            scope=QualityScope.SAMPLE,
        )

        self.assertTrue(decision.accepted)
        self.assertFalse(decision.evaluated & QualityReason.HAMILTONIAN_DRIFT)
        self.assertIsNone(metrics.maximum_hamiltonian_drift)


if __name__ == "__main__":
    unittest.main()
