"""Smoke-level tests for the paper-dataset acceptance decisions."""
from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.acceptance import (
    TrajectorySamples,
    evaluate_dense_trajectory_health,
    evaluate_production_trajectory,
)
from solver.gen_data.pipeline.quality import QualityReason


def make_trajectory() -> TrajectorySamples:
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x = 2.0 * np.pi * np.arange(8, dtype=np.float64) / 8.0
    time_grid = times[:, None]
    eta = (1.0 + 0.1 * time_grid) * np.cos(x)[None, :]
    xi = (0.5 - 0.05 * time_grid) * np.sin(x)[None, :]
    gxi = (0.25 + 0.02 * time_grid) * np.cos(x)[None, :]
    return TrajectorySamples(times=times, eta=eta, xi=xi, gxi=gxi)


class CompleteNumericalTrajectoryTest(unittest.TestCase):
    def test_finite_complete_trajectory_passes_one_required_check(self) -> None:
        trajectory = make_trajectory()

        decision = evaluate_production_trajectory(
            trajectory,
            depth=2.0,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.required,
            QualityReason.INCOMPLETE_TRAJECTORY,
        )

    def test_numerical_failure_is_a_diagnostic_cause_of_incompleteness(
        self,
    ) -> None:
        trajectory = make_trajectory()
        nonfinite_target = trajectory.gxi.copy()
        nonfinite_target[-1, 0] = np.nan
        failed_trajectory = TrajectorySamples(
            times=trajectory.times,
            eta=trajectory.eta,
            xi=trajectory.xi,
            gxi=nonfinite_target,
            reached_final_time=True,
            gl2_stages_solved=False,
        )

        decision = evaluate_production_trajectory(
            failed_trajectory,
            depth=2.0,
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(decision.failed & QualityReason.INCOMPLETE_TRAJECTORY)
        self.assertTrue(decision.failed & QualityReason.NONFINITE_TARGET)
        self.assertTrue(decision.failed & QualityReason.GL2_STAGE_RESIDUAL)


class InternalTrajectoryHealthTest(unittest.TestCase):
    def test_each_internal_requirement_rejects_independently(self) -> None:
        healthy = {
            "hamiltonian": np.asarray([2.0, 2.001], dtype=np.float64),
            "state_finite": np.asarray([True, True]),
            "dno_output_finite": np.asarray([True, True]),
            "minimum_water_column": np.asarray([1.0, 0.5]),
        }

        metrics, decision = evaluate_dense_trajectory_health(
            **healthy,
            hamiltonian_drift_threshold=1.0e-3,
        )

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(
            metrics.maximum_relative_hamiltonian_drift,
            5.0e-4,
        )
        defects = (
            ("hamiltonian", np.asarray([2.0, 2.01]), QualityReason.HAMILTONIAN_DRIFT),
            ("state_finite", np.asarray([True, False]), QualityReason.NONFINITE_STATE),
            (
                "dno_output_finite",
                np.asarray([True, False]),
                QualityReason.NONFINITE_TARGET,
            ),
            ("minimum_water_column", np.asarray([1.0, 0.0]), QualityReason.BOTTOM_CLEARANCE),
        )
        for name, value, reason in defects:
            inputs = {**healthy, name: value}
            with self.subTest(name=name):
                _, failed = evaluate_dense_trajectory_health(
                    **inputs,
                    hamiltonian_drift_threshold=1.0e-3,
                )
                self.assertFalse(failed.accepted)
                self.assertTrue(failed.failed & reason)
if __name__ == "__main__":
    unittest.main()
