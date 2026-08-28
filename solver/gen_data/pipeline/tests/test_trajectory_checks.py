"""Tests for generated-trajectory quality decisions."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.trajectory_checks import (
    evaluate_trajectory_health,
    evaluate_trajectory,
)
from solver.gen_data.pipeline.case_checks import CaseCheck


def make_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x = 2.0 * np.pi * np.arange(8, dtype=np.float64) / 8.0
    time_grid = times[:, None]
    eta = (1.0 + 0.1 * time_grid) * np.cos(x)[None, :]
    xi = (0.5 - 0.05 * time_grid) * np.sin(x)[None, :]
    gxi = (0.25 + 0.02 * time_grid) * np.cos(x)[None, :]
    return eta, xi, gxi


class CompleteNumericalTrajectoryTest(unittest.TestCase):
    def test_finite_complete_trajectory_passes_one_required_check(self) -> None:
        eta, xi, gxi = make_trajectory()

        decision = evaluate_trajectory(
            eta,
            xi,
            gxi,
            depth=2.0,
            gl2_stages_solved=True,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.required,
            CaseCheck.INCOMPLETE_TRAJECTORY,
        )

    def test_numerical_failure_is_a_diagnostic_cause_of_incompleteness(
        self,
    ) -> None:
        eta, xi, gxi = make_trajectory()
        nonfinite_target = gxi.copy()
        nonfinite_target[-1, 0] = np.nan

        decision = evaluate_trajectory(
            eta,
            xi,
            nonfinite_target,
            depth=2.0,
            gl2_stages_solved=False,
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(decision.failed & CaseCheck.INCOMPLETE_TRAJECTORY)
        self.assertTrue(decision.failed & CaseCheck.NONFINITE_TARGET)
        self.assertTrue(decision.failed & CaseCheck.GL2_STAGE_RESIDUAL)


class InternalTrajectoryHealthTest(unittest.TestCase):
    def test_each_internal_requirement_rejects_independently(self) -> None:
        healthy = {
            "hamiltonian": np.asarray([2.0, 2.001], dtype=np.float64),
            "state_finite": np.asarray([True, True]),
            "dno_output_finite": np.asarray([True, True]),
            "minimum_water_column": np.asarray([1.0, 0.5]),
        }

        metrics, decision = evaluate_trajectory_health(
            **healthy,
            hamiltonian_drift_threshold=1.0e-3,
        )

        self.assertTrue(decision.accepted)
        self.assertIsNotNone(metrics.maximum_relative_hamiltonian_drift)
        assert metrics.maximum_relative_hamiltonian_drift is not None
        self.assertAlmostEqual(
            metrics.maximum_relative_hamiltonian_drift,
            5.0e-4,
        )
        defects = (
            ("hamiltonian", np.asarray([2.0, 2.01]), CaseCheck.HAMILTONIAN_DRIFT),
            ("state_finite", np.asarray([True, False]), CaseCheck.NONFINITE_STATE),
            (
                "dno_output_finite",
                np.asarray([True, False]),
                CaseCheck.NONFINITE_TARGET,
            ),
            (
                "minimum_water_column",
                np.asarray([1.0, 0.0]),
                CaseCheck.BOTTOM_CLEARANCE,
            ),
        )
        for name, value, reason in defects:
            inputs = {**healthy, name: value}
            with self.subTest(name=name):
                _, failed = evaluate_trajectory_health(
                    **inputs,
                    hamiltonian_drift_threshold=1.0e-3,
                )
                self.assertFalse(failed.accepted)
                self.assertTrue(failed.failed & reason)


if __name__ == "__main__":
    unittest.main()
