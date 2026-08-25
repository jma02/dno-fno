"""Smoke-level tests for the paper-dataset acceptance decisions."""
from __future__ import annotations

import math
import unittest

import numpy as np

from solver.gen_data.pipeline.acceptance import (
    RefinementTrajectory,
    evaluate_complete_numerical_trajectory,
    evaluate_internal_trajectory_health,
    evaluate_temporal_refinement,
)
from solver.gen_data.pipeline.quality import QualityReason


def make_refinement_pair(
    relative_change: float,
) -> tuple[RefinementTrajectory, RefinementTrajectory]:
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x = 2.0 * np.pi * np.arange(8, dtype=np.float64) / 8.0
    time_grid = times[:, None]
    eta = (1.0 + 0.1 * time_grid) * np.cos(x)[None, :]
    xi = (0.5 - 0.05 * time_grid) * np.sin(x)[None, :]
    gxi = (0.25 + 0.02 * time_grid) * np.cos(x)[None, :]
    coarse = RefinementTrajectory(times=times, eta=eta, xi=xi, gxi=gxi)
    fine_scale = 1.0 + relative_change * time_grid
    fine = RefinementTrajectory(
        times=times,
        eta=eta * fine_scale,
        xi=xi * fine_scale,
        gxi=gxi * fine_scale,
    )
    return coarse, fine


class CompleteNumericalTrajectoryTest(unittest.TestCase):
    def test_finite_complete_trajectory_passes_one_required_check(self) -> None:
        trajectory, _ = make_refinement_pair(0.0)

        decision = evaluate_complete_numerical_trajectory(
            trajectory,
            depth=2.0,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.required,
            QualityReason.INCOMPLETE_TRAJECTORY,
        )
        self.assertFalse(decision.evaluated & QualityReason.TEMPORAL_DEFECT)

    def test_numerical_failure_is_a_diagnostic_cause_of_incompleteness(
        self,
    ) -> None:
        trajectory, _ = make_refinement_pair(0.0)
        nonfinite_target = trajectory.gxi.copy()
        nonfinite_target[-1, 0] = np.nan
        failed_trajectory = RefinementTrajectory(
            times=trajectory.times,
            eta=trajectory.eta,
            xi=trajectory.xi,
            gxi=nonfinite_target,
            complete=True,
            gl2_stages_solved=False,
        )

        decision = evaluate_complete_numerical_trajectory(
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
            "dno_finite": np.asarray([True, True]),
            "minimum_water_column": np.asarray([1.0, 0.5]),
        }

        metrics, decision = evaluate_internal_trajectory_health(
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
            ("dno_finite", np.asarray([True, False]), QualityReason.NONFINITE_TARGET),
            ("minimum_water_column", np.asarray([1.0, 0.0]), QualityReason.BOTTOM_CLEARANCE),
        )
        for name, value, reason in defects:
            inputs = {**healthy, name: value}
            with self.subTest(name=name):
                _, failed = evaluate_internal_trajectory_health(
                    **inputs,
                    hamiltonian_drift_threshold=1.0e-3,
                )
                self.assertFalse(failed.accepted)
                self.assertTrue(failed.failed & reason)


class TemporalRefinementSmokeTest(unittest.TestCase):
    def evaluate_pair(
        self,
        relative_change: float,
    ) -> tuple[float, bool]:
        coarse, fine = make_refinement_pair(relative_change)
        metrics, decision = evaluate_temporal_refinement(
            coarse,
            fine,
            depth=2.0,
            gravity=1.0,
            length=2.0 * np.pi,
            maximum_wavenumber=1.0,
        )
        return metrics.maximum_error, decision.accepted

    def test_agreeing_pair_passes(self) -> None:
        error, accepted = self.evaluate_pair(5e-4)

        self.assertTrue(accepted)
        self.assertLess(error, 1e-3)

    def test_disagreeing_pair_fails(self) -> None:
        error, accepted = self.evaluate_pair(2e-3)

        self.assertFalse(accepted)
        self.assertGreater(error, 1e-3)

    def test_difference_outside_delivered_band_is_not_a_defect(self) -> None:
        coarse, fine = make_refinement_pair(0.0)
        x = 2.0 * np.pi * np.arange(8, dtype=np.float64) / 8.0
        high_mode_difference = 0.1 * np.cos(3.0 * x)
        high_mode_coarse = RefinementTrajectory(
            times=coarse.times,
            eta=coarse.eta + high_mode_difference[None, :],
            xi=coarse.xi,
            gxi=coarse.gxi,
        )

        metrics, decision = evaluate_temporal_refinement(
            high_mode_coarse,
            fine,
            depth=2.0,
            gravity=1.0,
            length=2.0 * np.pi,
            maximum_wavenumber=1.0,
        )

        self.assertTrue(decision.accepted)
        self.assertLess(metrics.eta_error, 1e-14)

    def test_nonfinite_pair_has_infinite_defect(self) -> None:
        coarse, fine = make_refinement_pair(5e-4)
        fine_gxi = fine.gxi.copy()
        fine_gxi[-1, 0] = np.nan
        nonfinite_fine = RefinementTrajectory(
            times=fine.times,
            eta=fine.eta,
            xi=fine.xi,
            gxi=fine_gxi,
        )

        metrics, decision = evaluate_temporal_refinement(
            coarse,
            nonfinite_fine,
            depth=2.0,
            gravity=1.0,
            length=2.0 * np.pi,
            maximum_wavenumber=1.0,
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(math.isinf(metrics.maximum_error))
        self.assertTrue(decision.failed & QualityReason.NONFINITE_TARGET)

    def test_positive_water_column_below_old_half_depth_margin_passes(self) -> None:
        coarse, fine = make_refinement_pair(0.0)
        shifted_eta = np.full_like(coarse.eta, -1.2)
        shifted_coarse = RefinementTrajectory(
            times=coarse.times,
            eta=shifted_eta,
            xi=coarse.xi,
            gxi=coarse.gxi,
        )
        shifted_fine = RefinementTrajectory(
            times=fine.times,
            eta=shifted_eta,
            xi=fine.xi,
            gxi=fine.gxi,
        )

        metrics, decision = evaluate_temporal_refinement(
            shifted_coarse,
            shifted_fine,
            depth=2.0,
            gravity=1.0,
            length=2.0 * np.pi,
            maximum_wavenumber=1.0,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(metrics.maximum_error, 0.0)

    def test_relative_floor_is_invariant_under_spatial_replication(self) -> None:
        errors = []
        for nx in (4, 64):
            times = np.asarray([0.0], dtype=np.float64)
            zeros = np.zeros((1, nx), dtype=np.float64)
            tiny = np.full((1, nx), 1e-12, dtype=np.float64)
            coarse = RefinementTrajectory(
                times=times,
                eta=zeros,
                xi=zeros,
                gxi=zeros,
            )
            fine = RefinementTrajectory(
                times=times,
                eta=tiny,
                xi=zeros,
                gxi=zeros,
            )
            metrics, _ = evaluate_temporal_refinement(
                coarse,
                fine,
                depth=1.0,
                gravity=1.0,
                length=2.0 * np.pi,
                maximum_wavenumber=0.0,
            )
            errors.append(metrics.eta_error)

        np.testing.assert_allclose(errors, (0.5, 0.5), rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
