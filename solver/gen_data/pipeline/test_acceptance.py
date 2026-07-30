"""Smoke-level tests for the paper-corpus acceptance decisions."""
from __future__ import annotations

import math
import unittest

import numpy as np

from solver.gen_data.pipeline.acceptance import (
    RefinementTrajectory,
    evaluate_complete_numerical_trajectory,
    evaluate_finite_stokes_support,
    evaluate_temporal_refinement,
)
from solver.gen_data.pipeline.quality import QualityReason


STOKES_CASE_98_HARMONICS = np.asarray(
    [
        0.2474963451001154,
        0.13520417407096738,
        0.07006262814091847,
        0.05461649162577016,
        0.032900266559519276,
    ],
    dtype=np.float64,
)
def wave_height(elevation_harmonics: np.ndarray) -> float:
    phases = np.linspace(0.0, 2.0 * np.pi, 16384, endpoint=False)
    modes = np.arange(1, 6, dtype=np.float64)
    elevation = np.sum(
        elevation_harmonics[:, None] * np.cos(modes[:, None] * phases),
        axis=0,
    )
    return float(np.max(elevation) - np.min(elevation))


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


class FiniteStokesSupportSmokeTest(unittest.TestCase):
    def test_case_below_ursell_limit_passes(self) -> None:
        metrics, decision = evaluate_finite_stokes_support(
            wave_height_upper_bound=0.18,
            wavelength=10.0,
            depth=1.0,
        )

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(metrics.ursell_upper_bound, 18.0)

    def test_audited_case_98_is_outside_support(self) -> None:
        metrics, decision = evaluate_finite_stokes_support(
            wave_height_upper_bound=wave_height(STOKES_CASE_98_HARMONICS),
            wavelength=164.0 / 14.0,
            depth=1.0,
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(decision.failed & QualityReason.OUTSIDE_SUPPORT)
        self.assertAlmostEqual(
            metrics.ursell_upper_bound,
            96.87204675,
            places=5,
        )


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
