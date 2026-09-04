"""Tests for converting accepted trajectories into dataset rows."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.trajectory_config import TrajectoryFamily
from solver.gen_data.pipeline.trajectory_rollout import (
    TrajectorySamples,
    TrajectorySimulationResult,
)
from solver.gen_data.pipeline.trajectory_subsampling import subsample_trajectories


class TrajectorySubsamplingTest(unittest.TestCase):
    def test_each_family_keeps_its_declared_number_of_frames(self) -> None:
        times = 0.08 * np.arange(359, dtype=np.float64)
        x = 2.0 * np.pi * np.arange(8) / 8
        eta = times[:, None] + np.cos(x)[None]
        trajectory = TrajectorySamples(times, eta, 2.0 * eta, 3.0 * eta)
        simulation = TrajectorySimulationResult(
            SimulationCheckResult(accepted=True),
            trajectory,
        )

        families: tuple[tuple[TrajectoryFamily, int], ...] = (
            ("tanaka", 200),
            ("benjamin_feir", 200),
            ("jonswap_tma", 16),
        )
        for family, expected_count in families:
            with self.subTest(family=family):
                outcome = subsample_trajectories(
                    (simulation,),
                    np.asarray([2.0]),
                    family=family,
                    length=2.0 * np.pi,
                )[0]
                self.assertTrue(outcome.decision.accepted)
                assert outcome.rows is not None
                self.assertEqual(outcome.rows.eta.shape, (expected_count, 8))
                self.assertEqual(outcome.rows.depth, 2.0)
                self.assertEqual(
                    (outcome.rows.time[0], outcome.rows.time[-1]),
                    (times[0], times[-1]),
                )
                np.testing.assert_allclose(outcome.rows.xi, 2.0 * outcome.rows.eta)
                np.testing.assert_allclose(outcome.rows.gxi, 3.0 * outcome.rows.eta)

    def test_rejected_trajectory_produces_no_rows(self) -> None:
        decision = SimulationCheckResult(
            accepted=False,
            integration_failure=True,
            incomplete_trajectory=True,
        )
        outcome = subsample_trajectories(
            (TrajectorySimulationResult(decision, None),),
            np.asarray([1.0]),
            family="benjamin_feir",
            length=2.0 * np.pi,
        )[0]

        self.assertEqual(outcome.decision, decision)
        self.assertIsNone(outcome.rows)


if __name__ == "__main__":
    unittest.main()
