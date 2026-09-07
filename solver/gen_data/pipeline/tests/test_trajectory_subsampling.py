"""Tests for converting accepted trajectories into dataset rows."""

from __future__ import annotations

import unittest

import numpy as np

from solver.gen_data.pipeline.trajectory_config import TrajectoryFamily
from solver.gen_data.pipeline.trajectory_rollout import TrajectorySamples
from solver.gen_data.pipeline.time_selection import subsample_trajectories


class TrajectorySubsamplingTest(unittest.TestCase):
    def test_each_family_keeps_its_declared_number_of_frames(self) -> None:
        times = 0.08 * np.arange(359, dtype=np.float64)
        x = 2.0 * np.pi * np.arange(8) / 8
        eta = times[:, None] + np.cos(x)[None]
        trajectory = TrajectorySamples(times, eta, 2.0 * eta, 3.0 * eta)

        families: tuple[tuple[TrajectoryFamily, int], ...] = (
            ("tanaka", 200),
            ("benjamin_feir", 200),
            ("jonswap_tma", 16),
        )
        for family, expected_count in families:
            with self.subTest(family=family):
                rows = subsample_trajectories(
                    (trajectory,),
                    np.asarray([2.0]),
                    family=family,
                    length=2.0 * np.pi,
                )[0]
                assert rows is not None
                self.assertEqual(rows.eta.shape, (expected_count, 8))
                self.assertEqual(rows.depth, 2.0)
                self.assertEqual(
                    (rows.time[0], rows.time[-1]),
                    (times[0], times[-1]),
                )
                np.testing.assert_allclose(rows.xi, 2.0 * rows.eta)
                np.testing.assert_allclose(rows.gxi, 3.0 * rows.eta)

    def test_rejected_trajectory_produces_no_rows(self) -> None:
        rows = subsample_trajectories(
            (None,),
            np.asarray([1.0]),
            family="benjamin_feir",
            length=2.0 * np.pi,
        )[0]

        self.assertIsNone(rows)


if __name__ == "__main__":
    unittest.main()
