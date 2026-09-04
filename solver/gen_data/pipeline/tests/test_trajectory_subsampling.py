"""CPU integration test from one GL2 rollout through a committed view."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.trajectory_checks import (  # noqa: E402
    TrajectoryHealthMetrics,
)
from solver.gen_data.pipeline.batch_storage import batch_path  # noqa: E402
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402
from solver.gen_data.pipeline.trajectory_config import RolloutNumerics  # noqa: E402
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    TrajectorySimulationResult,
    TrajectorySamples,
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.simulation_checks import (  # noqa: E402
    SimulationCheckResult,
)
from solver.gen_data.pipeline.trajectory_subsampling import (  # noqa: E402
    subsample_trajectories,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    build_batch_plan,
    commit_simulation_outcomes,
)

jax.config.update("jax_enable_x64", True)


class TrajectoryWriterIntegrationTest(unittest.TestCase):
    def test_actual_rollout_time_selection_commit_and_view(self) -> None:
        rollout_config = RolloutNumerics(
            nx=16,
            target_nx=16,
            length=2.0 * math.pi,
            gravity=1.0,
            integration_dno_order=0,
            label_dno_order=0,
            pad_factor=1,
            maximum_wavenumber=3.0,
            target_maximum_wavenumber=3.0,
            saved_dt=0.02,
            substeps_per_saved_frame=1,
            gl2_residual_tolerance=1.0e-8,
            gl2_iteration_cap=8,
            internal_hamiltonian_drift_threshold=None,
        )
        x = 2.0 * math.pi * np.arange(rollout_config.nx) / rollout_config.nx
        eta0 = np.stack((0.001 * np.cos(x), 0.0015 * np.cos(2.0 * x)))
        xi0 = np.stack((0.002 * np.sin(2.0 * x), 0.001 * np.sin(x)))
        depths = np.asarray([1.0, 1.5])
        proposal = build_batch_plan(
            ("finite_a", "finite_b"),
            ({"amplitude": 0.001}, {"amplitude": 0.0015}),
            family_id=4,
            dataset_split=DatasetSplit.TEST,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = batch_path(
                root,
                family="jonswap_tma",
                split=DatasetSplit.TEST.value,
                batch_id=0,
            )

            saved_times = 0.02 * np.arange(16, dtype=np.float64)
            execution = execute_trajectory_batch(
                eta0,
                xi0,
                depths,
                (saved_times,) * eta0.shape[0],
                config=rollout_config,
            )
            outcomes = subsample_trajectories(
                execution,
                depths,
                family="jonswap_tma",
                length=rollout_config.length,
            )
            self.assertTrue(all(outcome.decision.accepted for outcome in outcomes))
            commit_simulation_outcomes(
                path,
                proposal,
                outcomes,
            )
            self.assertTrue(path.exists())
            view = build_dataset_view(
                root,
                (path,),
            )
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([16, 16]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_dataset_split"],
                    np.asarray([DatasetSplit.TEST.value] * 2),
                )

    def test_large_offline_hamiltonian_drift_does_not_gate_time_selection(
        self,
    ) -> None:
        times = 0.08 * np.arange(200, dtype=np.float64)
        eta = np.zeros((200, 4), dtype=np.float64)
        xi = np.ones_like(eta)
        gxi = np.ones_like(eta)
        gxi[100] *= 10.0
        trajectory = TrajectorySamples(
            times=times,
            eta=eta,
            xi=xi,
            gxi=gxi,
        )
        decision = SimulationCheckResult(accepted=True)
        simulations = (
            TrajectorySimulationResult(
                maximum_gl2_stage_residual=0.0,
                decision=decision,
                trajectory=trajectory,
            ),
        )

        outcome = subsample_trajectories(
            simulations,
            np.asarray([1.0]),
            family="tanaka",
            length=2.0 * math.pi,
        )[0]

        self.assertTrue(outcome.decision.accepted)
        self.assertIsNotNone(outcome.rows)
        assert outcome.rows is not None
        np.testing.assert_array_equal(outcome.rows.time, times)

    def test_rejected_simulation_persists_finite_internal_health_metrics(self) -> None:
        decision = SimulationCheckResult(
            accepted=False,
            hamiltonian_drift=True,
        )
        simulations = (
            TrajectorySimulationResult(
                maximum_gl2_stage_residual=0.0,
                decision=decision,
                trajectory=None,
                health_metrics=TrajectoryHealthMetrics(
                    state_finite=True,
                    dno_output_finite=True,
                    minimum_water_column=0.75,
                    initial_hamiltonian=2.5,
                    maximum_relative_hamiltonian_drift=2.0e-3,
                    hamiltonian_drift_threshold=1.0e-3,
                ),
            ),
        )

        outcome = subsample_trajectories(
            simulations,
            np.asarray([1.0]),
            family="benjamin_feir",
            length=2.0 * math.pi,
        )[0]

        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertEqual(
            outcome.metrics["maximum_internal_hamiltonian_drift"],
            2.0e-3,
        )
        self.assertEqual(
            outcome.metrics["initial_internal_hamiltonian"],
            2.5,
        )
        self.assertEqual(
            outcome.metrics["minimum_internal_water_column"],
            0.75,
        )
        json.dumps(outcome.metrics, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
