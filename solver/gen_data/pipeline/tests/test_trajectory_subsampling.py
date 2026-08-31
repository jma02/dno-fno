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
from solver.gen_data.pipeline.batch_storage import (  # noqa: E402
    BatchPaths,
    save_batch_plan,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    AttemptAssignment,
    SimulationKey,
    SplitId,
)
from solver.gen_data.pipeline.trajectory_config import RolloutConfig  # noqa: E402
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    TrajectorySimulationResult,
    TrajectorySamples,
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.simulation_checks import (  # noqa: E402
    SimulationCheckResult,
    SimulationCheck,
)
from solver.gen_data.pipeline.trajectory_subsampling import (  # noqa: E402
    TrajectoryFrameSelectionConfig,
    _select_subsample_time_indices,
    subsample_trajectories,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    build_batch_plan,
    commit_simulation_outcomes,
)

jax.config.update("jax_enable_x64", True)


class TrajectoryWriterIntegrationTest(unittest.TestCase):
    def test_benjamin_feir_retains_uniform_saved_time_indices(self) -> None:
        times = 0.08 * np.arange(10, dtype=np.float64)
        x = 2.0 * np.pi * np.arange(8, dtype=np.float64) / 8
        eta = np.arange(1.0, 11.0)[:, None] * np.cos(x)[None, :]
        trajectory = TrajectorySamples(
            times=times,
            eta=eta,
            xi=np.zeros_like(eta),
            gxi=np.zeros_like(eta),
        )
        indices = _select_subsample_time_indices(
            trajectory,
            family="benjamin_feir",
            length=2.0 * np.pi,
            frame_selection=TrajectoryFrameSelectionConfig(benjamin_feir_count=4),
        )
        np.testing.assert_array_equal(
            indices,
            np.asarray([0, 3, 6, 9], dtype=np.int32),
        )

    def test_actual_rollout_time_selection_commit_and_view(self) -> None:
        contract = RolloutConfig(
            nx=16,
            target_nx=16,
            dno_order=0,
            target_dno_order=0,
            pad_factor=1,
            maximum_wavenumber=3.0,
            target_maximum_wavenumber=3.0,
            dt=0.02,
            saved_dt=0.02,
            target_time_chunk_size=2,
        )
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        eta0 = np.stack((0.001 * np.cos(x), 0.0015 * np.cos(2.0 * x)))
        xi0 = np.stack((0.002 * np.sin(2.0 * x), 0.001 * np.sin(x)))
        depths = np.asarray([1.0, 1.5])
        assignments = tuple(
            AttemptAssignment(
                simulation_key=SimulationKey(
                    family_id=4,
                    revision_id=1,
                    split_id=SplitId.TEST,
                    stream_id=2,
                    attempt_index=index,
                ),
                cell_id=cell,
            )
            for index, cell in enumerate(("finite_a", "finite_b"))
        )
        proposal = build_batch_plan(
            assignments,
            ({"amplitude": 0.001}, {"amplitude": 0.0015}),
            cell_codes={"finite_a": 0, "finite_b": 1},
            batch_id=0,
            metadata={"smoke": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = BatchPaths.for_batch(
                root,
                family="jonswap_tma",
                split=SplitId.TEST.value,
                batch_id=0,
            )
            save_batch_plan(paths, proposal)
            self.assertTrue(paths.batch_plan.exists())

            saved_times = np.asarray([0.0, 0.02, 0.04])
            execution = execute_trajectory_batch(
                eta0,
                xi0,
                depths,
                (saved_times,) * eta0.shape[0],
                config=contract,
            )
            outcomes = subsample_trajectories(
                execution,
                depths,
                family="jonswap_tma",
                length=contract.length,
                frame_selection=TrajectoryFrameSelectionConfig(
                    tanaka_count=2,
                    benjamin_feir_count=2,
                    jonswap_tma_count=3,
                ),
            )
            self.assertTrue(all(outcome.decision.accepted for outcome in outcomes))
            self.assertTrue(
                all(
                    np.array_equal(
                        outcome.rows.selected_dense_index,
                        np.asarray([0, 1, 2]),
                    )
                    for outcome in outcomes
                    if outcome.rows is not None
                )
            )
            commit_simulation_outcomes(
                paths,
                proposal,
                outcomes,
                metadata={"accepted_simulations": 2},
            )
            view = build_dataset_view(
                root,
                (paths,),
            )
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([3, 3]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_split_id"],
                    np.asarray([2, 2], dtype=np.uint8),
                )

    def test_large_offline_hamiltonian_drift_does_not_gate_time_selection(
        self,
    ) -> None:
        times = np.asarray([0.0, 0.08, 0.16], dtype=np.float64)
        eta = np.zeros((3, 4), dtype=np.float64)
        xi = np.ones_like(eta)
        gxi = np.ones_like(eta)
        gxi[1] *= 10.0
        trajectory = TrajectorySamples(
            times=times,
            eta=eta,
            xi=xi,
            gxi=gxi,
        )
        reason = SimulationCheck.INCOMPLETE_TRAJECTORY
        decision = SimulationCheckResult(
            required=reason,
            evaluated=reason,
            failed=SimulationCheck.NONE,
        )
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
            frame_selection=TrajectoryFrameSelectionConfig(tanaka_count=2),
        )[0]

        self.assertTrue(outcome.decision.accepted)
        self.assertIsNotNone(outcome.rows)
        assert outcome.rows is not None
        np.testing.assert_array_equal(
            outcome.rows.selected_dense_index,
            np.asarray([0, 2], dtype=np.int32),
        )

    def test_rejected_simulation_persists_finite_internal_health_metrics(self) -> None:
        reason = SimulationCheck.HAMILTONIAN_DRIFT
        decision = SimulationCheckResult(
            required=reason,
            evaluated=reason,
            failed=reason,
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
            frame_selection=TrajectoryFrameSelectionConfig(benjamin_feir_count=2),
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
