"""CPU integration test from one GL2 production arm through a committed view."""

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

from solver.gen_data.pipeline.acceptance import (  # noqa: E402
    DenseTrajectoryHealthMetrics,
    TrajectorySamples,
)
from solver.gen_data.pipeline.archive import ensure_proposal  # noqa: E402
from solver.gen_data.pipeline.manifest import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    CaseGL2Telemetry,
    ProductionCaseResult,
    ProductionExecution,
    ResidualControlledGL2Contract,
    execute_production_trajectory,
)
from solver.gen_data.pipeline.quality import (  # noqa: E402
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
    _selected_indices,
    outcomes_from_production,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
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
        indices = _selected_indices(
            trajectory,
            family="benjamin_feir",
            length=2.0 * np.pi,
            policy=StoredTimePolicy(benjamin_feir_count=4),
        )
        np.testing.assert_array_equal(
            indices,
            np.asarray([0, 3, 6, 9], dtype=np.int32),
        )

    def test_actual_single_arm_time_selection_commit_and_view(self) -> None:
        contract = ResidualControlledGL2Contract(
            nx=16,
            dno_order=0,
            pad_factor=1,
            maximum_wavenumber=3.0,
            production_dt=0.02,
            saved_dt=0.02,
            target_time_chunk_size=2,
        )
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        eta0 = np.stack(
            (0.001 * np.cos(x), 0.0015 * np.cos(2.0 * x))
        )
        xi0 = np.stack(
            (0.002 * np.sin(2.0 * x), 0.001 * np.sin(x))
        )
        depths = np.asarray([1.0, 1.5])
        assignments = tuple(
            AttemptAssignment(
                case_key=CaseKey(
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
        fingerprint = "a" * 64
        proposal = build_proposal_arrays(
            assignments,
            ({"amplitude": 0.001}, {"amplitude": 0.0015}),
            cell_codes={"finite_a": 0, "finite_b": 1},
            batch_id=0,
            config_fingerprint=fingerprint,
            metadata={"smoke": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = batch_paths_for_assignments(
                root,
                assignments,
                family_name="jonswap_tma",
                batch_id=0,
            )
            ensure_proposal(paths, proposal)
            self.assertTrue(paths.proposal.exists())

            execution = execute_production_trajectory(
                eta0,
                xi0,
                depths,
                np.asarray([0.0, 0.02, 0.04]),
                contract=contract,
            )
            outcomes = outcomes_from_production(
                execution,
                depths,
                family="jonswap_tma",
                length=contract.length,
                policy=StoredTimePolicy(
                    tanaka_count=2,
                    benjamin_feir_count=2,
                    random_sea_count=3,
                ),
            )
            self.assertTrue(
                all(outcome.decision.accepted for outcome in outcomes)
            )
            self.assertTrue(
                all(
                    outcome.metrics["production_dt"] == contract.production_dt
                    for outcome in outcomes
                )
            )
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
            commit_case_outcomes(
                paths,
                proposal,
                outcomes,
                metadata={"accepted_cases": 2},
            )
            view = build_dataset_view(
                root,
                (paths,),
                expected_fingerprint=fingerprint,
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
        reason = QualityReason.INCOMPLETE_TRAJECTORY
        decision = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=reason,
            evaluated=reason,
            failed=QualityReason.NONE,
        )
        telemetry = CaseGL2Telemetry(
            residual=np.zeros(2, dtype=np.float64),
            iterations=np.ones(2, dtype=np.int32),
            converged=np.ones(2, dtype=np.bool_),
            stage_finite=np.ones(2, dtype=np.bool_),
            state_finite=np.ones(2, dtype=np.bool_),
            hit_iteration_cap=np.zeros(2, dtype=np.bool_),
            all_stages_solved=True,
            maximum_stage_residual=0.0,
        )
        execution = ProductionExecution(
            timing=None,
            cases=(
                ProductionCaseResult(
                    case_index=0,
                    dt=0.01,
                    telemetry=telemetry,
                    decision=decision,
                    retained_trajectory=trajectory,
                ),
            ),
        )

        outcome = outcomes_from_production(
            execution,
            np.asarray([1.0]),
            family="tanaka",
            length=2.0 * math.pi,
            policy=StoredTimePolicy(tanaka_count=2),
        )[0]

        self.assertTrue(outcome.decision.accepted)
        self.assertIsNotNone(outcome.rows)
        assert outcome.rows is not None
        np.testing.assert_array_equal(
            outcome.rows.selected_dense_index,
            np.asarray([0, 2], dtype=np.int32),
        )
        self.assertNotIn("hamiltonian_drift_evaluated", outcome.metrics)

    def test_rejected_case_persists_finite_internal_health_metrics(self) -> None:
        reason = QualityReason.HAMILTONIAN_DRIFT
        decision = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=reason,
            evaluated=reason,
            failed=reason,
        )
        telemetry = CaseGL2Telemetry(
            residual=np.zeros(2, dtype=np.float64),
            iterations=np.ones(2, dtype=np.int32),
            converged=np.ones(2, dtype=np.bool_),
            stage_finite=np.ones(2, dtype=np.bool_),
            state_finite=np.ones(2, dtype=np.bool_),
            hit_iteration_cap=np.zeros(2, dtype=np.bool_),
            all_stages_solved=True,
            maximum_stage_residual=0.0,
        )
        execution = ProductionExecution(
            timing=None,
            cases=(
                ProductionCaseResult(
                    case_index=0,
                    dt=0.01,
                    telemetry=telemetry,
                    decision=decision,
                    retained_trajectory=None,
                    internal_metrics=DenseTrajectoryHealthMetrics(
                        state_finite=True,
                        dno_output_finite=True,
                        minimum_water_column=0.75,
                        initial_hamiltonian=2.5,
                        maximum_relative_hamiltonian_drift=2.0e-3,
                        hamiltonian_drift_threshold=1.0e-3,
                    ),
                ),
            ),
        )

        outcome = outcomes_from_production(
            execution,
            np.asarray([1.0]),
            family="benjamin_feir",
            length=2.0 * math.pi,
            policy=StoredTimePolicy(benjamin_feir_count=2),
        )[0]

        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertTrue(outcome.metrics["internal_health_evaluated"])
        self.assertEqual(
            outcome.metrics["maximum_internal_hamiltonian_drift"],
            2.0e-3,
        )
        self.assertEqual(
            outcome.metrics["internal_hamiltonian_drift_threshold"],
            1.0e-3,
        )
        self.assertEqual(
            outcome.metrics["minimum_internal_water_column"],
            0.75,
        )
        json.dumps(outcome.metrics, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
