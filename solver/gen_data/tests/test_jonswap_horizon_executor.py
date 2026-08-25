"""CPU tests for horizon-bucketed JONSWAP/TMA execution."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from scripts import generate_paper_dataset_jonswap as bucketed  # noqa: E402
from scripts import generate_paper_dataset as base  # noqa: E402
from solver.gen_data.jonswap_horizon_executor import (  # noqa: E402
    BUCKETING_CONFIG_KEY,
    BucketingConfig,
    HorizonBucketedJonswapBatchExecutor,
    horizon_sorted_groups,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_SAMPLING_REVISION_V4,
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    ValidCaseTarget,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.valid_case_generation import (  # noqa: E402
    DatasetGenerationSpec,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    InternalTrajectoryTelemetry,
    NonlinearAdjustmentArm,
    PAPER_JONSWAP_GL2_CONTRACT,
    ResidualControlledArm,
    ResidualControlledGL2Contract,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    CaseTimeGrid,
    JONSWAP_ADJUSTMENT_FORMULA,
    PAPER_JONSWAP_ADJUSTMENT_POLICY,
    TrajectoryExecutionConfig,
    TrajectoryHorizonPolicy,
)


def _contract() -> ResidualControlledGL2Contract:
    return ResidualControlledGL2Contract(
        nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        production_dt=0.04,
        saved_dt=0.08,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        target_time_chunk_size=2,
    )


def _execution(
    contract: ResidualControlledGL2Contract | None = None,
) -> TrajectoryExecutionConfig:
    return TrajectoryExecutionConfig(
        family="jonswap_tma",
        role="reduced_wiring_evidence_only",
        numerical=_contract() if contract is None else contract,
        horizon=TrajectoryHorizonPolicy.jonswap_peak_periods(16),
        stored_time_policy=StoredTimePolicy(
            tanaka_count=3,
            tanaka_alpha=0.5,
            tanaka_sigma_steps=1.0,
            benjamin_feir_count=3,
            random_sea_count=3,
        ),
        jonswap_quadrature_order=4,
        jonswap_adjustment=PAPER_JONSWAP_ADJUSTMENT_POLICY,
    )


def _run_spec(
    root: Path,
    execution: TrajectoryExecutionConfig,
    *,
    outer_size: int,
    solver_size: int,
) -> DatasetGenerationSpec:
    cell_id = JONSWAP_TMA_SAMPLE_CELL_IDS[0]
    return DatasetGenerationSpec(
        root=root,
        family_name="jonswap_tma",
        family_id=PhysicalFamilyId.JONSWAP_TMA,
        revision_id=JONSWAP_TMA_SAMPLING_REVISION_V4,
        split_id=SplitId.TEST,
        stream_id=13,
        case_targets=(ValidCaseTarget(cell_id, outer_size),),
        cell_codes={cell_id: 0},
        batch_size=outer_size,
        first_attempt_index=0,
        configuration={
            "trajectory_execution": execution.to_json_record(),
            BUCKETING_CONFIG_KEY: BucketingConfig(
                outer_proposal_size=outer_size,
                solver_batch_size=solver_size,
            ).to_json_record(),
        },
    )


def _grid(saved_count: int) -> CaseTimeGrid:
    saved_times = np.arange(saved_count, dtype=np.float64) * 0.08
    terminal_time = float(saved_times[-1])
    return CaseTimeGrid(
        intended_terminal_time=terminal_time,
        realized_terminal_time=terminal_time,
        saved_times=saved_times,
    )


class RecordingArmExecutor:
    """Return exact constant trajectories and record numerical group shapes."""

    def __init__(self, *, hamiltonian_drift: float | None = None) -> None:
        self.hamiltonian_drift = hamiltonian_drift
        self.calls: list[tuple[int, int]] = []
        self.initial_eta: list[np.ndarray] = []
        self.saved_times: list[np.ndarray] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
    ) -> ResidualControlledArm:
        del depths
        batch_size, internal_nx = eta0.shape
        self.calls.append((batch_size, saved_times.size))
        self.initial_eta.append(np.asarray(eta0, dtype=np.float64).copy())
        self.saved_times.append(np.asarray(saved_times, dtype=np.float64).copy())
        saved_count = saved_times.size
        delivered_nx = contract.delivered_nx
        if delivered_nx == internal_nx:
            delivered_eta0 = eta0
            delivered_xi0 = xi0
        else:
            delivered_eta0 = np.repeat(
                np.mean(eta0, axis=-1)[:, None],
                delivered_nx,
                axis=1,
            )
            delivered_xi0 = np.repeat(
                np.mean(xi0, axis=-1)[:, None],
                delivered_nx,
                axis=1,
            )
        substeps = int(round(contract.saved_dt / dt))
        telemetry_shape = ((saved_count - 1) * substeps, batch_size)
        internal_telemetry = None
        if self.hamiltonian_drift is not None:
            hamiltonian = np.ones((saved_count, batch_size), dtype=np.float64)
            hamiltonian[-1] += self.hamiltonian_drift
            internal_telemetry = InternalTrajectoryTelemetry(
                hamiltonian=hamiltonian,
                state_finite=np.ones_like(hamiltonian, dtype=np.bool_),
                dno_finite=np.ones_like(hamiltonian, dtype=np.bool_),
                minimum_water_column=np.ones_like(
                    hamiltonian,
                    dtype=np.float64,
                ),
            )
        return ResidualControlledArm(
            dt=dt,
            times=np.asarray(saved_times, dtype=np.float64),
            eta=np.repeat(delivered_eta0[None, :, :], saved_count, axis=0),
            xi=np.repeat(delivered_xi0[None, :, :], saved_count, axis=0),
            q_ref=np.zeros(
                (saved_count, batch_size, delivered_nx),
                dtype=np.float64,
            ),
            complete=np.ones(batch_size, dtype=np.bool_),
            gl2_stage_residual=np.zeros(telemetry_shape, dtype=np.float64),
            gl2_iterations=np.ones(telemetry_shape, dtype=np.int32),
            gl2_converged=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_hit_iteration_cap=np.zeros(telemetry_shape, dtype=np.bool_),
            internal_telemetry=internal_telemetry,
        )


class RecordingAdjustmentArmExecutor:
    """Return deterministic ramp endpoints and optionally fail named markers."""

    def __init__(
        self,
        *,
        failing_markers: tuple[float, ...] = (),
        terminal_addition: np.ndarray | None = None,
    ) -> None:
        self.failing_markers = failing_markers
        self.terminal_addition = terminal_addition
        self.calls: list[tuple[int, int, np.ndarray, int]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
        nonlinear_ramp_times: np.ndarray,
        nonlinear_ramp_order: int,
    ) -> NonlinearAdjustmentArm:
        del depths
        batch_size, nx = eta0.shape
        ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
        self.calls.append(
            (
                batch_size,
                saved_times.size,
                ramp_times.copy(),
                nonlinear_ramp_order,
            )
        )
        saved_count = saved_times.size
        eta = np.repeat(eta0[None, :, :], saved_count, axis=0)
        eta += np.asarray(saved_times)[:, None, None]
        if self.terminal_addition is not None:
            addition = np.asarray(self.terminal_addition, dtype=np.float64)
            if addition.shape != (nx,):
                raise ValueError("terminal_addition has the wrong spatial shape")
            eta[1:] += addition[None, None, :]
        xi = np.repeat(xi0[None, :, :], saved_count, axis=0)
        substeps = int(round(contract.saved_dt / dt))
        telemetry_shape = ((saved_count - 1) * substeps, batch_size)
        residual = np.zeros(telemetry_shape, dtype=np.float64)
        converged = np.ones(telemetry_shape, dtype=np.bool_)
        markers = np.asarray(eta0[:, 0], dtype=np.float64)
        for marker in self.failing_markers:
            failed = np.isclose(markers, marker, rtol=0.0, atol=1.0e-14)
            residual[:, failed] = 1.0
            converged[:, failed] = False
        return NonlinearAdjustmentArm(
            dt=dt,
            times=np.asarray(saved_times, dtype=np.float64),
            eta=eta,
            xi=xi,
            gl2_stage_residual=residual,
            gl2_iterations=np.ones(telemetry_shape, dtype=np.int32),
            gl2_converged=converged,
            gl2_stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_hit_iteration_cap=np.zeros(telemetry_shape, dtype=np.bool_),
        )


def _jonswap_record(
    marker: int,
    *,
    depth: float = 1.0,
    peak_wavenumber: float = 4.0,
) -> dict[str, object]:
    return {
        "family_id": 3,
        "revision_id": 4,
        "marker": marker,
        "depth": depth,
        "peak_wavenumber": peak_wavenumber,
    }


class JonswapHorizonExecutorTests(unittest.TestCase):
    def test_groups_are_stable(self) -> None:
        grids = tuple(map(_grid, (9, 3, 8, 3, 10)))

        groups = horizon_sorted_groups(grids, solver_batch_size=2)

        self.assertEqual(groups, ((1, 3), (2, 0), (4,)))

    def test_executor_preserves_base_decisions_rows_and_time_metrics(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(
                Path(directory),
                execution,
                outer_size=5,
                solver_size=2,
            )
            arms = RecordingArmExecutor()
            adjustment_arms = RecordingAdjustmentArmExecutor()
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=arms,
                adjustment_arm_executor=adjustment_arms,
            )
            markers = np.arange(1, 6, dtype=np.float64) / 100.0
            initial = TrajectoryInitialBatch(
                eta0=np.repeat(markers[:, None], 64, axis=1),
                xi0=np.zeros((5, 64), dtype=np.float64),
                depths=np.ones(5, dtype=np.float64),
                specification_records=tuple(
                    _jonswap_record(index) for index in range(5)
                ),
            )
            grids = tuple(map(_grid, (9, 3, 8, 3, 10)))

            outcomes = executor._produce_jonswap(initial, grids)

        self.assertEqual(arms.calls, [(2, 3), (2, 9), (1, 10)])
        self.assertTrue(all(outcome.decision.accepted for outcome in outcomes))
        self.assertTrue(
            all(
                outcome.decision.evaluated & QualityReason.OUTSIDE_SUPPORT
                for outcome in outcomes
            )
        )
        self.assertEqual(
            [outcome.metrics["saved_time_count"] for outcome in outcomes],
            [9, 3, 8, 3, 10],
        )
        self.assertEqual(
            [
                float(outcome.rows.eta[0, 0])
                for outcome in outcomes
                if outcome.rows is not None
            ],
            [
                marker
                + float(outcome.metrics["nonlinear_adjustment_realized_terminal_time"])
                for marker, outcome in zip(markers, outcomes)
            ],
        )
        self.assertTrue(
            all(
                call[3] == PAPER_JONSWAP_ADJUSTMENT_POLICY.ramp_order
                for call in adjustment_arms.calls
            )
        )

    def test_mixed_burn_horizons_use_each_cases_own_endpoint(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=3, solver_size=3)
            production_arms = RecordingArmExecutor()
            adjustment_arms = RecordingAdjustmentArmExecutor()
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=production_arms,
                adjustment_arm_executor=adjustment_arms,
            )
            markers = np.asarray((0.01, 0.02, 0.03), dtype=np.float64)
            peak_wavenumbers = (4.0, 9.0, 16.0)
            initial = TrajectoryInitialBatch(
                eta0=np.repeat(markers[:, None], 64, axis=1),
                xi0=np.zeros((3, 64), dtype=np.float64),
                depths=np.ones(3, dtype=np.float64),
                specification_records=tuple(
                    _jonswap_record(index, peak_wavenumber=wavenumber)
                    for index, wavenumber in enumerate(peak_wavenumbers)
                ),
            )

            outcomes = executor._produce_jonswap(
                initial,
                tuple(map(_grid, (5, 5, 5))),
            )

        handed_off = production_arms.initial_eta[0][:, 0]
        realized = np.asarray(
            [
                outcome.metrics["nonlinear_adjustment_realized_terminal_time"]
                for outcome in outcomes
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(handed_off, markers + realized)
        self.assertGreater(len(set(realized.tolist())), 1)
        expected_ramps = (
            PAPER_JONSWAP_ADJUSTMENT_POLICY.ramp_time_peak_periods
            * realized
            / PAPER_JONSWAP_ADJUSTMENT_POLICY.burn_peak_periods
        )
        np.testing.assert_allclose(
            adjustment_arms.calls[0][2],
            expected_ramps,
            rtol=0.0,
            atol=0.08,
        )

    def test_full_band_adjustment_endpoint_is_not_projected_before_production(
        self,
    ) -> None:
        contract = PAPER_JONSWAP_GL2_CONTRACT
        execution = _execution(contract)
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        high_mode = 0.001 * np.cos(600.0 * x)
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            production_arms = RecordingArmExecutor(hamiltonian_drift=0.0)
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=production_arms,
                adjustment_arm_executor=RecordingAdjustmentArmExecutor(
                    terminal_addition=high_mode
                ),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, contract.nx), 0.01, dtype=np.float64),
                xi0=np.zeros((1, contract.nx), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            outcomes = executor._produce_jonswap(initial, (_grid(5),))

        handed_off = production_arms.initial_eta[0][0]
        coefficient = np.fft.rfft(handed_off - np.mean(handed_off))[600]
        self.assertGreater(abs(coefficient), 0.4)
        self.assertEqual(handed_off.shape, (2048,))
        assert outcomes[0].rows is not None
        self.assertEqual(outcomes[0].rows.eta.shape[-1], 1024)
        self.assertTrue(outcomes[0].decision.accepted)

    def test_burn_failure_skips_production_and_restores_proposal_order(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=5, solver_size=2)
            production_arms = RecordingArmExecutor()
            adjustment_arms = RecordingAdjustmentArmExecutor(
                failing_markers=(0.02, 0.05)
            )
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=production_arms,
                adjustment_arm_executor=adjustment_arms,
            )
            markers = np.arange(1, 6, dtype=np.float64) / 100.0
            initial = TrajectoryInitialBatch(
                eta0=np.repeat(markers[:, None], 64, axis=1),
                xi0=np.zeros((5, 64), dtype=np.float64),
                depths=np.ones(5, dtype=np.float64),
                specification_records=tuple(
                    _jonswap_record(index) for index in range(5)
                ),
                construction_metrics=tuple(
                    {"construction_marker": index} for index in range(5)
                ),
            )

            outcomes = executor._produce_jonswap(
                initial,
                tuple(map(_grid, (9, 3, 8, 3, 10))),
            )

        self.assertEqual(
            [outcome.decision.accepted for outcome in outcomes],
            [True, False, True, True, False],
        )
        self.assertIsNone(outcomes[1].rows)
        self.assertIsNone(outcomes[4].rows)
        self.assertEqual(
            outcomes[1].metrics["production_status"],
            "not_run_adjustment_failed",
        )
        self.assertTrue(outcomes[1].decision.failed & QualityReason.GL2_STAGE_RESIDUAL)
        self.assertEqual(outcomes[1].metrics["construction_marker"], 1)
        self.assertEqual(outcomes[4].metrics["construction_marker"], 4)
        self.assertEqual(
            sum(batch.shape[0] for batch in production_arms.initial_eta), 3
        )

    def test_burn_hamiltonian_is_not_an_acceptance_check_and_production_is_unramped(
        self,
    ) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            production_arms = RecordingArmExecutor()
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=production_arms,
                adjustment_arm_executor=RecordingAdjustmentArmExecutor(),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, 64), 0.01, dtype=np.float64),
                xi0=np.zeros((1, 64), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            (outcome,) = executor._produce_jonswap(initial, (_grid(5),))

        self.assertTrue(outcome.decision.accepted)
        self.assertNotIn("nonlinear_adjustment_hamiltonian", outcome.metrics)
        self.assertEqual(
            outcome.metrics["nonlinear_adjustment_production_ramp"],
            "disabled",
        )
        self.assertEqual(len(production_arms.calls), 1)
        self.assertEqual(float(production_arms.saved_times[0][0]), 0.0)

    def test_autonomous_hamiltonian_gate_rejects_after_accepted_burn(self) -> None:
        contract = replace(
            _contract(),
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
        execution = _execution(contract)
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            executor = HorizonBucketedJonswapBatchExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=RecordingArmExecutor(hamiltonian_drift=1.0e-2),
                adjustment_arm_executor=RecordingAdjustmentArmExecutor(),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, 64), 0.01, dtype=np.float64),
                xi0=np.zeros((1, 64), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            (outcome,) = executor._produce_jonswap(initial, (_grid(5),))

        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertTrue(outcome.decision.failed & QualityReason.HAMILTONIAN_DRIFT)
        self.assertTrue(outcome.decision.failed & QualityReason.INCOMPLETE_TRAJECTORY)
        self.assertTrue(outcome.metrics["nonlinear_adjustment_accepted"])
        self.assertEqual(outcome.metrics["production_status"], "completed")
        self.assertAlmostEqual(
            outcome.metrics["maximum_internal_hamiltonian_drift"], 1.0e-2
        )
        self.assertEqual(
            outcome.metrics["internal_hamiltonian_drift_threshold"],
            1.0e-3,
        )

    def test_bucketed_source_identity_extends_only_jonswap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = base.GenerationRequest(
                output_root=Path(directory),
                family="jonswap_tma",
                split=SplitId.TEST,
                accepted_cases=1024,
                batch_size=1024,
                platform="cpu",
            )
            baseline = base.build_run_spec(request)
            revised = bucketed.build_bucketed_run_spec(
                request,
                solver_batch_size=256,
            )
            with bucketed._bucketed_runtime(256):
                _, _, _, preflight = base.preflight(request)

        baseline_sources = dict(
            baseline.configuration["source_sha256"]  # type: ignore[arg-type]
        )
        revised_sources = dict(
            revised.configuration["source_sha256"]  # type: ignore[arg-type]
        )
        self.assertEqual(
            {path: revised_sources[path] for path in baseline_sources},
            baseline_sources,
        )
        self.assertEqual(
            set(revised_sources).difference(baseline_sources),
            {
                "scripts/generate_paper_dataset_jonswap.py",
                "solver/gen_data/jonswap_horizon_executor.py",
            },
        )
        self.assertNotEqual(
            baseline.config_fingerprint,
            revised.config_fingerprint,
        )
        numerical = revised.configuration["trajectory_execution"][  # type: ignore[index]
            "numerical"
        ]
        self.assertEqual(  # type: ignore[index]
            (
                numerical["nx"],
                numerical["target_nx"],
                numerical["maximum_wavenumber"],
                numerical["target_maximum_wavenumber"],
                numerical["dno_order"],
                numerical["target_dno_order"],
                numerical["gl2_iteration_cap"],
            ),
            (2048, 1024, 704.0, 128.0, 4, 6, 5),
        )
        self.assertEqual(
            preflight["expected_output"]["spatial_points_per_row"],  # type: ignore[index]
            1024,
        )
        self.assertEqual(
            preflight["expected_output"]["field_values_per_row"],  # type: ignore[index]
            3072,
        )
        changed_configuration = json.loads(
            json.dumps(
                revised.to_json_record()["configuration"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        changed_execution = dict(  # type: ignore[arg-type]
            changed_configuration["trajectory_execution"]
        )
        changed_numerical = dict(changed_execution["numerical"])  # type: ignore[arg-type]
        changed_numerical["target_nx"] = 512
        changed_execution["numerical"] = changed_numerical
        changed_configuration["trajectory_execution"] = changed_execution
        self.assertNotEqual(
            revised.config_fingerprint,
            replace(
                revised,
                configuration=changed_configuration,
            ).config_fingerprint,
        )
        executor = HorizonBucketedJonswapBatchExecutor(
            run_spec=revised,
            execution=TrajectoryExecutionConfig.paper("jonswap_tma"),
        )
        self.assertEqual(executor.solver_batch_size, 256)
        config = revised.configuration[BUCKETING_CONFIG_KEY]
        self.assertEqual(
            dict(config),  # type: ignore[arg-type]
            BucketingConfig(
                outer_proposal_size=1024,
                solver_batch_size=256,
            ).to_json_record(),
        )
        adjustment = config["nonlinear_adjustment"]  # type: ignore[index]
        self.assertEqual(  # type: ignore[index]
            adjustment["formula"],
            JONSWAP_ADJUSTMENT_FORMULA,
        )
        self.assertEqual(  # type: ignore[index]
            adjustment["ramp_time_peak_periods"],
            PAPER_JONSWAP_ADJUSTMENT_POLICY.ramp_time_peak_periods,
        )
        self.assertEqual(  # type: ignore[index]
            adjustment["burn_peak_periods"],
            PAPER_JONSWAP_ADJUSTMENT_POLICY.burn_peak_periods,
        )
        self.assertEqual(  # type: ignore[index]
            adjustment["autonomous_clock"],
            "restart_at_zero",
        )
        self.assertEqual(  # type: ignore[index]
            adjustment["hamiltonian_scope"],
            "autonomous_production_only",
        )

    def test_bucketed_preflight_rejects_nonpositive_solver_batch_size(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = base.GenerationRequest(
                output_root=Path(directory),
                family="jonswap_tma",
                split=SplitId.TEST,
                accepted_cases=27,
                batch_size=27,
                platform="cpu",
            )
            for solver_batch_size in (0, -1, True):
                with self.subTest(solver_batch_size=solver_batch_size):
                    with self.assertRaisesRegex(
                        ValueError,
                        "positive integer",
                    ):
                        bucketed.build_bucketed_run_spec(
                            request,
                            solver_batch_size=solver_batch_size,
                        )

    def test_executor_rejects_adjustment_policy_mismatch(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            valid = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            configuration = valid.to_json_record()["configuration"]
            assert isinstance(configuration, dict)
            config = configuration[BUCKETING_CONFIG_KEY]
            assert isinstance(config, dict)
            adjustment = config["nonlinear_adjustment"]
            assert isinstance(adjustment, dict)
            adjustment["formula"] = "different"
            invalid = DatasetGenerationSpec(
                root=valid.root,
                family_name=valid.family_name,
                family_id=valid.family_id,
                revision_id=valid.revision_id,
                split_id=valid.split_id,
                stream_id=valid.stream_id,
                case_targets=valid.case_targets,
                cell_codes=valid.cell_codes,
                batch_size=valid.batch_size,
                first_attempt_index=valid.first_attempt_index,
                maximum_attempts_per_accepted_case=(
                    valid.maximum_attempts_per_accepted_case
                ),
                configuration=configuration,
            )

            with self.assertRaisesRegex(ValueError, "config is inconsistent"):
                HorizonBucketedJonswapBatchExecutor(
                    run_spec=invalid,
                    execution=execution,
                    arm_executor=RecordingArmExecutor(),
                    adjustment_arm_executor=RecordingAdjustmentArmExecutor(),
                )


if __name__ == "__main__":
    unittest.main()
