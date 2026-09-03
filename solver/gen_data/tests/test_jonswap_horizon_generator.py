"""CPU tests for horizon-bucketed JONSWAP/TMA execution."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from solver.gen_data.jonswap_horizon_generator import (  # noqa: E402
    HorizonBucketedJonswapBatchGenerator,
    horizon_sorted_groups,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_JONSWAP_ADJUSTMENT_CONFIG,
    PAPER_JONSWAP_ROLLOUT_CONFIG,
    RolloutConfig,
    TrajectoryExecutionConfig,
    TrajectoryFrameSelectionConfig,
    paper_trajectory_execution,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedAdjustmentBatch,
    IntegratedTrajectoryBatch,
    GL2BatchTelemetry,
    InternalHealthTelemetry,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
)
from solver.gen_data.trajectory_batch_generator import (  # noqa: E402
    SimulationTimeGrid,
)


def _contract() -> RolloutConfig:
    return RolloutConfig(
        nx=64,
        target_nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        target_dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        target_maximum_wavenumber=16.0,
        dt=0.04,
        saved_dt=0.08,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        target_time_chunk_size=2,
    )


def _execution(
    contract: RolloutConfig | None = None,
) -> TrajectoryExecutionConfig:
    return TrajectoryExecutionConfig(
        family="jonswap_tma",
        numerical=_contract() if contract is None else contract,
        frame_selection=TrajectoryFrameSelectionConfig(
            tanaka_count=3,
            tanaka_alpha=0.5,
            tanaka_sigma_steps=1.0,
            benjamin_feir_count=3,
            jonswap_tma_count=3,
        ),
        period_count=16,
        jonswap_quadrature_order=4,
        jonswap_adjustment=PAPER_JONSWAP_ADJUSTMENT_CONFIG,
    )


def _run_spec(
    root: Path,
    execution: TrajectoryExecutionConfig,
    *,
    outer_size: int,
    solver_size: int,
) -> DatasetChunkConfig:
    parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[0]
    return DatasetChunkConfig(
        root=root,
        family_name="jonswap_tma",
        family_id=PhysicalFamilyId.JONSWAP_TMA,
        dataset_split=DatasetSplit.TEST,
        simulation_targets={parameter_group_id: outer_size},
        batch_size=outer_size,
        solver_batch_size=solver_size,
    )


def _grid(saved_count: int) -> SimulationTimeGrid:
    saved_times = np.arange(saved_count, dtype=np.float64) * 0.08
    terminal_time = float(saved_times[-1])
    return SimulationTimeGrid(
        intended_terminal_time=terminal_time,
        realized_terminal_time=terminal_time,
        saved_times=saved_times,
    )


class RecordingRolloutIntegrator:
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
        config: RolloutConfig,
    ) -> IntegratedTrajectoryBatch:
        del depths
        batch_size, internal_nx = eta0.shape
        self.calls.append((batch_size, saved_times.size))
        self.initial_eta.append(np.asarray(eta0, dtype=np.float64).copy())
        self.saved_times.append(np.asarray(saved_times, dtype=np.float64).copy())
        saved_count = saved_times.size
        delivered_nx = config.target_nx
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
        substeps = int(round(config.saved_dt / config.dt))
        telemetry_shape = ((saved_count - 1) * substeps, batch_size)
        internal_telemetry = None
        if self.hamiltonian_drift is not None:
            hamiltonian = np.ones((saved_count, batch_size), dtype=np.float64)
            hamiltonian[-1] += self.hamiltonian_drift
            internal_telemetry = InternalHealthTelemetry(
                hamiltonian=hamiltonian,
                state_finite=np.ones_like(hamiltonian, dtype=np.bool_),
                dno_output_finite=np.ones_like(hamiltonian, dtype=np.bool_),
                minimum_water_column=np.ones_like(
                    hamiltonian,
                    dtype=np.float64,
                ),
            )
        return IntegratedTrajectoryBatch(
            eta=np.repeat(delivered_eta0[None, :, :], saved_count, axis=0),
            xi=np.repeat(delivered_xi0[None, :, :], saved_count, axis=0),
            gxi=np.zeros(
                (saved_count, batch_size, delivered_nx),
                dtype=np.float64,
            ),
            gl2=GL2BatchTelemetry(
                stage_residual=np.zeros(telemetry_shape, dtype=np.float64),
                converged=np.ones(telemetry_shape, dtype=np.bool_),
                stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
                state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            ),
            internal_telemetry=internal_telemetry,
        )


class RecordingAdjustmentRolloutIntegrator:
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
        config: RolloutConfig,
        nonlinear_ramp_times: np.ndarray,
        nonlinear_ramp_order: int,
    ) -> IntegratedAdjustmentBatch:
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
        substeps = int(round(config.saved_dt / config.dt))
        telemetry_shape = ((saved_count - 1) * substeps, batch_size)
        residual = np.zeros(telemetry_shape, dtype=np.float64)
        converged = np.ones(telemetry_shape, dtype=np.bool_)
        markers = np.asarray(eta0[:, 0], dtype=np.float64)
        for marker in self.failing_markers:
            failed = np.isclose(markers, marker, rtol=0.0, atol=1.0e-14)
            residual[:, failed] = 1.0
            converged[:, failed] = False
        return IntegratedAdjustmentBatch(
            eta=eta,
            xi=xi,
            gl2=GL2BatchTelemetry(
                stage_residual=residual,
                converged=converged,
                stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
                state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            ),
        )


def _jonswap_record(
    marker: int,
    *,
    depth: float = 1.0,
    peak_wavenumber: float = 4.0,
) -> dict[str, object]:
    return {
        "family_id": 3,
        "marker": marker,
        "depth": depth,
        "peak_wavenumber": peak_wavenumber,
    }


class JonswapHorizonGeneratorTests(unittest.TestCase):
    def test_groups_are_stable(self) -> None:
        grids = tuple(map(_grid, (9, 3, 8, 3, 10)))

        groups = horizon_sorted_groups(grids, solver_batch_size=2)

        self.assertEqual(groups, ((1, 3), (2, 0), (4,)))

    def test_generator_preserves_base_decisions_rows_and_time_metrics(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(
                Path(directory),
                execution,
                outer_size=5,
                solver_size=2,
            )
            rollouts = RecordingRolloutIntegrator()
            adjustment_rollouts = RecordingAdjustmentRolloutIntegrator()
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=rollouts,
                adjustment_rollout_integrator=adjustment_rollouts,
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

            outcomes = generator._integrate_and_subsample_jonswap(initial, grids)

        self.assertEqual(rollouts.calls, [(2, 3), (2, 9), (1, 10)])
        self.assertTrue(all(outcome.decision.accepted for outcome in outcomes))
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
                call[3] == PAPER_JONSWAP_ADJUSTMENT_CONFIG.ramp_order
                for call in adjustment_rollouts.calls
            )
        )

    def test_mixed_burn_horizons_use_each_simulations_own_endpoint(self) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=3, solver_size=3)
            production_rollouts = RecordingRolloutIntegrator()
            adjustment_rollouts = RecordingAdjustmentRolloutIntegrator()
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=production_rollouts,
                adjustment_rollout_integrator=adjustment_rollouts,
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

            outcomes = generator._integrate_and_subsample_jonswap(
                initial,
                tuple(map(_grid, (5, 5, 5))),
            )

        handed_off = production_rollouts.initial_eta[0][:, 0]
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
            PAPER_JONSWAP_ADJUSTMENT_CONFIG.ramp_time_peak_periods
            * realized
            / PAPER_JONSWAP_ADJUSTMENT_CONFIG.burn_peak_periods
        )
        np.testing.assert_allclose(
            adjustment_rollouts.calls[0][2],
            expected_ramps,
            rtol=0.0,
            atol=0.08,
        )

    def test_full_band_adjustment_endpoint_is_not_projected_before_production(
        self,
    ) -> None:
        contract = PAPER_JONSWAP_ROLLOUT_CONFIG
        execution = _execution(contract)
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        high_mode = 0.001 * np.cos(600.0 * x)
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            production_rollouts = RecordingRolloutIntegrator(hamiltonian_drift=0.0)
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=production_rollouts,
                adjustment_rollout_integrator=RecordingAdjustmentRolloutIntegrator(
                    terminal_addition=high_mode
                ),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, contract.nx), 0.01, dtype=np.float64),
                xi0=np.zeros((1, contract.nx), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            outcomes = generator._integrate_and_subsample_jonswap(initial, (_grid(5),))

        handed_off = production_rollouts.initial_eta[0][0]
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
            production_rollouts = RecordingRolloutIntegrator()
            adjustment_rollouts = RecordingAdjustmentRolloutIntegrator(
                failing_markers=(0.02, 0.05)
            )
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=production_rollouts,
                adjustment_rollout_integrator=adjustment_rollouts,
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

            outcomes = generator._integrate_and_subsample_jonswap(
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
        self.assertTrue(outcomes[1].decision.integration_failure)
        self.assertEqual(outcomes[1].metrics["construction_marker"], 1)
        self.assertEqual(outcomes[4].metrics["construction_marker"], 4)
        self.assertEqual(
            sum(batch.shape[0] for batch in production_rollouts.initial_eta), 3
        )

    def test_burn_hamiltonian_is_not_an_acceptance_check_and_production_is_unramped(
        self,
    ) -> None:
        execution = _execution()
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            production_rollouts = RecordingRolloutIntegrator()
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=production_rollouts,
                adjustment_rollout_integrator=RecordingAdjustmentRolloutIntegrator(),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, 64), 0.01, dtype=np.float64),
                xi0=np.zeros((1, 64), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            (outcome,) = generator._integrate_and_subsample_jonswap(
                initial, (_grid(5),)
            )

        self.assertTrue(outcome.decision.accepted)
        self.assertNotIn("nonlinear_adjustment_hamiltonian", outcome.metrics)
        self.assertEqual(
            outcome.metrics["nonlinear_adjustment_production_ramp"],
            "disabled",
        )
        self.assertEqual(len(production_rollouts.calls), 1)
        self.assertEqual(float(production_rollouts.saved_times[0][0]), 0.0)

    def test_autonomous_hamiltonian_gate_rejects_after_accepted_burn(self) -> None:
        contract = replace(
            _contract(),
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
        execution = _execution(contract)
        with tempfile.TemporaryDirectory() as directory:
            spec = _run_spec(Path(directory), execution, outer_size=1, solver_size=1)
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=spec,
                execution=execution,
                rollout_integrator=RecordingRolloutIntegrator(hamiltonian_drift=1.0e-2),
                adjustment_rollout_integrator=RecordingAdjustmentRolloutIntegrator(),
            )
            initial = TrajectoryInitialBatch(
                eta0=np.full((1, 64), 0.01, dtype=np.float64),
                xi0=np.zeros((1, 64), dtype=np.float64),
                depths=np.ones(1, dtype=np.float64),
                specification_records=(_jonswap_record(0),),
            )

            (outcome,) = generator._integrate_and_subsample_jonswap(
                initial, (_grid(5),)
            )

        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertTrue(outcome.decision.hamiltonian_drift)
        self.assertTrue(outcome.decision.incomplete_trajectory)
        self.assertTrue(outcome.metrics["nonlinear_adjustment_accepted"])
        self.assertEqual(outcome.metrics["production_status"], "completed")
        self.assertAlmostEqual(
            outcome.metrics["maximum_internal_hamiltonian_drift"], 1.0e-2
        )

    def test_paper_generator_uses_configured_solver_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution = paper_trajectory_execution("jonswap_tma")
            config = _run_spec(
                Path(directory),
                execution,
                outer_size=1024,
                solver_size=256,
            )

        self.assertEqual(config.solver_batch_size, 256)
        numerical = execution.numerical
        self.assertEqual(
            (
                numerical.nx,
                numerical.target_nx,
                numerical.maximum_wavenumber,
                numerical.target_maximum_wavenumber,
                numerical.dno_order,
                numerical.target_dno_order,
                numerical.gl2_iteration_cap,
            ),
            (2048, 1024, 704.0, 128.0, 4, 6, 5),
        )
        generator = HorizonBucketedJonswapBatchGenerator(
            chunk_config=config,
            execution=execution,
        )
        self.assertEqual(generator.chunk_config.solver_batch_size, 256)


if __name__ == "__main__":
    unittest.main()
