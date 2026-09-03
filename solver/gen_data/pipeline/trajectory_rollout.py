"""Integrate wave trajectories and retain numerically valid simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.trajectory_config import RolloutConfig
from solver.gen_data.pipeline.trajectory_integration import (
    AdjustmentBatchIntegrator,
    BatchIntegrator,
    GL2BatchTelemetry,
    IntegratedAdjustmentBatch,
    IntegratedTrajectoryBatch,
    integrate_adjustment_batch,
    integrate_batch,
    validate_adjustment_rollout,
    validate_integrated_batch,
)
from solver.gen_data.pipeline.trajectory_checks import (
    TrajectoryHealthMetrics,
    evaluate_adjustment_trajectory,
    evaluate_trajectory,
    evaluate_trajectory_health,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectorySamples:
    """Saved times, states, and DNO targets for one trajectory."""

    times: FloatArray
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray


@dataclass(frozen=True)
class TrajectorySimulationResult:
    """Acceptance decision and retained data for one trajectory."""

    maximum_gl2_stage_residual: float
    decision: SimulationCheckResult
    trajectory: TrajectorySamples | None
    health_metrics: TrajectoryHealthMetrics | None = None


@dataclass(frozen=True)
class AdjustmentSimulationResult:
    """Acceptance decision and final state for one JONSWAP warm-up."""

    maximum_gl2_stage_residual: float
    decision: SimulationCheckResult
    terminal_eta: FloatArray | None
    terminal_xi: FloatArray | None
    minimum_water_column: float | None


def _validate_initial_conditions(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    *,
    nx: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate and normalize one batch of initial wave states."""

    eta = np.asarray(eta0, dtype=np.float64)
    xi = np.asarray(xi0, dtype=np.float64)
    depth_values = np.asarray(depths, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[-1] != nx:
        raise ValueError(f"eta0 must have shape (batch, {nx}), got {eta.shape}")
    if xi.shape != eta.shape:
        raise ValueError("xi0 must have the same shape as eta0")
    if depth_values.shape != (eta.shape[0],):
        raise ValueError(f"depths must have shape ({eta.shape[0]},)")
    if not np.isfinite(depth_values).all() or np.any(depth_values <= 0.0):
        raise ValueError("depths must be finite and positive")
    return eta, xi, depth_values


def _evaluate_gl2_steps(
    telemetry: GL2BatchTelemetry,
    simulation_index: int,
    config: RolloutConfig,
    *,
    saved_time_count: int,
) -> tuple[bool, float]:
    step_count = (saved_time_count - 1) * config.substeps_per_saved_frame
    residual = np.asarray(
        telemetry.stage_residual[:step_count, simulation_index],
        dtype=np.float64,
    )
    successful_steps = (
        np.asarray(telemetry.converged[:step_count, simulation_index], dtype=np.bool_)
        & np.asarray(
            telemetry.stage_finite[:step_count, simulation_index], dtype=np.bool_
        )
        & np.asarray(
            telemetry.state_finite[:step_count, simulation_index], dtype=np.bool_
        )
        & np.isfinite(residual)
        & (residual <= config.gl2_residual_tolerance)
    )
    maximum_stage_residual = float(np.max(residual)) if residual.size else 0.0
    return bool(np.all(successful_steps)), maximum_stage_residual


def _evaluate_trajectory_simulation(
    rollout: IntegratedTrajectoryBatch,
    simulation_index: int,
    depth: float,
    saved_times: FloatArray,
    config: RolloutConfig,
) -> TrajectorySimulationResult:
    gl2_succeeded, maximum_stage_residual = _evaluate_gl2_steps(
        rollout.gl2,
        simulation_index,
        config,
        saved_time_count=saved_times.size,
    )
    count = saved_times.size
    trajectory = TrajectorySamples(
        times=np.asarray(saved_times, dtype=np.float64),
        eta=np.asarray(rollout.eta[:count, simulation_index], dtype=np.float64),
        xi=np.asarray(rollout.xi[:count, simulation_index], dtype=np.float64),
        gxi=np.asarray(rollout.gxi[:count, simulation_index], dtype=np.float64),
    )
    decision = evaluate_trajectory(
        trajectory.eta,
        trajectory.xi,
        trajectory.gxi,
        depth=depth,
        gl2_succeeded=gl2_succeeded,
    )

    health_metrics = None
    drift_threshold = config.internal_hamiltonian_drift_threshold
    if drift_threshold is not None:
        solver_grid = rollout.internal_telemetry
        if solver_grid is None:
            raise RuntimeError("the rollout config requires internal telemetry")
        health_metrics, health_decision = evaluate_trajectory_health(
            solver_grid.hamiltonian[:count, simulation_index],
            solver_grid.state_finite[:count, simulation_index],
            solver_grid.dno_output_finite[:count, simulation_index],
            solver_grid.minimum_water_column[:count, simulation_index],
            hamiltonian_drift_threshold=drift_threshold,
        )
        decision = SimulationCheckResult(
            accepted=decision.accepted and health_decision.accepted,
            nonfinite_state=(
                decision.nonfinite_state or health_decision.nonfinite_state
            ),
            nonfinite_target=(
                decision.nonfinite_target or health_decision.nonfinite_target
            ),
            nonpositive_water_height=(
                decision.nonpositive_water_height
                or health_decision.nonpositive_water_height
            ),
            hamiltonian_drift=(
                decision.hamiltonian_drift or health_decision.hamiltonian_drift
            ),
            integration_failure=(
                decision.integration_failure or health_decision.integration_failure
            ),
            outside_support=(
                decision.outside_support or health_decision.outside_support
            ),
            incomplete_trajectory=(
                decision.incomplete_trajectory or not health_decision.accepted
            ),
        )

    return TrajectorySimulationResult(
        maximum_gl2_stage_residual=maximum_stage_residual,
        decision=decision,
        trajectory=trajectory if decision.accepted else None,
        health_metrics=health_metrics,
    )


def _evaluate_adjustment_simulation(
    rollout: IntegratedAdjustmentBatch,
    simulation_index: int,
    depth: float,
    saved_time_count: int,
    config: RolloutConfig,
) -> AdjustmentSimulationResult:
    gl2_succeeded, maximum_stage_residual = _evaluate_gl2_steps(
        rollout.gl2,
        simulation_index,
        config,
        saved_time_count=saved_time_count,
    )
    eta = np.asarray(rollout.eta[:saved_time_count, simulation_index], dtype=np.float64)
    xi = np.asarray(rollout.xi[:saved_time_count, simulation_index], dtype=np.float64)
    decision, minimum_water_column = evaluate_adjustment_trajectory(
        eta,
        xi,
        depth=depth,
        gl2_succeeded=gl2_succeeded,
    )
    return AdjustmentSimulationResult(
        maximum_gl2_stage_residual=maximum_stage_residual,
        decision=decision,
        terminal_eta=eta[-1] if decision.accepted else None,
        terminal_xi=xi[-1] if decision.accepted else None,
        minimum_water_column=minimum_water_column,
    )


def _prepare_time_grids(
    time_grids: tuple[FloatArray, ...],
    *,
    batch_size: int,
) -> tuple[tuple[FloatArray, ...], FloatArray]:
    if len(time_grids) != batch_size:
        raise ValueError("time_grids must contain one grid per input simulation")
    grids = tuple(np.asarray(times, dtype=np.float64) for times in time_grids)
    integration_times = max(grids, key=len)
    if any(
        not np.array_equal(times, integration_times[: times.size]) for times in grids
    ):
        raise ValueError("every time grid must be a prefix of the longest grid")
    return grids, integration_times


def execute_trajectory_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    time_grids: tuple[FloatArray, ...],
    *,
    config: RolloutConfig,
    integrator: BatchIntegrator = integrate_batch,
) -> tuple[TrajectorySimulationResult, ...]:
    """Integrate a batch, then evaluate each simulation on its requested time grid."""

    eta, xi, depth_values = _validate_initial_conditions(
        eta0,
        xi0,
        depths,
        nx=config.nx,
    )
    grids, integration_times = _prepare_time_grids(
        time_grids,
        batch_size=eta.shape[0],
    )
    rollout = integrator(
        eta0=eta,
        xi0=xi,
        depths=depth_values,
        saved_times=integration_times,
        config=config,
    )
    validate_integrated_batch(
        rollout,
        batch_size=eta.shape[0],
        saved_time_count=integration_times.size,
        config=config,
    )
    return tuple(
        _evaluate_trajectory_simulation(
            rollout,
            index,
            float(depth_values[index]),
            time_grid,
            config,
        )
        for index, time_grid in enumerate(grids)
    )


def execute_adjustment_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    time_grids: tuple[FloatArray, ...],
    *,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
    config: RolloutConfig,
    integrator: AdjustmentBatchIntegrator = integrate_adjustment_batch,
) -> tuple[AdjustmentSimulationResult, ...]:
    """Warm up JONSWAP simulations and return each valid nonlinear endpoint."""

    eta, xi, depth_values = _validate_initial_conditions(
        eta0,
        xi0,
        depths,
        nx=config.nx,
    )
    grids, integration_times = _prepare_time_grids(
        time_grids,
        batch_size=eta.shape[0],
    )
    ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
    if ramp_times.shape != depth_values.shape:
        raise ValueError(f"nonlinear_ramp_times must have shape {depth_values.shape}")
    if not np.isfinite(ramp_times).all() or np.any(ramp_times <= 0.0):
        raise ValueError("nonlinear_ramp_times must be finite and positive")
    if nonlinear_ramp_order < 1:
        raise ValueError("nonlinear_ramp_order must be positive")

    rollout = integrator(
        eta0=eta,
        xi0=xi,
        depths=depth_values,
        saved_times=integration_times,
        config=config,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    validate_adjustment_rollout(
        rollout,
        batch_size=eta.shape[0],
        saved_time_count=integration_times.size,
        config=config,
    )
    return tuple(
        _evaluate_adjustment_simulation(
            rollout,
            index,
            float(depth_values[index]),
            time_grid.size,
            config,
        )
        for index, time_grid in enumerate(grids)
    )
