"""Evaluate integrated wave trajectories and retain numerically valid cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.case_checks import CaseCheck, CaseCheckResult
from solver.gen_data.pipeline.trajectory_config import RolloutConfig
from solver.gen_data.pipeline.trajectory_integration import (
    AdjustmentBatchIntegrator,
    BatchIntegrator,
    CompletedAdjustmentRollout,
    GL2BatchTelemetry,
    IntegratedTrajectoryBatch,
    integrate_adjustment_batch,
    integrate_batch,
    validate_adjustment_rollout,
    validate_integrated_batch,
)
from solver.gen_data.pipeline.trajectory_checks import (
    TrajectoryHealthMetrics,
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
class TrajectoryCaseResult:
    """Quality decision and retained data for one trajectory."""

    maximum_gl2_stage_residual: float
    decision: CaseCheckResult
    trajectory: TrajectorySamples | None
    health_metrics: TrajectoryHealthMetrics | None = None


@dataclass(frozen=True)
class AdjustmentCaseResult:
    """Quality decision and final state for one JONSWAP warm-up."""

    maximum_gl2_stage_residual: float
    decision: CaseCheckResult
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
    depth = np.asarray(depths, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[-1] != nx:
        raise ValueError(f"eta0 must have shape (batch, {nx}), got {eta.shape}")
    if xi.shape != eta.shape:
        raise ValueError("xi0 must have the same shape as eta0")
    if depth.shape != (eta.shape[0],):
        raise ValueError(f"depths must have shape ({eta.shape[0]},)")
    if not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ValueError("depths must be finite and positive")
    return eta, xi, depth


def _validate_saved_times(saved_times: FloatArray, *, saved_dt: float) -> FloatArray:
    """Validate and normalize one regular grid of stored times."""

    times = np.asarray(saved_times, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("saved_times must contain at least two times")
    intervals = np.diff(times)
    if not np.isfinite(times).all() or np.any(intervals <= 0.0):
        raise ValueError("saved_times must be finite and strictly increasing")
    if not np.allclose(intervals, saved_dt, rtol=0.0, atol=1.0e-13):
        raise ValueError("saved_times must use the configured saved_dt")
    return times


def _summarize_case_gl2(
    telemetry: GL2BatchTelemetry,
    case_index: int,
    config: RolloutConfig,
    *,
    saved_time_count: int,
) -> tuple[bool, float]:
    step_count = (saved_time_count - 1) * config.substeps_per_saved_frame
    residual = np.asarray(
        telemetry.stage_residual[:step_count, case_index],
        dtype=np.float64,
    )
    solved = (
        np.asarray(telemetry.converged[:step_count, case_index], dtype=np.bool_)
        & np.asarray(telemetry.stage_finite[:step_count, case_index], dtype=np.bool_)
        & np.asarray(telemetry.state_finite[:step_count, case_index], dtype=np.bool_)
        & np.isfinite(residual)
        & (residual <= config.gl2_residual_tolerance)
    )
    maximum_residual = float(np.max(residual)) if residual.size else 0.0
    return bool(np.all(solved)), maximum_residual


def _trajectory_samples(
    rollout: IntegratedTrajectoryBatch,
    case_index: int,
    saved_times: FloatArray,
) -> TrajectorySamples:
    count = saved_times.size
    return TrajectorySamples(
        times=np.asarray(saved_times, dtype=np.float64),
        eta=np.asarray(rollout.eta[:count, case_index], dtype=np.float64),
        xi=np.asarray(rollout.xi[:count, case_index], dtype=np.float64),
        gxi=np.asarray(rollout.gxi[:count, case_index], dtype=np.float64),
    )


def _combine_trajectory_checks(
    trajectory_decision: CaseCheckResult,
    health_decision: CaseCheckResult,
) -> CaseCheckResult:
    failed = trajectory_decision.failed | health_decision.failed
    if health_decision.failed:
        failed |= CaseCheck.INCOMPLETE_TRAJECTORY
    return CaseCheckResult(
        required=trajectory_decision.required | health_decision.required,
        evaluated=trajectory_decision.evaluated | health_decision.evaluated,
        failed=failed,
    )


def _evaluate_trajectory_case(
    rollout: IntegratedTrajectoryBatch,
    case_index: int,
    depth: float,
    saved_times: FloatArray,
    config: RolloutConfig,
) -> TrajectoryCaseResult:
    all_stages_solved, maximum_residual = _summarize_case_gl2(
        rollout.gl2,
        case_index,
        config,
        saved_time_count=saved_times.size,
    )
    trajectory = _trajectory_samples(rollout, case_index, saved_times)
    decision = evaluate_trajectory(
        trajectory.eta,
        trajectory.xi,
        trajectory.gxi,
        depth=depth,
        gl2_stages_solved=all_stages_solved,
    )

    health_metrics = None
    threshold = config.internal_hamiltonian_drift_threshold
    if threshold is not None:
        internal = rollout.internal_telemetry
        if internal is None:
            raise RuntimeError("the rollout config requires internal telemetry")
        count = saved_times.size
        health_metrics, health_decision = evaluate_trajectory_health(
            internal.hamiltonian[:count, case_index],
            internal.state_finite[:count, case_index],
            internal.dno_output_finite[:count, case_index],
            internal.minimum_water_column[:count, case_index],
            hamiltonian_drift_threshold=threshold,
        )
        decision = _combine_trajectory_checks(decision, health_decision)

    return TrajectoryCaseResult(
        maximum_gl2_stage_residual=maximum_residual,
        decision=decision,
        trajectory=trajectory if decision.accepted else None,
        health_metrics=health_metrics,
    )


def _evaluate_adjustment_case(
    rollout: CompletedAdjustmentRollout,
    case_index: int,
    depth: float,
    saved_times: FloatArray,
    config: RolloutConfig,
) -> AdjustmentCaseResult:
    all_stages_solved, maximum_residual = _summarize_case_gl2(
        rollout.gl2,
        case_index,
        config,
        saved_time_count=saved_times.size,
    )
    count = saved_times.size
    eta = np.asarray(rollout.eta[:count, case_index], dtype=np.float64)
    xi = np.asarray(rollout.xi[:count, case_index], dtype=np.float64)
    evaluated = (
        CaseCheck.INCOMPLETE_TRAJECTORY
        | CaseCheck.GL2_STAGE_RESIDUAL
        | CaseCheck.NONFINITE_STATE
    )
    failed = CaseCheck.NONE if all_stages_solved else CaseCheck.GL2_STAGE_RESIDUAL
    state_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    minimum_water_column = None
    if not state_finite:
        failed |= CaseCheck.NONFINITE_STATE
    else:
        evaluated |= CaseCheck.BOTTOM_CLEARANCE
        minimum_water_column = float(np.min(depth + eta))
        if minimum_water_column <= 0.0:
            failed |= CaseCheck.BOTTOM_CLEARANCE
    if failed:
        failed |= CaseCheck.INCOMPLETE_TRAJECTORY
    decision = CaseCheckResult(
        required=CaseCheck.INCOMPLETE_TRAJECTORY,
        evaluated=evaluated,
        failed=failed,
    )
    return AdjustmentCaseResult(
        maximum_gl2_stage_residual=maximum_residual,
        decision=decision,
        terminal_eta=np.asarray(eta[-1]) if decision.accepted else None,
        terminal_xi=np.asarray(xi[-1]) if decision.accepted else None,
        minimum_water_column=minimum_water_column,
    )


def _integrate_and_evaluate_batch(
    eta: FloatArray,
    xi: FloatArray,
    depths: FloatArray,
    saved_times_by_case: tuple[FloatArray, ...],
    longest_times: FloatArray,
    config: RolloutConfig,
    rollout_executor: BatchIntegrator,
) -> tuple[TrajectoryCaseResult, ...]:
    rollout = rollout_executor(
        eta0=eta,
        xi0=xi,
        depths=depths,
        saved_times=longest_times,
        config=config,
    )
    validate_integrated_batch(
        rollout,
        batch_size=eta.shape[0],
        saved_time_count=longest_times.size,
        config=config,
    )
    return tuple(
        _evaluate_trajectory_case(
            rollout,
            index,
            float(depths[index]),
            saved_times,
            config,
        )
        for index, saved_times in enumerate(saved_times_by_case)
    )


def execute_trajectory_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    *,
    config: RolloutConfig,
    rollout_executor: BatchIntegrator = integrate_batch,
) -> tuple[TrajectoryCaseResult, ...]:
    """Integrate and evaluate a fixed-duration trajectory batch."""

    eta, xi, depth = _validate_initial_conditions(
        eta0,
        xi0,
        depths,
        nx=config.nx,
    )
    times = _validate_saved_times(saved_times, saved_dt=config.saved_dt)
    return _integrate_and_evaluate_batch(
        eta,
        xi,
        depth,
        (times,) * eta.shape[0],
        times,
        config,
        rollout_executor,
    )


def _validate_variable_horizons(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times_by_case: tuple[FloatArray, ...],
    config: RolloutConfig,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    tuple[FloatArray, ...],
    FloatArray,
]:
    eta, xi, depth = _validate_initial_conditions(eta0, xi0, depths, nx=config.nx)
    if len(saved_times_by_case) != eta.shape[0]:
        raise ValueError("saved_times_by_case must contain one grid per input case")
    grids = tuple(
        _validate_saved_times(times, saved_dt=config.saved_dt)
        for times in saved_times_by_case
    )
    if not grids:
        raise ValueError("saved_times_by_case must not be empty")
    longest = max(grids, key=len)
    if any(not np.array_equal(times, longest[: times.size]) for times in grids):
        raise ValueError(
            "every variable saved-time grid must be a prefix of the longest grid"
        )
    return eta, xi, depth, grids, longest


def execute_variable_horizon_trajectory_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times_by_case: tuple[FloatArray, ...],
    *,
    config: RolloutConfig,
    rollout_executor: BatchIntegrator = integrate_batch,
) -> tuple[TrajectoryCaseResult, ...]:
    """Integrate once, then evaluate each case on its requested time prefix."""

    eta, xi, depth, grids, longest = _validate_variable_horizons(
        eta0,
        xi0,
        depths,
        saved_times_by_case,
        config,
    )
    return _integrate_and_evaluate_batch(
        eta,
        xi,
        depth,
        grids,
        longest,
        config,
        rollout_executor,
    )


def execute_variable_horizon_adjustment(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times_by_case: tuple[FloatArray, ...],
    *,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
    config: RolloutConfig,
    rollout_executor: AdjustmentBatchIntegrator = integrate_adjustment_batch,
) -> tuple[AdjustmentCaseResult, ...]:
    """Warm up JONSWAP cases and return each valid nonlinear endpoint."""

    eta, xi, depth, grids, longest = _validate_variable_horizons(
        eta0,
        xi0,
        depths,
        saved_times_by_case,
        config,
    )
    ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
    if ramp_times.shape != depth.shape:
        raise ValueError(f"nonlinear_ramp_times must have shape {depth.shape}")
    if not np.isfinite(ramp_times).all() or np.any(ramp_times <= 0.0):
        raise ValueError("nonlinear_ramp_times must be finite and positive")
    if nonlinear_ramp_order < 1:
        raise ValueError("nonlinear_ramp_order must be positive")

    rollout = rollout_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=longest,
        config=config,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    validate_adjustment_rollout(
        rollout,
        batch_size=eta.shape[0],
        saved_time_count=longest.size,
        config=config,
    )
    return tuple(
        _evaluate_adjustment_case(
            rollout,
            index,
            float(depth[index]),
            times,
            config,
        )
        for index, times in enumerate(grids)
    )
