"""Run numerical checks on generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheck,
    SimulationCheckResult,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class TrajectoryHealthMetrics:
    """Dense diagnostics computed before projecting a trajectory for storage."""

    state_finite: bool
    dno_output_finite: bool
    minimum_water_column: float | None
    initial_hamiltonian: float | None
    maximum_relative_hamiltonian_drift: float | None
    hamiltonian_drift_threshold: float


def evaluate_trajectory_health(
    hamiltonian: FloatArray,
    state_finite: BoolArray,
    dno_output_finite: BoolArray,
    minimum_water_column: FloatArray,
    *,
    hamiltonian_drift_threshold: float,
) -> tuple[TrajectoryHealthMetrics, SimulationCheckResult]:
    """Evaluate full-grid finiteness, clearance, and Hamiltonian accuracy."""

    required = (
        SimulationCheck.NONFINITE_STATE
        | SimulationCheck.NONFINITE_TARGET
        | SimulationCheck.BOTTOM_CLEARANCE
        | SimulationCheck.HAMILTONIAN_DRIFT
    )
    failed = SimulationCheck.NONE
    state_is_finite = bool(np.all(state_finite))
    dno_output_is_finite = bool(np.all(dno_output_finite))
    if not state_is_finite:
        failed |= SimulationCheck.NONFINITE_STATE
    if not dno_output_is_finite:
        failed |= SimulationCheck.NONFINITE_TARGET

    minimum_water_column_value = (
        float(np.min(minimum_water_column))
        if np.isfinite(minimum_water_column).all()
        else None
    )
    if minimum_water_column_value is None or minimum_water_column_value <= 0.0:
        failed |= SimulationCheck.BOTTOM_CLEARANCE

    initial_hamiltonian: float | None = None
    maximum_relative_hamiltonian_drift: float | None = None
    if np.isfinite(hamiltonian).all():
        initial_hamiltonian = float(hamiltonian[0])
        maximum_relative_hamiltonian_drift = float(
            np.max(
                np.abs(hamiltonian - initial_hamiltonian)
                / max(abs(initial_hamiltonian), np.finfo(np.float64).tiny)
            )
        )
    if (
        maximum_relative_hamiltonian_drift is None
        or not np.isfinite(maximum_relative_hamiltonian_drift)
        or maximum_relative_hamiltonian_drift > hamiltonian_drift_threshold
    ):
        failed |= SimulationCheck.HAMILTONIAN_DRIFT

    return (
        TrajectoryHealthMetrics(
            state_finite=state_is_finite,
            dno_output_finite=dno_output_is_finite,
            minimum_water_column=minimum_water_column_value,
            initial_hamiltonian=initial_hamiltonian,
            maximum_relative_hamiltonian_drift=maximum_relative_hamiltonian_drift,
            hamiltonian_drift_threshold=float(hamiltonian_drift_threshold),
        ),
        SimulationCheckResult(
            required=required,
            evaluated=required,
            failed=failed,
        ),
    )


def evaluate_trajectory(
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
    *,
    depth: float,
    gl2_succeeded: bool,
) -> SimulationCheckResult:
    """Reject failed integration, nonfinite values, or a dry point."""

    evaluated = (
        SimulationCheck.INCOMPLETE_TRAJECTORY
        | SimulationCheck.GL2_STAGE_RESIDUAL
        | SimulationCheck.NONFINITE_STATE
        | SimulationCheck.NONFINITE_TARGET
    )
    failed = SimulationCheck.NONE
    if not gl2_succeeded:
        failed |= SimulationCheck.GL2_STAGE_RESIDUAL

    state_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    target_finite = bool(np.isfinite(gxi).all())
    if not state_finite:
        failed |= SimulationCheck.NONFINITE_STATE
    if not target_finite:
        failed |= SimulationCheck.NONFINITE_TARGET

    if state_finite:
        evaluated |= SimulationCheck.BOTTOM_CLEARANCE
        if float(np.min(depth + eta)) <= 0.0:
            failed |= SimulationCheck.BOTTOM_CLEARANCE

    if failed != SimulationCheck.NONE:
        failed |= SimulationCheck.INCOMPLETE_TRAJECTORY
    return SimulationCheckResult(
        required=SimulationCheck.INCOMPLETE_TRAJECTORY,
        evaluated=evaluated,
        failed=failed,
    )


def evaluate_adjustment_trajectory(
    eta: FloatArray,
    xi: FloatArray,
    *,
    depth: float,
    gl2_succeeded: bool,
) -> tuple[SimulationCheckResult, float | None]:
    """Reject a failed, nonfinite, or dry nonlinear-adjustment trajectory."""

    evaluated = (
        SimulationCheck.INCOMPLETE_TRAJECTORY
        | SimulationCheck.GL2_STAGE_RESIDUAL
        | SimulationCheck.NONFINITE_STATE
    )
    failed = (
        SimulationCheck.NONE if gl2_succeeded else SimulationCheck.GL2_STAGE_RESIDUAL
    )
    state_is_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    minimum_water_column = None
    if not state_is_finite:
        failed |= SimulationCheck.NONFINITE_STATE
    else:
        evaluated |= SimulationCheck.BOTTOM_CLEARANCE
        minimum_water_column = float(np.min(depth + eta))
        if minimum_water_column <= 0.0:
            failed |= SimulationCheck.BOTTOM_CLEARANCE
    if failed != SimulationCheck.NONE:
        failed |= SimulationCheck.INCOMPLETE_TRAJECTORY

    return (
        SimulationCheckResult(
            required=SimulationCheck.INCOMPLETE_TRAJECTORY,
            evaluated=evaluated,
            failed=failed,
        ),
        minimum_water_column,
    )
