"""Run numerical checks on generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult

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

    state_is_finite = bool(np.all(state_finite))
    dno_output_is_finite = bool(np.all(dno_output_finite))

    minimum_water_column_value = (
        float(np.min(minimum_water_column))
        if np.isfinite(minimum_water_column).all()
        else None
    )
    nonpositive_water_height = (
        minimum_water_column_value is None or minimum_water_column_value <= 0.0
    )

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
    hamiltonian_drift = (
        maximum_relative_hamiltonian_drift is None
        or not np.isfinite(maximum_relative_hamiltonian_drift)
        or maximum_relative_hamiltonian_drift > hamiltonian_drift_threshold
    )
    accepted = (
        state_is_finite
        and dno_output_is_finite
        and not nonpositive_water_height
        and not hamiltonian_drift
    )

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
            accepted=accepted,
            nonfinite_state=not state_is_finite,
            nonfinite_target=not dno_output_is_finite,
            nonpositive_water_height=nonpositive_water_height,
            hamiltonian_drift=hamiltonian_drift,
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

    state_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    target_finite = bool(np.isfinite(gxi).all())
    nonpositive_water_height = state_finite and float(np.min(depth + eta)) <= 0.0
    accepted = (
        gl2_succeeded
        and state_finite
        and target_finite
        and not nonpositive_water_height
    )
    return SimulationCheckResult(
        accepted=accepted,
        nonfinite_state=not state_finite,
        nonfinite_target=not target_finite,
        nonpositive_water_height=nonpositive_water_height,
        integration_failure=not gl2_succeeded,
        incomplete_trajectory=not accepted,
    )


def evaluate_adjustment_trajectory(
    eta: FloatArray,
    xi: FloatArray,
    *,
    depth: float,
    gl2_succeeded: bool,
) -> tuple[SimulationCheckResult, float | None]:
    """Reject a failed, nonfinite, or dry nonlinear-adjustment trajectory."""

    state_is_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    minimum_water_column = None
    if state_is_finite:
        minimum_water_column = float(np.min(depth + eta))
    nonpositive_water_height = (
        minimum_water_column is not None and minimum_water_column <= 0.0
    )
    accepted = gl2_succeeded and state_is_finite and not nonpositive_water_height

    return (
        SimulationCheckResult(
            accepted=accepted,
            nonfinite_state=not state_is_finite,
            nonpositive_water_height=nonpositive_water_height,
            integration_failure=not gl2_succeeded,
            incomplete_trajectory=not accepted,
        ),
        minimum_water_column,
    )
