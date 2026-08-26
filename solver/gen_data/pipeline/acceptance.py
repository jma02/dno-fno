"""Paper-dataset support, trajectory, and refinement decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectorySamples:
    """One trajectory sampled at the common saved comparison times."""

    times: FloatArray
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    reached_final_time: bool = True
    gl2_stages_solved: bool = True


@dataclass(frozen=True)
class DenseTrajectoryHealthMetrics:
    """Dense diagnostics computed before projecting a trajectory for storage."""

    state_finite: bool
    dno_output_finite: bool
    minimum_water_column: float | None
    initial_hamiltonian: float | None
    maximum_relative_hamiltonian_drift: float | None
    hamiltonian_drift_threshold: float


def evaluate_dense_trajectory_health(
    hamiltonian: FloatArray,
    state_finite: NDArray[np.bool_],
    dno_output_finite: NDArray[np.bool_],
    minimum_water_column: FloatArray,
    *,
    hamiltonian_drift_threshold: float,
    tiny: float = np.finfo(np.float64).tiny,
) -> tuple[DenseTrajectoryHealthMetrics, QualityDecision]:
    """Require dense internal finiteness, clearance, and Hamiltonian accuracy."""

    hamiltonian_array = np.asarray(hamiltonian, dtype=np.float64)
    state_finite_array = np.asarray(state_finite, dtype=np.bool_)
    dno_output_finite_array = np.asarray(dno_output_finite, dtype=np.bool_)
    water_column_array = np.asarray(
        minimum_water_column,
        dtype=np.float64,
    )
    arrays = (
        hamiltonian_array,
        state_finite_array,
        dno_output_finite_array,
        water_column_array,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("internal trajectory telemetry arrays must be one-dimensional")
    if hamiltonian_array.size == 0:
        raise ValueError("internal trajectory telemetry cannot be empty")
    if any(array.shape != hamiltonian_array.shape for array in arrays[1:]):
        raise ValueError("internal trajectory telemetry arrays must have one shape")
    if (
        not np.isfinite(hamiltonian_drift_threshold)
        or hamiltonian_drift_threshold < 0.0
    ):
        raise ValueError(
            "hamiltonian_drift_threshold must be finite and nonnegative"
        )
    if not np.isfinite(tiny) or tiny <= 0.0:
        raise ValueError("tiny must be finite and positive")

    required = (
        QualityReason.NONFINITE_STATE
        | QualityReason.NONFINITE_TARGET
        | QualityReason.BOTTOM_CLEARANCE
        | QualityReason.HAMILTONIAN_DRIFT
    )
    failed = QualityReason.NONE
    all_state_finite = bool(np.all(state_finite_array))
    all_dno_output_finite = bool(np.all(dno_output_finite_array))
    if not all_state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not all_dno_output_finite:
        failed |= QualityReason.NONFINITE_TARGET

    minimum_water: float | None = None
    if np.isfinite(water_column_array).all():
        minimum_water = float(np.min(water_column_array))
    if minimum_water is None or minimum_water <= 0.0:
        failed |= QualityReason.BOTTOM_CLEARANCE

    initial_hamiltonian: float | None = None
    maximum_drift: float | None = None
    if np.isfinite(hamiltonian_array).all():
        initial_hamiltonian = float(hamiltonian_array[0])
        denominator = max(abs(initial_hamiltonian), tiny)
        maximum_drift = float(
            np.max(
                np.abs(hamiltonian_array - initial_hamiltonian)
                / denominator
            )
        )
    if (
        maximum_drift is None
        or not np.isfinite(maximum_drift)
        or maximum_drift > hamiltonian_drift_threshold
    ):
        failed |= QualityReason.HAMILTONIAN_DRIFT

    return (
        DenseTrajectoryHealthMetrics(
            state_finite=all_state_finite,
            dno_output_finite=all_dno_output_finite,
            minimum_water_column=minimum_water,
            initial_hamiltonian=initial_hamiltonian,
            maximum_relative_hamiltonian_drift=maximum_drift,
            hamiltonian_drift_threshold=float(
                hamiltonian_drift_threshold
            ),
        ),
        QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=required,
            evaluated=required,
            failed=failed,
        ),
    )


def _validate_trajectory_arrays(
    trajectory: TrajectorySamples,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    times = np.asarray(trajectory.times, dtype=np.float64)
    eta = np.asarray(trajectory.eta, dtype=np.float64)
    xi = np.asarray(trajectory.xi, dtype=np.float64)
    gxi = np.asarray(trajectory.gxi, dtype=np.float64)

    if times.ndim != 1:
        raise ValueError(f"times must have shape (time,), got {times.shape}")
    if eta.ndim != 2:
        raise ValueError(f"eta must have shape (time, space), got {eta.shape}")
    if xi.shape != eta.shape or gxi.shape != eta.shape:
        raise ValueError("eta, xi, and gxi must have identical shapes")
    if eta.shape[0] != times.size:
        raise ValueError("trajectory fields and times must have equal lengths")
    if eta.shape[0] == 0 or eta.shape[1] == 0:
        raise ValueError("trajectory samples cannot be empty")
    return times, eta, xi, gxi


def evaluate_production_trajectory(
    trajectory: TrajectorySamples,
    *,
    depth: float,
) -> QualityDecision:
    """Require one complete admissible trajectory on its declared interval.

    ``INCOMPLETE_TRAJECTORY`` is the single required production check.  The
    other bits record why that check failed; they are diagnostics rather than
    additional empirical rejection rules.
    """

    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be finite and positive")

    _, eta, xi, gxi = _validate_trajectory_arrays(trajectory)
    evaluated = (
        QualityReason.INCOMPLETE_TRAJECTORY
        | QualityReason.GL2_STAGE_RESIDUAL
        | QualityReason.NONFINITE_STATE
        | QualityReason.NONFINITE_TARGET
    )
    failed = QualityReason.NONE
    if not trajectory.reached_final_time:
        failed |= QualityReason.INCOMPLETE_TRAJECTORY
    if not trajectory.gl2_stages_solved:
        failed |= QualityReason.GL2_STAGE_RESIDUAL

    state_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    target_finite = bool(np.isfinite(gxi).all())
    if not state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not target_finite:
        failed |= QualityReason.NONFINITE_TARGET

    if state_finite:
        evaluated |= QualityReason.BOTTOM_CLEARANCE
        if float(np.min(depth + eta)) <= 0.0:
            failed |= QualityReason.BOTTOM_CLEARANCE

    if failed != QualityReason.NONE:
        failed |= QualityReason.INCOMPLETE_TRAJECTORY
    return QualityDecision(
        scope=QualityScope.TRAJECTORY,
        required=QualityReason.INCOMPLETE_TRAJECTORY,
        evaluated=evaluated,
        failed=failed,
    )
