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
class RefinementTrajectory:
    """One trajectory restricted to the common saved comparison times."""

    times: FloatArray
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    complete: bool = True
    gl2_stages_solved: bool = True


@dataclass(frozen=True)
class TemporalRefinementMetrics:
    """Relative delivered-band discrepancies for a consecutive pair."""

    eta_error: float
    xi_error: float
    gxi_error: float
    maximum_error: float


@dataclass(frozen=True)
class InternalTrajectoryMetrics:
    """Dense diagnostics computed before projecting a trajectory for storage."""

    state_finite: bool
    dno_finite: bool
    minimum_water_column: float | None
    initial_hamiltonian: float | None
    maximum_relative_hamiltonian_drift: float | None
    hamiltonian_drift_threshold: float


def evaluate_internal_trajectory_health(
    hamiltonian: FloatArray,
    state_finite: NDArray[np.bool_],
    dno_finite: NDArray[np.bool_],
    minimum_water_column: FloatArray,
    *,
    hamiltonian_drift_threshold: float,
    tiny: float = np.finfo(np.float64).tiny,
) -> tuple[InternalTrajectoryMetrics, QualityDecision]:
    """Require dense internal finiteness, clearance, and Hamiltonian accuracy."""

    hamiltonian_array = np.asarray(hamiltonian, dtype=np.float64)
    state_finite_array = np.asarray(state_finite, dtype=np.bool_)
    dno_finite_array = np.asarray(dno_finite, dtype=np.bool_)
    water_column_array = np.asarray(
        minimum_water_column,
        dtype=np.float64,
    )
    arrays = (
        hamiltonian_array,
        state_finite_array,
        dno_finite_array,
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
    all_dno_finite = bool(np.all(dno_finite_array))
    if not all_state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not all_dno_finite:
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
        InternalTrajectoryMetrics(
            state_finite=all_state_finite,
            dno_finite=all_dno_finite,
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


def _as_trajectory_arrays(
    trajectory: RefinementTrajectory,
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
        raise ValueError("a refinement trajectory cannot be empty")
    return times, eta, xi, gxi


def _project_to_wavenumber_band(
    field: FloatArray,
    *,
    length: float,
    maximum_wavenumber: float,
) -> FloatArray:
    nx = field.shape[-1]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    coefficients = np.fft.fft(field, axis=-1)
    coefficients[..., np.abs(wavenumbers) > maximum_wavenumber] = 0.0
    return np.fft.ifft(coefficients, axis=-1).real


def _relative_trajectory_error(
    coarse: FloatArray,
    fine: FloatArray,
    *,
    relative_floor: float,
) -> float:
    normalization = np.sqrt(coarse.shape[-1])
    numerator = float(
        np.max(np.linalg.norm(coarse - fine, axis=-1) / normalization)
    )
    denominator = float(
        np.max(np.linalg.norm(fine, axis=-1) / normalization)
    )
    return numerator / (denominator + relative_floor)


def _infinite_refinement_metrics() -> TemporalRefinementMetrics:
    return TemporalRefinementMetrics(
        eta_error=float("inf"),
        xi_error=float("inf"),
        gxi_error=float("inf"),
        maximum_error=float("inf"),
    )


def evaluate_complete_numerical_trajectory(
    trajectory: RefinementTrajectory,
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

    _, eta, xi, gxi = _as_trajectory_arrays(trajectory)
    evaluated = (
        QualityReason.INCOMPLETE_TRAJECTORY
        | QualityReason.GL2_STAGE_RESIDUAL
        | QualityReason.NONFINITE_STATE
        | QualityReason.NONFINITE_TARGET
    )
    failed = QualityReason.NONE
    if not trajectory.complete:
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


def evaluate_temporal_refinement(
    coarse: RefinementTrajectory,
    fine: RefinementTrajectory,
    *,
    depth: float,
    gravity: float,
    length: float,
    maximum_wavenumber: float,
    tolerance: float = 1e-3,
    relative_floor: float = 1e-12,
) -> tuple[TemporalRefinementMetrics, QualityDecision]:
    """Compare two consecutive time refinements on the delivered Fourier band.

    ``coarse`` and ``fine`` must already be restricted to identical saved
    times and use the same spatial grid. A missing numerical trajectory
    produces an infinite temporal defect rather than an additional empirical
    rejection rule.
    """

    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be finite and positive")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not np.isfinite(maximum_wavenumber) or maximum_wavenumber < 0.0:
        raise ValueError("maximum_wavenumber must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")

    coarse_arrays = _as_trajectory_arrays(coarse)
    fine_arrays = _as_trajectory_arrays(fine)
    coarse_times, coarse_eta, coarse_xi, coarse_gxi = coarse_arrays
    fine_times, fine_eta, fine_xi, fine_gxi = fine_arrays

    evaluated = (
        QualityReason.TEMPORAL_DEFECT
        | QualityReason.INCOMPLETE_TRAJECTORY
        | QualityReason.GL2_STAGE_RESIDUAL
        | QualityReason.NONFINITE_STATE
        | QualityReason.NONFINITE_TARGET
    )
    failed = QualityReason.NONE
    if not coarse.complete or not fine.complete:
        failed |= QualityReason.INCOMPLETE_TRAJECTORY
    if not coarse.gl2_stages_solved or not fine.gl2_stages_solved:
        failed |= QualityReason.GL2_STAGE_RESIDUAL

    state_finite = bool(
        np.isfinite(coarse_eta).all()
        and np.isfinite(coarse_xi).all()
        and np.isfinite(fine_eta).all()
        and np.isfinite(fine_xi).all()
    )
    target_finite = bool(
        np.isfinite(coarse_gxi).all() and np.isfinite(fine_gxi).all()
    )
    if not state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not target_finite:
        failed |= QualityReason.NONFINITE_TARGET

    if state_finite:
        evaluated |= QualityReason.BOTTOM_CLEARANCE
        if (
            np.min(depth + coarse_eta) <= 0.0
            or np.min(depth + fine_eta) <= 0.0
        ):
            failed |= QualityReason.BOTTOM_CLEARANCE

    existence_failures = failed & ~QualityReason.TEMPORAL_DEFECT
    if existence_failures:
        failed |= QualityReason.TEMPORAL_DEFECT
        decision = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=QualityReason.TEMPORAL_DEFECT,
            evaluated=evaluated,
            failed=failed,
        )
        return _infinite_refinement_metrics(), decision

    if coarse_eta.shape != fine_eta.shape:
        raise ValueError("coarse and fine fields must have identical shapes")
    if not np.array_equal(coarse_times, fine_times):
        raise ValueError("coarse and fine saved times must be identical")

    scale = np.sqrt(gravity * depth)
    coarse_fields = (
        _project_to_wavenumber_band(
            coarse_eta / depth,
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
        _project_to_wavenumber_band(
            coarse_xi / (depth * scale),
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
        _project_to_wavenumber_band(
            coarse_gxi / scale,
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
    )
    fine_fields = (
        _project_to_wavenumber_band(
            fine_eta / depth,
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
        _project_to_wavenumber_band(
            fine_xi / (depth * scale),
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
        _project_to_wavenumber_band(
            fine_gxi / scale,
            length=length,
            maximum_wavenumber=maximum_wavenumber,
        ),
    )
    errors = tuple(
        _relative_trajectory_error(
            coarse_field,
            fine_field,
            relative_floor=relative_floor,
        )
        for coarse_field, fine_field in zip(
            coarse_fields, fine_fields
        )
    )
    maximum_error = max(errors)
    if maximum_error > tolerance:
        failed |= QualityReason.TEMPORAL_DEFECT

    metrics = TemporalRefinementMetrics(
        eta_error=errors[0],
        xi_error=errors[1],
        gxi_error=errors[2],
        maximum_error=maximum_error,
    )
    decision = QualityDecision(
        scope=QualityScope.TRAJECTORY,
        required=QualityReason.TEMPORAL_DEFECT,
        evaluated=evaluated,
        failed=failed,
    )
    return metrics, decision
