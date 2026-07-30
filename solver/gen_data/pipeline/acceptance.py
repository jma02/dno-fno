"""Paper-corpus support, trajectory, and refinement decisions."""
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
class StokesSupportMetrics:
    """Quantities in the conservative finite-Stokes Ursell condition."""

    wave_height_upper_bound: float
    wavelength: float
    depth: float
    ursell_upper_bound: float
    ursell_limit: float


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


def evaluate_finite_stokes_support(
    *,
    wave_height_upper_bound: float,
    wavelength: float,
    depth: float,
    ursell_limit: float = 26.0,
) -> tuple[StokesSupportMetrics, QualityDecision]:
    """Evaluate the sufficient condition ``H_+ lambda^2/h^3 <= 26``."""

    values = np.asarray(
        [wave_height_upper_bound, wavelength, depth, ursell_limit],
        dtype=np.float64,
    )
    if wave_height_upper_bound < 0.0:
        raise ValueError("wave_height_upper_bound must be nonnegative")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if ursell_limit <= 0.0:
        raise ValueError("ursell_limit must be positive")

    ursell_upper_bound = float(
        wave_height_upper_bound * wavelength**2 / depth**3
    )
    in_support = bool(
        np.isfinite(values).all() and ursell_upper_bound <= ursell_limit
    )

    reason = QualityReason.OUTSIDE_SUPPORT
    decision = QualityDecision(
        scope=QualityScope.SAMPLE,
        required=reason,
        evaluated=reason,
        failed=QualityReason.NONE if in_support else reason,
    )
    return (
        StokesSupportMetrics(
            wave_height_upper_bound=float(wave_height_upper_bound),
            wavelength=float(wavelength),
            depth=float(depth),
            ursell_upper_bound=ursell_upper_bound,
            ursell_limit=float(ursell_limit),
        ),
        decision,
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
