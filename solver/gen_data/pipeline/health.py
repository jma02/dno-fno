"""Shared numerical-health checks for stored reference trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)


@dataclass(frozen=True)
class StoredTrajectoryThresholds:
    """Diagnostic thresholds for checks reconstructable from stored frames."""

    hamiltonian_drift: float = 1e-3
    minimum_water_column_fraction: float = 0.0
    relative_hamiltonian_floor: float = 1e-14

    def __post_init__(self) -> None:
        if self.hamiltonian_drift < 0.0:
            raise ValueError("hamiltonian_drift must be nonnegative")
        if not 0.0 <= self.minimum_water_column_fraction < 1.0:
            raise ValueError(
                "minimum_water_column_fraction must lie in [0, 1)"
            )
        if self.relative_hamiltonian_floor <= 0.0:
            raise ValueError("relative_hamiltonian_floor must be positive")


@dataclass(frozen=True)
class StoredTrajectoryMetrics:
    """Scalar diagnostics reconstructed from one stored trajectory."""

    minimum_water_column_fraction: float | None
    maximum_hamiltonian_drift: float | None


@dataclass(frozen=True)
class StoredFrameStatistics:
    """Sufficient statistics for checks reconstructable from stored frames."""

    state_finite: bool
    target_finite: bool
    minimum_water_column_fraction: float | None
    initial_hamiltonian: float | None
    minimum_hamiltonian: float | None
    maximum_hamiltonian: float | None
    hamiltonian_scale: float | None


STORED_SAMPLE_REQUIRED = (
    QualityReason.NONFINITE_STATE
    | QualityReason.NONFINITE_TARGET
    | QualityReason.BOTTOM_CLEARANCE
)


STORED_TRAJECTORY_REQUIRED = (
    STORED_SAMPLE_REQUIRED
)


def evaluate_stored_frame_statistics(
    statistics: StoredFrameStatistics,
    *,
    scope: QualityScope,
    thresholds: StoredTrajectoryThresholds = StoredTrajectoryThresholds(),
) -> tuple[StoredTrajectoryMetrics, QualityDecision]:
    """Turn streamed sufficient statistics into one quality decision.

    Static samples use ``QualityScope.SAMPLE`` and do not make a Hamiltonian-
    drift claim. Rollout trajectories use ``QualityScope.TRAJECTORY``. Other
    scopes are invalid because these checks are reconstructed from stored rows.
    Hamiltonian drift is evaluated for trajectories but is diagnostic rather
    than required. Final production acceptance additionally requires the
    integrator to have produced a complete admissible trajectory, including
    its GL2-stage diagnostics; that fact cannot be reconstructed from retained
    frames alone.
    """

    if scope not in (QualityScope.SAMPLE, QualityScope.TRAJECTORY):
        raise ValueError("stored-frame checks require sample or trajectory scope")

    required = (
        STORED_SAMPLE_REQUIRED
        if scope is QualityScope.SAMPLE
        else STORED_TRAJECTORY_REQUIRED
    )
    evaluated = QualityReason.NONFINITE_STATE | QualityReason.NONFINITE_TARGET
    failed = QualityReason.NONE
    if not statistics.state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not statistics.target_finite:
        failed |= QualityReason.NONFINITE_TARGET

    minimum_water_column_fraction: float | None = None
    if statistics.state_finite:
        evaluated |= QualityReason.BOTTOM_CLEARANCE
        minimum_water_column_fraction = statistics.minimum_water_column_fraction
        if (
            minimum_water_column_fraction is None
            or not np.isfinite(minimum_water_column_fraction)
            or minimum_water_column_fraction
            <= thresholds.minimum_water_column_fraction
        ):
            failed |= QualityReason.BOTTOM_CLEARANCE

    maximum_hamiltonian_drift: float | None = None
    if (
        scope is QualityScope.TRAJECTORY
        and statistics.state_finite
        and statistics.target_finite
    ):
        evaluated |= QualityReason.HAMILTONIAN_DRIFT
        hamiltonian_values = (
            statistics.initial_hamiltonian,
            statistics.minimum_hamiltonian,
            statistics.maximum_hamiltonian,
            statistics.hamiltonian_scale,
        )
        if any(
            value is None or not np.isfinite(value)
            for value in hamiltonian_values
        ):
            failed |= QualityReason.HAMILTONIAN_DRIFT
        else:
            initial, minimum, maximum, scale = (
                float(value) for value in hamiltonian_values if value is not None
            )
            denominator = (
                abs(initial) + thresholds.relative_hamiltonian_floor * scale
            )
            maximum_hamiltonian_drift = max(
                abs(minimum - initial), abs(maximum - initial)
            ) / denominator
            if (
                not np.isfinite(maximum_hamiltonian_drift)
                or maximum_hamiltonian_drift > thresholds.hamiltonian_drift
            ):
                failed |= QualityReason.HAMILTONIAN_DRIFT

    metrics = StoredTrajectoryMetrics(
        minimum_water_column_fraction=minimum_water_column_fraction,
        maximum_hamiltonian_drift=maximum_hamiltonian_drift,
    )
    decision = QualityDecision(
        scope=scope,
        required=required,
        evaluated=evaluated,
        failed=failed,
    )
    return metrics, decision


def evaluate_stored_trajectory(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    *,
    depth: float,
    gravity: float,
    dx: float,
    thresholds: StoredTrajectoryThresholds = StoredTrajectoryThresholds(),
) -> tuple[StoredTrajectoryMetrics, QualityDecision]:
    """Evaluate checks available on a trajectory's retained time grid.

    This does not test integration steps between supplied frames. GL2-stage
    diagnostics and full-horizon completeness therefore remain unevaluated
    here.
    """

    eta_array = np.asarray(eta, dtype=np.float64)
    xi_array = np.asarray(xi, dtype=np.float64)
    gxi_array = np.asarray(gxi, dtype=np.float64)
    if eta_array.ndim != 2:
        raise ValueError(f"eta must have shape (time, space), got {eta_array.shape}")
    if xi_array.shape != eta_array.shape or gxi_array.shape != eta_array.shape:
        raise ValueError("eta, xi, and gxi must have identical shapes")
    if eta_array.shape[0] == 0:
        raise ValueError("a trajectory must contain at least one frame")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and positive")

    state_finite = bool(
        np.isfinite(eta_array).all() and np.isfinite(xi_array).all()
    )
    target_finite = bool(np.isfinite(gxi_array).all())

    minimum_water_column_fraction: float | None = None
    valid_depth = bool(np.isfinite(depth) and depth > 0.0)
    if state_finite:
        if valid_depth:
            minimum_water_column_fraction = float(
                (depth + np.min(eta_array)) / depth
            )

    initial_hamiltonian: float | None = None
    minimum_hamiltonian: float | None = None
    maximum_hamiltonian: float | None = None
    hamiltonian_scale: float | None = None
    if state_finite and target_finite:
        hamiltonian = 0.5 * dx * np.sum(
            xi_array * gxi_array + gravity * eta_array**2,
            axis=1,
        )
        initial_hamiltonian = float(hamiltonian[0])
        minimum_hamiltonian = float(np.min(hamiltonian))
        maximum_hamiltonian = float(np.max(hamiltonian))
        if valid_depth:
            length = dx * eta_array.shape[1]
            hamiltonian_scale = float(gravity * depth**2 * length)

    return evaluate_stored_frame_statistics(
        StoredFrameStatistics(
            state_finite=state_finite,
            target_finite=target_finite,
            minimum_water_column_fraction=minimum_water_column_fraction,
            initial_hamiltonian=initial_hamiltonian,
            minimum_hamiltonian=minimum_hamiltonian,
            maximum_hamiltonian=maximum_hamiltonian,
            hamiltonian_scale=hamiltonian_scale,
        ),
        scope=QualityScope.TRAJECTORY,
        thresholds=thresholds,
    )
