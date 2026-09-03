"""Numerical settings shared by trajectory integration and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias


TrajectoryFamily: TypeAlias = Literal[
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]
JONSWAP_ADJUSTMENT_SCHEMA = "dommermuth_nonlinear_adjustment_v1"


@dataclass(frozen=True)
class RolloutConfig:
    """Numerical settings for one batch of trajectories."""

    nx: int = 1024
    target_nx: int = 1024
    length: float = 2.0 * math.pi
    gravity: float = 1.0
    dno_order: int = 6
    target_dno_order: int = 6
    pad_factor: int = 8
    maximum_wavenumber: float = 256.0
    target_maximum_wavenumber: float = 128.0
    dt: float = 0.01
    saved_dt: float = 0.08
    gl2_residual_tolerance: float = 1.0e-8
    gl2_iteration_cap: int = 4
    target_time_chunk_size: int = 8
    internal_hamiltonian_drift_threshold: float | None = None

    def __post_init__(self) -> None:
        if any(size <= 0 or size % 2 for size in (self.nx, self.target_nx)):
            raise ValueError("nx and target_nx must be positive even integers")
        if min(self.dno_order, self.target_dno_order) < 0:
            raise ValueError("DNO orders must be nonnegative")
        if self.pad_factor < 1 or self.target_time_chunk_size < 1:
            raise ValueError("pad_factor and target_time_chunk_size must be positive")
        if self.gl2_iteration_cap < 0:
            raise ValueError("gl2_iteration_cap must be nonnegative")
        positive_values = (
            self.length,
            self.gravity,
            self.maximum_wavenumber,
            self.target_maximum_wavenumber,
            self.dt,
            self.saved_dt,
            self.gl2_residual_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError(
                "rollout scales and tolerances must be finite and positive"
            )
        nyquist = math.pi * self.nx / self.length
        if self.maximum_wavenumber >= nyquist:
            raise ValueError("maximum_wavenumber must lie strictly below Nyquist")
        if self.target_maximum_wavenumber > self.maximum_wavenumber:
            raise ValueError(
                "target_maximum_wavenumber must be positive and no larger "
                "than maximum_wavenumber"
            )
        target_nyquist = math.pi * self.target_nx / self.length
        if self.target_maximum_wavenumber >= target_nyquist:
            raise ValueError(
                "target maximum_wavenumber must lie strictly below the "
                "target-grid Nyquist"
            )
        self.substeps_per_saved_frame
        if self.internal_hamiltonian_drift_threshold is not None and (
            not math.isfinite(self.internal_hamiltonian_drift_threshold)
            or self.internal_hamiltonian_drift_threshold < 0.0
        ):
            raise ValueError(
                "internal_hamiltonian_drift_threshold must be finite and nonnegative"
            )

    @property
    def filter_fraction(self) -> float:
        """Fraction of the internal Nyquist band retained by the solver."""

        return self.maximum_wavenumber / (math.pi * self.nx / self.length)

    @property
    def substeps_per_saved_frame(self) -> int:
        """Number of solver steps between stored trajectory frames."""

        substeps = int(round(self.saved_dt / self.dt))
        if not math.isclose(
            substeps * self.dt,
            self.saved_dt,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("saved_dt must be an integer multiple of dt")
        return substeps


@dataclass(frozen=True)
class TrajectoryFrameSelectionConfig:
    """Number and weighting parameters for retained trajectory frames."""

    tanaka_count: int = 200
    tanaka_alpha: float = 0.5
    tanaka_sigma_steps: float = 50.0
    benjamin_feir_count: int = 200
    jonswap_tma_count: int = 16


@dataclass(frozen=True)
class JonswapNonlinearAdjustmentConfig:
    """Ramp and warm-up lengths for JONSWAP/TMA initial states."""

    ramp_order: int
    ramp_time_peak_periods: int
    burn_peak_periods: int

    def __post_init__(self) -> None:
        values = (
            self.ramp_order,
            self.ramp_time_peak_periods,
            self.burn_peak_periods,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("JONSWAP nonlinear-adjustment values must be positive")
        if self.burn_peak_periods < self.ramp_time_peak_periods:
            raise ValueError("JONSWAP adjustment burn must cover the ramp time")


@dataclass(frozen=True)
class TrajectoryExecutionConfig:
    """Numerical, duration, and row-selection settings for one trajectory family."""

    family: TrajectoryFamily
    numerical: RolloutConfig
    frame_selection: TrajectoryFrameSelectionConfig
    fixed_terminal_time: float | None = None
    period_count: int | None = None
    jonswap_quadrature_order: int | None = None
    jonswap_adjustment: JonswapNonlinearAdjustmentConfig | None = None

    def __post_init__(self) -> None:
        if self.family not in ("tanaka", "benjamin_feir", "jonswap_tma"):
            raise ValueError(f"unknown trajectory family: {self.family}")
        if self.family == "tanaka":
            if (
                self.fixed_terminal_time is None
                or not math.isfinite(self.fixed_terminal_time)
                or self.fixed_terminal_time <= 0.0
            ):
                raise ValueError("Tanaka terminal time must be finite and positive")
            if self.period_count is not None:
                raise ValueError("Tanaka does not use a period-count duration")
        else:
            if self.fixed_terminal_time is not None:
                raise ValueError(f"{self.family} does not use a fixed terminal time")
            if (
                isinstance(self.period_count, bool)
                or not isinstance(self.period_count, int)
                or self.period_count <= 0
            ):
                raise ValueError(f"{self.family} period_count must be positive")

        if self.family == "jonswap_tma":
            if (
                isinstance(self.jonswap_quadrature_order, bool)
                or not isinstance(self.jonswap_quadrature_order, int)
                or self.jonswap_quadrature_order < 2
            ):
                raise ValueError("JONSWAP/TMA quadrature_order must be at least two")
        elif self.jonswap_quadrature_order is not None:
            raise ValueError("only JONSWAP/TMA may set a quadrature order")
        elif self.jonswap_adjustment is not None:
            raise ValueError("only JONSWAP/TMA may set nonlinear adjustment settings")


PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG = RolloutConfig(
    dno_order=4,
    internal_hamiltonian_drift_threshold=1.0e-3,
)
PAPER_JONSWAP_ROLLOUT_CONFIG = RolloutConfig(
    nx=2048,
    dno_order=4,
    maximum_wavenumber=704.0,
    gl2_iteration_cap=5,
    internal_hamiltonian_drift_threshold=1.0e-3,
)
PAPER_TANAKA_ROLLOUT_CONFIG = RolloutConfig()

PAPER_JONSWAP_ADJUSTMENT_CONFIG = JonswapNonlinearAdjustmentConfig(
    ramp_order=4,
    ramp_time_peak_periods=10,
    burn_peak_periods=20,
)


def paper_trajectory_execution(
    family: TrajectoryFamily,
) -> TrajectoryExecutionConfig:
    """Return the paper-dataset settings for one trajectory family."""

    if family == "tanaka":
        return TrajectoryExecutionConfig(
            family=family,
            numerical=PAPER_TANAKA_ROLLOUT_CONFIG,
            frame_selection=TrajectoryFrameSelectionConfig(),
            fixed_terminal_time=200.0,
        )
    if family == "benjamin_feir":
        return TrajectoryExecutionConfig(
            family=family,
            numerical=PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG,
            frame_selection=TrajectoryFrameSelectionConfig(),
            period_count=100,
        )
    return TrajectoryExecutionConfig(
        family=family,
        numerical=PAPER_JONSWAP_ROLLOUT_CONFIG,
        frame_selection=TrajectoryFrameSelectionConfig(),
        period_count=16,
        jonswap_quadrature_order=16,
        jonswap_adjustment=PAPER_JONSWAP_ADJUSTMENT_CONFIG,
    )
