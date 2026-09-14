"""Build output times and keep selected trajectory frames as dataset rows."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_config import TrajectoryFamily
from solver.gen_data.pipeline.trajectory_rollout import TrajectorySamples
from solver.gen_data.pipeline.types import SimulationRows

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]


def floor_saved_time_grid(
    terminal_time: float,
    *,
    saved_dt: float,
) -> FloatArray:
    """Return 0, saved_dt, 2*saved_dt, ... without exceeding terminal_time."""

    step_count = math.floor(terminal_time / saved_dt)
    # Correct division rounding so the last time fits and the next one does not.
    while step_count * saved_dt > terminal_time:
        step_count -= 1
    while (step_count + 1) * saved_dt <= terminal_time:
        step_count += 1
    return saved_dt * np.arange(step_count + 1, dtype=np.float64)


def select_tanaka_times(eta: FloatArray, *, length: float) -> IntArray:
    """Return 200 distinct frame indices, including the first and last.

    Half the selection weight is spread equally across frames; half follows
    smoothed relative changes in the sum of squared surface slopes.
    """

    surface = np.asarray(eta, dtype=np.float64)

    nx = surface.shape[-1]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    derivative = np.fft.ifft(
        1j * wavenumbers * np.fft.fft(surface, axis=-1),
        axis=-1,
    ).real
    # This is total squared surface slope, not physical wave energy.
    energy = np.sum(derivative**2, axis=-1)
    activity = np.abs(np.gradient(energy)) / (energy + 1.0e-12)

    # Smooth across saved frames: Gaussian standard deviation 50, cutoff 150.
    sigma_steps = 50.0
    radius = math.ceil(3.0 * sigma_steps)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_steps) ** 2)
    kernel /= np.sum(kernel)
    smoothed = np.asarray(
        np.convolve(
            np.pad(activity, (radius, radius), mode="edge"),
            kernel,
            mode="valid",
        ),
        dtype=np.float64,
    )
    uniform = np.full(surface.shape[0], 1.0 / surface.shape[0], dtype=np.float64)
    total_activity = float(np.sum(smoothed))
    normalized_activity = smoothed / total_activity if total_activity > 0.0 else uniform
    density = 0.5 * uniform + 0.5 * normalized_activity
    density /= np.sum(density)

    keep_samples = 200
    # Space picks by accumulated weight, giving high-weight intervals more picks.
    quantiles = (np.arange(keep_samples, dtype=np.float64) + 0.5) / keep_samples
    raw = np.searchsorted(np.cumsum(density), quantiles, side="left")
    raw[0] = 0
    raw[-1] = surface.shape[0] - 1
    # Reserve room for all 200 indices, then move repeated selections forward.
    lower = np.arange(keep_samples, dtype=np.int64)
    upper = surface.shape[0] - keep_samples + lower
    indices = np.clip(raw, lower, upper)
    for position in range(1, keep_samples):
        indices[position] = max(indices[position], indices[position - 1] + 1)
    return np.asarray(indices, dtype=np.int32)


def select_uniform_times(
    number_of_times: int,
    *,
    keep_samples: int,
) -> IntArray:
    """Round evenly spaced positions from first to last frame; round ties upward."""

    numerator = np.arange(keep_samples, dtype=np.int64) * (number_of_times - 1)
    return np.floor(numerator / (keep_samples - 1) + 0.5).astype(np.int32)


def subsample_trajectories(
    simulations: tuple[TrajectorySamples | None, ...],
    depths: FloatArray,
    *,
    family: TrajectoryFamily,
    length: float,
) -> tuple[SimulationRows | None, ...]:
    """Subsample accepted trajectories into dataset rows."""

    rows_by_simulation: list[SimulationRows | None] = []
    for trajectory, depth in zip(simulations, depths, strict=True):
        rows = None
        if trajectory is not None:
            if family == "tanaka":
                indices = select_tanaka_times(
                    trajectory.eta,
                    length=length,
                )
            else:
                indices = select_uniform_times(
                    trajectory.times.size,
                    keep_samples=200,
                )
            rows = SimulationRows(
                eta=trajectory.eta[indices],
                xi=trajectory.xi[indices],
                gxi=trajectory.gxi[indices],
                depth=float(depth),
                time=trajectory.times[indices],
            )

        rows_by_simulation.append(rows)
    return tuple(rows_by_simulation)
