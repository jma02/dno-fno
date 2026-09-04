"""Choose the saved frames retained from accepted trajectories."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]


def floor_saved_time_grid(
    terminal_time: float,
    *,
    saved_dt: float,
    horizon_name: str,
) -> FloatArray:
    """Return the saved-time prefix ending immediately before a horizon."""

    step_count = math.floor(terminal_time / saved_dt)
    while step_count * saved_dt > terminal_time:
        step_count -= 1
    while (step_count + 1) * saved_dt <= terminal_time:
        step_count += 1
    if step_count < 1:
        raise ValueError(f"{horizon_name} horizon is shorter than one saved step")
    saved_times = saved_dt * np.arange(step_count + 1, dtype=np.float64)
    realized = float(saved_times[-1])
    if not (realized <= terminal_time and terminal_time - realized < saved_dt):
        raise RuntimeError(
            f"{horizon_name} saved-grid horizon was not strictly floored"
        )
    return saved_times


def select_tanaka_times(eta: FloatArray, *, length: float) -> IntArray:
    """Select 200 frames, concentrating half the density on rapid evolution."""

    surface = np.asarray(eta, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 2:
        raise ValueError("eta must have nonempty shape (time, space)")
    if surface.shape[0] < 200:
        raise ValueError("Tanaka trajectories must contain at least 200 frames")
    if not np.isfinite(surface).all():
        raise ValueError("eta must contain only finite values")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")

    nx = surface.shape[-1]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    derivative = np.fft.ifft(
        1j * wavenumbers * np.fft.fft(surface, axis=-1),
        axis=-1,
    ).real
    energy = np.sum(derivative**2, axis=-1)
    activity = np.abs(np.gradient(energy)) / (energy + 1.0e-12)

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
    quantiles = (np.arange(keep_samples, dtype=np.float64) + 0.5) / keep_samples
    raw = np.searchsorted(np.cumsum(density), quantiles, side="left")
    raw[0] = 0
    raw[-1] = surface.shape[0] - 1
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
    """Select the nearest dense-grid indices to an endpoint-uniform grid."""

    if not 2 <= keep_samples <= number_of_times:
        raise ValueError("keep_samples must lie between two and the time count")
    numerator = np.arange(keep_samples, dtype=np.int64) * (number_of_times - 1)
    return np.floor(numerator / (keep_samples - 1) + 0.5).astype(np.int32)
