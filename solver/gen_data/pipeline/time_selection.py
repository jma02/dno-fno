"""Deterministic post-acceptance time selection for the paper dataset."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]


@dataclass(frozen=True)
class TemporalSelection:
    """Selected dense-time indices and the density that selected them."""

    indices: IntArray
    activity: FloatArray
    smoothed_activity: FloatArray
    density: FloatArray


def _validated_surface(eta: FloatArray, length: float) -> FloatArray:
    surface = np.asarray(eta, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 2:
        raise ValueError("eta must have nonempty shape (time, space)")
    if not np.isfinite(surface).all():
        raise ValueError("eta must contain only finite values")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    return surface


def _gaussian_smooth(activity: FloatArray, sigma_steps: float) -> FloatArray:
    if not math.isfinite(sigma_steps) or sigma_steps < 0.0:
        raise ValueError("sigma_steps must be finite and nonnegative")
    values = np.asarray(activity, dtype=np.float64)
    if sigma_steps == 0.0:
        return values.copy()
    radius = int(max(1, math.ceil(3.0 * sigma_steps)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_steps) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def _selection_density(
    activity: FloatArray,
    *,
    alpha: float,
    sigma_steps: float,
) -> tuple[FloatArray, FloatArray]:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("activity must contain at least two times")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("activity must be finite and nonnegative")

    smoothed = _gaussian_smooth(values, sigma_steps)
    uniform = np.full(values.size, 1.0 / values.size, dtype=np.float64)
    total = float(np.sum(smoothed))
    normalized_activity = smoothed / total if total > 0.0 else uniform
    density = (1.0 - alpha) * uniform + alpha * normalized_activity
    density /= np.sum(density)
    return smoothed, density


def _midpoint_quantile_indices(
    density: FloatArray,
    *,
    keep_samples: int,
) -> IntArray:
    """Return endpoint-pinned, strictly increasing midpoint quantiles."""

    probabilities = np.asarray(density, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("density must contain at least two times")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("density must be a finite probability vector")
    if not 2 <= keep_samples <= probabilities.size:
        raise ValueError("keep_samples must lie between two and the time count")
    if keep_samples == probabilities.size:
        return np.arange(probabilities.size, dtype=np.int32)

    quantiles = (np.arange(keep_samples, dtype=np.float64) + 0.5) / keep_samples
    raw = np.searchsorted(np.cumsum(probabilities), quantiles, side="left")
    raw[0] = 0
    raw[-1] = probabilities.size - 1

    # Coordinate bounds reserve one unused integer for every later quantile.
    lower = np.arange(keep_samples, dtype=np.int64)
    upper = probabilities.size - keep_samples + lower
    indices = np.clip(raw, lower, upper)
    for position in range(1, keep_samples):
        indices[position] = max(indices[position], indices[position - 1] + 1)
    return np.asarray(indices, dtype=np.int32)


def _selection_from_activity(
    activity: FloatArray,
    *,
    keep_samples: int,
    alpha: float,
    sigma_steps: float,
) -> TemporalSelection:
    smoothed, density = _selection_density(
        activity,
        alpha=alpha,
        sigma_steps=sigma_steps,
    )
    indices = _midpoint_quantile_indices(
        density,
        keep_samples=keep_samples,
    )
    return TemporalSelection(
        indices=indices,
        activity=np.asarray(activity, dtype=np.float64),
        smoothed_activity=smoothed,
        density=density,
    )


def select_tanaka_times(
    eta: FloatArray,
    *,
    length: float,
    keep_samples: int = 200,
    alpha: float = 0.5,
    sigma_steps: float = 50.0,
    relative_floor: float = 1.0e-12,
) -> TemporalSelection:
    """Select times from relative changes in surface-gradient energy.

    For dense-time index ``j``, the activity is

    ``A_j = |D S_j| / (S_j + relative_floor)``,

    where ``S_j = sum_x |partial_x eta_j|^2`` and ``D`` is the centered
    index difference with one-sided endpoint differences.
    """

    surface = _validated_surface(eta, length)
    if not math.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    nx = surface.shape[-1]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    derivative = np.fft.ifft(
        1j * wavenumbers * np.fft.fft(surface, axis=-1),
        axis=-1,
    ).real
    energy = np.sum(derivative**2, axis=-1)
    activity = np.abs(np.gradient(energy)) / (energy + relative_floor)
    return _selection_from_activity(
        activity,
        keep_samples=keep_samples,
        alpha=alpha,
        sigma_steps=sigma_steps,
    )


def select_uniform_times(
    number_of_times: int,
    *,
    keep_samples: int = 16,
) -> IntArray:
    """Select the nearest dense-grid indices to an endpoint-uniform grid."""

    if number_of_times < 2:
        raise ValueError("number_of_times must be at least two")
    if not 2 <= keep_samples <= number_of_times:
        raise ValueError("keep_samples must lie between two and the time count")
    numerator = np.arange(keep_samples, dtype=np.int64) * (number_of_times - 1)
    indices = np.floor(numerator / (keep_samples - 1) + 0.5).astype(np.int32)
    if np.any(np.diff(indices) <= 0):
        raise RuntimeError("uniform time selection did not produce unique indices")
    return indices
