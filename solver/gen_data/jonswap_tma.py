"""Build resolved-band JONSWAP/TMA random-phase initial conditions."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]

PAPER_PEAK_ENHANCEMENTS = (1.0, 3.3, 5.0)
PAPER_RIGHT_MOVING_FRACTIONS = (0.0, 0.5, 1.0)
PAPER_SHALLOW_PEAK_MODES = (16, 18, 20, 22, 24)
PAPER_PEAK_STEEPNESS_MAXIMUM = 0.08
PAPER_RELATIVE_FREQUENCY_MINIMUM = 0.5
PAPER_RELATIVE_FREQUENCY_MAXIMUM = 2.5
PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER = 128.0
PAPER_RESOLVED_BAND_QUADRATURE_ORDER = 16
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER
)


JonswapTmaParameters = NamedTuple(
    "JonswapTmaParameters",
    [
        ("depth", float),
        ("significant_height", float),
        ("peak_wavenumber", float),
        ("peak_enhancement", float),
        ("right_moving_fraction", float),
    ],
)

ResolvedBand = NamedTuple(
    "ResolvedBand",
    [
        ("length", float),
        ("maximum_wavenumber", float),
    ],
)

JonswapTmaSpectrum = NamedTuple(
    "JonswapTmaSpectrum",
    [
        ("wavenumbers", FloatArray),
        ("energy_fractions", FloatArray),
    ],
)

JonswapTmaState = NamedTuple(
    "JonswapTmaState",
    [
        ("eta", FloatArray),
        ("xi", FloatArray),
    ],
)


def finite_depth_angular_frequency(
    wavenumber: FloatArray,
    *,
    depth: float,
    gravity: float,
) -> FloatArray:
    """Return ``sqrt(g k tanh(k h))`` for positive wavenumbers ``k``."""

    values = np.asarray(wavenumber, dtype=np.float64)
    return np.sqrt(gravity * values * np.tanh(values * depth))


def tma_depth_factor(depth_wavenumber: FloatArray) -> FloatArray:
    """Return ``tanh(z)^2 / (1 + 2 z / sinh(2 z))`` for ``z = k h``."""

    z = np.asarray(depth_wavenumber, dtype=np.float64)
    correction = np.ones_like(z)
    regular = (np.abs(z) >= 1.0e-6) & (np.abs(z) < 350.0)
    correction[regular] = 2.0 * z[regular] / np.sinh(2.0 * z[regular])
    small = np.abs(z) < 1.0e-6
    correction[small] = 1.0 - 2.0 * z[small] ** 2 / 3.0
    correction[np.abs(z) >= 350.0] = 0.0
    return np.tanh(z) ** 2 / (1.0 + correction)


def finite_depth_group_velocity(
    wavenumber: FloatArray,
    *,
    depth: float,
    gravity: float,
) -> FloatArray:
    """Return the derivative of angular frequency with respect to wavenumber."""

    values = np.asarray(wavenumber, dtype=np.float64)
    z = values * depth
    angular_frequency = finite_depth_angular_frequency(
        values, depth=depth, gravity=gravity
    )
    sech_squared = np.zeros_like(z)
    moderate = z < 350.0
    sech_squared[moderate] = 1.0 / np.cosh(z[moderate]) ** 2
    return gravity * (np.tanh(z) + z * sech_squared) / (2.0 * angular_frequency)


def positive_mode_wavenumbers(*, band: ResolvedBand) -> FloatArray:
    """Return the positive periodic wavenumbers in the resolved band."""

    spacing = 2.0 * np.pi / band.length
    largest_mode = int(np.floor(band.maximum_wavenumber / spacing + 1.0e-12))
    modes = np.arange(1, largest_mode + 1, dtype=np.float64)
    wavenumbers = spacing * modes
    return wavenumbers[wavenumbers <= band.maximum_wavenumber + 1.0e-12]


def jonswap_tma_spectrum(
    parameters: JonswapTmaParameters,
    *,
    band: ResolvedBand,
    gravity: float = 1.0,
) -> JonswapTmaSpectrum:
    """Integrate the paper's sharp relative-frequency density over Fourier cells."""

    centers = positive_mode_wavenumbers(band=band)
    spacing = 2.0 * np.pi / band.length
    lower = np.maximum(0.0, centers - 0.5 * spacing)
    upper = np.minimum(band.maximum_wavenumber, centers + 0.5 * spacing)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    cell_wavenumbers = (
        midpoint[:, None] + half_width[:, None] * _QUADRATURE_NODES[None, :]
    )

    depth = parameters.depth
    angular_frequency = finite_depth_angular_frequency(
        cell_wavenumbers, depth=depth, gravity=gravity
    )
    peak_frequency = float(
        finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber]),
            depth=depth,
            gravity=gravity,
        )[0]
    )
    sigma = np.where(angular_frequency <= peak_frequency, 0.07, 0.09)
    peak_shape = np.exp(
        -((angular_frequency - peak_frequency) ** 2)
        / (2.0 * sigma**2 * peak_frequency**2)
    )
    jonswap_density = (
        gravity**2
        * angular_frequency ** (-5)
        * np.exp(-1.25 * (peak_frequency / angular_frequency) ** 4)
        * parameters.peak_enhancement**peak_shape
    )
    relative_frequency = angular_frequency / peak_frequency
    sharp_band = np.asarray(
        (relative_frequency >= PAPER_RELATIVE_FREQUENCY_MINIMUM)
        & (relative_frequency <= PAPER_RELATIVE_FREQUENCY_MAXIMUM),
        dtype=np.float64,
    )
    density = (
        jonswap_density
        * tma_depth_factor(cell_wavenumbers * depth)
        * finite_depth_group_velocity(
            cell_wavenumbers,
            depth=depth,
            gravity=gravity,
        )
        * sharp_band
    )
    cell_energy = half_width * np.sum(
        _QUADRATURE_WEIGHTS[None, :] * density, axis=1
    )
    total_energy = float(np.sum(cell_energy))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("resolved JONSWAP/TMA spectrum has no finite energy")
    return JonswapTmaSpectrum(
        np.asarray(centers, dtype=np.float64),
        np.asarray(cell_energy / total_energy, dtype=np.float64),
    )


def build_jonswap_tma_initial_condition(
    x: FloatArray,
    *,
    parameters: JonswapTmaParameters,
    phase_right: FloatArray,
    phase_left: FloatArray,
    band: ResolvedBand,
    gravity: float = 1.0,
) -> JonswapTmaState:
    """Construct one resolved-band random-phase ``(eta, xi)`` state."""

    coordinates = np.asarray(x, dtype=np.float64)
    spectrum = jonswap_tma_spectrum(parameters, band=band, gravity=gravity)
    right_phases = np.asarray(phase_right, dtype=np.float64)
    left_phases = np.asarray(phase_left, dtype=np.float64)
    variance = (parameters.significant_height / 4.0) ** 2
    amplitude_right = np.sqrt(
        2.0 * parameters.right_moving_fraction * variance * spectrum.energy_fractions
    )
    amplitude_left = np.sqrt(
        2.0
        * (1.0 - parameters.right_moving_fraction)
        * variance
        * spectrum.energy_fractions
    )
    right_argument = (
        spectrum.wavenumbers[:, None] * coordinates[None, :] + right_phases[:, None]
    )
    left_argument = (
        spectrum.wavenumbers[:, None] * coordinates[None, :] + left_phases[:, None]
    )
    eta = np.sum(
        amplitude_right[:, None] * np.cos(right_argument)
        + amplitude_left[:, None] * np.cos(left_argument),
        axis=0,
    )

    flat_dno = spectrum.wavenumbers * np.tanh(spectrum.wavenumbers * parameters.depth)
    transfer = np.sqrt(gravity * flat_dno) / flat_dno
    xi = np.sum(
        transfer[:, None]
        * (
            amplitude_right[:, None] * np.sin(right_argument)
            - amplitude_left[:, None] * np.sin(left_argument)
        ),
        axis=0,
    )
    return JonswapTmaState(
        np.asarray(eta, dtype=np.float64),
        np.asarray(xi, dtype=np.float64),
    )
