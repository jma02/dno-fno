"""Resolved-band JONSWAP/TMA random-phase initial conditions.

The constructor in this module is independent of the historical Gaussian
random-sea generator.  It uses a JONSWAP frequency density, the TMA
finite-depth multiplier, and the finite-depth linear traveling-wave relation
between surface elevation and surface potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
RandomSeaStratum: TypeAlias = Literal["shallow", "finite", "deep"]

PAPER_PEAK_ENHANCEMENTS = (1.0, 3.3, 5.0)
PAPER_RIGHT_MOVING_FRACTIONS = (0.0, 0.5, 1.0)
PAPER_SHALLOW_PEAK_MODES = (16, 18, 20, 22, 24)
PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER = 128.0
PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER = 96.0
PAPER_RESOLVED_BAND_TRANSITION_FRACTION = (
    PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER
    / PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER
)
PAPER_RESOLVED_BAND_QUADRATURE_ORDER = 16
PAPER_RESOLVED_BAND_WINDOW = "cosine_squared_density_v1"


@dataclass(frozen=True)
class JonswapTmaParameters:
    """The five physical/statistical parameters defining one spectrum.

    ``significant_height`` is the ensemble significant wave height ``H_s``.
    ``peak_wavenumber`` is the wavenumber ``k_p`` whose finite-depth
    frequency is used in the JONSWAP formula. ``peak_enhancement`` is
    JONSWAP's ``gamma``. ``right_moving_fraction`` is the fraction ``r_d`` of
    the phase-independent linear wave energy assigned to right-moving modes.
    """

    depth: float
    significant_height: float
    peak_wavenumber: float
    peak_enhancement: float
    right_moving_fraction: float


@dataclass(frozen=True)
class ResolvedBand:
    """Periodic domain and smooth spectral truncation used by the constructor."""

    length: float = 2.0 * np.pi
    maximum_wavenumber: float = PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER
    transition_wavenumber: float = PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER
    quadrature_order: int = PAPER_RESOLVED_BAND_QUADRATURE_ORDER

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if (
            not np.isfinite(self.transition_wavenumber)
            or not np.isfinite(self.maximum_wavenumber)
            or not 0.0 < self.transition_wavenumber < self.maximum_wavenumber
        ):
            raise ValueError(
                "wavenumbers must satisfy 0 < transition_wavenumber "
                "< maximum_wavenumber"
            )
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")


@dataclass(frozen=True)
class JonswapTmaSpectrum:
    """Discrete positive-mode representation of a windowed spectrum."""

    wavenumbers: FloatArray
    energy_fractions: FloatArray


@dataclass(frozen=True)
class JonswapTmaState:
    """One random-phase Zakharov state and its reproducible modal data."""

    eta: FloatArray
    xi: FloatArray
    spectrum: JonswapTmaSpectrum
    amplitude_right: FloatArray
    amplitude_left: FloatArray
    phase_right: FloatArray
    phase_left: FloatArray


def finite_depth_angular_frequency(
    wavenumber: FloatArray,
    *,
    depth: float,
    gravity: float,
) -> FloatArray:
    """Return ``sqrt(g k tanh(k h))`` for positive wavenumbers ``k``."""

    values = np.asarray(wavenumber, dtype=np.float64)
    return np.sqrt(gravity * values * np.tanh(values * depth))


def _depth_correction(depth_wavenumber: FloatArray) -> FloatArray:
    """Return ``2 z / sinh(2 z)`` without overflow at large ``z``."""

    z = np.asarray(depth_wavenumber, dtype=np.float64)
    correction = np.ones_like(z)
    regular = (np.abs(z) >= 1.0e-6) & (np.abs(z) < 350.0)
    correction[regular] = 2.0 * z[regular] / np.sinh(2.0 * z[regular])
    small = np.abs(z) < 1.0e-6
    correction[small] = 1.0 - 2.0 * z[small] ** 2 / 3.0
    correction[np.abs(z) >= 350.0] = 0.0
    return correction


def tma_depth_factor(depth_wavenumber: FloatArray) -> FloatArray:
    """Return the standard TMA multiplier as a function of ``z = k h``.

    The multiplier is

    ``tanh(z)^2 / (1 + 2 z / sinh(2 z))``.
    """

    z = np.asarray(depth_wavenumber, dtype=np.float64)
    return np.tanh(z) ** 2 / (1.0 + _depth_correction(z))


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


def resolved_band_window(
    wavenumber: FloatArray,
    *,
    band: ResolvedBand,
) -> FloatArray:
    """Return the common cosine-squared density window.

    The window is one through ``band.transition_wavenumber``, decreases
    smoothly to zero, and is zero from ``band.maximum_wavenumber`` onward.
    """

    values = np.asarray(wavenumber, dtype=np.float64)
    transition = band.transition_wavenumber
    maximum = band.maximum_wavenumber
    interior = values <= transition
    taper = (values > transition) & (values < maximum)
    result = np.zeros_like(values)
    result[interior] = 1.0
    result[taper] = (
        np.cos(0.5 * np.pi * (values[taper] - transition) / (maximum - transition)) ** 2
    )
    return result


def positive_mode_wavenumbers(*, band: ResolvedBand) -> FloatArray:
    """Return all positive periodic wavenumbers not exceeding the band limit."""

    spacing = 2.0 * np.pi / band.length
    largest_mode = int(np.floor(band.maximum_wavenumber / spacing + 1.0e-12))
    modes = np.arange(1, largest_mode + 1, dtype=np.float64)
    wavenumbers = spacing * modes
    return wavenumbers[wavenumbers <= band.maximum_wavenumber + 1.0e-12]


def paper_support_violations(
    parameters: JonswapTmaParameters,
    *,
    stratum: RandomSeaStratum,
    length: float = 2.0 * np.pi,
) -> tuple[str, ...]:
    """Return violations of the predeclared paper-corpus parameter support.

    This predicate depends only on the five parameters and the named stratum.
    It does not inspect random phases or a realized surface profile.
    """

    h = parameters.depth
    significant_height = parameters.significant_height
    peak_wavenumber = parameters.peak_wavenumber
    peak_enhancement = parameters.peak_enhancement
    right_fraction = parameters.right_moving_fraction
    violations: list[str] = []

    if not np.isfinite(
        (h, significant_height, peak_wavenumber, peak_enhancement, right_fraction)
    ).all():
        return ("all parameters must be finite",)
    if h <= 0.0:
        violations.append("depth must be positive")
    if significant_height <= 0.0:
        violations.append("significant_height must be positive")
    if peak_wavenumber <= 0.0:
        violations.append("peak_wavenumber must be positive")
    if not any(
        np.isclose(peak_enhancement, value, rtol=0.0, atol=1.0e-12)
        for value in PAPER_PEAK_ENHANCEMENTS
    ):
        violations.append("peak_enhancement must be one of {1, 3.3, 5}")
    if not any(
        np.isclose(right_fraction, value, rtol=0.0, atol=1.0e-12)
        for value in PAPER_RIGHT_MOVING_FRACTIONS
    ):
        violations.append("right_moving_fraction must be one of {0, 0.5, 1}")

    if h <= 0.0 or significant_height <= 0.0 or peak_wavenumber <= 0.0:
        return tuple(violations)

    depth_wavenumber = peak_wavenumber * h
    relative_height = significant_height / (2.0 * h)
    peak_steepness = peak_wavenumber * significant_height / 2.0

    if stratum == "shallow":
        peak_mode = peak_wavenumber * length / (2.0 * np.pi)
        if not any(
            np.isclose(peak_mode, value, rtol=0.0, atol=1.0e-12)
            for value in PAPER_SHALLOW_PEAK_MODES
        ):
            violations.append("shallow peak mode must be one of {16, 18, 20, 22, 24}")
        if not 0.2 <= depth_wavenumber <= 1.5:
            violations.append("shallow k_p h must lie in [0.2, 1.5]")
        if not 0.03 <= relative_height <= 0.16:
            violations.append("shallow H_s/(2h) must lie in [0.03, 0.16]")
        if peak_steepness > 0.15:
            violations.append("shallow k_p H_s/2 must not exceed 0.15")
    elif stratum == "finite":
        if not 2.0 <= peak_wavenumber <= 12.0:
            violations.append("finite peak_wavenumber must lie in [2, 12]")
        if not 0.1 <= h <= 1.5:
            violations.append("finite depth must lie in [0.1, 1.5]")
        if not 0.005 <= significant_height <= 0.03:
            violations.append("finite significant_height must lie in [0.005, 0.03]")
    elif stratum == "deep":
        if not 2.0 <= peak_wavenumber <= 12.0:
            violations.append("deep peak_wavenumber must lie in [2, 12]")
        if not 5.0 <= h <= 25.0:
            violations.append("deep depth must lie in [5, 25]")
        if not 0.005 <= significant_height <= 0.03:
            violations.append("deep significant_height must lie in [0.005, 0.03]")
    else:
        raise ValueError(f"unknown random-sea stratum: {stratum}")
    return tuple(violations)


def is_in_paper_support(
    parameters: JonswapTmaParameters,
    *,
    stratum: RandomSeaStratum,
    length: float = 2.0 * np.pi,
) -> bool:
    """Return whether parameters belong to the named paper-corpus stratum."""

    return not paper_support_violations(parameters, stratum=stratum, length=length)


def jonswap_tma_spectrum(
    parameters: JonswapTmaParameters,
    *,
    band: ResolvedBand,
    gravity: float = 1.0,
) -> JonswapTmaSpectrum:
    """Integrate the windowed JONSWAP/TMA density over Fourier cells."""

    if parameters.depth <= 0.0:
        raise ValueError("depth must be positive")
    if parameters.significant_height <= 0.0:
        raise ValueError("significant_height must be positive")
    if parameters.peak_wavenumber <= 0.0:
        raise ValueError("peak_wavenumber must be positive")
    if parameters.peak_enhancement < 1.0:
        raise ValueError("peak_enhancement must be at least one")
    if not 0.0 <= parameters.right_moving_fraction <= 1.0:
        raise ValueError("right_moving_fraction must lie in [0, 1]")
    if parameters.peak_wavenumber > band.transition_wavenumber:
        raise ValueError("peak_wavenumber must lie in the untapered spectral interior")
    if gravity <= 0.0 or not np.isfinite(gravity):
        raise ValueError("gravity must be finite and positive")

    centers = positive_mode_wavenumbers(band=band)
    spacing = 2.0 * np.pi / band.length
    lower = np.maximum(0.0, centers - 0.5 * spacing)
    upper = np.minimum(band.maximum_wavenumber, centers + 0.5 * spacing)
    nodes, weights = np.polynomial.legendre.leggauss(band.quadrature_order)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    cell_wavenumbers = midpoint[:, None] + half_width[:, None] * nodes[None, :]

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
    tma_factor = tma_depth_factor(cell_wavenumbers * depth)
    group_velocity = finite_depth_group_velocity(
        cell_wavenumbers, depth=depth, gravity=gravity
    )
    window = resolved_band_window(cell_wavenumbers, band=band)
    density = jonswap_density * tma_factor * group_velocity * window
    cell_energy = half_width * np.sum(weights[None, :] * density, axis=1)
    total_energy = float(np.sum(cell_energy))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("resolved JONSWAP/TMA spectrum has no finite energy")
    fractions = np.asarray(cell_energy / total_energy, dtype=np.float64)
    return JonswapTmaSpectrum(
        wavenumbers=np.asarray(centers, dtype=np.float64),
        energy_fractions=fractions,
    )


def sample_jonswap_tma_phases(
    rng: np.random.Generator,
    *,
    band: ResolvedBand,
) -> tuple[FloatArray, FloatArray]:
    """Draw independent right- and left-moving phases for every mode."""

    n_modes = positive_mode_wavenumbers(band=band).size
    return (
        rng.uniform(0.0, 2.0 * np.pi, size=n_modes).astype(np.float64),
        rng.uniform(0.0, 2.0 * np.pi, size=n_modes).astype(np.float64),
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
    """Construct one resolved-band random-phase ``(eta, xi)`` state.

    The phases are explicit inputs, so a state is reproducible without
    depending on a global random-number generator. No realization-dependent
    rescaling or profile rejection is performed.
    """

    coordinates = np.asarray(x, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    if not np.isfinite(coordinates).all():
        raise ValueError("x must contain only finite values")

    spectrum = jonswap_tma_spectrum(parameters, band=band, gravity=gravity)
    right_phases = np.asarray(phase_right, dtype=np.float64)
    left_phases = np.asarray(phase_left, dtype=np.float64)
    expected_shape = spectrum.wavenumbers.shape
    if right_phases.shape != expected_shape or left_phases.shape != expected_shape:
        raise ValueError(f"phase arrays must both have shape {expected_shape}")
    if not np.isfinite(right_phases).all() or not np.isfinite(left_phases).all():
        raise ValueError("phase arrays must contain only finite values")

    variance = (parameters.significant_height / 4.0) ** 2
    right_fraction = parameters.right_moving_fraction
    amplitude_right = np.sqrt(
        2.0 * right_fraction * variance * spectrum.energy_fractions
    )
    amplitude_left = np.sqrt(
        2.0 * (1.0 - right_fraction) * variance * spectrum.energy_fractions
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
    angular_frequency = np.sqrt(gravity * flat_dno)
    transfer = angular_frequency / flat_dno
    xi = np.sum(
        transfer[:, None]
        * (
            amplitude_right[:, None] * np.sin(right_argument)
            - amplitude_left[:, None] * np.sin(left_argument)
        ),
        axis=0,
    )
    return JonswapTmaState(
        eta=np.asarray(eta, dtype=np.float64),
        xi=np.asarray(xi, dtype=np.float64),
        spectrum=spectrum,
        amplitude_right=np.asarray(amplitude_right, dtype=np.float64),
        amplitude_left=np.asarray(amplitude_left, dtype=np.float64),
        phase_right=right_phases.copy(),
        phase_left=left_phases.copy(),
    )
