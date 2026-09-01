"""Parameter sampling for the paper-dataset JONSWAP/TMA family.

The 27 parameter groups are the Cartesian product of three depth strata,
three peak-enhancement values, and three right-moving energy fractions.
Each attempted simulation owns one PCG64 stream determined by its complete
``SimulationKey.seed_words`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.jonswap_tma import (
    PAPER_PEAK_ENHANCEMENTS,
    PAPER_PEAK_STEEPNESS_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RIGHT_MOVING_FRACTIONS,
    PAPER_SHALLOW_PEAK_MODES,
    JonswapTmaParameters,
    RandomSeaStratum,
    ResolvedBand,
    find_jonswap_parameter_violations,
    relative_frequency_interval_fits,
    sample_jonswap_tma_phases,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    random_generator_for_simulation,
)


FloatArray: TypeAlias = NDArray[np.float64]
JsonRecord: TypeAlias = dict[str, object]

FINITE_PEAK_WAVENUMBER_BOUNDS = (2.0, 12.0)
FINITE_DEPTH_BOUNDS = (0.1, 1.5)
DEEP_PEAK_WAVENUMBER_BOUNDS = (2.0, 12.0)
DEEP_DEPTH_BOUNDS = (5.0, 25.0)
SIGNIFICANT_HEIGHT_BOUNDS = (0.005, 0.03)
SHALLOW_DEPTH_WAVENUMBER_BOUNDS = (0.2, 1.5)
SHALLOW_RELATIVE_HEIGHT_BOUNDS = (0.03, 0.16)


JonswapTmaParameterGroup: TypeAlias = tuple[RandomSeaStratum, float, float]
JONSWAP_TMA_PARAMETER_GROUPS: dict[str, JonswapTmaParameterGroup] = {
    (
        f"{stratum}__gamma_{format(peak_enhancement, 'g').replace('.', 'p')}"
        f"__right_{format(right_moving_fraction, 'g').replace('.', 'p')}"
    ): (stratum, peak_enhancement, right_moving_fraction)
    for stratum in ("shallow", "finite", "deep")
    for peak_enhancement in PAPER_PEAK_ENHANCEMENTS
    for right_moving_fraction in PAPER_RIGHT_MOVING_FRACTIONS
}
JONSWAP_TMA_PARAMETER_GROUP_IDS = tuple(JONSWAP_TMA_PARAMETER_GROUPS)


@dataclass(frozen=True)
class JonswapTmaSample:
    """One sampled parameter specification and its two explicit phase arrays."""

    assignment: AttemptAssignment
    stratum: RandomSeaStratum
    parameters: JonswapTmaParameters
    phase_right: FloatArray
    phase_left: FloatArray

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON-ready record sufficient for exact replay."""

        return {
            **self.assignment.to_json_record(),
            "stratum": self.stratum,
            "depth": self.parameters.depth,
            "significant_height": self.parameters.significant_height,
            "peak_wavenumber": self.parameters.peak_wavenumber,
            "peak_enhancement": self.parameters.peak_enhancement,
            "right_moving_fraction": self.parameters.right_moving_fraction,
            "phase_right": self.phase_right.tolist(),
            "phase_left": self.phase_left.tolist(),
        }


def _sample_shallow_parameters(
    rng: np.random.Generator,
    *,
    peak_enhancement: float,
    right_moving_fraction: float,
    length: float,
    band: ResolvedBand,
) -> JonswapTmaParameters:
    """Draw uniformly from the declared shallow admissible set."""

    peak_mode = int(rng.choice(PAPER_SHALLOW_PEAK_MODES))
    while True:
        depth_wavenumber = float(rng.uniform(*SHALLOW_DEPTH_WAVENUMBER_BOUNDS))
        relative_height = float(rng.uniform(*SHALLOW_RELATIVE_HEIGHT_BOUNDS))
        peak_wavenumber = 2.0 * np.pi * peak_mode / length
        depth = depth_wavenumber / peak_wavenumber
        candidate = JonswapTmaParameters(
            depth=depth,
            significant_height=2.0 * depth * relative_height,
            peak_wavenumber=peak_wavenumber,
            peak_enhancement=peak_enhancement,
            right_moving_fraction=right_moving_fraction,
        )
        frequency_resolved = relative_frequency_interval_fits(
            candidate,
            band=band,
            relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )
        if (
            depth_wavenumber * relative_height <= PAPER_PEAK_STEEPNESS_MAXIMUM
            and frequency_resolved
        ):
            return candidate


def _sample_finite_or_deep_parameters(
    rng: np.random.Generator,
    *,
    stratum: RandomSeaStratum,
    peak_enhancement: float,
    right_moving_fraction: float,
    band: ResolvedBand,
) -> JonswapTmaParameters:
    """Draw independent uniform variables for a finite- or deep-water group."""

    if stratum == "finite":
        peak_bounds = FINITE_PEAK_WAVENUMBER_BOUNDS
        depth_bounds = FINITE_DEPTH_BOUNDS
    elif stratum == "deep":
        peak_bounds = DEEP_PEAK_WAVENUMBER_BOUNDS
        depth_bounds = DEEP_DEPTH_BOUNDS
    else:
        raise ValueError("parameter sampling requires a finite or deep group")

    while True:
        candidate = JonswapTmaParameters(
            depth=float(rng.uniform(*depth_bounds)),
            significant_height=float(rng.uniform(*SIGNIFICANT_HEIGHT_BOUNDS)),
            peak_wavenumber=float(rng.uniform(*peak_bounds)),
            peak_enhancement=peak_enhancement,
            right_moving_fraction=right_moving_fraction,
        )
        frequency_resolved = relative_frequency_interval_fits(
            candidate,
            band=band,
            relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )
        if (
            candidate.peak_wavenumber * candidate.significant_height / 2.0
            <= PAPER_PEAK_STEEPNESS_MAXIMUM
            and frequency_resolved
        ):
            return candidate


def sample_jonswap_tma_simulation(
    assignment: AttemptAssignment,
    *,
    band: ResolvedBand,
) -> JonswapTmaSample:
    """Sample one complete JONSWAP/TMA specification for an attempted simulation.

    The assignment fixes the parameter group and random stream before any
    parameter or phase is drawn. The returned specification includes both
    phase arrays and passes the paper-support predicate by construction.
    """

    try:
        stratum, peak_enhancement, right_moving_fraction = JONSWAP_TMA_PARAMETER_GROUPS[
            assignment.parameter_group_id
        ]
    except KeyError as error:
        raise ValueError(
            f"unknown JONSWAP/TMA parameter group: {assignment.parameter_group_id}"
        ) from error
    rng = random_generator_for_simulation(assignment.simulation_key)
    if stratum == "shallow":
        parameters = _sample_shallow_parameters(
            rng,
            peak_enhancement=peak_enhancement,
            right_moving_fraction=right_moving_fraction,
            length=band.length,
            band=band,
        )
    else:
        parameters = _sample_finite_or_deep_parameters(
            rng,
            stratum=stratum,
            peak_enhancement=peak_enhancement,
            right_moving_fraction=right_moving_fraction,
            band=band,
        )
    phase_right, phase_left = sample_jonswap_tma_phases(rng, band=band)

    violations = find_jonswap_parameter_violations(
        parameters,
        stratum=stratum,
        length=band.length,
        band=band,
        relative_frequency_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    )
    if violations:
        raise RuntimeError(
            "sampled JONSWAP/TMA parameters violate declared support: "
            + "; ".join(violations)
        )
    return JonswapTmaSample(
        assignment=assignment,
        stratum=stratum,
        parameters=parameters,
        phase_right=phase_right,
        phase_left=phase_left,
    )
