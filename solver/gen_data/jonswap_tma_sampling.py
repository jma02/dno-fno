"""Parameter sampling for the paper-dataset JONSWAP/TMA family.

The 27 parameter groups are the Cartesian product of three depth strata,
three peak-enhancement values, and three right-moving energy fractions.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

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
    relative_frequency_interval_fits,
    sample_jonswap_tma_phases,
)
from solver.gen_data.pipeline.types import (
    ROOT_SEED_BY_DATASET_SPLIT,
    DatasetSplit,
    PhysicalFamilyId,
)


FloatArray: TypeAlias = NDArray[np.float64]

PEAK_WAVENUMBER_BOUNDS = (2.0, 12.0)
FINITE_DEPTH_BOUNDS = (0.1, 1.5)
DEEP_DEPTH_BOUNDS = (5.0, 25.0)
SIGNIFICANT_HEIGHT_BOUNDS = (0.005, 0.03)
SHALLOW_DEPTH_WAVENUMBER_BOUNDS = (0.2, 1.5)
SHALLOW_RELATIVE_HEIGHT_BOUNDS = (0.03, 0.16)


JONSWAP_TMA_PARAMETER_GROUPS: dict[str, tuple[RandomSeaStratum, float, float]] = {
    (
        f"{stratum}__gamma_{format(peak_enhancement, 'g').replace('.', 'p')}"
        f"__right_{format(right_moving_fraction, 'g').replace('.', 'p')}"
    ): (stratum, peak_enhancement, right_moving_fraction)
    for stratum in ("shallow", "finite", "deep")
    for peak_enhancement in PAPER_PEAK_ENHANCEMENTS
    for right_moving_fraction in PAPER_RIGHT_MOVING_FRACTIONS
}
JONSWAP_TMA_PARAMETER_GROUP_IDS = tuple(JONSWAP_TMA_PARAMETER_GROUPS)


JonswapTmaSample = NamedTuple(
    "JonswapTmaSample",
    [
        ("parameters", JonswapTmaParameters),
        ("phase_right", FloatArray),
        ("phase_left", FloatArray),
    ],
)


def sample_jonswap_tma_simulation(
    parameter_group_id: str,
    *,
    dataset_split: DatasetSplit,
    attempt_number: int,
    band: ResolvedBand,
) -> JonswapTmaSample:
    """Sample parameters and both phase arrays from one JONSWAP/TMA group."""

    stratum, peak_enhancement, right_moving_fraction = JONSWAP_TMA_PARAMETER_GROUPS[
        parameter_group_id
    ]
    rng = np.random.Generator(
        np.random.PCG64(
            (
                ROOT_SEED_BY_DATASET_SPLIT[dataset_split],
                PhysicalFamilyId.JONSWAP_TMA,
                attempt_number,
            )
        )
    )
    if stratum == "shallow":
        peak_mode = int(rng.choice(PAPER_SHALLOW_PEAK_MODES))
        peak_wavenumber = 2.0 * np.pi * peak_mode / band.length
        while True:
            depth_wavenumber = float(rng.uniform(*SHALLOW_DEPTH_WAVENUMBER_BOUNDS))
            relative_height = float(rng.uniform(*SHALLOW_RELATIVE_HEIGHT_BOUNDS))
            depth = depth_wavenumber / peak_wavenumber
            parameters = JonswapTmaParameters(
                depth=depth,
                significant_height=2.0 * depth * relative_height,
                peak_wavenumber=peak_wavenumber,
                peak_enhancement=peak_enhancement,
                right_moving_fraction=right_moving_fraction,
            )
            if (
                depth_wavenumber * relative_height <= PAPER_PEAK_STEEPNESS_MAXIMUM
                and relative_frequency_interval_fits(
                    parameters,
                    band=band,
                    relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
                )
            ):
                break
    else:
        depth_bounds = FINITE_DEPTH_BOUNDS if stratum == "finite" else DEEP_DEPTH_BOUNDS
        while True:
            parameters = JonswapTmaParameters(
                depth=float(rng.uniform(*depth_bounds)),
                significant_height=float(rng.uniform(*SIGNIFICANT_HEIGHT_BOUNDS)),
                peak_wavenumber=float(rng.uniform(*PEAK_WAVENUMBER_BOUNDS)),
                peak_enhancement=peak_enhancement,
                right_moving_fraction=right_moving_fraction,
            )
            if (
                parameters.peak_wavenumber * parameters.significant_height / 2.0
                <= PAPER_PEAK_STEEPNESS_MAXIMUM
                and relative_frequency_interval_fits(
                    parameters,
                    band=band,
                    relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
                )
            ):
                break
    phase_right, phase_left = sample_jonswap_tma_phases(rng, band=band)
    return JonswapTmaSample(parameters, phase_right, phase_left)
