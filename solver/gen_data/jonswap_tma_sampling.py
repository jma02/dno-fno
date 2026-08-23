"""Parameter sampling for the paper-dataset JONSWAP/TMA family.

The 27 allocation cells are the Cartesian product of three depth strata,
three peak-enhancement values, and three right-moving energy fractions.
Each attempted case owns one PCG64 stream determined by its complete
``CaseKey.seed_words`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    paper_support_violations,
    relative_frequency_interval_fits,
    sample_jonswap_tma_phases,
)
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    random_generator_for_case,
)


FloatArray: TypeAlias = NDArray[np.float64]
JsonRecord: TypeAlias = dict[str, object]

JONSWAP_TMA_SAMPLING_REVISION_V4 = 4

FINITE_PEAK_WAVENUMBER_BOUNDS = (2.0, 12.0)
FINITE_DEPTH_BOUNDS = (0.1, 1.5)
DEEP_PEAK_WAVENUMBER_BOUNDS = (2.0, 12.0)
DEEP_DEPTH_BOUNDS = (5.0, 25.0)
SIGNIFICANT_HEIGHT_BOUNDS = (0.005, 0.03)
SHALLOW_DEPTH_WAVENUMBER_BOUNDS = (0.2, 1.5)
SHALLOW_RELATIVE_HEIGHT_BOUNDS = (0.03, 0.16)


def _number_label(value: float) -> str:
    """Return a stable, path-safe label for one discrete cell value."""

    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class JonswapTmaSampleCell:
    """One fixed stratum, peak enhancement, and direction fraction."""

    stratum: RandomSeaStratum
    peak_enhancement: float
    right_moving_fraction: float

    def __post_init__(self) -> None:
        if self.stratum not in ("shallow", "finite", "deep"):
            raise ValueError(f"unknown JONSWAP/TMA stratum: {self.stratum}")
        if self.peak_enhancement not in PAPER_PEAK_ENHANCEMENTS:
            raise ValueError("peak_enhancement is not a declared cell value")
        if self.right_moving_fraction not in PAPER_RIGHT_MOVING_FRACTIONS:
            raise ValueError("right_moving_fraction is not a declared cell value")

    @property
    def cell_id(self) -> str:
        """Return the stable identifier used by ``AttemptAssignment``."""

        gamma = _number_label(self.peak_enhancement)
        direction = _number_label(self.right_moving_fraction)
        return f"{self.stratum}__gamma_{gamma}__right_{direction}"


JONSWAP_TMA_SAMPLE_CELLS = tuple(
    JonswapTmaSampleCell(
        stratum=stratum,
        peak_enhancement=peak_enhancement,
        right_moving_fraction=right_moving_fraction,
    )
    for stratum in ("shallow", "finite", "deep")
    for peak_enhancement in PAPER_PEAK_ENHANCEMENTS
    for right_moving_fraction in PAPER_RIGHT_MOVING_FRACTIONS
)
_CELL_BY_ID = {cell.cell_id: cell for cell in JONSWAP_TMA_SAMPLE_CELLS}


@dataclass(frozen=True)
class JonswapTmaSample:
    """One sampled parameter specification and its two explicit phase arrays."""

    assignment: AttemptAssignment
    cell: JonswapTmaSampleCell
    parameters: JonswapTmaParameters
    phase_right: FloatArray
    phase_left: FloatArray

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON-ready record sufficient for exact replay."""

        key = self.assignment.case_key
        record: JsonRecord = {
            "case_id": key.case_id,
            "family_id": key.family_id,
            "revision_id": key.revision_id,
            "split_id": key.split_id.value,
            "root_seed": key.root_seed,
            "stream_id": key.stream_id,
            "attempt_index": key.attempt_index,
            "seed_words": list(key.seed_words),
            "cell_id": self.cell.cell_id,
            "stratum": self.cell.stratum,
            "depth": self.parameters.depth,
            "significant_height": self.parameters.significant_height,
            "peak_wavenumber": self.parameters.peak_wavenumber,
            "peak_enhancement": self.parameters.peak_enhancement,
            "right_moving_fraction": self.parameters.right_moving_fraction,
            "phase_right": self.phase_right.tolist(),
            "phase_left": self.phase_left.tolist(),
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


def _require_sample_cell(cell_id: str) -> JonswapTmaSampleCell:
    """Return the declared sample cell named by ``cell_id``."""

    try:
        return _CELL_BY_ID[cell_id]
    except KeyError as error:
        raise ValueError(f"unknown JONSWAP/TMA sample cell: {cell_id}") from error


def _sample_shallow_parameters(
    rng: np.random.Generator,
    *,
    cell: JonswapTmaSampleCell,
    length: float,
    band: ResolvedBand,
    relative_frequency_maximum: float | None,
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
            peak_enhancement=cell.peak_enhancement,
            right_moving_fraction=cell.right_moving_fraction,
        )
        frequency_resolved = (
            relative_frequency_maximum is None
            or relative_frequency_interval_fits(
                candidate,
                band=band,
                relative_maximum=relative_frequency_maximum,
            )
        )
        if (
            depth_wavenumber * relative_height
            <= PAPER_PEAK_STEEPNESS_MAXIMUM
            and frequency_resolved
        ):
            return candidate


def _sample_rectangular_parameters(
    rng: np.random.Generator,
    *,
    cell: JonswapTmaSampleCell,
    band: ResolvedBand,
    relative_frequency_maximum: float | None,
) -> JonswapTmaParameters:
    """Draw independent uniform variables in a finite- or deep-water cell."""

    if cell.stratum == "finite":
        peak_bounds = FINITE_PEAK_WAVENUMBER_BOUNDS
        depth_bounds = FINITE_DEPTH_BOUNDS
    elif cell.stratum == "deep":
        peak_bounds = DEEP_PEAK_WAVENUMBER_BOUNDS
        depth_bounds = DEEP_DEPTH_BOUNDS
    else:
        raise ValueError("rectangular sampling requires a finite or deep cell")

    while True:
        candidate = JonswapTmaParameters(
            depth=float(rng.uniform(*depth_bounds)),
            significant_height=float(rng.uniform(*SIGNIFICANT_HEIGHT_BOUNDS)),
            peak_wavenumber=float(rng.uniform(*peak_bounds)),
            peak_enhancement=cell.peak_enhancement,
            right_moving_fraction=cell.right_moving_fraction,
        )
        frequency_resolved = (
            relative_frequency_maximum is None
            or relative_frequency_interval_fits(
                candidate,
                band=band,
                relative_maximum=relative_frequency_maximum,
            )
        )
        if (
            candidate.peak_wavenumber * candidate.significant_height / 2.0
            <= PAPER_PEAK_STEEPNESS_MAXIMUM
            and frequency_resolved
        ):
            return candidate


def sample_jonswap_tma_case(
    assignment: AttemptAssignment,
    *,
    band: ResolvedBand,
) -> JonswapTmaSample:
    """Sample one complete JONSWAP/TMA specification for an attempted case.

    The assignment fixes the allocation cell and random stream before any
    parameter or phase is drawn. The returned specification includes both
    phase arrays and passes the paper-support predicate by construction.
    """

    cell = _require_sample_cell(assignment.cell_id)
    rng = random_generator_for_case(assignment.case_key)
    relative_frequency_maximum = (
        PAPER_RELATIVE_FREQUENCY_MAXIMUM
        if assignment.case_key.revision_id == JONSWAP_TMA_SAMPLING_REVISION_V4
        else None
    )
    if cell.stratum == "shallow":
        parameters = _sample_shallow_parameters(
            rng,
            cell=cell,
            length=band.length,
            band=band,
            relative_frequency_maximum=relative_frequency_maximum,
        )
    else:
        parameters = _sample_rectangular_parameters(
            rng,
            cell=cell,
            band=band,
            relative_frequency_maximum=relative_frequency_maximum,
        )
    phase_right, phase_left = sample_jonswap_tma_phases(rng, band=band)

    violations = paper_support_violations(
        parameters,
        stratum=cell.stratum,
        length=band.length,
        band=band if relative_frequency_maximum is not None else None,
        relative_frequency_maximum=relative_frequency_maximum,
    )
    if violations:
        raise RuntimeError(
            "sampled JONSWAP/TMA parameters violate declared support: "
            + "; ".join(violations)
        )
    return JonswapTmaSample(
        assignment=assignment,
        cell=cell,
        parameters=parameters,
        phase_right=phase_right,
        phase_left=phase_left,
    )
