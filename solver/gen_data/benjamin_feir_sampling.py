"""Parameter sampling for the paper-dataset Benjamin--Feir family.

Each of the 66 feasible integer pairs ``(n_c, Delta n)`` is one allocation
cell.  Conditional on that pair, the carrier steepness is uniform on the
part of ``[0.05, 0.13]`` inside the leading deep-water instability band and
below the declared focused-steepness limit.  The sideband-to-carrier amplitude
ratio is uniform on ``[0.05, 0.10]``, and the global translation is uniform on
``[0, L)``.  Relative to the translated coordinate, both sidebands have the
fixed JCP09 phase shift ``-pi/4``.  Each attempted case uses one PCG64 stream
constructed from every word in ``CaseKey.seed_words``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, TypeAlias

import numpy as np

from solver.gen_data.benjamin_feir_jcp09 import (
    BENJAMIN_FEIR_MODE_PAIRS,
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    PERTURBATION_RATIO_MIN,
    JCP09_RELATIVE_SIDEBAND_PHASE,
    ParameterArrays,
    deep_water_proxy_depth,
    focused_steepness_carrier_upper_bound,
    focused_steepness_proxy,
    instability_band_fraction,
)
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    random_generator_for_case,
)


JsonRecord: TypeAlias = dict[str, object]
PAPER_FOCUSED_STEEPNESS_LIMIT: Final[float] = (1.0 + math.sqrt(2.0)) / 10.0
PAPER_PERTURBATION_RATIO_MAX: Final[float] = 0.10


BenjaminFeirCell: TypeAlias = tuple[int, int]


def _conditional_steepness_bounds(
    carrier_mode: int,
    sideband_offset: int,
) -> tuple[float, float]:
    """Return the admissible carrier-steepness interval for one mode pair."""

    lower = max(
        CARRIER_STEEPNESS_MIN,
        sideband_offset / (2.0 * math.sqrt(2.0) * carrier_mode),
    )
    upper = min(
        CARRIER_STEEPNESS_MAX,
        float(
            focused_steepness_carrier_upper_bound(
                carrier_mode,
                sideband_offset,
                focused_steepness_limit=PAPER_FOCUSED_STEEPNESS_LIMIT,
            )
        ),
    )
    return lower, upper


BENJAMIN_FEIR_SAMPLE_CELLS: dict[str, BenjaminFeirCell] = {
    f"n_c_{carrier_mode:02d}__delta_n_{sideband_offset:02d}": (
        carrier_mode,
        sideband_offset,
    )
    for carrier_mode, sideband_offset in BENJAMIN_FEIR_MODE_PAIRS
}
BENJAMIN_FEIR_SAMPLE_CELL_IDS = tuple(BENJAMIN_FEIR_SAMPLE_CELLS)


@dataclass(frozen=True)
class BenjaminFeirSample:
    """One complete sampled Benjamin--Feir parameter specification."""

    assignment: AttemptAssignment
    domain_length: float
    carrier_mode: int
    sideband_offset: int
    carrier_steepness: float
    perturbation_ratio: float
    translation: float

    @property
    def conditional_steepness_bounds(self) -> tuple[float, float]:
        """Return this mode pair's admissible carrier-steepness interval."""

        return _conditional_steepness_bounds(
            self.carrier_mode,
            self.sideband_offset,
        )

    @property
    def left_mode(self) -> int:
        """Return the lower sideband mode ``n_c - Delta n``."""

        return self.carrier_mode - self.sideband_offset

    @property
    def right_mode(self) -> int:
        """Return the upper sideband mode ``n_c + Delta n``."""

        return self.carrier_mode + self.sideband_offset

    @property
    def depth(self) -> float:
        """Return the fixed deep-water proxy depth."""

        return deep_water_proxy_depth(self.domain_length)

    @property
    def fundamental_wavenumber(self) -> float:
        """Return the domain's fundamental wavenumber ``2 pi / L``."""

        return 2.0 * math.pi / self.domain_length

    @property
    def carrier_wavenumber(self) -> float:
        """Return the carrier wavenumber."""

        return self.carrier_mode * self.fundamental_wavenumber

    @property
    def carrier_amplitude(self) -> float:
        """Return the prescribed first-harmonic carrier amplitude."""

        return self.carrier_steepness / self.carrier_wavenumber

    @property
    def band_fraction(self) -> float:
        """Return the leading-order instability-band fraction."""

        return float(
            instability_band_fraction(
                self.carrier_mode,
                self.sideband_offset,
                self.carrier_steepness,
            )
        )

    @property
    def focused_steepness(self) -> float:
        """Return the leading-NLS focused-envelope steepness proxy."""

        return float(
            focused_steepness_proxy(
                self.carrier_mode,
                self.sideband_offset,
                self.carrier_steepness,
            )
        )

    def to_parameter_arrays(self) -> ParameterArrays:
        """Return a one-case batch for ``build_initial_conditions``."""

        return {
            "n_carr": np.asarray([self.carrier_mode], dtype=np.int32),
            "side_offset": np.asarray(
                [self.sideband_offset],
                dtype=np.int32,
            ),
            "n_l": np.asarray([self.left_mode], dtype=np.int32),
            "n_r": np.asarray([self.right_mode], dtype=np.int32),
            "eps_carrier": np.asarray(
                [self.carrier_steepness],
                dtype=np.float64,
            ),
            "eps_pert": np.asarray(
                [self.perturbation_ratio],
                dtype=np.float64,
            ),
            "translation": np.asarray(
                [self.translation],
                dtype=np.float64,
            ),
            "depth": np.asarray([self.depth], dtype=np.float64),
        }

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON record containing every sampled parameter."""

        violations = find_benjamin_feir_sample_violations(self)
        if violations:
            raise ValueError(
                "cannot serialize unsupported Benjamin--Feir sample: "
                + "; ".join(violations)
            )

        fundamental = self.fundamental_wavenumber
        record: JsonRecord = {
            **self.assignment.to_json_record(),
            "domain_length": self.domain_length,
            "depth": self.depth,
            "carrier_mode": self.carrier_mode,
            "sideband_offset": self.sideband_offset,
            "left_mode": self.left_mode,
            "right_mode": self.right_mode,
            "first_harmonic_carrier_steepness": self.carrier_steepness,
            "first_harmonic_sideband_ratio": self.perturbation_ratio,
            "translation": self.translation,
            "carrier_phase_in_translated_frame": 0.0,
            "relative_sideband_phase": JCP09_RELATIVE_SIDEBAND_PHASE,
            "fundamental_wavenumber": fundamental,
            "carrier_wavenumber": self.carrier_wavenumber,
            "left_wavenumber": self.left_mode * fundamental,
            "right_wavenumber": self.right_mode * fundamental,
            "carrier_amplitude": self.carrier_amplitude,
            "sideband_amplitude": (self.perturbation_ratio * self.carrier_amplitude),
            "instability_band_fraction": self.band_fraction,
            "focused_steepness": self.focused_steepness,
            "focused_steepness_limit": PAPER_FOCUSED_STEEPNESS_LIMIT,
        }
        return record


def find_benjamin_feir_sample_violations(
    sample: BenjaminFeirSample,
) -> tuple[str, ...]:
    """Return every violation of the declared sampling law."""

    violations: list[str] = []
    expected_cell = BENJAMIN_FEIR_SAMPLE_CELLS.get(sample.assignment.cell_id)
    if expected_cell != (sample.carrier_mode, sample.sideband_offset):
        violations.append("sample parameters do not match the assigned cell")
    if not math.isfinite(sample.domain_length) or sample.domain_length <= 0.0:
        violations.append("domain_length must be finite and positive")
    steepness_lower, steepness_upper = sample.conditional_steepness_bounds
    if (
        not math.isfinite(sample.carrier_steepness)
        or sample.carrier_steepness <= steepness_lower
        or sample.carrier_steepness > steepness_upper
    ):
        violations.append("carrier_steepness is outside its conditional support")
    if (
        not math.isfinite(sample.perturbation_ratio)
        or sample.perturbation_ratio < PERTURBATION_RATIO_MIN
        or sample.perturbation_ratio > PAPER_PERTURBATION_RATIO_MAX
    ):
        violations.append("perturbation_ratio is outside declared support")
    if (
        not math.isfinite(sample.translation)
        or sample.translation < 0.0
        or sample.translation >= sample.domain_length
    ):
        violations.append("translation must lie in [0, L)")
    if math.isfinite(sample.carrier_steepness):
        band_fraction = sample.band_fraction
        if not math.isfinite(band_fraction) or not 0.0 < band_fraction < 1.0:
            violations.append("carrier and sidebands are outside the instability band")
    return tuple(violations)


def sample_benjamin_feir_case(
    assignment: AttemptAssignment,
    *,
    domain_length: float = 2.0 * math.pi,
) -> BenjaminFeirSample:
    """Sample one complete specification in the assignment's fixed pair cell."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be finite and positive")

    cell = BENJAMIN_FEIR_SAMPLE_CELLS.get(assignment.cell_id)
    if cell is None:
        raise ValueError(f"unknown Benjamin--Feir sample cell: {assignment.cell_id}")
    carrier_mode, sideband_offset = cell
    rng = random_generator_for_case(assignment.case_key)
    steepness_lower, steepness_upper = _conditional_steepness_bounds(
        carrier_mode,
        sideband_offset,
    )
    carrier_steepness = steepness_upper - (steepness_upper - steepness_lower) * float(
        rng.random()
    )
    if carrier_steepness <= steepness_lower:
        carrier_steepness = float(np.nextafter(steepness_lower, steepness_upper))
    perturbation_ratio = float(
        rng.uniform(PERTURBATION_RATIO_MIN, PAPER_PERTURBATION_RATIO_MAX)
    )
    translation = float(rng.uniform(0.0, domain_length))
    sample = BenjaminFeirSample(
        assignment=assignment,
        domain_length=domain_length,
        carrier_mode=carrier_mode,
        sideband_offset=sideband_offset,
        carrier_steepness=carrier_steepness,
        perturbation_ratio=perturbation_ratio,
        translation=translation,
    )
    violations = find_benjamin_feir_sample_violations(sample)
    if violations:
        raise RuntimeError(
            "sampled Benjamin--Feir parameters violate declared support: "
            + "; ".join(violations)
        )
    return sample
