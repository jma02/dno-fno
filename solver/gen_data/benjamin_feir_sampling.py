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
import json
import math
from typing import Final, TypeAlias

import numpy as np

from solver.gen_data.benjamin_feir_jcp09 import (
    CARRIER_MODE_MAX,
    CARRIER_MODE_MIN,
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    PERTURBATION_RATIO_MIN,
    JCP09_RELATIVE_SIDEBAND_PHASE,
    ParameterArrays,
    deep_water_proxy_depth,
    feasible_mode_pairs,
    focused_steepness_carrier_upper_bound,
    focused_steepness_proxy,
    instability_band_fraction,
)
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    random_generator_for_case,
)


JsonRecord: TypeAlias = dict[str, object]
PAPER_FOCUSED_STEEPNESS_LIMIT: Final[float] = (
    1.0 + math.sqrt(2.0)
) / 10.0
PAPER_PERTURBATION_RATIO_MAX: Final[float] = 0.10


@dataclass(frozen=True)
class BenjaminFeirSampleCell:
    """One fixed carrier mode and symmetric sideband separation."""

    carrier_mode: int
    sideband_offset: int

    def __post_init__(self) -> None:
        if not CARRIER_MODE_MIN <= self.carrier_mode <= CARRIER_MODE_MAX:
            raise ValueError("carrier_mode is outside the declared support")
        if not 1 <= self.sideband_offset < self.carrier_mode:
            raise ValueError(
                "sideband_offset must be positive and below carrier_mode"
            )
        if (
            self.conditional_steepness_lower_bound
            >= self.conditional_steepness_upper_bound
        ):
            raise ValueError(
                "cell does not intersect the declared focused support"
            )

    @property
    def cell_id(self) -> str:
        """Return the stable identifier used by ``AttemptAssignment``."""

        return (
            f"n_c_{self.carrier_mode:02d}"
            f"__delta_n_{self.sideband_offset:02d}"
        )

    @property
    def conditional_steepness_lower_bound(self) -> float:
        """Return the excluded lower endpoint for carrier steepness."""

        instability_threshold = self.sideband_offset / (
            2.0 * math.sqrt(2.0) * self.carrier_mode
        )
        return max(CARRIER_STEEPNESS_MIN, instability_threshold)

    @property
    def conditional_steepness_upper_bound(self) -> float:
        """Return the included focused-support endpoint for steepness."""

        focused_upper = float(
            focused_steepness_carrier_upper_bound(
                self.carrier_mode,
                self.sideband_offset,
                focused_steepness_limit=PAPER_FOCUSED_STEEPNESS_LIMIT,
            )
        )
        return min(CARRIER_STEEPNESS_MAX, focused_upper)


BENJAMIN_FEIR_SAMPLE_CELLS = tuple(
    BenjaminFeirSampleCell(
        carrier_mode=int(carrier_mode),
        sideband_offset=int(sideband_offset),
    )
    for carrier_mode, sideband_offset in feasible_mode_pairs().tolist()
)
_CELL_BY_ID = {
    cell.cell_id: cell for cell in BENJAMIN_FEIR_SAMPLE_CELLS
}


@dataclass(frozen=True)
class BenjaminFeirSample:
    """One complete sampled Benjamin--Feir parameter specification."""

    assignment: AttemptAssignment
    cell: BenjaminFeirSampleCell
    domain_length: float
    carrier_steepness: float
    perturbation_ratio: float
    translation: float

    @property
    def left_mode(self) -> int:
        """Return the lower sideband mode ``n_c - Delta n``."""

        return self.cell.carrier_mode - self.cell.sideband_offset

    @property
    def right_mode(self) -> int:
        """Return the upper sideband mode ``n_c + Delta n``."""

        return self.cell.carrier_mode + self.cell.sideband_offset

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

        return self.cell.carrier_mode * self.fundamental_wavenumber

    @property
    def carrier_amplitude(self) -> float:
        """Return the prescribed first-harmonic carrier amplitude."""

        return self.carrier_steepness / self.carrier_wavenumber

    @property
    def band_fraction(self) -> float:
        """Return the leading-order instability-band fraction."""

        return float(
            instability_band_fraction(
                self.cell.carrier_mode,
                self.cell.sideband_offset,
                self.carrier_steepness,
            )
        )

    @property
    def focused_steepness(self) -> float:
        """Return the leading-NLS focused-envelope steepness proxy."""

        return float(
            focused_steepness_proxy(
                self.cell.carrier_mode,
                self.cell.sideband_offset,
                self.carrier_steepness,
            )
        )

    def to_parameter_arrays(self) -> ParameterArrays:
        """Return a one-case batch for ``build_initial_conditions``."""

        return {
            "n_carr": np.asarray([self.cell.carrier_mode], dtype=np.int32),
            "side_offset": np.asarray(
                [self.cell.sideband_offset],
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

        key = self.assignment.case_key
        fundamental = self.fundamental_wavenumber
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
            "domain_length": self.domain_length,
            "depth": self.depth,
            "carrier_mode": self.cell.carrier_mode,
            "sideband_offset": self.cell.sideband_offset,
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
            "sideband_amplitude": (
                self.perturbation_ratio * self.carrier_amplitude
            ),
            "instability_band_fraction": self.band_fraction,
            "focused_steepness": self.focused_steepness,
            "focused_steepness_limit": PAPER_FOCUSED_STEEPNESS_LIMIT,
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


def _require_sample_cell(cell_id: str) -> BenjaminFeirSampleCell:
    """Return the declared Benjamin--Feir cell named by ``cell_id``."""

    try:
        return _CELL_BY_ID[cell_id]
    except KeyError as error:
        raise ValueError(
            f"unknown Benjamin--Feir sample cell: {cell_id}"
        ) from error


def _sample_open_closed_uniform(
    rng: np.random.Generator,
    *,
    lower: float,
    upper: float,
) -> float:
    """Sample the half-open interval ``(lower, upper]``."""

    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError("uniform bounds must be finite and increasing")
    value = upper - (upper - lower) * float(rng.random())
    if value <= lower:
        value = float(np.nextafter(lower, upper))
    return value


def find_benjamin_feir_sample_violations(
    sample: BenjaminFeirSample,
) -> tuple[str, ...]:
    """Return every violation of the declared sampling law."""

    violations: list[str] = []
    if sample.assignment.cell_id != sample.cell.cell_id:
        violations.append("assignment and sample cells differ")
    if _CELL_BY_ID.get(sample.cell.cell_id) != sample.cell:
        violations.append("sample cell is not one of the 66 declared pairs")
    if not math.isfinite(sample.domain_length) or sample.domain_length <= 0.0:
        violations.append("domain_length must be finite and positive")
    if (
        not math.isfinite(sample.carrier_steepness)
        or sample.carrier_steepness
        <= sample.cell.conditional_steepness_lower_bound
        or sample.carrier_steepness
        > sample.cell.conditional_steepness_upper_bound
    ):
        violations.append(
            "carrier_steepness is outside its conditional support"
        )
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
            violations.append(
                "carrier and sidebands are outside the instability band"
            )
    return tuple(violations)


def sample_benjamin_feir_case(
    assignment: AttemptAssignment,
    *,
    domain_length: float = 2.0 * math.pi,
) -> BenjaminFeirSample:
    """Sample one complete specification in the assignment's fixed pair cell."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be finite and positive")

    cell = _require_sample_cell(assignment.cell_id)
    rng = random_generator_for_case(assignment.case_key)
    carrier_steepness = _sample_open_closed_uniform(
        rng,
        lower=cell.conditional_steepness_lower_bound,
        upper=cell.conditional_steepness_upper_bound,
    )
    perturbation_ratio = float(
        rng.uniform(PERTURBATION_RATIO_MIN, PAPER_PERTURBATION_RATIO_MAX)
    )
    translation = float(rng.uniform(0.0, domain_length))
    sample = BenjaminFeirSample(
        assignment=assignment,
        cell=cell,
        domain_length=domain_length,
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
