"""Parameter sampling for the paper-dataset Tanaka family.

A parameter category fixes the regime, crest count ``m``, and number ``q`` of
right-moving crests.  The main regime has ``m = 1, 2, 3`` and
``q = 0, ..., m``; the steep regime has ``m = 1`` and ``q = 0, 1``.  These
give eleven equally balanced categories.  Conditional on ``q``, the sampler
uniformly chooses which ``q`` of the ``m`` crests move right.

Revision 2 draws depth independently before amplitude.  Revision 3 draws the
amplitudes first and then draws depth log-uniformly above the elementary
profile-resolution lower bound.  The revision-2 order remains available only
for exact replay of historical proposals.

Here ``alpha_i = a_i / h``, where ``a_i`` is the dimensional crest
amplitude.  Each attempted case uses one PCG64 stream constructed from all
five words in ``AttemptAssignment.case_key.seed_words``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Final, Literal, TypeAlias

import numpy as np

from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    random_generator_for_case,
)


JsonRecord: TypeAlias = dict[str, object]
TanakaRegime: TypeAlias = Literal["main", "steep"]

MAIN_DEPTH_BOUNDS = (0.01, 0.30)
MAIN_TOTAL_ALPHA_BOUNDS = (0.10, 0.35)
STEEP_DEPTH_BOUNDS = (0.20, 0.35)
STEEP_ALPHA_BOUNDS = (0.25, 0.45)
SEPARATION_TO_DEPTH_RATIO = 3.0
TANAKA_SAMPLING_REVISION_V2: Final[int] = 2
TANAKA_SAMPLING_REVISION_V3: Final[int] = 3
TANAKA_DELIVERED_MAXIMUM_WAVENUMBER: Final[float] = 128.0
TANAKA_MINIMUM_RESOLUTION_RATIO: Final[float] = 10.0
_RESOLUTION_ROUNDOFF_RELATIVE_TOLERANCE: Final[float] = 1.0e-12
_RESOLUTION_ROUNDOFF_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12


@dataclass(frozen=True)
class TanakaSampleCell:
    """One fixed Tanaka regime, crest count, and direction composition."""

    cell_id: str
    regime: TanakaRegime
    crest_count: int
    right_moving_count: int

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must not be empty")
        if self.regime not in ("main", "steep"):
            raise ValueError(f"unknown Tanaka regime: {self.regime}")
        if isinstance(self.crest_count, bool) or not isinstance(
            self.crest_count,
            int,
        ):
            raise TypeError("crest_count must be an integer")
        if self.crest_count not in (1, 2, 3):
            raise ValueError("crest_count must be one, two, or three")
        if self.regime == "steep" and self.crest_count != 1:
            raise ValueError("the steep Tanaka cell must contain one crest")
        if (
            isinstance(self.right_moving_count, bool)
            or not isinstance(self.right_moving_count, int)
        ):
            raise TypeError("right_moving_count must be an integer")
        if not 0 <= self.right_moving_count <= self.crest_count:
            raise ValueError(
                "right_moving_count must lie between zero and crest_count"
            )


TANAKA_SAMPLE_CELLS = (
    *tuple(
        TanakaSampleCell(
            f"main_m{crest_count}_q{right_moving_count}",
            "main",
            crest_count,
            right_moving_count,
        )
        for crest_count in (1, 2, 3)
        for right_moving_count in range(crest_count + 1)
    ),
    TanakaSampleCell("steep_m1_q0", "steep", 1, 0),
    TanakaSampleCell("steep_m1_q1", "steep", 1, 1),
)
_CELL_BY_ID = {cell.cell_id: cell for cell in TANAKA_SAMPLE_CELLS}


@dataclass(frozen=True)
class TanakaCrest:
    """Parameters of one translated Tanaka crest."""

    alpha: float
    center: float
    direction: int


@dataclass(frozen=True)
class TanakaSample:
    """One complete sampled Tanaka parameter specification."""

    assignment: AttemptAssignment
    cell: TanakaSampleCell
    domain_length: float
    depth: float
    crests: tuple[TanakaCrest, ...]

    @property
    def total_dimensionless_amplitude(self) -> float:
        """Return ``sum_i alpha_i`` using the persisted crest values."""

        return sum(crest.alpha for crest in self.crests)

    @property
    def maximum_dimensionless_crest_amplitude(self) -> float:
        """Return the largest dimensionless crest amplitude."""

        if not self.crests:
            raise ValueError("a Tanaka sample must contain at least one crest")
        return max(crest.alpha for crest in self.crests)

    @property
    def maximum_inverse_width(self) -> float:
        """Return the inverse-width scale of the narrowest crest."""

        return tanaka_inverse_width(
            alpha=self.maximum_dimensionless_crest_amplitude,
            depth=self.depth,
        )

    @property
    def resolution_ratio(self) -> float:
        """Return delivered wavenumbers per narrowest inverse width."""

        return tanaka_resolution_ratio(
            alpha=self.maximum_dimensionless_crest_amplitude,
            depth=self.depth,
        )

    @property
    def conditional_depth_lower_bound(self) -> float:
        """Return the revision-3 lower depth bound for these amplitudes."""

        return tanaka_conditional_depth_bounds(
            self.cell,
            tuple(crest.alpha for crest in self.crests),
        )[0]

    @property
    def required_minimum_separation(self) -> float:
        """Return the required periodic center separation ``3 h``."""

        return SEPARATION_TO_DEPTH_RATIO * self.depth

    @property
    def cyclic_gaps(self) -> tuple[float, ...]:
        """Return the positive gaps between sorted centers around the circle."""

        centers = sorted(crest.center for crest in self.crests)
        if len(centers) == 1:
            return (self.domain_length,)
        return tuple(
            (
                centers[(index + 1) % len(centers)]
                - center
            )
            % self.domain_length
            for index, center in enumerate(centers)
        )

    @property
    def achieved_minimum_separation(self) -> float:
        """Return the smallest cyclic center gap."""

        return min(self.cyclic_gaps)

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON-ready record sufficient for exact replay."""

        key = self.assignment.case_key
        if key.revision_id == TANAKA_SAMPLING_REVISION_V2:
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
                "regime": self.cell.regime,
                "domain_length": self.domain_length,
                "depth": self.depth,
                "crest_count": self.cell.crest_count,
                "right_moving_count": self.cell.right_moving_count,
                "total_dimensionless_amplitude": (
                    self.total_dimensionless_amplitude
                ),
                "total_amplitude": (
                    self.depth * self.total_dimensionless_amplitude
                ),
                "required_minimum_separation": (
                    self.required_minimum_separation
                ),
                "achieved_minimum_separation": (
                    self.achieved_minimum_separation
                ),
                "cyclic_gaps": list(self.cyclic_gaps),
                "crests": [
                    {
                        "alpha": crest.alpha,
                        "amplitude": self.depth * crest.alpha,
                        "center": crest.center,
                        "direction": crest.direction,
                    }
                    for crest in self.crests
                ],
            }
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return record
        if key.revision_id != TANAKA_SAMPLING_REVISION_V3:
            raise ValueError(
                f"unsupported Tanaka sampling revision: {key.revision_id}"
            )

        record = {
            "case_id": key.case_id,
            "family_id": key.family_id,
            "revision_id": key.revision_id,
            "split_id": key.split_id.value,
            "root_seed": key.root_seed,
            "stream_id": key.stream_id,
            "attempt_index": key.attempt_index,
            "seed_words": list(key.seed_words),
            "cell_id": self.cell.cell_id,
            "regime": self.cell.regime,
            "domain_length": self.domain_length,
            "depth": self.depth,
            "crest_count": self.cell.crest_count,
            "right_moving_count": self.cell.right_moving_count,
            "total_dimensionless_amplitude": (
                self.total_dimensionless_amplitude
            ),
            "total_amplitude": self.depth * self.total_dimensionless_amplitude,
            "required_minimum_separation": self.required_minimum_separation,
            "achieved_minimum_separation": self.achieved_minimum_separation,
            "cyclic_gaps": list(self.cyclic_gaps),
            "crests": [
                {
                    "alpha": crest.alpha,
                    "amplitude": self.depth * crest.alpha,
                    "center": crest.center,
                    "direction": crest.direction,
                }
                for crest in self.crests
            ],
            "profile_resolution": {
                "delivered_maximum_wavenumber": (
                    TANAKA_DELIVERED_MAXIMUM_WAVENUMBER
                ),
                "maximum_dimensionless_crest_amplitude": (
                    self.maximum_dimensionless_crest_amplitude
                ),
                "maximum_inverse_width": self.maximum_inverse_width,
                "wavenumbers_per_inverse_width": self.resolution_ratio,
                "required_minimum_wavenumbers_per_inverse_width": (
                    TANAKA_MINIMUM_RESOLUTION_RATIO
                ),
                "conditional_depth_lower_bound": (
                    self.conditional_depth_lower_bound
                ),
            },
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


def _require_sample_cell(cell_id: str) -> TanakaSampleCell:
    """Return the declared Tanaka sample cell named by ``cell_id``."""

    try:
        return _CELL_BY_ID[cell_id]
    except KeyError as error:
        raise ValueError(f"unknown Tanaka sample cell: {cell_id}") from error


def tanaka_inverse_width(*, alpha: float, depth: float) -> float:
    """Return ``sqrt(3 alpha) / (2 h)`` for one crest."""

    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    if isinstance(depth, bool) or not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be positive and finite")
    return math.sqrt(3.0 * alpha) / (2.0 * depth)


def tanaka_resolution_ratio(*, alpha: float, depth: float) -> float:
    """Return delivered wavenumbers per crest inverse width."""

    return TANAKA_DELIVERED_MAXIMUM_WAVENUMBER / tanaka_inverse_width(
        alpha=alpha,
        depth=depth,
    )


def tanaka_conditional_depth_bounds(
    cell: TanakaSampleCell,
    alphas: tuple[float, ...],
) -> tuple[float, float]:
    """Return the revision-3 depth interval resolved by construction."""

    if not isinstance(cell, TanakaSampleCell):
        raise TypeError("cell must be a TanakaSampleCell")
    if len(alphas) != cell.crest_count:
        raise ValueError("alphas must contain one value per crest")
    if any(
        isinstance(alpha, bool)
        or not math.isfinite(alpha)
        or alpha <= 0.0
        for alpha in alphas
    ):
        raise ValueError("every alpha must be positive and finite")

    base_lower, base_upper = (
        MAIN_DEPTH_BOUNDS
        if cell.regime == "main"
        else STEEP_DEPTH_BOUNDS
    )
    maximum_alpha = max(alphas)
    resolution_lower = (
        TANAKA_MINIMUM_RESOLUTION_RATIO
        * math.sqrt(3.0 * maximum_alpha)
        / (2.0 * TANAKA_DELIVERED_MAXIMUM_WAVENUMBER)
    )
    lower = max(base_lower, resolution_lower)
    if lower > base_upper:
        raise ValueError(
            "the requested profile resolution leaves no admissible depth"
        )
    return (lower, base_upper)


def _sample_log_uniform(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> float:
    """Geometrically interpolate between positive bounds using a uniform draw."""

    lower, upper = bounds
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def _split_total_alpha(
    rng: np.random.Generator,
    *,
    total_alpha: float,
    crest_count: int,
) -> tuple[float, ...]:
    """Split a positive total by a uniform-simplex Dirichlet draw."""

    if crest_count == 1:
        return (total_alpha,)

    weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
    alphas = [float(total_alpha * weight) for weight in weights]
    alphas[-1] = total_alpha - sum(alphas[:-1])
    if any(alpha <= 0.0 for alpha in alphas):
        raise RuntimeError("Dirichlet split produced a nonpositive crest amplitude")
    return tuple(alphas)


def _sample_depth_and_alphas_v2(
    rng: np.random.Generator,
    cell: TanakaSampleCell,
) -> tuple[float, tuple[float, ...]]:
    """Replay the historical revision-2 draw order exactly."""

    if cell.regime == "main":
        depth = _sample_log_uniform(rng, MAIN_DEPTH_BOUNDS)
        sampled_total = float(rng.uniform(*MAIN_TOTAL_ALPHA_BOUNDS))
        alphas = _split_total_alpha(
            rng,
            total_alpha=sampled_total,
            crest_count=cell.crest_count,
        )
    else:
        depth = _sample_log_uniform(rng, STEEP_DEPTH_BOUNDS)
        alphas = (float(rng.uniform(*STEEP_ALPHA_BOUNDS)),)
    return depth, alphas


def _sample_depth_and_alphas_v3(
    rng: np.random.Generator,
    cell: TanakaSampleCell,
) -> tuple[float, tuple[float, ...]]:
    """Draw amplitudes first and depth directly on resolved support."""

    if cell.regime == "main":
        sampled_total = float(rng.uniform(*MAIN_TOTAL_ALPHA_BOUNDS))
        alphas = _split_total_alpha(
            rng,
            total_alpha=sampled_total,
            crest_count=cell.crest_count,
        )
    else:
        alphas = (float(rng.uniform(*STEEP_ALPHA_BOUNDS)),)
    depth = _sample_log_uniform(
        rng,
        tanaka_conditional_depth_bounds(cell, alphas),
    )
    return depth, alphas


def _sample_periodic_centers(
    rng: np.random.Generator,
    *,
    crest_count: int,
    domain_length: float,
    minimum_separation: float,
) -> tuple[float, ...]:
    """Sample a rotation-invariant separated point set on a periodic domain.

    For ``m = crest_count``, draw

    ``w ~ Dirichlet(1, ..., 1)`` and ``theta ~ Uniform[0, L)``.

    The cyclic gaps are

    ``g_i = d + (L - m d) w_i``,

    where ``L`` is the domain length and ``d`` the required separation.
    Centers are ``theta`` plus the successive partial sums of the gaps,
    reduced modulo ``L``.  The uniform ``theta`` makes the law invariant
    under every periodic translation.
    """

    if crest_count == 1:
        return (float(rng.uniform(0.0, domain_length)),)

    slack = domain_length - crest_count * minimum_separation
    if slack < 0.0:
        raise ValueError(
            "crest count and required separation do not fit on the domain"
        )
    weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
    gaps = minimum_separation + slack * weights
    origin = float(rng.uniform(0.0, domain_length))
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(gaps[:-1]))
    )
    return tuple(
        float(center)
        for center in np.mod(origin + offsets, domain_length)
    )


def tanaka_support_violations(
    sample: TanakaSample,
) -> tuple[str, ...]:
    """Return every violation of the declared Tanaka sampling support."""

    violations: list[str] = []
    cell = sample.cell
    revision_id = sample.assignment.case_key.revision_id
    if revision_id not in (
        TANAKA_SAMPLING_REVISION_V2,
        TANAKA_SAMPLING_REVISION_V3,
    ):
        violations.append("unsupported Tanaka sampling revision")
    if sample.assignment.cell_id != cell.cell_id:
        violations.append("assignment cell_id does not match sample cell")
    if not math.isfinite(sample.domain_length) or sample.domain_length <= 0.0:
        violations.append("domain_length must be positive and finite")
    if not math.isfinite(sample.depth) or sample.depth <= 0.0:
        violations.append("depth must be positive and finite")
    if len(sample.crests) != cell.crest_count:
        violations.append("crest count does not match the parameter category")

    if cell.regime == "main":
        depth_bounds = MAIN_DEPTH_BOUNDS
        total_bounds = MAIN_TOTAL_ALPHA_BOUNDS
    else:
        depth_bounds = STEEP_DEPTH_BOUNDS
        total_bounds = STEEP_ALPHA_BOUNDS

    if math.isfinite(sample.depth) and not (
        depth_bounds[0] <= sample.depth <= depth_bounds[1]
    ):
        violations.append("depth lies outside the cell support")

    crest_values_are_finite = all(
        math.isfinite(value)
        for crest in sample.crests
        for value in (crest.alpha, crest.center, float(crest.direction))
    )
    if not crest_values_are_finite:
        violations.append("crest parameters must be finite")

    if any(crest.alpha <= 0.0 for crest in sample.crests):
        violations.append("every dimensionless crest amplitude must be positive")
    if any(crest.direction not in (-1, 1) for crest in sample.crests):
        violations.append("every crest direction must equal -1 or +1")
    if (
        sum(crest.direction == 1 for crest in sample.crests)
        != cell.right_moving_count
    ):
        violations.append(
            "number of right-moving crests does not match the parameter category"
        )
    if math.isfinite(sample.domain_length) and sample.domain_length > 0.0:
        if any(
            not 0.0 <= crest.center < sample.domain_length
            for crest in sample.crests
        ):
            violations.append("crest centers must lie in [0, L)")

    total_alpha = sample.total_dimensionless_amplitude
    if not math.isfinite(total_alpha) or not (
        total_bounds[0] <= total_alpha <= total_bounds[1]
    ):
        violations.append("total dimensionless amplitude lies outside support")

    if (
        len(sample.crests) > 1
        and math.isfinite(sample.domain_length)
        and sample.domain_length > 0.0
        and math.isfinite(sample.depth)
    ):
        required = sample.required_minimum_separation
        if cell.crest_count * required > sample.domain_length:
            violations.append("required separated centers do not fit on the domain")
        elif sample.achieved_minimum_separation < required:
            violations.append("periodic crest-center separation is below 3h")

    resolution_inputs_are_valid = (
        revision_id == TANAKA_SAMPLING_REVISION_V3
        and len(sample.crests) == cell.crest_count
        and all(
            math.isfinite(crest.alpha) and crest.alpha > 0.0
            for crest in sample.crests
        )
        and math.isfinite(sample.depth)
        and sample.depth > 0.0
        and math.isfinite(total_alpha)
        and total_bounds[0] <= total_alpha <= total_bounds[1]
    )
    if resolution_inputs_are_valid:
        lower, _ = tanaka_conditional_depth_bounds(
            cell,
            tuple(crest.alpha for crest in sample.crests),
        )
        if sample.depth < lower and not math.isclose(
            sample.depth,
            lower,
            rel_tol=_RESOLUTION_ROUNDOFF_RELATIVE_TOLERANCE,
            abs_tol=_RESOLUTION_ROUNDOFF_ABSOLUTE_TOLERANCE,
        ):
            violations.append("depth is below the profile-resolution support")
        ratio = sample.resolution_ratio
        if ratio < TANAKA_MINIMUM_RESOLUTION_RATIO and not math.isclose(
            ratio,
            TANAKA_MINIMUM_RESOLUTION_RATIO,
            rel_tol=_RESOLUTION_ROUNDOFF_RELATIVE_TOLERANCE,
            abs_tol=_RESOLUTION_ROUNDOFF_ABSOLUTE_TOLERANCE,
        ):
            violations.append(
                "profile resolution ratio is below the required minimum"
            )

    return tuple(violations)


def sample_tanaka_case(
    assignment: AttemptAssignment,
    *,
    domain_length: float = 2.0 * np.pi,
) -> TanakaSample:
    """Sample one complete Tanaka specification for an attempted case."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")

    cell = _require_sample_cell(assignment.cell_id)
    rng = random_generator_for_case(assignment.case_key)
    revision_id = assignment.case_key.revision_id
    if revision_id == TANAKA_SAMPLING_REVISION_V2:
        depth, alphas = _sample_depth_and_alphas_v2(rng, cell)
    elif revision_id == TANAKA_SAMPLING_REVISION_V3:
        depth, alphas = _sample_depth_and_alphas_v3(rng, cell)
    else:
        raise ValueError(
            f"unsupported Tanaka sampling revision: {revision_id}"
        )

    minimum_separation = SEPARATION_TO_DEPTH_RATIO * depth
    centers = _sample_periodic_centers(
        rng,
        crest_count=cell.crest_count,
        domain_length=domain_length,
        minimum_separation=minimum_separation,
    )
    direction_multiset = np.concatenate(
        (
            -np.ones(
                cell.crest_count - cell.right_moving_count,
                dtype=np.int8,
            ),
            np.ones(cell.right_moving_count, dtype=np.int8),
        )
    )
    directions = tuple(
        int(direction) for direction in rng.permutation(direction_multiset)
    )
    sample = TanakaSample(
        assignment=assignment,
        cell=cell,
        domain_length=float(domain_length),
        depth=depth,
        crests=tuple(
            TanakaCrest(alpha=alpha, center=center, direction=direction)
            for alpha, center, direction in zip(
                alphas,
                centers,
                directions,
            )
        ),
    )
    violations = tanaka_support_violations(sample)
    if violations:
        raise RuntimeError(
            "sampled Tanaka parameters violate declared support: "
            + "; ".join(violations)
        )
    return sample
