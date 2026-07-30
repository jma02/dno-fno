"""Population sampling for the paper-corpus Tanaka family.

A parameter category fixes the regime, crest count ``m``, and number ``q`` of
right-moving crests.  The main regime has ``m = 1, 2, 3`` and
``q = 0, ..., m``; the steep regime has ``m = 1`` and ``q = 0, 1``.  These
give eleven equally balanced categories.  Conditional on ``q``, the sampler
uniformly chooses which ``q`` of the ``m`` crests move right.

In the main regime, if ``U`` is uniform on ``[0, 1]``, the depth is the
geometric interpolation ``h = 0.01**(1-U) * 0.30**U``.  A total dimensionless
amplitude is uniform on ``[0.10, 0.35]`` and is split among the crests by a
``Dirichlet(1, ..., 1)`` draw.  The steep regime has one crest, geometric
depth interpolation between ``0.20`` and ``0.35``, and dimensionless
amplitude ``alpha`` uniform on ``[0.25, 0.45]``.

Here ``alpha_i = a_i / h``, where ``a_i`` is the dimensional crest
amplitude.  Each attempted case uses one PCG64 stream constructed from all
five words in ``AttemptAssignment.case_key.seed_words``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Literal, TypeAlias

import numpy as np

from solver.gen_data.pipeline.production import AttemptAssignment, CaseKey


JsonRecord: TypeAlias = dict[str, object]
TanakaRegime: TypeAlias = Literal["main", "steep"]

MAIN_DEPTH_BOUNDS = (0.01, 0.30)
MAIN_TOTAL_ALPHA_BOUNDS = (0.10, 0.35)
STEEP_DEPTH_BOUNDS = (0.20, 0.35)
STEEP_ALPHA_BOUNDS = (0.25, 0.45)
SEPARATION_TO_DEPTH_RATIO = 3.0


@dataclass(frozen=True)
class TanakaPopulationCell:
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


TANAKA_POPULATION_CELLS = (
    *tuple(
        TanakaPopulationCell(
            f"main_m{crest_count}_q{right_moving_count}",
            "main",
            crest_count,
            right_moving_count,
        )
        for crest_count in (1, 2, 3)
        for right_moving_count in range(crest_count + 1)
    ),
    TanakaPopulationCell("steep_m1_q0", "steep", 1, 0),
    TanakaPopulationCell("steep_m1_q1", "steep", 1, 1),
)
_CELL_BY_ID = {cell.cell_id: cell for cell in TANAKA_POPULATION_CELLS}


@dataclass(frozen=True)
class TanakaCrest:
    """Parameters of one translated Tanaka crest."""

    alpha: float
    center: float
    direction: int


@dataclass(frozen=True)
class TanakaPopulationSample:
    """One complete sampled Tanaka parameter specification."""

    assignment: AttemptAssignment
    cell: TanakaPopulationCell
    domain_length: float
    depth: float
    crests: tuple[TanakaCrest, ...]

    @property
    def total_dimensionless_amplitude(self) -> float:
        """Return ``sum_i alpha_i`` using the persisted crest values."""

        return sum(crest.alpha for crest in self.crests)

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
        record: JsonRecord = {
            "schema": "tanaka_population_spec_v2",
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
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


def population_cell(cell_id: str) -> TanakaPopulationCell:
    """Return the declared Tanaka population cell named by ``cell_id``."""

    try:
        return _CELL_BY_ID[cell_id]
    except KeyError as error:
        raise ValueError(f"unknown Tanaka population cell: {cell_id}") from error


def pcg64_for_case(case_key: CaseKey) -> np.random.Generator:
    """Construct the case's independent PCG64 generator from all seed words."""

    seed_sequence = np.random.SeedSequence(case_key.seed_words)
    return np.random.Generator(np.random.PCG64(seed_sequence))


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
    sample: TanakaPopulationSample,
) -> tuple[str, ...]:
    """Return every violation of the declared Tanaka population support."""

    violations: list[str] = []
    cell = sample.cell
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

    return tuple(violations)


def sample_tanaka_population(
    assignment: AttemptAssignment,
    *,
    domain_length: float = 2.0 * np.pi,
) -> TanakaPopulationSample:
    """Sample one complete Tanaka specification for an attempted case."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")

    cell = population_cell(assignment.cell_id)
    rng = pcg64_for_case(assignment.case_key)
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
    sample = TanakaPopulationSample(
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
