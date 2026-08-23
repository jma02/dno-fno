"""Parameter sampling for the paper-dataset fifth-order Stokes family.

The sampling scheme has four allocation cells: finite- and deep-water branches,
each crossed with the steepness intervals ``[0.005, 0.03)`` and
``[0.03, 0.15]``.  An ``AttemptAssignment`` fixes the cell and all five
PCG64 seed words before any parameter is drawn.

For a finite-depth attempt, the carrier mode, depth, phase, and steepness cell
remain fixed while the amplitude is redrawn within that cell until the
conservative fifth-order support condition ``Ur_+ <= 26`` holds.  Every
amplitude and ``Ur_+`` evaluation is retained.  Exhaustion returns no sample:
it raises an exception carrying a strict-JSON-ready failure record.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from solver.reference_solutions.stokes_wave import (
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
)
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    random_generator_for_case,
)


JsonRecord: TypeAlias = dict[str, object]
StokesBranch: TypeAlias = Literal["finite", "deep"]

PAPER_GRAVITY = 1.0
PAPER_STOKES_STEEPNESS_CELLS = np.asarray(
    ((0.005, 0.03), (0.03, 0.15)),
    dtype=np.float64,
)
PAPER_DOMAIN_LENGTH = 2.0 * np.pi
PAPER_AMPLITUDE_BOUNDS = (0.000766, 0.011494)
FINITE_CARRIER_MODE_BOUNDS = (14, 26)
DEEP_CARRIER_MODE_BOUNDS = (1, 20)
FINITE_DEPTH_BOUNDS = (0.02, 1.5)
DEEP_DEPTH_BOUNDS = (4.0, 50.0)
FINITE_DEPTH_WAVENUMBER_BOUNDS = (0.5, 5.0)
DEEP_DEPTH_WAVENUMBER_MINIMUM = 5.0
DEFAULT_MAXIMUM_URSELL_REDRAWS = 1000


@dataclass(frozen=True)
class StokesSampleCell:
    """One fixed Stokes branch and steepness interval."""

    cell_id: str
    branch: StokesBranch
    steepness_cell_index: int

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must not be empty")
        if self.branch not in ("finite", "deep"):
            raise ValueError(f"unknown Stokes branch: {self.branch}")
        if not 0 <= self.steepness_cell_index < len(
            PAPER_STOKES_STEEPNESS_CELLS
        ):
            raise ValueError("steepness_cell_index is out of range")

    @property
    def steepness_bounds(self) -> tuple[float, float]:
        """Return the lower and upper ``ka`` bounds of this cell."""

        bounds = PAPER_STOKES_STEEPNESS_CELLS[
            self.steepness_cell_index
        ]
        return float(bounds[0]), float(bounds[1])

    @property
    def upper_steepness_is_inclusive(self) -> bool:
        """Return whether the mathematical support includes its upper bound."""

        return self.steepness_cell_index == 1


STOKES_SAMPLE_CELLS = (
    StokesSampleCell("finite_low", "finite", 0),
    StokesSampleCell("finite_moderate", "finite", 1),
    StokesSampleCell("deep_low", "deep", 0),
    StokesSampleCell("deep_moderate", "deep", 1),
)
_CELL_BY_ID = {cell.cell_id: cell for cell in STOKES_SAMPLE_CELLS}


@dataclass(frozen=True)
class StokesAmplitudeAttempt:
    """One amplitude proposal and its finite-depth support result."""

    attempt_index: int
    amplitude: float
    ursell_upper_bound: float | None
    accepted: bool


@dataclass(frozen=True)
class StokesSample:
    """One complete accepted Stokes parameter specification."""

    assignment: AttemptAssignment
    cell: StokesSampleCell
    domain_length: float
    gravity: float
    carrier_mode: int
    carrier_mode_support: tuple[int, ...]
    depth: float
    depth_draw_bounds: tuple[float, float]
    phase: float
    amplitude_draw_bounds: tuple[float, float]
    amplitude_attempts: tuple[StokesAmplitudeAttempt, ...]

    @property
    def wavenumber(self) -> float:
        """Return the carrier wavenumber ``2 pi n / L``."""

        return 2.0 * math.pi * self.carrier_mode / self.domain_length

    @property
    def amplitude(self) -> float:
        """Return the accepted amplitude."""

        return self.amplitude_attempts[-1].amplitude

    @property
    def steepness(self) -> float:
        """Return the accepted carrier steepness ``ka``."""

        return self.wavenumber * self.amplitude

    @property
    def support_resampling_count(self) -> int:
        """Return the number of rejected amplitudes before acceptance."""

        return len(self.amplitude_attempts) - 1

    @property
    def ursell_upper_bound(self) -> float | None:
        """Return the accepted finite-depth ``Ur_+``, or ``None`` in deep water."""

        return self.amplitude_attempts[-1].ursell_upper_bound

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON-ready record sufficient for exact replay."""

        record = {
            **_assignment_record(self.assignment),
            "status": "accepted",
            "constructor": "project_fifth_order_stokes_fixed_phase_v1",
            "cell_id": self.cell.cell_id,
            "branch": self.cell.branch,
            "steepness_cell_index": self.cell.steepness_cell_index,
            "steepness_cell_lower": self.cell.steepness_bounds[0],
            "steepness_cell_upper": self.cell.steepness_bounds[1],
            "steepness_cell_upper_inclusive": (
                self.cell.upper_steepness_is_inclusive
            ),
            "domain_length": self.domain_length,
            "gravity": self.gravity,
            "carrier_mode": self.carrier_mode,
            "carrier_mode_support": list(self.carrier_mode_support),
            "wavenumber": self.wavenumber,
            "depth": self.depth,
            "depth_draw_lower": self.depth_draw_bounds[0],
            "depth_draw_upper": self.depth_draw_bounds[1],
            "phase": self.phase,
            "amplitude": self.amplitude,
            "steepness": self.steepness,
            "amplitude_draw_lower": self.amplitude_draw_bounds[0],
            "amplitude_draw_upper": self.amplitude_draw_bounds[1],
            "finite_depth_ursell_limit": (
                FINITE_DEPTH_STOKES_URSELL_LIMIT
                if self.cell.branch == "finite"
                else None
            ),
            "support_resampling_count": self.support_resampling_count,
            "amplitude_attempts": _attempt_records(
                self.amplitude_attempts,
                wavenumber=self.wavenumber,
                ursell_was_evaluated=self.cell.branch == "finite",
            ),
        }
        _check_strict_json(record)
        return record


class StokesSamplingError(ValueError):
    """A fixed finite-depth attempt exhausted its same-cell redraws."""

    def __init__(self, message: str, *, failure_record: JsonRecord) -> None:
        super().__init__(message)
        _check_strict_json(failure_record)
        self.failure_record = failure_record


def _require_sample_cell(cell_id: str) -> StokesSampleCell:
    """Return the declared Stokes sample cell named by ``cell_id``."""

    try:
        return _CELL_BY_ID[cell_id]
    except KeyError as error:
        raise ValueError(f"unknown Stokes sample cell: {cell_id}") from error


def _assignment_record(assignment: AttemptAssignment) -> JsonRecord:
    """Return the common identity fields for success and failure records."""

    key = assignment.case_key
    return {
        "case_id": key.case_id,
        "family_id": key.family_id,
        "revision_id": key.revision_id,
        "split_id": key.split_id.value,
        "root_seed": key.root_seed,
        "stream_id": key.stream_id,
        "attempt_index": key.attempt_index,
        "seed_words": list(key.seed_words),
    }


def _check_strict_json(record: JsonRecord) -> None:
    """Raise if a record cannot be serialized without nonstandard numbers."""

    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_json_number(value: float | None) -> float | None:
    """Represent a finite scalar exactly and a nonfinite scalar as JSON null."""

    if value is None or not math.isfinite(value):
        return None
    return value


def _attempt_records(
    attempts: tuple[StokesAmplitudeAttempt, ...],
    *,
    wavenumber: float,
    ursell_was_evaluated: bool,
) -> list[JsonRecord]:
    """Return strict-JSON records for every amplitude proposal."""

    return [
        {
            "attempt_index": attempt.attempt_index,
            "amplitude": attempt.amplitude,
            "steepness": wavenumber * attempt.amplitude,
            "ursell_was_evaluated": ursell_was_evaluated,
            "ursell_upper_bound": _finite_json_number(
                attempt.ursell_upper_bound
            ),
            "ursell_upper_bound_was_finite": (
                attempt.ursell_upper_bound is not None
                and math.isfinite(attempt.ursell_upper_bound)
            ),
            "accepted": attempt.accepted,
        }
        for attempt in attempts
    ]


def _carrier_mode_bounds(branch: StokesBranch) -> tuple[int, int]:
    """Return the inclusive carrier-mode bounds for one branch."""

    if branch == "finite":
        return FINITE_CARRIER_MODE_BOUNDS
    return DEEP_CARRIER_MODE_BOUNDS


def effective_amplitude_bounds(
    cell: StokesSampleCell,
    *,
    wavenumber: float,
) -> tuple[float, float]:
    """Intersect the global amplitude support with an assigned ``ka`` cell."""

    if not math.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive and finite")
    steepness_lower, steepness_upper = cell.steepness_bounds
    return (
        max(PAPER_AMPLITUDE_BOUNDS[0], steepness_lower / wavenumber),
        min(PAPER_AMPLITUDE_BOUNDS[1], steepness_upper / wavenumber),
    )


def effective_depth_bounds(
    cell: StokesSampleCell,
    *,
    wavenumber: float,
) -> tuple[float, float]:
    """Return the branch depth bounds after imposing its ``kh`` bounds."""

    if not math.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive and finite")
    if cell.branch == "finite":
        return (
            max(
                FINITE_DEPTH_BOUNDS[0],
                FINITE_DEPTH_WAVENUMBER_BOUNDS[0] / wavenumber,
            ),
            min(
                FINITE_DEPTH_BOUNDS[1],
                FINITE_DEPTH_WAVENUMBER_BOUNDS[1] / wavenumber,
            ),
        )
    return (
        max(
            DEEP_DEPTH_BOUNDS[0],
            DEEP_DEPTH_WAVENUMBER_MINIMUM / wavenumber,
        ),
        DEEP_DEPTH_BOUNDS[1],
    )


def feasible_carrier_modes(
    cell: StokesSampleCell,
    *,
    domain_length: float = PAPER_DOMAIN_LENGTH,
) -> tuple[int, ...]:
    """Return modes having nonempty depth and amplitude intervals."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")
    first_mode, last_mode = _carrier_mode_bounds(cell.branch)

    def is_feasible(carrier_mode: int) -> bool:
        wavenumber = 2.0 * math.pi * carrier_mode / domain_length
        amplitude_lower, amplitude_upper = effective_amplitude_bounds(
            cell,
            wavenumber=wavenumber,
        )
        depth_lower, depth_upper = effective_depth_bounds(
            cell,
            wavenumber=wavenumber,
        )
        return amplitude_lower < amplitude_upper and depth_lower <= depth_upper

    return tuple(filter(is_feasible, range(first_mode, last_mode + 1)))


def _sample_log_uniform(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> float:
    """Draw a positive variable uniformly in its logarithm."""

    lower, upper = bounds
    if lower == upper:
        return lower
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


@jax.jit
def _compiled_finite_depth_ursell(
    wavenumber: jax.Array,
    depth: jax.Array,
    gravity: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    """JIT wrapper around the shared conservative fifth-order calculation."""

    return finite_depth_stokes_ursell_upper_bound(
        wavenumber,
        depth,
        gravity,
        amplitude,
    )


def _evaluate_finite_depth_ursell(
    *,
    wavenumber: float,
    depth: float,
    gravity: float,
    amplitude: float,
) -> float:
    """Evaluate ``Ur_+`` in float64 using the shared Stokes formula."""

    with jax.enable_x64():
        value = _compiled_finite_depth_ursell(
            jnp.asarray(wavenumber, dtype=jnp.float64),
            jnp.asarray(depth, dtype=jnp.float64),
            jnp.asarray(gravity, dtype=jnp.float64),
            jnp.asarray(amplitude, dtype=jnp.float64),
        )
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _amplitude_is_in_cell(
    cell: StokesSampleCell,
    *,
    wavenumber: float,
    amplitude: float,
) -> bool:
    """Return whether an amplitude obeys global and cell support."""

    if not math.isfinite(amplitude):
        return False
    steepness = wavenumber * amplitude
    steepness_lower, steepness_upper = cell.steepness_bounds
    upper_ok = (
        steepness <= steepness_upper
        if cell.upper_steepness_is_inclusive
        else steepness < steepness_upper
    )
    return (
        PAPER_AMPLITUDE_BOUNDS[0]
        <= amplitude
        <= PAPER_AMPLITUDE_BOUNDS[1]
        and steepness_lower <= steepness
        and upper_ok
    )


def stokes_support_violations(
    sample: StokesSample,
) -> tuple[str, ...]:
    """Return every violation of the declared Stokes sampling support."""

    violations: list[str] = []
    if _CELL_BY_ID.get(sample.cell.cell_id) != sample.cell:
        violations.append("sample cell is not a declared canonical cell")
    if sample.assignment.cell_id != sample.cell.cell_id:
        violations.append("assignment cell_id does not match sample cell")
    if not math.isfinite(sample.domain_length) or sample.domain_length <= 0.0:
        violations.append("domain_length must be positive and finite")
    if not math.isfinite(sample.gravity) or sample.gravity <= 0.0:
        violations.append("gravity must be positive and finite")

    try:
        expected_modes = feasible_carrier_modes(
            sample.cell,
            domain_length=sample.domain_length,
        )
    except ValueError:
        expected_modes = ()
    if sample.carrier_mode_support != expected_modes:
        violations.append("stored carrier-mode support is inconsistent")
    if sample.carrier_mode not in expected_modes:
        violations.append("carrier mode lies outside the cell support")

    if expected_modes and sample.carrier_mode in expected_modes:
        expected_depth_bounds = effective_depth_bounds(
            sample.cell,
            wavenumber=sample.wavenumber,
        )
        expected_amplitude_bounds = effective_amplitude_bounds(
            sample.cell,
            wavenumber=sample.wavenumber,
        )
        if sample.depth_draw_bounds != expected_depth_bounds:
            violations.append("stored depth-draw bounds are inconsistent")
        if sample.amplitude_draw_bounds != expected_amplitude_bounds:
            violations.append("stored amplitude-draw bounds are inconsistent")
        if not math.isfinite(sample.depth) or not (
            expected_depth_bounds[0]
            <= sample.depth
            <= expected_depth_bounds[1]
        ):
            violations.append("depth lies outside the branch support")

    if not math.isfinite(sample.phase) or not 0.0 <= sample.phase < 2.0 * math.pi:
        violations.append("phase must lie in [0, 2 pi)")
    if not sample.amplitude_attempts:
        violations.append("at least one amplitude attempt is required")
        return tuple(violations)

    expected_attempt_indices = tuple(range(len(sample.amplitude_attempts)))
    observed_attempt_indices = tuple(
        attempt.attempt_index for attempt in sample.amplitude_attempts
    )
    if observed_attempt_indices != expected_attempt_indices:
        violations.append("amplitude attempt indices must be consecutive")
    if any(attempt.accepted for attempt in sample.amplitude_attempts[:-1]):
        violations.append("a rejected prefix attempt is marked accepted")
    if not sample.amplitude_attempts[-1].accepted:
        violations.append("the final amplitude attempt must be accepted")

    for attempt in sample.amplitude_attempts:
        if not _amplitude_is_in_cell(
            sample.cell,
            wavenumber=sample.wavenumber,
            amplitude=attempt.amplitude,
        ):
            violations.append("amplitude attempt lies outside its assigned cell")
            break

    if sample.cell.branch == "finite":
        for attempt in sample.amplitude_attempts[:-1]:
            ursell = attempt.ursell_upper_bound
            if ursell is None or (
                math.isfinite(ursell)
                and ursell <= FINITE_DEPTH_STOKES_URSELL_LIMIT
            ):
                violations.append(
                    "a rejected amplitude satisfies the finite-depth support"
                )
                break
        accepted_ursell = sample.amplitude_attempts[-1].ursell_upper_bound
        if (
            accepted_ursell is None
            or not math.isfinite(accepted_ursell)
            or accepted_ursell > FINITE_DEPTH_STOKES_URSELL_LIMIT
        ):
            violations.append(
                "accepted amplitude violates the finite-depth Ursell support"
            )
    elif len(sample.amplitude_attempts) != 1:
        violations.append("deep-water sampling must use exactly one amplitude")
    elif sample.amplitude_attempts[0].ursell_upper_bound is not None:
        violations.append("deep-water sampling must not evaluate an Ursell number")

    return tuple(violations)


def _failure_record(
    *,
    assignment: AttemptAssignment,
    cell: StokesSampleCell,
    domain_length: float,
    gravity: float,
    carrier_mode: int,
    carrier_mode_support: tuple[int, ...],
    depth: float,
    depth_draw_bounds: tuple[float, float],
    phase: float,
    amplitude_draw_bounds: tuple[float, float],
    attempts: tuple[StokesAmplitudeAttempt, ...],
) -> JsonRecord:
    """Build the complete strict-JSON record for an exhausted attempt."""

    wavenumber = 2.0 * math.pi * carrier_mode / domain_length
    record = {
        **_assignment_record(assignment),
        "status": "failed_ursell_redraw_limit",
        "constructor": "project_fifth_order_stokes_fixed_phase_v1",
        "cell_id": cell.cell_id,
        "branch": cell.branch,
        "steepness_cell_index": cell.steepness_cell_index,
        "steepness_cell_lower": cell.steepness_bounds[0],
        "steepness_cell_upper": cell.steepness_bounds[1],
        "steepness_cell_upper_inclusive": (
            cell.upper_steepness_is_inclusive
        ),
        "domain_length": domain_length,
        "gravity": gravity,
        "carrier_mode": carrier_mode,
        "carrier_mode_support": list(carrier_mode_support),
        "wavenumber": wavenumber,
        "depth": depth,
        "depth_draw_lower": depth_draw_bounds[0],
        "depth_draw_upper": depth_draw_bounds[1],
        "phase": phase,
        "amplitude": attempts[-1].amplitude,
        "steepness": wavenumber * attempts[-1].amplitude,
        "amplitude_draw_lower": amplitude_draw_bounds[0],
        "amplitude_draw_upper": amplitude_draw_bounds[1],
        "finite_depth_ursell_limit": FINITE_DEPTH_STOKES_URSELL_LIMIT,
        "support_resampling_count": len(attempts) - 1,
        "amplitude_attempts": _attempt_records(
            attempts,
            wavenumber=wavenumber,
            ursell_was_evaluated=True,
        ),
    }
    _check_strict_json(record)
    return record


def sample_stokes_case(
    assignment: AttemptAssignment,
    *,
    domain_length: float = PAPER_DOMAIN_LENGTH,
    gravity: float = PAPER_GRAVITY,
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS,
) -> StokesSample:
    """Sample one complete paper-dataset Stokes specification.

    Carrier mode is uniform on the modes having nonempty branch, depth, and
    amplitude intervals.  Depth is log-uniform on its mode-conditioned
    interval, phase is uniform on ``[0, 2 pi)``, and amplitude is uniform on
    the intersection of the global amplitude range and the assigned
    steepness cell.

    A finite-depth ``Ur_+`` failure redraws only amplitude.  The initial draw
    plus at most ``maximum_ursell_redraws`` redraws are evaluated.
    """

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be positive and finite")
    if maximum_ursell_redraws < 0:
        raise ValueError("maximum_ursell_redraws must be nonnegative")

    cell = _require_sample_cell(assignment.cell_id)
    carrier_mode_support = feasible_carrier_modes(
        cell,
        domain_length=domain_length,
    )
    if not carrier_mode_support:
        raise ValueError("the assigned Stokes cell has no feasible carrier mode")

    rng = random_generator_for_case(assignment.case_key)
    carrier_mode = carrier_mode_support[
        int(rng.integers(0, len(carrier_mode_support)))
    ]
    wavenumber = 2.0 * math.pi * carrier_mode / domain_length
    depth_draw_bounds = effective_depth_bounds(
        cell,
        wavenumber=wavenumber,
    )
    amplitude_draw_bounds = effective_amplitude_bounds(
        cell,
        wavenumber=wavenumber,
    )
    depth = _sample_log_uniform(rng, depth_draw_bounds)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    attempts: list[StokesAmplitudeAttempt] = []
    for amplitude_attempt_index in range(maximum_ursell_redraws + 1):
        amplitude = float(rng.uniform(*amplitude_draw_bounds))
        ursell_upper_bound = (
            _evaluate_finite_depth_ursell(
                wavenumber=wavenumber,
                depth=depth,
                gravity=gravity,
                amplitude=amplitude,
            )
            if cell.branch == "finite"
            else None
        )
        accepted = cell.branch == "deep" or (
            ursell_upper_bound is not None
            and math.isfinite(ursell_upper_bound)
            and ursell_upper_bound <= FINITE_DEPTH_STOKES_URSELL_LIMIT
        )
        attempts.append(
            StokesAmplitudeAttempt(
                attempt_index=amplitude_attempt_index,
                amplitude=amplitude,
                ursell_upper_bound=ursell_upper_bound,
                accepted=accepted,
            )
        )
        if accepted:
            sample = StokesSample(
                assignment=assignment,
                cell=cell,
                domain_length=float(domain_length),
                gravity=float(gravity),
                carrier_mode=carrier_mode,
                carrier_mode_support=carrier_mode_support,
                depth=depth,
                depth_draw_bounds=depth_draw_bounds,
                phase=phase,
                amplitude_draw_bounds=amplitude_draw_bounds,
                amplitude_attempts=tuple(attempts),
            )
            violations = stokes_support_violations(sample)
            if violations:
                raise RuntimeError(
                    "sampled Stokes parameters violate declared support: "
                    + "; ".join(violations)
                )
            return sample

    failure_record = _failure_record(
        assignment=assignment,
        cell=cell,
        domain_length=float(domain_length),
        gravity=float(gravity),
        carrier_mode=carrier_mode,
        carrier_mode_support=carrier_mode_support,
        depth=depth,
        depth_draw_bounds=depth_draw_bounds,
        phase=phase,
        amplitude_draw_bounds=amplitude_draw_bounds,
        attempts=tuple(attempts),
    )
    raise StokesSamplingError(
        "finite-depth Stokes attempt exhausted same-cell amplitude redraws",
        failure_record=failure_record,
    )
