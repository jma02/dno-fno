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
import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from solver.reference_solutions.stokes_wave import (
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
)
from solver.gen_data.pipeline.case_allocation import (
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


StokesCell: TypeAlias = tuple[StokesBranch, int]
AmplitudeAttempt: TypeAlias = tuple[float, float | None]
STOKES_SAMPLE_CELLS: dict[str, StokesCell] = {
    "finite_low": ("finite", 0),
    "finite_moderate": ("finite", 1),
    "deep_low": ("deep", 0),
    "deep_moderate": ("deep", 1),
}
STOKES_SAMPLE_CELL_IDS = tuple(STOKES_SAMPLE_CELLS)


@dataclass(frozen=True)
class StokesSample:
    """One complete accepted Stokes parameter specification."""

    assignment: AttemptAssignment
    branch: StokesBranch
    steepness_cell_index: int
    domain_length: float
    gravity: float
    carrier_mode: int
    carrier_mode_support: tuple[int, ...]
    depth: float
    depth_draw_bounds: tuple[float, float]
    phase: float
    amplitude_draw_bounds: tuple[float, float]
    amplitude_attempts: tuple[AmplitudeAttempt, ...]

    @property
    def steepness_bounds(self) -> tuple[float, float]:
        """Return the lower and upper ``ka`` bounds of this sample's cell."""

        bounds = PAPER_STOKES_STEEPNESS_CELLS[self.steepness_cell_index]
        return float(bounds[0]), float(bounds[1])

    @property
    def wavenumber(self) -> float:
        """Return the carrier wavenumber ``2 pi n / L``."""

        return 2.0 * math.pi * self.carrier_mode / self.domain_length

    @property
    def amplitude(self) -> float:
        """Return the accepted amplitude."""

        return self.amplitude_attempts[-1][0]

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

        return self.amplitude_attempts[-1][1]

    def to_json_record(self) -> JsonRecord:
        """Return a strict-JSON-ready record sufficient for exact replay."""

        record = {
            **self.assignment.to_json_record(),
            "status": "accepted",
            "constructor": "project_fifth_order_stokes_fixed_phase_v1",
            "branch": self.branch,
            "steepness_cell_index": self.steepness_cell_index,
            "steepness_cell_lower": self.steepness_bounds[0],
            "steepness_cell_upper": self.steepness_bounds[1],
            "steepness_cell_upper_inclusive": self.steepness_cell_index == 1,
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
                FINITE_DEPTH_STOKES_URSELL_LIMIT if self.branch == "finite" else None
            ),
            "support_resampling_count": self.support_resampling_count,
            "amplitude_attempts": [
                {
                    "attempt_index": index,
                    "amplitude": amplitude,
                    "steepness": self.wavenumber * amplitude,
                    "ursell_was_evaluated": self.branch == "finite",
                    "ursell_upper_bound": (
                        ursell if ursell is not None and math.isfinite(ursell) else None
                    ),
                    "ursell_upper_bound_was_finite": (
                        ursell is not None and math.isfinite(ursell)
                    ),
                    "accepted": index == len(self.amplitude_attempts) - 1,
                }
                for index, (amplitude, ursell) in enumerate(self.amplitude_attempts)
            ],
        }
        return record


class UrsellRedrawLimitReached(ValueError):
    """All amplitude draws for one finite-depth attempt failed support."""

    def __init__(self, failure_record: JsonRecord) -> None:
        super().__init__("finite-depth Ursell redraw limit reached")
        self.failure_record = failure_record


def effective_amplitude_bounds(
    steepness_cell_index: int,
    *,
    wavenumber: float,
) -> tuple[float, float]:
    """Intersect the global amplitude support with an assigned ``ka`` cell."""

    if not math.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive and finite")
    if not 0 <= steepness_cell_index < len(PAPER_STOKES_STEEPNESS_CELLS):
        raise ValueError("steepness_cell_index is out of range")
    steepness_lower, steepness_upper = PAPER_STOKES_STEEPNESS_CELLS[
        steepness_cell_index
    ]
    return (
        max(PAPER_AMPLITUDE_BOUNDS[0], steepness_lower / wavenumber),
        min(PAPER_AMPLITUDE_BOUNDS[1], steepness_upper / wavenumber),
    )


def effective_depth_bounds(
    branch: StokesBranch,
    *,
    wavenumber: float,
) -> tuple[float, float]:
    """Return the branch depth bounds after imposing its ``kh`` bounds."""

    if not math.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive and finite")
    if branch == "finite":
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
    if branch == "deep":
        return (
            max(
                DEEP_DEPTH_BOUNDS[0],
                DEEP_DEPTH_WAVENUMBER_MINIMUM / wavenumber,
            ),
            DEEP_DEPTH_BOUNDS[1],
        )
    raise ValueError(f"unknown Stokes branch: {branch}")


def feasible_carrier_modes(
    branch: StokesBranch,
    steepness_cell_index: int,
    *,
    domain_length: float = PAPER_DOMAIN_LENGTH,
) -> tuple[int, ...]:
    """Return modes having nonempty depth and amplitude intervals."""

    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")
    if branch not in ("finite", "deep"):
        raise ValueError(f"unknown Stokes branch: {branch}")
    first_mode, last_mode = (
        FINITE_CARRIER_MODE_BOUNDS if branch == "finite" else DEEP_CARRIER_MODE_BOUNDS
    )

    def is_feasible(carrier_mode: int) -> bool:
        wavenumber = 2.0 * math.pi * carrier_mode / domain_length
        amplitude_lower, amplitude_upper = effective_amplitude_bounds(
            steepness_cell_index,
            wavenumber=wavenumber,
        )
        depth_lower, depth_upper = effective_depth_bounds(
            branch,
            wavenumber=wavenumber,
        )
        return amplitude_lower < amplitude_upper and depth_lower <= depth_upper

    return tuple(filter(is_feasible, range(first_mode, last_mode + 1)))


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
    steepness_cell_index: int,
    *,
    wavenumber: float,
    amplitude: float,
) -> bool:
    """Return whether an amplitude obeys global and cell support."""

    if not math.isfinite(amplitude) or not 0 <= steepness_cell_index < len(
        PAPER_STOKES_STEEPNESS_CELLS
    ):
        return False
    steepness = wavenumber * amplitude
    steepness_lower, steepness_upper = PAPER_STOKES_STEEPNESS_CELLS[
        steepness_cell_index
    ]
    upper_ok = (
        steepness <= steepness_upper
        if steepness_cell_index == 1
        else steepness < steepness_upper
    )
    return (
        PAPER_AMPLITUDE_BOUNDS[0] <= amplitude <= PAPER_AMPLITUDE_BOUNDS[1]
        and steepness_lower <= steepness
        and upper_ok
    )


def stokes_support_violations(
    sample: StokesSample,
) -> tuple[str, ...]:
    """Return every violation of the declared Stokes sampling support."""

    violations: list[str] = []
    if STOKES_SAMPLE_CELLS.get(sample.assignment.cell_id) != (
        sample.branch,
        sample.steepness_cell_index,
    ):
        violations.append("sample parameters do not match the assigned cell")
    if not math.isfinite(sample.domain_length) or sample.domain_length <= 0.0:
        violations.append("domain_length must be positive and finite")
    if not math.isfinite(sample.gravity) or sample.gravity <= 0.0:
        violations.append("gravity must be positive and finite")

    try:
        expected_modes = feasible_carrier_modes(
            sample.branch,
            sample.steepness_cell_index,
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
            sample.branch,
            wavenumber=sample.wavenumber,
        )
        expected_amplitude_bounds = effective_amplitude_bounds(
            sample.steepness_cell_index,
            wavenumber=sample.wavenumber,
        )
        if sample.depth_draw_bounds != expected_depth_bounds:
            violations.append("stored depth-draw bounds are inconsistent")
        if sample.amplitude_draw_bounds != expected_amplitude_bounds:
            violations.append("stored amplitude-draw bounds are inconsistent")
        if not math.isfinite(sample.depth) or not (
            expected_depth_bounds[0] <= sample.depth <= expected_depth_bounds[1]
        ):
            violations.append("depth lies outside the branch support")

    if not math.isfinite(sample.phase) or not 0.0 <= sample.phase < 2.0 * math.pi:
        violations.append("phase must lie in [0, 2 pi)")
    if not sample.amplitude_attempts:
        violations.append("at least one amplitude attempt is required")
        return tuple(violations)

    for amplitude, _ in sample.amplitude_attempts:
        if not _amplitude_is_in_cell(
            sample.steepness_cell_index,
            wavenumber=sample.wavenumber,
            amplitude=amplitude,
        ):
            violations.append("amplitude attempt lies outside its assigned cell")
            break

    if sample.branch == "finite":
        for _, ursell in sample.amplitude_attempts[:-1]:
            if ursell is None or (
                math.isfinite(ursell) and ursell <= FINITE_DEPTH_STOKES_URSELL_LIMIT
            ):
                violations.append(
                    "a rejected amplitude satisfies the finite-depth support"
                )
                break
        accepted_ursell = sample.amplitude_attempts[-1][1]
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
    elif sample.amplitude_attempts[0][1] is not None:
        violations.append("deep-water sampling must not evaluate an Ursell number")

    return tuple(violations)


def _failure_record(
    *,
    assignment: AttemptAssignment,
    branch: StokesBranch,
    steepness_cell_index: int,
    domain_length: float,
    gravity: float,
    carrier_mode: int,
    carrier_mode_support: tuple[int, ...],
    depth: float,
    depth_draw_bounds: tuple[float, float],
    phase: float,
    amplitude_draw_bounds: tuple[float, float],
    attempts: tuple[AmplitudeAttempt, ...],
) -> JsonRecord:
    """Build the complete strict-JSON record for an exhausted attempt."""

    wavenumber = 2.0 * math.pi * carrier_mode / domain_length
    steepness_lower, steepness_upper = PAPER_STOKES_STEEPNESS_CELLS[
        steepness_cell_index
    ]
    record = {
        **assignment.to_json_record(),
        "status": "failed_ursell_redraw_limit",
        "constructor": "project_fifth_order_stokes_fixed_phase_v1",
        "branch": branch,
        "steepness_cell_index": steepness_cell_index,
        "steepness_cell_lower": float(steepness_lower),
        "steepness_cell_upper": float(steepness_upper),
        "steepness_cell_upper_inclusive": steepness_cell_index == 1,
        "domain_length": domain_length,
        "gravity": gravity,
        "carrier_mode": carrier_mode,
        "carrier_mode_support": list(carrier_mode_support),
        "wavenumber": wavenumber,
        "depth": depth,
        "depth_draw_lower": depth_draw_bounds[0],
        "depth_draw_upper": depth_draw_bounds[1],
        "phase": phase,
        "amplitude": attempts[-1][0],
        "steepness": wavenumber * attempts[-1][0],
        "amplitude_draw_lower": amplitude_draw_bounds[0],
        "amplitude_draw_upper": amplitude_draw_bounds[1],
        "finite_depth_ursell_limit": FINITE_DEPTH_STOKES_URSELL_LIMIT,
        "support_resampling_count": len(attempts) - 1,
        "amplitude_attempts": [
            {
                "attempt_index": index,
                "amplitude": amplitude,
                "steepness": wavenumber * amplitude,
                "ursell_was_evaluated": True,
                "ursell_upper_bound": (
                    ursell if ursell is not None and math.isfinite(ursell) else None
                ),
                "ursell_upper_bound_was_finite": (
                    ursell is not None and math.isfinite(ursell)
                ),
                "accepted": False,
            }
            for index, (amplitude, ursell) in enumerate(attempts)
        ],
    }
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

    cell = STOKES_SAMPLE_CELLS.get(assignment.cell_id)
    if cell is None:
        raise ValueError(f"unknown Stokes sample cell: {assignment.cell_id}")
    branch, steepness_cell_index = cell
    carrier_mode_support = feasible_carrier_modes(
        branch,
        steepness_cell_index,
        domain_length=domain_length,
    )
    if not carrier_mode_support:
        raise ValueError("the assigned Stokes cell has no feasible carrier mode")

    rng = random_generator_for_case(assignment.case_key)
    carrier_mode = carrier_mode_support[int(rng.integers(0, len(carrier_mode_support)))]
    wavenumber = 2.0 * math.pi * carrier_mode / domain_length
    depth_draw_bounds = effective_depth_bounds(
        branch,
        wavenumber=wavenumber,
    )
    amplitude_draw_bounds = effective_amplitude_bounds(
        steepness_cell_index,
        wavenumber=wavenumber,
    )
    depth_lower, depth_upper = depth_draw_bounds
    depth = (
        depth_lower
        if depth_lower == depth_upper
        else float(np.exp(rng.uniform(np.log(depth_lower), np.log(depth_upper))))
    )
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    attempts: list[AmplitudeAttempt] = []
    for _ in range(maximum_ursell_redraws + 1):
        amplitude = float(rng.uniform(*amplitude_draw_bounds))
        ursell_upper_bound = (
            _evaluate_finite_depth_ursell(
                wavenumber=wavenumber,
                depth=depth,
                gravity=gravity,
                amplitude=amplitude,
            )
            if branch == "finite"
            else None
        )
        accepted = branch == "deep" or (
            ursell_upper_bound is not None
            and math.isfinite(ursell_upper_bound)
            and ursell_upper_bound <= FINITE_DEPTH_STOKES_URSELL_LIMIT
        )
        attempts.append((amplitude, ursell_upper_bound))
        if accepted:
            sample = StokesSample(
                assignment=assignment,
                branch=branch,
                steepness_cell_index=steepness_cell_index,
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
        branch=branch,
        steepness_cell_index=steepness_cell_index,
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
    raise UrsellRedrawLimitReached(failure_record)
