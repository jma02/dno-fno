"""Parameter sampling for the paper-dataset fifth-order Stokes family.

The four parameter groups cross finite- and deep-water branches with low and
moderate carrier-steepness intervals.  Finite-depth amplitudes are redrawn in
their assigned interval until the conservative fifth-order support condition
``Ur_+ <= 26`` holds.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from solver.reference_solutions.stokes_wave import (
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
)
from solver.gen_data.pipeline.types import (
    ROOT_SEED_BY_DATASET_SPLIT,
    DatasetSplit,
    PhysicalFamilyId,
)


StokesBranch: TypeAlias = Literal["finite", "deep"]
PAPER_GRAVITY = 1.0
PAPER_DOMAIN_LENGTH = 2.0 * np.pi
PAPER_AMPLITUDE_BOUNDS = (0.000766, 0.011494)
FINITE_DEPTH_BOUNDS = (0.02, 1.5)
DEEP_DEPTH_BOUNDS = (4.0, 50.0)
FINITE_DEPTH_WAVENUMBER_BOUNDS = (0.5, 5.0)
DEEP_DEPTH_WAVENUMBER_MINIMUM = 5.0
MAXIMUM_URSELL_REDRAWS = 1000

_FINITE_CARRIER_MODES = tuple(range(14, 27))
_DEEP_LOW_CARRIER_MODES = tuple(range(1, 21))
_DEEP_MODERATE_CARRIER_MODES = tuple(range(3, 21))
STOKES_PARAMETER_GROUPS: dict[
    str, tuple[StokesBranch, tuple[float, float], tuple[int, ...]]
] = {
    "finite_low": ("finite", (0.005, 0.03), _FINITE_CARRIER_MODES),
    "finite_moderate": ("finite", (0.03, 0.15), _FINITE_CARRIER_MODES),
    "deep_low": ("deep", (0.005, 0.03), _DEEP_LOW_CARRIER_MODES),
    "deep_moderate": ("deep", (0.03, 0.15), _DEEP_MODERATE_CARRIER_MODES),
}

_compiled_finite_depth_ursell = jax.jit(finite_depth_stokes_ursell_upper_bound)


StokesSample = NamedTuple(
    "StokesSample",
    [
        ("branch", StokesBranch),
        ("carrier_mode", int),
        ("depth", float),
        ("phase", float),
        ("amplitude", float),
    ],
)


def sample_stokes_simulation(
    parameter_group_id: str,
    *,
    dataset_split: DatasetSplit,
    attempt_number: int,
) -> StokesSample | None:
    """Sample one Stokes state, or return ``None`` if amplitude redraws expire."""

    branch, steepness_bounds, carrier_modes = STOKES_PARAMETER_GROUPS[
        parameter_group_id
    ]

    rng = np.random.Generator(
        np.random.PCG64(
            (
                ROOT_SEED_BY_DATASET_SPLIT[dataset_split],
                PhysicalFamilyId.STOKES,
                attempt_number,
            )
        )
    )
    carrier_mode = carrier_modes[int(rng.integers(0, len(carrier_modes)))]
    wavenumber = 2.0 * math.pi * carrier_mode / PAPER_DOMAIN_LENGTH
    if branch == "finite":
        depth_lower = max(
            FINITE_DEPTH_BOUNDS[0],
            FINITE_DEPTH_WAVENUMBER_BOUNDS[0] / wavenumber,
        )
        depth_upper = min(
            FINITE_DEPTH_BOUNDS[1],
            FINITE_DEPTH_WAVENUMBER_BOUNDS[1] / wavenumber,
        )
    else:
        depth_lower = max(
            DEEP_DEPTH_BOUNDS[0],
            DEEP_DEPTH_WAVENUMBER_MINIMUM / wavenumber,
        )
        depth_upper = DEEP_DEPTH_BOUNDS[1]
    amplitude_lower = max(
        PAPER_AMPLITUDE_BOUNDS[0],
        steepness_bounds[0] / wavenumber,
    )
    amplitude_upper = min(
        PAPER_AMPLITUDE_BOUNDS[1],
        steepness_bounds[1] / wavenumber,
    )
    depth = (
        depth_lower
        if depth_lower == depth_upper
        else float(np.exp(rng.uniform(np.log(depth_lower), np.log(depth_upper))))
    )
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    for _ in range(MAXIMUM_URSELL_REDRAWS + 1):
        amplitude = float(rng.uniform(amplitude_lower, amplitude_upper))
        if branch == "deep":
            return StokesSample(branch, carrier_mode, depth, phase, amplitude)
        with jax.enable_x64():
            ursell = _compiled_finite_depth_ursell(
                jnp.asarray(wavenumber, dtype=jnp.float64),
                jnp.asarray(depth, dtype=jnp.float64),
                jnp.asarray(PAPER_GRAVITY, dtype=jnp.float64),
                jnp.asarray(amplitude, dtype=jnp.float64),
            )
        ursell_value = float(np.asarray(jax.device_get(ursell), dtype=np.float64))
        if (
            math.isfinite(ursell_value)
            and ursell_value <= FINITE_DEPTH_STOKES_URSELL_LIMIT
        ):
            return StokesSample(branch, carrier_mode, depth, phase, amplitude)
    return None
