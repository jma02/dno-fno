"""Parameter sampling for the paper-dataset Benjamin--Feir family.

Each of the 66 feasible integer pairs ``(n_c, Delta n)`` is one parameter
group. Conditional on that pair, the carrier steepness is uniform on the
part of ``[0.05, 0.13]`` inside the leading deep-water instability band and
below the focused-steepness limit. The sideband-to-carrier amplitude ratio is
uniform on ``[0.05, 0.10]``, and the translation is uniform on ``[0, 2 pi)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from solver.gen_data.pipeline.types import (
    ROOT_SEED_BY_DATASET_SPLIT,
    DatasetSplit,
    PhysicalFamilyId,
)


PAPER_DOMAIN_LENGTH = 2.0 * np.pi
CARRIER_MODE_MIN = 4
CARRIER_MODE_MAX = 20
CARRIER_STEEPNESS_MIN = 0.05
CARRIER_STEEPNESS_MAX = 0.13
PERTURBATION_RATIO_MIN = 0.05
PAPER_FOCUSED_STEEPNESS_LIMIT = (1.0 + math.sqrt(2.0)) / 10.0
PAPER_PERTURBATION_RATIO_MAX = 0.10

BENJAMIN_FEIR_PARAMETER_GROUPS: dict[str, tuple[int, int]] = {
    f"n_c_{carrier_mode:02d}__delta_n_{sideband_offset:02d}": (
        carrier_mode,
        sideband_offset,
    )
    for carrier_mode in range(CARRIER_MODE_MIN, CARRIER_MODE_MAX + 1)
    for sideband_offset in range(1, carrier_mode)
    if sideband_offset / carrier_mode < 2.0 * math.sqrt(2.0) * CARRIER_STEEPNESS_MAX
}

BenjaminFeirSample = NamedTuple(
    "BenjaminFeirSample",
    [
        ("carrier_mode", int),
        ("sideband_offset", int),
        ("carrier_steepness", float),
        ("perturbation_ratio", float),
        ("translation", float),
    ],
)


def sample_benjamin_feir_simulation(
    parameter_group_id: str,
    *,
    dataset_split: DatasetSplit,
    attempt_number: int,
) -> BenjaminFeirSample:
    """Sample one Benjamin--Feir state in the assigned mode-pair group."""

    carrier_mode, sideband_offset = BENJAMIN_FEIR_PARAMETER_GROUPS[parameter_group_id]
    rng = np.random.Generator(
        np.random.PCG64(
            (
                ROOT_SEED_BY_DATASET_SPLIT[dataset_split],
                PhysicalFamilyId.BENJAMIN_FEIR,
                attempt_number,
            )
        )
    )
    steepness_lower = max(
        CARRIER_STEEPNESS_MIN,
        sideband_offset / (2.0 * math.sqrt(2.0) * carrier_mode),
    )
    instability_threshold = sideband_offset / (2.0 * np.sqrt(2.0) * carrier_mode)
    steepness_upper = min(
        CARRIER_STEEPNESS_MAX,
        float(
            (
                -PAPER_FOCUSED_STEEPNESS_LIMIT
                + 2.0
                * np.sqrt(
                    PAPER_FOCUSED_STEEPNESS_LIMIT**2 + 3.0 * instability_threshold**2
                )
            )
            / 3.0
        ),
    )
    carrier_steepness = steepness_upper - (steepness_upper - steepness_lower) * float(
        rng.random()
    )
    if carrier_steepness <= steepness_lower:
        carrier_steepness = float(np.nextafter(steepness_lower, steepness_upper))

    return BenjaminFeirSample(
        carrier_mode,
        sideband_offset,
        carrier_steepness,
        float(rng.uniform(PERTURBATION_RATIO_MIN, PAPER_PERTURBATION_RATIO_MAX)),
        float(rng.uniform(0.0, PAPER_DOMAIN_LENGTH)),
    )
