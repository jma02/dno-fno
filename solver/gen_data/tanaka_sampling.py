"""Parameter sampling for the paper-dataset Tanaka family.

The eleven parameter groups fix the amplitude regime, crest count, and number
of right-moving crests. Amplitudes are drawn first, followed by a log-uniform
depth that keeps the narrowest crest resolved. Crest centers are uniformly
rotated around the periodic domain with at least three depths of separation.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np

from solver.gen_data.pipeline.types import (
    ROOT_SEED_BY_DATASET_SPLIT,
    DatasetSplit,
    PhysicalFamilyId,
)


PAPER_DOMAIN_LENGTH = 2.0 * np.pi
MAIN_DEPTH_BOUNDS = (0.01, 0.30)
MAIN_TOTAL_ALPHA_BOUNDS = (0.10, 0.35)
STEEP_DEPTH_BOUNDS = (0.20, 0.35)
STEEP_ALPHA_BOUNDS = (0.25, 0.45)
SEPARATION_TO_DEPTH_RATIO = 3.0
TANAKA_DELIVERED_MAXIMUM_WAVENUMBER = 128.0
TANAKA_MINIMUM_RESOLUTION_RATIO = 10.0

TANAKA_PARAMETER_GROUPS: dict[str, tuple[Literal["main", "steep"], int, int]] = {
    **{
        f"main_m{crest_count}_q{right_moving_count}": (
            "main",
            crest_count,
            right_moving_count,
        )
        for crest_count in (1, 2, 3)
        for right_moving_count in range(crest_count + 1)
    },
    "steep_m1_q0": ("steep", 1, 0),
    "steep_m1_q1": ("steep", 1, 1),
}

TanakaCrest = NamedTuple(
    "TanakaCrest",
    [("alpha", float), ("center", float), ("direction", int)],
)
TanakaSample = NamedTuple(
    "TanakaSample",
    [("depth", float), ("crests", tuple[TanakaCrest, ...])],
)


def sample_tanaka_simulation(
    parameter_group_id: str,
    *,
    dataset_split: DatasetSplit,
    attempt_number: int,
) -> TanakaSample:
    """Sample one Tanaka initial-condition specification."""

    regime, crest_count, right_moving_count = TANAKA_PARAMETER_GROUPS[
        parameter_group_id
    ]
    rng = np.random.Generator(
        np.random.PCG64(
            (
                ROOT_SEED_BY_DATASET_SPLIT[dataset_split],
                PhysicalFamilyId.TANAKA,
                attempt_number,
            )
        )
    )

    if regime == "main":
        total_alpha = float(rng.uniform(*MAIN_TOTAL_ALPHA_BOUNDS))
        weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
        split_alphas = [float(total_alpha * weight) for weight in weights]
        split_alphas[-1] = total_alpha - sum(split_alphas[:-1])
        alphas = tuple(split_alphas)
        base_depth_lower, depth_upper = MAIN_DEPTH_BOUNDS
    else:
        alphas = (float(rng.uniform(*STEEP_ALPHA_BOUNDS)),)
        base_depth_lower, depth_upper = STEEP_DEPTH_BOUNDS

    depth_lower = max(
        base_depth_lower,
        TANAKA_MINIMUM_RESOLUTION_RATIO
        * math.sqrt(3.0 * max(alphas))
        / (2.0 * TANAKA_DELIVERED_MAXIMUM_WAVENUMBER),
    )
    depth = float(np.exp(rng.uniform(np.log(depth_lower), np.log(depth_upper))))

    minimum_separation = SEPARATION_TO_DEPTH_RATIO * depth
    if crest_count == 1:
        centers = (float(rng.uniform(0.0, PAPER_DOMAIN_LENGTH)),)
    else:
        slack = PAPER_DOMAIN_LENGTH - crest_count * minimum_separation
        weights = rng.dirichlet(np.ones(crest_count, dtype=np.float64))
        gaps = minimum_separation + slack * weights
        origin = float(rng.uniform(0.0, PAPER_DOMAIN_LENGTH))
        offsets = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(gaps[:-1])))
        centers = tuple(
            float(center) for center in np.mod(origin + offsets, PAPER_DOMAIN_LENGTH)
        )

    directions = tuple(
        int(direction)
        for direction in rng.permutation(
            np.concatenate(
                (
                    -np.ones(
                        crest_count - right_moving_count,
                        dtype=np.int8,
                    ),
                    np.ones(right_moving_count, dtype=np.int8),
                )
            )
        )
    )
    return TanakaSample(
        depth,
        tuple(
            TanakaCrest(alpha, center, direction)
            for alpha, center, direction in zip(alphas, centers, directions)
        ),
    )
