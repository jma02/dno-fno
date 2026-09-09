"""Numerical settings for paper-dataset trajectory rollouts."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, TypeAlias


TrajectoryFamily: TypeAlias = Literal[
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]

RolloutNumerics = NamedTuple(
    "RolloutNumerics",
    [
        ("nx", int),
        ("target_nx", int),
        ("length", float),
        ("gravity", float),
        # Lower-cost approximation used repeatedly while advancing a rollout.
        ("integration_dno_order", int),
        # Re-evaluate saved states at higher accuracy instead of reusing the
        # lower-order DNO values computed while advancing the rollout.
        ("label_dno_order", int),
        ("pad_factor", int),
        ("maximum_wavenumber", float),
        ("target_maximum_wavenumber", float),
        ("saved_dt", float),
        ("substeps_per_saved_frame", int),
        ("gl2_residual_tolerance", float),
        ("gl2_iteration_cap", int),
        ("internal_hamiltonian_drift_threshold", float | None),
    ],
)


_BASE_ROLLOUT_NUMERICS = RolloutNumerics(
    nx=1024,
    target_nx=1024,
    length=2.0 * math.pi,
    gravity=1.0,
    integration_dno_order=4,
    label_dno_order=6,
    pad_factor=8,
    maximum_wavenumber=256.0,
    target_maximum_wavenumber=128.0,
    saved_dt=0.08,
    substeps_per_saved_frame=8,
    gl2_residual_tolerance=1.0e-8,
    gl2_iteration_cap=4,
    internal_hamiltonian_drift_threshold=1.0e-3,
)

PAPER_ROLLOUT_NUMERICS: dict[TrajectoryFamily, RolloutNumerics] = {
    "tanaka": _BASE_ROLLOUT_NUMERICS._replace(
        integration_dno_order=6,
        internal_hamiltonian_drift_threshold=None,
    ),
    "benjamin_feir": _BASE_ROLLOUT_NUMERICS,
    "jonswap_tma": _BASE_ROLLOUT_NUMERICS._replace(
        nx=2048,
        maximum_wavenumber=704.0,
        gl2_iteration_cap=5,
    ),
}
