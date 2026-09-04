"""Subsample completed trajectories and build dataset rows."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_rollout import TrajectorySamples
from solver.gen_data.pipeline.time_selection import (
    select_tanaka_times,
    select_uniform_times,
)
from solver.gen_data.pipeline.trajectory_config import TrajectoryFamily
from solver.gen_data.pipeline.types import SimulationRows


FloatArray: TypeAlias = NDArray[np.float64]


def subsample_trajectories(
    simulations: tuple[TrajectorySamples | None, ...],
    depths: FloatArray,
    *,
    family: TrajectoryFamily,
    length: float,
) -> tuple[SimulationRows | None, ...]:
    """Subsample accepted trajectories into dataset rows."""

    rows_by_simulation: list[SimulationRows | None] = []
    for trajectory, depth in zip(simulations, depths, strict=True):
        rows = None
        if trajectory is not None:
            if family == "tanaka":
                indices = select_tanaka_times(
                    trajectory.eta,
                    length=length,
                )
            else:
                indices = select_uniform_times(
                    trajectory.times.size,
                    keep_samples=200 if family == "benjamin_feir" else 16,
                )
            rows = SimulationRows(
                eta=trajectory.eta[indices],
                xi=trajectory.xi[indices],
                gxi=trajectory.gxi[indices],
                depth=float(depth),
                time=trajectory.times[indices],
            )

        rows_by_simulation.append(rows)
    return tuple(rows_by_simulation)
