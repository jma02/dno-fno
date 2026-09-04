"""Subsample completed trajectories and build dataset rows."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_rollout import (
    TrajectorySimulationResult,
)
from solver.gen_data.pipeline.time_selection import (
    select_tanaka_times,
    select_uniform_times,
)
from solver.gen_data.pipeline.trajectory_config import TrajectoryFamily
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
)


FloatArray: TypeAlias = NDArray[np.float64]


def subsample_trajectories(
    simulations: tuple[TrajectorySimulationResult, ...],
    depths: FloatArray,
    *,
    family: TrajectoryFamily,
    length: float,
) -> tuple[SimulationOutcome, ...]:
    """Subsample accepted trajectories into dataset rows."""

    outcomes: list[SimulationOutcome] = []
    for simulation, depth in zip(simulations, depths, strict=True):
        rows = None
        trajectory = simulation.trajectory
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
            rows = AcceptedSimulationRows(
                eta=trajectory.eta[indices],
                xi=trajectory.xi[indices],
                gxi=trajectory.gxi[indices],
                depth=float(depth),
                time=trajectory.times[indices],
            )

        outcomes.append(
            SimulationOutcome(
                decision=simulation.decision,
                rows=rows,
                metrics={},
            )
        )
    return tuple(outcomes)
