"""Subsample completed trajectories and build dataset rows."""

from __future__ import annotations

import math
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
        if simulation.decision.accepted != (trajectory is not None):
            raise RuntimeError(
                "trajectory data must exist exactly when a simulation is accepted"
            )
        if trajectory is not None:
            if family == "tanaka":
                indices = select_tanaka_times(
                    trajectory.eta,
                    length=length,
                    keep_samples=200,
                    alpha=0.5,
                    sigma_steps=50.0,
                ).indices
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

        residual = float(simulation.maximum_gl2_stage_residual)
        diagnostics: dict[str, float | None] = {
            "maximum_stage_residual": residual if math.isfinite(residual) else None,
        }
        if simulation.health_metrics is not None:
            diagnostics.update(
                minimum_internal_water_column=(
                    simulation.health_metrics.minimum_water_column
                ),
                initial_internal_hamiltonian=simulation.health_metrics.initial_hamiltonian,
                maximum_internal_hamiltonian_drift=(
                    simulation.health_metrics.maximum_relative_hamiltonian_drift
                ),
            )
        outcomes.append(
            SimulationOutcome(
                decision=simulation.decision,
                rows=rows,
                metrics=diagnostics,
            )
        )
    return tuple(outcomes)
