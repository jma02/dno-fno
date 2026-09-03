"""Subsample completed trajectories and build dataset rows."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_rollout import (
    TrajectorySimulationResult,
    TrajectorySamples,
)
from solver.gen_data.pipeline.time_selection import (
    select_tanaka_times,
    select_uniform_times,
)
from solver.gen_data.pipeline.trajectory_config import (
    TrajectoryFamily,
    TrajectoryFrameSelectionConfig,
)
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
)


FloatArray: TypeAlias = NDArray[np.float64]


def _select_subsample_time_indices(
    trajectory: TrajectorySamples,
    *,
    family: TrajectoryFamily,
    length: float,
    frame_selection: TrajectoryFrameSelectionConfig,
) -> NDArray[np.int32]:
    if family == "tanaka":
        return select_tanaka_times(
            trajectory.eta,
            length=length,
            keep_samples=frame_selection.tanaka_count,
            alpha=frame_selection.tanaka_alpha,
            sigma_steps=frame_selection.tanaka_sigma_steps,
        ).indices
    subsample_count = (
        frame_selection.benjamin_feir_count
        if family == "benjamin_feir"
        else frame_selection.jonswap_tma_count
    )
    return select_uniform_times(
        trajectory.times.size,
        keep_samples=subsample_count,
    )


def subsample_trajectories(
    simulations: tuple[TrajectorySimulationResult, ...],
    depths: FloatArray,
    *,
    family: TrajectoryFamily,
    length: float,
    frame_selection: TrajectoryFrameSelectionConfig,
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
            indices = _select_subsample_time_indices(
                trajectory,
                family=family,
                length=length,
                frame_selection=frame_selection,
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
