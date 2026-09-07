"""Run JONSWAP simulations in small batches grouped by rollout length."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.pipeline.trajectory_rollout import (
    execute_adjustment_batch,
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.trajectory_subsampling import subsample_trajectories
from solver.gen_data.pipeline.time_selection import floor_saved_time_grid
from solver.gen_data.pipeline.types import SimulationRows
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch


FloatArray: TypeAlias = NDArray[np.float64]

_ADJUSTMENT_RAMP_ORDER = 4
_ADJUSTMENT_RAMP_PEAK_PERIODS = 10
_ADJUSTMENT_BURN_PEAK_PERIODS = 20


def integrate_and_subsample_jonswap(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[FloatArray, ...],
    peak_periods: FloatArray,
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationRows | None, ...]:
    """Run JONSWAP adjustment and production in similar-length groups."""

    ordered_indices = sorted(
        range(len(time_grids)), key=lambda index: time_grids[index].size
    )
    rows_by_simulation: list[SimulationRows | None] = [None] * len(time_grids)
    for start in range(0, len(ordered_indices), solver_batch_size):
        indices = ordered_indices[start : start + solver_batch_size]
        group_peak_periods = peak_periods[indices]
        adjustment_grids = tuple(
            floor_saved_time_grid(
                _ADJUSTMENT_BURN_PEAK_PERIODS * peak_period,
                saved_dt=numerical.saved_dt,
            )
            for peak_period in group_peak_periods
        )
        adjustment_simulations = execute_adjustment_batch(
            initial.eta0[indices],
            initial.xi0[indices],
            initial.depths[indices],
            adjustment_grids,
            nonlinear_ramp_times=(_ADJUSTMENT_RAMP_PEAK_PERIODS * group_peak_periods),
            nonlinear_ramp_order=_ADJUSTMENT_RAMP_ORDER,
            config=numerical,
        )
        accepted = tuple(
            (index, endpoint)
            for index, endpoint in zip(indices, adjustment_simulations, strict=True)
            if endpoint is not None
        )
        if not accepted:
            continue
        accepted_indices = [index for index, _ in accepted]
        depths = initial.depths[accepted_indices]
        subsampled_rows = subsample_trajectories(
            execute_trajectory_batch(
                np.stack(tuple(endpoint[0] for _, endpoint in accepted)),
                np.stack(tuple(endpoint[1] for _, endpoint in accepted)),
                depths,
                tuple(time_grids[index] for index in accepted_indices),
                config=numerical,
            ),
            depths,
            family="jonswap_tma",
            length=numerical.length,
        )
        for index, rows in zip(accepted_indices, subsampled_rows, strict=True):
            rows_by_simulation[index] = rows
    return tuple(rows_by_simulation)
