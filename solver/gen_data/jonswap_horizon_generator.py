"""Run JONSWAP simulations in small batches grouped by rollout length."""

from __future__ import annotations

from collections.abc import Sequence
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


def horizon_sorted_groups(
    time_grids: Sequence[FloatArray],
    *,
    solver_batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Group simulation indices by saved-time count without changing ties."""

    if solver_batch_size <= 0:
        raise ValueError("solver_batch_size must be positive")
    ordered = tuple(
        sorted(
            range(len(time_grids)),
            key=lambda index: (int(time_grids[index].size), index),
        )
    )
    return tuple(
        ordered[start : start + solver_batch_size]
        for start in range(0, len(ordered), solver_batch_size)
    )


def _select_initial_batch(
    initial: TrajectoryInitialBatch,
    indices: Sequence[int],
) -> TrajectoryInitialBatch:
    selected = np.asarray(indices, dtype=np.int64)
    return TrajectoryInitialBatch(
        np.take(initial.eta0, selected, axis=0),
        np.take(initial.xi0, selected, axis=0),
        np.take(initial.depths, selected, axis=0),
    )


def integrate_and_subsample_jonswap(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[FloatArray, ...],
    peak_periods: FloatArray,
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationRows | None, ...]:
    """Run JONSWAP adjustment and production in similar-length groups."""

    groups = horizon_sorted_groups(time_grids, solver_batch_size=solver_batch_size)
    ordered_rows: list[SimulationRows | None] = [None] * len(time_grids)
    for indices in groups:
        group_initial = _select_initial_batch(initial, indices)
        group_time_grids = tuple(time_grids[index] for index in indices)
        group_peak_periods = np.take(peak_periods, indices)
        adjustment_grids = tuple(
            floor_saved_time_grid(
                _ADJUSTMENT_BURN_PEAK_PERIODS * peak_period,
                saved_dt=numerical.saved_dt,
                horizon_name="nonlinear-adjustment",
            )
            for peak_period in group_peak_periods
        )
        adjustment_simulations = execute_adjustment_batch(
            group_initial.eta0,
            group_initial.xi0,
            group_initial.depths,
            adjustment_grids,
            nonlinear_ramp_times=(_ADJUSTMENT_RAMP_PEAK_PERIODS * group_peak_periods),
            nonlinear_ramp_order=_ADJUSTMENT_RAMP_ORDER,
            config=numerical,
        )
        accepted_indices = tuple(
            index
            for index, endpoint in enumerate(adjustment_simulations)
            if endpoint is not None
        )
        endpoints = tuple(
            endpoint for endpoint in adjustment_simulations if endpoint is not None
        )
        production_by_index: dict[int, SimulationRows | None] = {}
        if accepted_indices:
            selected = _select_initial_batch(group_initial, accepted_indices)
            endpoint_batch = TrajectoryInitialBatch(
                np.stack(tuple(endpoint[0] for endpoint in endpoints)),
                np.stack(tuple(endpoint[1] for endpoint in endpoints)),
                selected.depths,
            )
            production = subsample_trajectories(
                execute_trajectory_batch(
                    endpoint_batch.eta0,
                    endpoint_batch.xi0,
                    endpoint_batch.depths,
                    tuple(group_time_grids[index] for index in accepted_indices),
                    config=numerical,
                ),
                endpoint_batch.depths,
                family="jonswap_tma",
                length=numerical.length,
            )
            production_by_index = dict(zip(accepted_indices, production, strict=True))
        group_rows = tuple(
            production_by_index.get(index)
            for index in range(len(adjustment_simulations))
        )
        for index, rows in zip(indices, group_rows, strict=True):
            ordered_rows[index] = rows
    return tuple(ordered_rows)
