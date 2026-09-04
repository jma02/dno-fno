"""Run JONSWAP simulations in small batches grouped by rollout length."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.pipeline.trajectory_rollout import (
    AdjustmentSimulationResult,
    execute_adjustment_batch,
)
from solver.gen_data.pipeline.writer import SimulationOutcome
from solver.gen_data.trajectory_batch_generator import (
    floor_saved_time_grid,
    integrate_and_subsample_trajectories,
)
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


def _adjustment_metrics(
    simulation: AdjustmentSimulationResult,
    *,
    peak_period: float,
    saved_times: FloatArray,
) -> dict[str, float | int | bool | None]:
    decision = simulation.decision
    return {
        "nonlinear_adjustment_accepted": decision.accepted,
        "nonlinear_adjustment_complete_admissible_handoff": (
            not decision.incomplete_trajectory
        ),
        "nonlinear_adjustment_state_finite": not decision.nonfinite_state,
        "nonlinear_adjustment_positive_water_column": (
            not decision.nonpositive_water_height
            if simulation.minimum_water_column is not None
            else None
        ),
        "nonlinear_adjustment_all_stages_solved": not decision.integration_failure,
        "nonlinear_adjustment_maximum_stage_residual": (
            simulation.maximum_gl2_stage_residual
            if math.isfinite(simulation.maximum_gl2_stage_residual)
            else None
        ),
        "nonlinear_adjustment_minimum_water_column": simulation.minimum_water_column,
        "nonlinear_adjustment_ramp_order": _ADJUSTMENT_RAMP_ORDER,
        "nonlinear_adjustment_ramp_time": _ADJUSTMENT_RAMP_PEAK_PERIODS
        * peak_period,
        "nonlinear_adjustment_intended_terminal_time": (
            _ADJUSTMENT_BURN_PEAK_PERIODS * peak_period
        ),
        "nonlinear_adjustment_realized_terminal_time": float(saved_times[-1]),
        "nonlinear_adjustment_saved_time_count": int(saved_times.size),
    }


def integrate_and_subsample_jonswap(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[FloatArray, ...],
    peak_periods: FloatArray,
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationOutcome, ...]:
    """Run JONSWAP adjustment and production in similar-length groups."""

    groups = horizon_sorted_groups(time_grids, solver_batch_size=solver_batch_size)
    ordered_outcomes: list[SimulationOutcome | None] = [None] * len(time_grids)
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
            nonlinear_ramp_times=(
                _ADJUSTMENT_RAMP_PEAK_PERIODS * group_peak_periods
            ),
            nonlinear_ramp_order=_ADJUSTMENT_RAMP_ORDER,
            config=numerical,
        )
        accepted_indices = tuple(
            index
            for index, simulation in enumerate(adjustment_simulations)
            if simulation.decision.accepted
        )
        production_by_index: dict[int, SimulationOutcome] = {}
        if accepted_indices:
            endpoints = tuple(
                (
                    adjustment_simulations[index].terminal_eta,
                    adjustment_simulations[index].terminal_xi,
                )
                for index in accepted_indices
            )
            if any(eta is None or xi is None for eta, xi in endpoints):
                raise RuntimeError(
                    "accepted nonlinear adjustment omitted a full-band endpoint"
                )
            selected = _select_initial_batch(group_initial, accepted_indices)
            production = integrate_and_subsample_trajectories(
                TrajectoryInitialBatch(
                    np.stack(tuple(eta for eta, _ in endpoints if eta is not None)),
                    np.stack(tuple(xi for _, xi in endpoints if xi is not None)),
                    selected.depths,
                ),
                tuple(group_time_grids[index] for index in accepted_indices),
                family="jonswap_tma",
                numerical=numerical,
            )
            production_by_index = dict(
                zip(accepted_indices, production, strict=True)
            )
        outcomes = tuple(
            SimulationOutcome(
                decision=production_by_index[index].decision,
                rows=production_by_index[index].rows,
                metrics={
                    **production_by_index[index].metrics,
                    **_adjustment_metrics(
                        simulation,
                        peak_period=float(group_peak_periods[index]),
                        saved_times=adjustment_grids[index],
                    ),
                },
            )
            if simulation.decision.accepted
            else SimulationOutcome(
                decision=simulation.decision,
                rows=None,
                metrics=_adjustment_metrics(
                    simulation,
                    peak_period=float(group_peak_periods[index]),
                    saved_times=adjustment_grids[index],
                ),
            )
            for index, simulation in enumerate(adjustment_simulations)
        )
        for index, outcome in zip(indices, outcomes, strict=True):
            ordered_outcomes[index] = outcome
    if any(outcome is None for outcome in ordered_outcomes):
        raise RuntimeError("every JONSWAP simulation must produce an outcome")
    return tuple(outcome for outcome in ordered_outcomes if outcome is not None)
