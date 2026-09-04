"""Run JONSWAP simulations in small batches grouped by rollout length.

A batch may contain more simulations than fit in GPU memory. Sort them by
saved timestep count, solve them in smaller groups, then restore the original
order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math

import numpy as np

from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.pipeline.trajectory_rollout import (
    AdjustmentSimulationResult,
    execute_adjustment_batch,
)
from solver.gen_data.pipeline.writer import SimulationOutcome
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch
from solver.gen_data.trajectory_batch_generator import (
    SimulationTimeGrid,
    floor_saved_time_grid,
    integrate_and_subsample_trajectories,
)


_ADJUSTMENT_RAMP_ORDER = 4
_ADJUSTMENT_RAMP_PEAK_PERIODS = 10
_ADJUSTMENT_BURN_PEAK_PERIODS = 20


def horizon_sorted_groups(
    grids: Sequence[SimulationTimeGrid],
    *,
    solver_batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition simulation indices into stable horizon-near solver groups."""

    if (
        isinstance(solver_batch_size, bool)
        or not isinstance(solver_batch_size, int)
        or solver_batch_size <= 0
    ):
        raise ValueError("solver_batch_size must be a positive integer")
    if not grids:
        raise ValueError("grids must not be empty")

    ordered = tuple(
        sorted(
            range(len(grids)),
            key=lambda index: (int(grids[index].saved_times.size), index),
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
        eta0=np.take(initial.eta0, selected, axis=0),
        xi0=np.take(initial.xi0, selected, axis=0),
        depths=np.take(initial.depths, selected, axis=0),
        specification_records=tuple(
            initial.specification_records[index] for index in indices
        ),
        construction_metrics=(
            tuple(initial.construction_metrics[index] for index in indices)
            if initial.construction_metrics
            else ()
        ),
    )


def _peak_periods(
    initial: TrajectoryInitialBatch,
    *,
    gravity: float,
) -> np.ndarray:
    """Compute each simulation's finite-depth peak period."""

    depths = np.asarray(initial.depths, dtype=np.float64)
    peak_wavenumbers = np.asarray(
        [record["peak_wavenumber"] for record in initial.specification_records],
        dtype=np.float64,
    )
    angular_frequencies = np.asarray(
        [
            finite_depth_angular_frequency(
                np.asarray([peak_wavenumber], dtype=np.float64),
                depth=float(depth),
                gravity=gravity,
            )[0]
            for peak_wavenumber, depth in zip(peak_wavenumbers, depths)
        ],
        dtype=np.float64,
    )
    periods = 2.0 * math.pi / np.asarray(angular_frequencies, dtype=np.float64)
    if periods.shape != depths.shape or not np.isfinite(periods).all():
        raise RuntimeError("failed to compute finite JONSWAP peak periods")
    return periods


def _adjustment_metrics(
    simulation: AdjustmentSimulationResult,
    *,
    peak_period: float,
    saved_times: np.ndarray,
) -> dict[str, float | int | bool | str | None]:
    intended_terminal = _ADJUSTMENT_BURN_PEAK_PERIODS * peak_period
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
        "nonlinear_adjustment_ramp_time": (_ADJUSTMENT_RAMP_PEAK_PERIODS * peak_period),
        "nonlinear_adjustment_intended_terminal_time": intended_terminal,
        "nonlinear_adjustment_realized_terminal_time": float(saved_times[-1]),
        "nonlinear_adjustment_saved_time_count": int(saved_times.size),
    }


def integrate_and_subsample_jonswap(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[SimulationTimeGrid, ...],
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationOutcome, ...]:
    """Run JONSWAP adjustment and production in horizon-near groups."""

    if len(time_grids) != initial.eta0.shape[0]:
        raise ValueError("JONSWAP/TMA requires one time grid per initial condition")
    if any(grid.saved_times.size < 16 for grid in time_grids):
        raise ValueError("trajectory horizon has fewer saved times than storage policy")

    groups = horizon_sorted_groups(time_grids, solver_batch_size=solver_batch_size)
    ordered_outcomes: list[SimulationOutcome | None] = [None] * len(time_grids)
    for indices in groups:
        group_initial = _select_initial_batch(initial, indices)
        group_time_grids = tuple(time_grids[index] for index in indices)
        peak_periods = _peak_periods(group_initial, gravity=numerical.gravity)
        adjustment_grids = tuple(
            floor_saved_time_grid(
                _ADJUSTMENT_BURN_PEAK_PERIODS * peak_period,
                saved_dt=numerical.saved_dt,
                horizon_name="nonlinear-adjustment",
            )
            for peak_period in peak_periods
        )
        adjustment_simulations = execute_adjustment_batch(
            group_initial.eta0,
            group_initial.xi0,
            group_initial.depths,
            adjustment_grids,
            nonlinear_ramp_times=_ADJUSTMENT_RAMP_PEAK_PERIODS * peak_periods,
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
            adjusted_initial = replace(
                selected,
                eta0=np.stack(tuple(eta for eta, _ in endpoints if eta is not None)),
                xi0=np.stack(tuple(xi for _, xi in endpoints if xi is not None)),
            )
            production = integrate_and_subsample_trajectories(
                adjusted_initial,
                tuple(group_time_grids[index] for index in accepted_indices),
                family="jonswap_tma",
                numerical=numerical,
            )
            production_by_index = dict(zip(accepted_indices, production))
        outcomes = tuple(
            SimulationOutcome(
                decision=production_by_index[index].decision,
                rows=production_by_index[index].rows,
                metrics={
                    **production_by_index[index].metrics,
                    **_adjustment_metrics(
                        simulation,
                        peak_period=float(peak_periods[index]),
                        saved_times=adjustment_grids[index],
                    ),
                    "production_status": "completed",
                },
            )
            if simulation.decision.accepted
            else SimulationOutcome(
                decision=simulation.decision,
                rows=None,
                metrics={
                    **(
                        group_initial.construction_metrics[index]
                        if group_initial.construction_metrics
                        else {}
                    ),
                    **_adjustment_metrics(
                        simulation,
                        peak_period=float(peak_periods[index]),
                        saved_times=adjustment_grids[index],
                    ),
                    "production_status": "not_run_adjustment_failed",
                    "intended_terminal_time": (
                        group_time_grids[index].intended_terminal_time
                    ),
                    "realized_terminal_time": (
                        group_time_grids[index].realized_terminal_time
                    ),
                    "saved_time_count": int(group_time_grids[index].saved_times.size),
                },
            )
            for index, simulation in enumerate(adjustment_simulations)
        )
        for index, outcome in zip(indices, outcomes):
            ordered_outcomes[index] = outcome
    if any(outcome is None for outcome in ordered_outcomes):
        raise RuntimeError("every JONSWAP simulation must produce an outcome")
    return tuple(outcome for outcome in ordered_outcomes if outcome is not None)
