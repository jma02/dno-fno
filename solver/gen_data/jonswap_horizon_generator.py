"""Run JONSWAP simulations in small batches grouped by rollout length.

A batch may contain more simulations than fit in GPU memory. Sort them by
saved timestep count, solve them in smaller groups, then restore the original
order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import ClassVar

import numpy as np

from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.pipeline.trajectory_config import (
    JONSWAP_ADJUSTMENT_SCHEMA,
    JonswapNonlinearAdjustmentConfig,
)
from solver.gen_data.pipeline.trajectory_integration import (
    AdjustmentBatchIntegrator,
    integrate_adjustment_batch,
)
from solver.gen_data.pipeline.trajectory_rollout import (
    AdjustmentSimulationResult,
    execute_adjustment_batch,
)
from solver.gen_data.pipeline.writer import SimulationOutcome, JsonScalar
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch
from solver.gen_data.trajectory_batch_generator import (
    SimulationTimeGrid,
    TrajectoryBatchGenerator,
    floor_saved_time_grid,
)


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


def _accepted_adjustment_batch(
    initial: TrajectoryInitialBatch,
    simulations: Sequence[AdjustmentSimulationResult],
) -> tuple[tuple[int, ...], TrajectoryInitialBatch | None]:
    """Build the production batch from accepted full-band endpoints."""

    accepted_indices = tuple(
        index
        for index, simulation in enumerate(simulations)
        if simulation.decision.accepted
    )
    if not accepted_indices:
        return (), None

    endpoints: list[tuple[np.ndarray, np.ndarray]] = []
    for index in accepted_indices:
        eta = simulations[index].terminal_eta
        xi = simulations[index].terminal_xi
        if eta is None or xi is None:
            raise RuntimeError(
                "accepted nonlinear adjustment omitted a full-band endpoint"
            )
        endpoints.append((eta, xi))

    selected = _select_initial_batch(initial, accepted_indices)
    return accepted_indices, replace(
        selected,
        eta0=np.stack(tuple(eta for eta, _ in endpoints)),
        xi0=np.stack(tuple(xi for _, xi in endpoints)),
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
    adjustment_config: JonswapNonlinearAdjustmentConfig,
) -> dict[str, float | int | bool | str | None]:
    intended_terminal = adjustment_config.burn_peak_periods * peak_period
    decision = simulation.decision
    return {
        "nonlinear_adjustment_schema": JONSWAP_ADJUSTMENT_SCHEMA,
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
        "nonlinear_adjustment_ramp_order": adjustment_config.ramp_order,
        "nonlinear_adjustment_ramp_time": (
            adjustment_config.ramp_time_peak_periods * peak_period
        ),
        "nonlinear_adjustment_intended_terminal_time": intended_terminal,
        "nonlinear_adjustment_realized_terminal_time": float(saved_times[-1]),
        "nonlinear_adjustment_saved_time_count": int(saved_times.size),
        "nonlinear_adjustment_production_ramp": "disabled",
    }


def _failed_adjustment_outcome(
    simulation: AdjustmentSimulationResult,
    *,
    construction_metrics: Mapping[str, JsonScalar],
    peak_period: float,
    saved_times: np.ndarray,
    production_grid: SimulationTimeGrid,
    adjustment_config: JonswapNonlinearAdjustmentConfig,
) -> SimulationOutcome:
    return SimulationOutcome(
        decision=simulation.decision,
        rows=None,
        metrics={
            **construction_metrics,
            **_adjustment_metrics(
                simulation,
                peak_period=peak_period,
                saved_times=saved_times,
                adjustment_config=adjustment_config,
            ),
            "production_status": "not_run_adjustment_failed",
            "intended_terminal_time": production_grid.intended_terminal_time,
            "realized_terminal_time": production_grid.realized_terminal_time,
            "saved_time_count": int(production_grid.saved_times.size),
        },
    )


def _with_adjustment_metrics(
    outcome: SimulationOutcome,
    simulation: AdjustmentSimulationResult,
    *,
    peak_period: float,
    saved_times: np.ndarray,
    adjustment_config: JonswapNonlinearAdjustmentConfig,
) -> SimulationOutcome:
    return SimulationOutcome(
        decision=outcome.decision,
        rows=outcome.rows,
        metrics={
            **outcome.metrics,
            **_adjustment_metrics(
                simulation,
                peak_period=peak_period,
                saved_times=saved_times,
                adjustment_config=adjustment_config,
            ),
            "production_status": "completed",
        },
    )


@dataclass(frozen=True)
class HorizonBucketedJonswapBatchGenerator(TrajectoryBatchGenerator):
    """Solve one JONSWAP batch in groups with similar rollout lengths."""

    adjustment_rollout_integrator: AdjustmentBatchIntegrator = (
        integrate_adjustment_batch
    )
    runs_jonswap_adjustment: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.execution.family != "jonswap_tma":
            raise ValueError("horizon bucketing is implemented only for JONSWAP/TMA")
        if self.execution.jonswap_adjustment is None:
            raise ValueError(
                "horizon-bucketed JONSWAP execution requires adjustment settings"
            )
        if self.chunk_config.solver_batch_size is None:
            raise ValueError("JONSWAP generation requires solver_batch_size")

        super().__post_init__()

    def _produce_adjusted_group(
        self,
        initial: TrajectoryInitialBatch,
        time_grids: tuple[SimulationTimeGrid, ...],
        adjustment_config: JonswapNonlinearAdjustmentConfig,
    ) -> tuple[SimulationOutcome, ...]:
        """Adjust one horizon-near group, then run accepted simulations."""

        peak_periods = _peak_periods(
            initial,
            gravity=self.execution.numerical.gravity,
        )
        adjustment_grids = tuple(
            floor_saved_time_grid(
                adjustment_config.burn_peak_periods * peak_period,
                saved_dt=self.execution.numerical.saved_dt,
                horizon_name="nonlinear-adjustment",
            )
            for peak_period in peak_periods
        )
        adjustment_simulations = execute_adjustment_batch(
            initial.eta0,
            initial.xi0,
            initial.depths,
            adjustment_grids,
            nonlinear_ramp_times=(
                adjustment_config.ramp_time_peak_periods * peak_periods
            ),
            nonlinear_ramp_order=adjustment_config.ramp_order,
            config=self.execution.numerical,
            integrator=self.adjustment_rollout_integrator,
        )
        accepted_indices, adjusted_initial = _accepted_adjustment_batch(
            initial,
            adjustment_simulations,
        )
        production_by_index: dict[int, SimulationOutcome] = {}
        if adjusted_initial is not None:
            production = super()._integrate_and_subsample_jonswap(
                adjusted_initial,
                tuple(time_grids[index] for index in accepted_indices),
            )
            production_by_index = dict(zip(accepted_indices, production))

        return tuple(
            _with_adjustment_metrics(
                production_by_index[index],
                simulation,
                peak_period=float(peak_periods[index]),
                saved_times=adjustment_grids[index],
                adjustment_config=adjustment_config,
            )
            if simulation.decision.accepted
            else _failed_adjustment_outcome(
                simulation,
                construction_metrics=(
                    initial.construction_metrics[index]
                    if initial.construction_metrics
                    else {}
                ),
                peak_period=float(peak_periods[index]),
                saved_times=adjustment_grids[index],
                production_grid=time_grids[index],
                adjustment_config=adjustment_config,
            )
            for index, simulation in enumerate(adjustment_simulations)
        )

    def _integrate_and_subsample_jonswap(
        self,
        initial: TrajectoryInitialBatch,
        time_grids: tuple[SimulationTimeGrid, ...],
    ) -> tuple[SimulationOutcome, ...]:
        if len(time_grids) != initial.eta0.shape[0]:
            raise ValueError("JONSWAP/TMA requires one time grid per initial condition")
        minimum_stored = self.execution.frame_selection.jonswap_tma_count
        adjustment_config = self.execution.jonswap_adjustment
        assert adjustment_config is not None
        if any(grid.saved_times.size < minimum_stored for grid in time_grids):
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )

        solver_batch_size = self.chunk_config.solver_batch_size
        assert solver_batch_size is not None
        groups = horizon_sorted_groups(
            time_grids,
            solver_batch_size=solver_batch_size,
        )
        ordered_outcomes: list[SimulationOutcome | None] = [None] * len(time_grids)
        for indices in groups:
            outcomes = self._produce_adjusted_group(
                _select_initial_batch(initial, indices),
                tuple(time_grids[index] for index in indices),
                adjustment_config,
            )
            for index, outcome in zip(indices, outcomes):
                ordered_outcomes[index] = outcome
        if any(outcome is None for outcome in ordered_outcomes):
            raise RuntimeError("every JONSWAP simulation must produce an outcome")
        return tuple(outcome for outcome in ordered_outcomes if outcome is not None)
