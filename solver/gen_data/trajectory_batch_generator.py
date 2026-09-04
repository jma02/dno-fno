"""Generate and validate one trajectory batch for each rollout family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from solver.gen_data.pipeline.artifact_io import json_text
from solver.gen_data.jonswap_tma import (
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
    finite_depth_angular_frequency,
)
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.trajectory_config import (
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_rollout import (
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.trajectory_subsampling import subsample_trajectories
from solver.gen_data.pipeline.writer import (
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)
from solver.gen_data.tanaka_initial_conditions import TanakaPotentialRadicandError
from solver.gen_data.tanaka_sampling import TanakaSample
from solver.gen_data.trajectory_family_adapters import (
    JonswapInitialStateDomainError,
    SampledSimulations,
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
    sample_benjamin_feir_simulations,
    sample_jonswap_tma_simulations,
    sample_tanaka_simulations,
)


JsonScalar: TypeAlias = str | int | float | bool | None

_FAMILY_IDS = {
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}


@dataclass(frozen=True)
class SimulationTimeGrid:
    """Intended and realized terminal times for one sampled simulation."""

    intended_terminal_time: float
    realized_terminal_time: float
    saved_times: np.ndarray


def floor_saved_time_grid(
    terminal_time: float,
    *,
    saved_dt: float,
    horizon_name: str,
) -> np.ndarray:
    """Return the saved-time prefix ending immediately before a horizon."""

    step_count = math.floor(terminal_time / saved_dt)
    while step_count * saved_dt > terminal_time:
        step_count -= 1
    while (step_count + 1) * saved_dt <= terminal_time:
        step_count += 1
    if step_count < 1:
        raise ValueError(f"{horizon_name} horizon is shorter than one saved step")
    saved_times = saved_dt * np.arange(step_count + 1, dtype=np.float64)
    realized = float(saved_times[-1])
    if not (realized <= terminal_time and terminal_time - realized < saved_dt):
        raise RuntimeError(
            f"{horizon_name} saved-grid horizon was not strictly floored"
        )
    return saved_times


def _construction_rejection(
    *,
    failure_reason: str,
    failure_record: Mapping[str, object],
    original_local_index: int,
    nonfinite_state: bool = False,
    nonpositive_water_height: bool = False,
) -> SimulationOutcome:
    metrics: dict[str, JsonScalar] = {
        "construction_status": "declared_outside_support",
        "construction_failure_reason": failure_reason,
        "original_local_index": original_local_index,
        "construction_failure_json": json_text(failure_record),
    }
    return SimulationOutcome(
        decision=SimulationCheckResult(
            accepted=False,
            nonfinite_state=nonfinite_state,
            nonpositive_water_height=nonpositive_water_height,
            outside_support=True,
        ),
        rows=None,
        metrics=metrics,
    )


def _finalize_outcome(
    outcome: SimulationOutcome,
    grid: SimulationTimeGrid,
    *,
    construction_metrics: Mapping[str, JsonScalar] | None = None,
) -> SimulationOutcome:
    """Attach rollout metadata to a simulation outcome."""

    return SimulationOutcome(
        decision=outcome.decision,
        rows=outcome.rows,
        metrics={
            **outcome.metrics,
            **(construction_metrics or {}),
            "intended_terminal_time": grid.intended_terminal_time,
            "realized_terminal_time": grid.realized_terminal_time,
            "saved_time_count": int(grid.saved_times.size),
        },
    )


def _check_batch_size(
    initial: TrajectoryInitialBatch,
    expected: int,
    family: TrajectoryFamily,
) -> None:
    if initial.eta0.shape[0] != expected:
        raise RuntimeError(f"{family} constructor returned the wrong batch size")


def _remapped_failure_record(
    failure: TanakaPotentialRadicandError | JonswapInitialStateDomainError,
    active_indices: tuple[int, ...],
) -> tuple[dict[str, object], tuple[int, ...]]:
    """Start rewriting constructor-subset indices into batch-plan indices."""

    positions = failure.invalid_simulation_indices
    if (
        not positions
        or positions != tuple(sorted(set(positions)))
        or positions[0] < 0
        or positions[-1] >= len(active_indices)
    ):
        raise RuntimeError("construction failure has invalid sub-batch indices")
    record = failure.to_json_record()
    original_indices = tuple(active_indices[index] for index in positions)
    record["constructor_subbatch_invalid_simulation_indices"] = list(positions)
    record["invalid_simulation_indices"] = list(original_indices)
    record["index_space"] = "original_batch_plan_local_index"
    return record, original_indices


def _remap_jonswap_failure(
    *,
    active_indices: tuple[int, ...],
    failure: JonswapInitialStateDomainError,
) -> tuple[dict[str, object], tuple[int, ...]]:
    """Return saved JONSWAP diagnostics using batch-plan indices."""

    record, original_indices = _remapped_failure_record(failure, active_indices)
    simulations = []
    for failed_simulation in failure.failures:
        subbatch_index = failed_simulation.local_simulation_index
        simulation = failed_simulation.to_json_record()
        simulation["constructor_subbatch_local_simulation_index"] = subbatch_index
        simulation["local_simulation_index"] = active_indices[subbatch_index]
        simulations.append(simulation)
    record["simulations"] = simulations
    return record, original_indices


def _remap_tanaka_failure(
    sampled: SampledSimulations[Any],
    *,
    active_indices: tuple[int, ...],
    failure: TanakaPotentialRadicandError,
) -> tuple[dict[str, object], tuple[int, ...]]:
    """Return saved Tanaka diagnostics using batch-plan indices."""

    samples = tuple(sampled.samples)
    if not all(isinstance(sample, TanakaSample) for sample in samples):
        raise TypeError("Tanaka failure remapping requires Tanaka samples")
    crest_counts = [len(sample.crests) for sample in samples]
    full_offsets = list(accumulate([0, *crest_counts[:-1]]))
    subbatch_offsets = list(
        accumulate([0, *(crest_counts[index] for index in active_indices[:-1])])
    )

    record, original_indices = _remapped_failure_record(failure, active_indices)
    components = []
    for failed_component in failure.components:
        subbatch_simulation = failed_component.local_simulation_index
        within = failed_component.component_within_simulation
        subbatch_global = failed_component.global_component_index
        if not 0 <= subbatch_simulation < len(active_indices):
            raise RuntimeError(
                "Tanaka component refers to an absent constructor-subset simulation"
            )
        original_simulation = active_indices[subbatch_simulation]
        if not 0 <= within < crest_counts[original_simulation]:
            raise RuntimeError(
                "Tanaka component refers to an absent crest within its simulation"
            )
        if subbatch_global != subbatch_offsets[subbatch_simulation] + within:
            raise RuntimeError(
                "Tanaka component global index is inconsistent with its simulation"
            )
        component = failed_component.to_json_record()
        component["constructor_subbatch_local_simulation_index"] = subbatch_simulation
        component["constructor_subbatch_global_component_index"] = subbatch_global
        component["local_simulation_index"] = original_simulation
        component["global_component_index"] = full_offsets[original_simulation] + within
        components.append(component)

    record["components"] = components
    return record, original_indices


def _construct_initial_states(
    sampled: SampledSimulations[Any],
    *,
    family: TrajectoryFamily,
    selected_local_indices: tuple[int, ...] | None,
) -> TrajectoryInitialBatch:
    if family == "benjamin_feir":
        if selected_local_indices is not None:
            raise ValueError(
                "selected construction is not supported for Benjamin--Feir"
            )
        return construct_benjamin_feir_trajectory_batch(sampled)
    construct = (
        construct_tanaka_trajectory_batch
        if family == "tanaka"
        else construct_jonswap_tma_trajectory_batch
    )
    return construct(sampled, selected_local_indices=selected_local_indices)


def integrate_and_subsample_trajectories(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[SimulationTimeGrid, ...],
    *,
    family: TrajectoryFamily,
    numerical: RolloutNumerics,
) -> tuple[SimulationOutcome, ...]:
    """Integrate constructed initial states and subsample the saved frames."""

    if len(time_grids) != initial.eta0.shape[0]:
        raise ValueError(f"{family} requires one time grid per initial condition")
    if any(
        grid.saved_times.size < (16 if family == "jonswap_tma" else 200)
        for grid in time_grids
    ):
        raise ValueError("trajectory horizon has fewer saved times than storage policy")
    simulations = execute_trajectory_batch(
        initial.eta0,
        initial.xi0,
        initial.depths,
        tuple(grid.saved_times for grid in time_grids),
        config=numerical,
    )
    outcomes = subsample_trajectories(
        simulations,
        initial.depths,
        family=family,
        length=numerical.length,
    )
    return tuple(
        _finalize_outcome(
            outcome,
            grid,
            construction_metrics=(
                initial.construction_metrics[index]
                if initial.construction_metrics
                else None
            ),
        )
        for index, (outcome, grid) in enumerate(zip(outcomes, time_grids))
    )


def generate_trajectory_batch(
    parameter_group_ids: tuple[str, ...],
    first_attempt_number: int,
    output_path: Path,
    *,
    dataset_split: DatasetSplit,
    family: TrajectoryFamily,
    numerical: RolloutNumerics,
    solver_batch_size: int | None = None,
) -> None:
    """Generate and save one trajectory batch."""

    if not parameter_group_ids:
        raise ValueError("trajectory attempt batches must not be empty")
    if family == "jonswap_tma" and solver_batch_size is None:
        raise ValueError("JONSWAP generation requires solver_batch_size")
    sampled: SampledSimulations[Any]
    if family == "tanaka":
        sampled = sample_tanaka_simulations(
            parameter_group_ids,
            dataset_split=dataset_split,
            first_attempt_number=first_attempt_number,
            config=numerical,
        )
        terminal_time = 200.0
        step_count = int(round(terminal_time / numerical.saved_dt))
        if step_count < 1 or not math.isclose(
            step_count * numerical.saved_dt,
            terminal_time,
            rel_tol=0.0,
            abs_tol=2.0e-13,
        ):
            raise ValueError(
                "fixed terminal time must lie exactly on the saved-time grid"
            )
        saved_times = numerical.saved_dt * np.arange(
            step_count + 1,
            dtype=np.float64,
        )
        grid = SimulationTimeGrid(
            intended_terminal_time=terminal_time,
            realized_terminal_time=float(saved_times[-1]),
            saved_times=saved_times,
        )
        time_grids = tuple(grid for _ in sampled.samples)
    elif family == "benjamin_feir":
        sampled = sample_benjamin_feir_simulations(
            parameter_group_ids,
            dataset_split=dataset_split,
            first_attempt_number=first_attempt_number,
            config=numerical,
        )
        grids = []
        for sample in sampled.samples:
            carrier_wavenumber = 2.0 * math.pi * sample.carrier_mode / numerical.length
            intended = (
                100 * 2.0 * math.pi / math.sqrt(numerical.gravity * carrier_wavenumber)
            )
            saved_times = floor_saved_time_grid(
                intended,
                saved_dt=numerical.saved_dt,
                horizon_name="Benjamin--Feir",
            )
            grids.append(
                SimulationTimeGrid(
                    intended_terminal_time=intended,
                    realized_terminal_time=float(saved_times[-1]),
                    saved_times=saved_times,
                )
            )
        time_grids = tuple(grids)
    else:
        sampled = sample_jonswap_tma_simulations(
            parameter_group_ids,
            dataset_split=dataset_split,
            first_attempt_number=first_attempt_number,
            config=numerical,
            quadrature_order=PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
        )
        grids = []
        for sample in sampled.samples:
            parameters = sample.parameters
            angular_frequency = finite_depth_angular_frequency(
                np.asarray([parameters.peak_wavenumber], dtype=np.float64),
                depth=parameters.depth,
                gravity=numerical.gravity,
            )[0]
            intended = 16 * 2.0 * math.pi / float(angular_frequency)
            saved_times = floor_saved_time_grid(
                intended,
                saved_dt=numerical.saved_dt,
                horizon_name="JONSWAP/TMA",
            )
            grids.append(
                SimulationTimeGrid(
                    intended_terminal_time=intended,
                    realized_terminal_time=float(saved_times[-1]),
                    saved_times=saved_times,
                )
            )
        time_grids = tuple(grids)

    batch_plan = build_batch_plan(
        sampled.parameter_group_ids,
        sampled.specification_records,
        family_id=_FAMILY_IDS[family],
        dataset_split=dataset_split,
    )
    valid_indices = tuple(range(len(sampled.parameter_group_ids)))
    rejected: dict[int, SimulationOutcome] = {}
    if family == "benjamin_feir":
        initial = _construct_initial_states(
            sampled,
            family=family,
            selected_local_indices=None,
        )
        _check_batch_size(initial, len(valid_indices), family)
    else:
        initial = None
        while valid_indices:
            try:
                initial = _construct_initial_states(
                    sampled,
                    family=family,
                    selected_local_indices=valid_indices,
                )
                break
            except Exception as error:
                if family == "tanaka":
                    if not isinstance(error, TanakaPotentialRadicandError):
                        raise
                    if not error.is_recoverable:
                        raise
                    positions = error.invalid_simulation_indices
                    record, original_indices = _remap_tanaka_failure(
                        sampled,
                        active_indices=valid_indices,
                        failure=error,
                    )
                    rejected.update(
                        (
                            index,
                            _construction_rejection(
                                failure_reason=error.reason,
                                failure_record=record,
                                original_local_index=index,
                            ),
                        )
                        for index in original_indices
                    )
                elif isinstance(error, JonswapInitialStateDomainError):
                    positions = error.invalid_simulation_indices
                    record, original_indices = _remap_jonswap_failure(
                        active_indices=valid_indices,
                        failure=error,
                    )
                    failures_by_position = {
                        failure.local_simulation_index: failure
                        for failure in error.failures
                    }
                    rejected.update(
                        (
                            original_index,
                            _construction_rejection(
                                failure_reason="invalid_initial_graph_state",
                                failure_record=record,
                                original_local_index=original_index,
                                nonfinite_state=(
                                    not failures_by_position[position].state_finite
                                ),
                                nonpositive_water_height=(
                                    failures_by_position[position].state_finite
                                ),
                            ),
                        )
                        for position, original_index in zip(
                            positions,
                            original_indices,
                            strict=True,
                        )
                    )
                else:
                    raise
                failed = set(positions)
                valid_indices = tuple(
                    index
                    for position, index in enumerate(valid_indices)
                    if position not in failed
                )
        if initial is not None:
            _check_batch_size(initial, len(valid_indices), family)
    ordered = {
        index: _finalize_outcome(outcome, time_grids[index])
        for index, outcome in rejected.items()
    }
    if initial is not None:
        valid_time_grids = tuple(time_grids[index] for index in valid_indices)
        if family == "jonswap_tma":
            assert solver_batch_size is not None
            from solver.gen_data.jonswap_horizon_generator import (
                integrate_and_subsample_jonswap,
            )

            produced = integrate_and_subsample_jonswap(
                initial,
                valid_time_grids,
                numerical=numerical,
                solver_batch_size=solver_batch_size,
            )
        else:
            produced = integrate_and_subsample_trajectories(
                initial,
                valid_time_grids,
                family=family,
                numerical=numerical,
            )
        ordered.update(zip(valid_indices, produced))
    if len(ordered) != len(time_grids):
        raise RuntimeError("every proposed trajectory simulation must have an outcome")
    outcomes = tuple(ordered[index] for index in range(len(time_grids)))
    commit_simulation_outcomes(output_path, batch_plan, outcomes)
