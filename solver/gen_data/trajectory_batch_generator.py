"""Generate and validate one trajectory batch for each rollout family."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.benjamin_feir_sampling import sample_benjamin_feir_simulation
from solver.gen_data.jonswap_tma import (
    ResolvedBand,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import sample_jonswap_tma_simulation
from solver.gen_data.jonswap_horizon_generator import integrate_and_subsample_jonswap
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.trajectory_config import (
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_rollout import execute_trajectory_batch
from solver.gen_data.pipeline.trajectory_subsampling import subsample_trajectories
from solver.gen_data.pipeline.time_selection import floor_saved_time_grid
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId
from solver.gen_data.pipeline.writer import (
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)
from solver.gen_data.tanaka_initial_conditions import TanakaPotentialRadicandError
from solver.gen_data.tanaka_sampling import sample_tanaka_simulation
from solver.gen_data.trajectory_family_adapters import (
    JonswapInitialStateDomainError,
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
)


FloatArray: TypeAlias = NDArray[np.float64]


def _construction_rejection(
    *,
    nonfinite_state: bool = False,
    nonpositive_water_height: bool = False,
) -> SimulationOutcome:
    return SimulationOutcome(
        decision=SimulationCheckResult(
            accepted=False,
            nonfinite_state=nonfinite_state,
            nonpositive_water_height=nonpositive_water_height,
            outside_support=True,
        ),
        rows=None,
        metrics={},
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
    """Sample, simulate, and save one trajectory batch."""

    valid_indices = tuple(range(len(parameter_group_ids)))
    rejected: dict[int, SimulationOutcome] = {}
    peak_periods: FloatArray | None = None

    if family == "tanaka":
        tanaka_samples = tuple(
            sample_tanaka_simulation(
                parameter_group_id,
                dataset_split=dataset_split,
                attempt_number=first_attempt_number + offset,
            )
            for offset, parameter_group_id in enumerate(parameter_group_ids)
        )
        specifications = tuple(
            {
                "depth": sample.depth,
                "crests": [
                    {
                        "alpha": crest.alpha,
                        "center": crest.center,
                        "direction": crest.direction,
                    }
                    for crest in sample.crests
                ],
            }
            for sample in tanaka_samples
        )
        saved_times = numerical.saved_dt * np.arange(
            int(round(200.0 / numerical.saved_dt)) + 1,
            dtype=np.float64,
        )
        time_grids = (saved_times,) * len(tanaka_samples)
        try:
            initial: TrajectoryInitialBatch | None = construct_tanaka_trajectory_batch(
                tanaka_samples, numerical
            )
        except TanakaPotentialRadicandError as error:
            failed = set(error.invalid_simulation_indices)
            rejected.update(
                (
                    index,
                    _construction_rejection(),
                )
                for index in error.invalid_simulation_indices
            )
            valid_indices = tuple(
                index for index in valid_indices if index not in failed
            )
            initial = (
                construct_tanaka_trajectory_batch(
                    tuple(tanaka_samples[index] for index in valid_indices),
                    numerical,
                )
                if valid_indices
                else None
            )
    elif family == "benjamin_feir":
        benjamin_feir_samples = tuple(
            sample_benjamin_feir_simulation(
                parameter_group_id,
                dataset_split=dataset_split,
                attempt_number=first_attempt_number + offset,
            )
            for offset, parameter_group_id in enumerate(parameter_group_ids)
        )
        specifications = tuple(
            {
                "carrier_mode": sample.carrier_mode,
                "sideband_offset": sample.sideband_offset,
                "carrier_steepness": sample.carrier_steepness,
                "perturbation_ratio": sample.perturbation_ratio,
                "translation": sample.translation,
            }
            for sample in benjamin_feir_samples
        )
        time_grids_list: list[FloatArray] = []
        for sample in benjamin_feir_samples:
            carrier_wavenumber = 2.0 * math.pi * sample.carrier_mode / numerical.length
            time_grids_list.append(
                floor_saved_time_grid(
                    100.0
                    * 2.0
                    * math.pi
                    / math.sqrt(numerical.gravity * carrier_wavenumber),
                    saved_dt=numerical.saved_dt,
                    horizon_name="Benjamin--Feir",
                )
            )
        time_grids = tuple(time_grids_list)
        initial = construct_benjamin_feir_trajectory_batch(
            benjamin_feir_samples,
            numerical,
        )
    else:
        if solver_batch_size is None:
            raise ValueError("JONSWAP generation requires solver_batch_size")
        maximum_wavenumber = numerical.target_maximum_wavenumber
        band = ResolvedBand(
            numerical.length,
            maximum_wavenumber,
        )
        jonswap_samples = tuple(
            sample_jonswap_tma_simulation(
                parameter_group_id,
                dataset_split=dataset_split,
                attempt_number=first_attempt_number + offset,
                band=band,
            )
            for offset, parameter_group_id in enumerate(parameter_group_ids)
        )
        specifications = tuple(
            {
                "depth": sample.parameters.depth,
                "significant_height": sample.parameters.significant_height,
                "peak_wavenumber": sample.parameters.peak_wavenumber,
                "peak_enhancement": sample.parameters.peak_enhancement,
                "right_moving_fraction": sample.parameters.right_moving_fraction,
                "phase_right": sample.phase_right.tolist(),
                "phase_left": sample.phase_left.tolist(),
            }
            for sample in jonswap_samples
        )
        peak_periods = np.asarray(
            [
                2.0
                * math.pi
                / float(
                    finite_depth_angular_frequency(
                        np.asarray(
                            [sample.parameters.peak_wavenumber], dtype=np.float64
                        ),
                        depth=sample.parameters.depth,
                        gravity=numerical.gravity,
                    )[0]
                )
                for sample in jonswap_samples
            ],
            dtype=np.float64,
        )
        time_grids = tuple(
            floor_saved_time_grid(
                16.0 * peak_period,
                saved_dt=numerical.saved_dt,
                horizon_name="JONSWAP/TMA",
            )
            for peak_period in peak_periods
        )
        try:
            initial = construct_jonswap_tma_trajectory_batch(
                jonswap_samples,
                numerical,
                band=band,
            )
        except JonswapInitialStateDomainError as error:
            failed = set(error.invalid_simulation_indices)
            rejected.update(
                (
                    index,
                    _construction_rejection(
                        nonfinite_state=nonfinite_state,
                        nonpositive_water_height=nonpositive_water_height,
                    ),
                )
                for index, nonfinite_state, nonpositive_water_height in zip(
                    error.invalid_simulation_indices,
                    error.nonfinite_state_flags,
                    error.nonpositive_water_height_flags,
                    strict=True,
                )
            )
            valid_indices = tuple(
                index for index in valid_indices if index not in failed
            )
            initial = (
                construct_jonswap_tma_trajectory_batch(
                    tuple(jonswap_samples[index] for index in valid_indices),
                    numerical,
                    band=band,
                )
                if valid_indices
                else None
            )

    batch_plan = build_batch_plan(
        parameter_group_ids,
        specifications,
        family_id=PhysicalFamilyId[family.upper()],
        dataset_split=dataset_split,
    )
    produced: tuple[SimulationOutcome, ...] = ()
    if initial is not None:
        valid_time_grids = tuple(time_grids[index] for index in valid_indices)
        if family == "jonswap_tma":
            assert solver_batch_size is not None and peak_periods is not None
            produced = integrate_and_subsample_jonswap(
                initial,
                valid_time_grids,
                np.asarray(peak_periods[list(valid_indices)], dtype=np.float64),
                numerical=numerical,
                solver_batch_size=solver_batch_size,
            )
        else:
            produced = subsample_trajectories(
                execute_trajectory_batch(
                    initial.eta0,
                    initial.xi0,
                    initial.depths,
                    valid_time_grids,
                    config=numerical,
                ),
                initial.depths,
                family=family,
                length=numerical.length,
            )
    outcomes_by_index = {
        **rejected,
        **dict(zip(valid_indices, produced, strict=True)),
    }
    commit_simulation_outcomes(
        output_path,
        batch_plan,
        tuple(outcomes_by_index[index] for index in range(len(parameter_group_ids))),
    )
