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
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.trajectory_config import (
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_rollout import execute_trajectory_batch
from solver.gen_data.pipeline.trajectory_subsampling import subsample_trajectories
from solver.gen_data.pipeline.time_selection import floor_saved_time_grid
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)
from solver.gen_data.tanaka_initial_conditions import TanakaPotentialRadicandError
from solver.gen_data.tanaka_sampling import sample_tanaka_simulation
from solver.gen_data.trajectory_family_adapters import (
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
)


FloatArray: TypeAlias = NDArray[np.float64]


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
            )
            for peak_period in peak_periods
        )
        initial, valid_indices = construct_jonswap_tma_trajectory_batch(
            jonswap_samples,
            numerical,
            band=band,
        )

    produced: tuple[SimulationRows | None, ...] = ()
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
    rows_by_index = dict(zip(valid_indices, produced, strict=True))
    save_completed_batch(
        output_path,
        parameter_group_ids,
        tuple(rows_by_index.get(index) for index in range(len(parameter_group_ids))),
        family_id=PhysicalFamilyId[family.upper()],
        dataset_split=dataset_split,
    )
