"""Build a loader-facing view from completed paper-dataset batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_storage import load_completed_batch
from solver.gen_data.pipeline.artifact_io import (
    write_json_atomic,
    write_npz_atomic,
)
from solver.gen_data.pipeline.batch_artifacts import (
    SimulationCommitRecord,
    compute_batch_simulation_ids,
    parse_batch_plan_metadata,
)
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit
from solver.gen_data.pipeline.types import (
    BatchPlanArrays,
    JsonObject,
    JsonValue,
    SimulationIndexArrays,
)


DATASET_VIEW_SCHEMA_VERSION = 2
TRAJECTORY_MAP_SCHEMA_VERSION = 2
_STORED_DTYPES = {
    "eta": "float32",
    "xi": "float32",
    "gxi": "float32",
    "depth": "float64",
    "time": "float64",
}


@dataclass(frozen=True)
class DatasetViewPaths:
    """Files created by :func:`build_dataset_view`."""

    # JSON listing the dataset shards, grid settings, and split sizes.
    manifest: Path
    # NPZ mapping stored rows to simulations and recording each simulation's result.
    trajectory_map: Path


@dataclass(frozen=True)
class _TrainingTarget:
    """Grid and DNO settings shared by every shard in a dataset view."""

    role: str
    nx: int
    length: float
    gravity: float
    dno_order: int
    pad_factor: int
    maximum_wavenumber: float
    dtype: str = "float64"


@dataclass(frozen=True)
class _BatchSettings:
    """Generator settings read from one batch plan."""

    family_id: int
    training_target: _TrainingTarget
    # Full generator settings, compared to prevent mixing unlike batches.
    generation_record: JsonObject
    # Trajectory integrator settings copied into the dataset manifest.
    integration_record: JsonObject | None


def _training_target(
    numerical: JsonObject,
    *,
    role: object,
) -> _TrainingTarget:
    required_fields = {
        "nx",
        "length",
        "gravity",
        "dno_order",
        "pad_factor",
        "maximum_wavenumber",
    }
    missing = required_fields.difference(numerical)
    if missing:
        raise ValueError(
            f"generation settings are missing training target fields {sorted(missing)}"
        )
    if not isinstance(role, str):
        raise TypeError("generation role must be a string")
    return _TrainingTarget(
        role=role,
        nx=cast(int, numerical.get("target_nx", numerical["nx"])),
        length=cast(float, numerical["length"]),
        gravity=cast(float, numerical["gravity"]),
        dno_order=cast(
            int,
            numerical.get("target_dno_order", numerical["dno_order"]),
        ),
        pad_factor=cast(int, numerical["pad_factor"]),
        maximum_wavenumber=cast(
            float,
            numerical.get(
                "target_maximum_wavenumber",
                numerical["maximum_wavenumber"],
            ),
        ),
    )


def _batch_settings(
    batch_plan: BatchPlanArrays,
) -> _BatchSettings | None:
    metadata = parse_batch_plan_metadata(batch_plan["metadata_json"])
    simulation_type = metadata.get("simulation_type")
    if simulation_type not in ("static", "trajectory"):
        if "contract" in metadata or "trajectory_execution" in metadata:
            raise ValueError(
                "batch-plan metadata has an unknown or missing simulation_type"
            )
        return None

    is_trajectory = simulation_type == "trajectory"
    execution_name = "trajectory_execution" if is_trajectory else "contract"
    execution = metadata.get(execution_name)
    if not isinstance(execution, dict):
        raise ValueError(
            f"{simulation_type} metadata must contain an object field {execution_name!r}"
        )
    numerical = execution.get("numerical") if is_trajectory else execution
    if not isinstance(numerical, dict):
        raise ValueError(
            "trajectory execution must contain an object field 'numerical'"
        )
    return _BatchSettings(
        family_id=int(batch_plan["family_id"]),
        training_target=_training_target(numerical, role=execution.get("role")),
        generation_record=execution,
        integration_record=numerical if is_trajectory else None,
    )


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def _record_family_settings(
    settings_by_family: dict[int, JsonObject],
    family_id: int,
    settings: JsonObject,
    *,
    mismatch_message: str,
) -> None:
    previous = settings_by_family.setdefault(family_id, settings)
    if previous != settings:
        raise ValueError(mismatch_message)


def _validate_view_request(
    name: str,
    length: float,
    batches: Sequence[Path],
) -> None:
    if not name or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in name
    ):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not batches:
        raise ValueError("at least one completed batch is required")


def _dataset_settings(
    batches: Sequence[_BatchSettings | None],
    *,
    spatial_size: int,
    length: float,
) -> JsonObject | None:
    settings = [
        batch_settings for batch_settings in batches if batch_settings is not None
    ]
    if not settings:
        return None
    if len(settings) != len(batches):
        raise ValueError(
            "all batches must record generator settings when any batch does"
        )

    shared_target = settings[0].training_target
    generation_by_family: dict[int, JsonObject] = {}
    integration_by_family: dict[int, JsonObject] = {}
    for batch_settings in settings:
        if batch_settings.training_target != shared_target:
            raise ValueError(
                "all batches in one dataset view must use the same DNO target settings"
            )
        _record_family_settings(
            generation_by_family,
            batch_settings.family_id,
            batch_settings.generation_record,
            mismatch_message="one family cannot mix generator settings",
        )
        if batch_settings.integration_record is not None:
            _record_family_settings(
                integration_by_family,
                batch_settings.family_id,
                batch_settings.integration_record,
                mismatch_message=("one family cannot mix integration settings"),
            )

    if shared_target.nx != spatial_size:
        raise ValueError("shared target nx differs from the stored spatial grid")
    if not math.isclose(
        shared_target.length,
        length,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("shared target length differs from the dataset-view length")

    numerical_values = tuple(integration_by_family.values())
    common_numerical = (
        dict(numerical_values[0])
        if numerical_values
        and all(value == numerical_values[0] for value in numerical_values[1:])
        else None
    )
    return {
        "target": {
            "role": shared_target.role,
            "nx": shared_target.nx,
            "length": shared_target.length,
            "gravity": shared_target.gravity,
            "dno_order": shared_target.dno_order,
            "pad_factor": shared_target.pad_factor,
            "maximum_wavenumber": shared_target.maximum_wavenumber,
            "dtype": shared_target.dtype,
        },
        "trajectory_numerical": common_numerical,
        "trajectory_numerical_by_family": [
            {
                "family_id": family_id,
                "numerical": dict(numerical),
            }
            for family_id, numerical in sorted(integration_by_family.items())
        ],
        "stored_dtypes": dict(_STORED_DTYPES),
        "whole_simulation_rows": True,
    }


def build_dataset_view(
    root: Path,
    batches: Sequence[Path],
    *,
    name: str = "paper_dataset",
    length: float = 2.0 * math.pi,
) -> DatasetViewPaths:
    """Write a training manifest and attempted-simulation trajectory map.

    ``batches`` defines the immutable shard order. A batch with no accepted
    simulations legitimately contains no row arrays.
    """

    _validate_view_request(name, length, batches)
    output_paths = DatasetViewPaths(
        manifest=root / f"{name}.dataset.json",
        trajectory_map=root / f"{name}.trajectory_map.npz",
    )
    row_trajectory_parts: list[NDArray[np.int32]] = []
    row_frame_parts: list[NDArray[np.int32]] = []
    row_shard_parts: list[NDArray[np.int32]] = []
    row_shard_row_parts: list[NDArray[np.int64]] = []
    family_ids: list[int] = []
    dataset_splits: list[str] = []
    simulation_ids: list[int] = []
    parameter_group_ids: list[str] = []
    simulation_results: list[SimulationCommitRecord] = []
    first_rows: list[int] = []
    shard_records: list[JsonValue] = []
    batch_records: list[JsonValue] = []
    batch_settings: list[_BatchSettings | None] = []
    global_row_count = 0
    trajectory_offset = 0
    spatial_size: int | None = None

    for batch_index, batch_path in enumerate(batches):
        batch = load_completed_batch(batch_path)
        batch_plan = batch.plan
        current_settings = _batch_settings(batch_plan)
        family_id = int(batch_plan["family_id"])
        batch_settings.append(current_settings)

        planned_simulation_ids = compute_batch_simulation_ids(batch_plan)
        number_of_simulations = int(planned_simulation_ids.size)

        shard_index: int | None = None
        shard_row_count = 0
        if batch.shard is not None:
            shard = batch.shard
            current_spatial_size = int(shard["eta"].shape[1])
            if spatial_size is None:
                spatial_size = current_spatial_size
            elif current_spatial_size != spatial_size:
                raise ValueError("all dataset shards must use the same spatial grid")
            shard_index = len(shard_records)
            shard_row_count = int(shard["eta"].shape[0])
            shard_records.append(
                {
                    "path": _relative_path(batch_path, output_paths.manifest.parent),
                    "n_rows": shard_row_count,
                    "batch_index": batch_index,
                }
            )
            local_index = shard["simulation_local_index"]
            row_trajectory_parts.append(
                np.asarray(local_index + trajectory_offset, dtype=np.int32)
            )
            row_frame_parts.append(np.asarray(shard["frame_index"], dtype=np.int32))
            row_shard_parts.append(
                np.full(shard_row_count, shard_index, dtype=np.int32)
            )
            row_shard_row_parts.append(np.arange(shard_row_count, dtype=np.int64))
        dataset_split = DatasetSplit(str(batch_plan["dataset_split"].item()))
        simulations = batch.simulations
        accepted_in_batch = sum(simulation.accepted for simulation in simulations)
        family_ids.extend([family_id] * number_of_simulations)
        dataset_splits.extend([dataset_split.value] * number_of_simulations)
        simulation_ids.extend(map(int, planned_simulation_ids))
        parameter_group_ids.extend(map(str, batch_plan["parameter_group_id"]))
        simulation_results.extend(simulations)
        first_rows.extend(
            global_row_count + simulation.first_row if simulation.accepted else -1
            for simulation in simulations
        )

        batch_records.append(
            {
                "path": _relative_path(batch_path, output_paths.manifest.parent),
                "shard_index": shard_index,
                "family_id": family_id,
                "dataset_split": dataset_split.value,
                "n_attempted_trajectories": number_of_simulations,
                "n_accepted_trajectories": accepted_in_batch,
                "n_rows": shard_row_count,
            }
        )

        global_row_count += shard_row_count
        trajectory_offset += number_of_simulations

    if spatial_size is None:
        raise ValueError("a dataset view must contain at least one accepted row")
    dataset_settings = _dataset_settings(
        batch_settings,
        spatial_size=spatial_size,
        length=length,
    )
    if trajectory_offset > np.iinfo(np.int32).max:
        raise ValueError("trajectory count exceeds the int32 map capacity")
    if len(set(zip(family_ids, simulation_ids))) != trajectory_offset:
        raise ValueError("compound simulation IDs (family, simulation) must be unique")

    dataset_split_array = np.asarray(dataset_splits)
    accepted_array = np.asarray(
        [simulation.accepted for simulation in simulation_results],
        dtype=np.bool_,
    )
    trajectory_map: SimulationIndexArrays = {
        "schema_version": np.asarray(
            TRAJECTORY_MAP_SCHEMA_VERSION,
            dtype=np.int16,
        ),
        "trajectory_index": np.concatenate(row_trajectory_parts),
        "frame_index": np.concatenate(row_frame_parts),
        "shard_index": np.concatenate(row_shard_parts),
        "shard_row": np.concatenate(row_shard_row_parts),
        "trajectory_family_id": np.asarray(family_ids, dtype=np.int16),
        "trajectory_dataset_split": dataset_split_array,
        "trajectory_simulation_id": np.asarray(simulation_ids, dtype=np.int64),
        "trajectory_parameter_group_id": np.asarray(parameter_group_ids),
        "trajectory_accepted": accepted_array,
        "trajectory_required_bits": np.asarray(
            [simulation.required_bits for simulation in simulation_results],
            dtype=np.uint32,
        ),
        "trajectory_evaluated_bits": np.asarray(
            [simulation.evaluated_bits for simulation in simulation_results],
            dtype=np.uint32,
        ),
        "trajectory_failed_bits": np.asarray(
            [simulation.failed_bits for simulation in simulation_results],
            dtype=np.uint32,
        ),
        "trajectory_first_row": np.asarray(first_rows, dtype=np.int64),
        "trajectory_row_count": np.asarray(
            [simulation.row_count for simulation in simulation_results],
            dtype=np.int32,
        ),
    }
    write_npz_atomic(
        output_paths.trajectory_map,
        cast(Mapping[str, NDArray[Any]], trajectory_map),
    )

    split_counts: JsonObject = {
        dataset_split.value: {
            "attempted": int(
                np.count_nonzero(dataset_split_array == dataset_split.value)
            ),
            "accepted": int(
                np.count_nonzero(
                    (dataset_split_array == dataset_split.value) & accepted_array
                )
            ),
        }
        for dataset_split in DatasetSplit
    }
    manifest: JsonObject = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "dataset_settings": dataset_settings,
        "dataset_batches": batch_records,
        "dataset_shards": shard_records,
        "trajectory_map_npz": output_paths.trajectory_map.name,
        "requires_trajectory_map": True,
        "n_rows": global_row_count,
        "n_trajectories": trajectory_offset,
        "n_accepted_trajectories": int(np.count_nonzero(accepted_array)),
        "n_accepted_rows": global_row_count,
        "grid": {
            "length": float(length),
            "nx": spatial_size,
        },
        "split_counts": split_counts,
    }
    write_json_atomic(output_paths.manifest, manifest)
    return output_paths
