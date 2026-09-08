"""Read and write completed NPZ simulation batches."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.types import (
    DatasetShardArrays,
    PhysicalFamilyId,
    SimulationRows,
)

_SHARD_DTYPES = {
    "eta": np.dtype(np.float32),
    "xi": np.dtype(np.float32),
    "gxi": np.dtype(np.float32),
    "depth": np.dtype(np.float64),
    "time": np.dtype(np.float64),
    "simulation_local_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
}
_SHARD_FIELDS = frozenset(_SHARD_DTYPES)

CompletedBatch = NamedTuple(
    "CompletedBatch",
    [
        ("family_id", PhysicalFamilyId),
        ("seed", int),
        ("parameter_group_ids", tuple[str, ...]),
        ("accepted_simulations", NDArray[np.bool_]),
        ("shard", DatasetShardArrays | None),
    ],
)


def simulation_row_blocks(
    simulation_local_index: NDArray[Any],
    *,
    number_of_simulations: int,
) -> dict[int, tuple[int, int]]:
    """Return each simulation's contiguous ``(first_row, row_count)`` block."""

    indices = np.asarray(simulation_local_index)
    if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "simulation_local_index must be a one-dimensional integer array"
        )
    if indices.size == 0:
        return {}
    if np.any(indices < 0) or int(np.max(indices)) >= number_of_simulations:
        raise ValueError(
            "simulation_local_index references a simulation absent from the batch"
        )
    if np.any(indices[1:] < indices[:-1]):
        raise ValueError("rows from each simulation must form one ordered block")

    starts = np.concatenate(
        (np.asarray([0]), np.flatnonzero(indices[1:] != indices[:-1]) + 1)
    )
    ends = np.concatenate((starts[1:], np.asarray([indices.size])))
    return {
        int(indices[first_row]): (int(first_row), int(end - first_row))
        for first_row, end in zip(starts, ends, strict=True)
    }


def _validate_shard(
    shard: DatasetShardArrays, number_of_simulations: int
) -> dict[int, tuple[int, int]]:
    eta = shard["eta"]
    if eta.ndim != 2 or min(eta.shape) == 0:
        raise ValueError("eta must have nonempty shape (row, space)")
    row_count = eta.shape[0]
    for name, dtype in _SHARD_DTYPES.items():
        field = shard[name]
        if field.dtype != dtype:
            raise TypeError(f"array {name!r} has dtype {field.dtype}; expected {dtype}")
        shape = eta.shape if name in {"eta", "xi", "gxi"} else (row_count,)
        if field.shape != shape:
            raise ValueError(f"array {name!r} must have shape {shape}")
        if not np.isfinite(field).all():
            raise ValueError(f"shard array {name!r} contains nonfinite values")
    if np.any(shard["depth"] <= 0.0):
        raise ValueError("every stored depth must be positive")

    row_blocks = simulation_row_blocks(
        shard["simulation_local_index"], number_of_simulations=number_of_simulations
    )
    for first_row, frame_count in row_blocks.values():
        rows = slice(first_row, first_row + frame_count)
        if not np.array_equal(
            shard["frame_index"][rows], np.arange(frame_count, dtype=np.int32)
        ):
            raise ValueError("frame_index must be 0, 1, ... within every simulation")
        if np.any(np.diff(shard["time"][rows]) <= 0.0):
            raise ValueError("stored times must increase within every simulation")
        depths = shard["depth"][rows]
        if not np.all(depths == depths[0]):
            raise ValueError("depth must remain constant within each simulation")
    return row_blocks


def save_completed_batch(
    path: Path,
    parameter_group_ids: Sequence[str],
    rows_by_simulation: Sequence[SimulationRows | None],
    *,
    family_id: PhysicalFamilyId,
    seed: int,
) -> None:
    """Atomically save one finished batch without replacing an existing batch."""

    simulation_count = len(parameter_group_ids)

    parts: dict[str, list[NDArray[Any]]] = {name: [] for name in _SHARD_DTYPES}
    for local_index, rows in enumerate(rows_by_simulation):
        if rows is None:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            eta = np.asarray(rows.eta, dtype=np.float32)
            xi = np.asarray(rows.xi, dtype=np.float32)
            gxi = np.asarray(rows.gxi, dtype=np.float32)
        row_count = eta.shape[0]
        shard = DatasetShardArrays(
            eta=eta,
            xi=xi,
            gxi=gxi,
            depth=np.full(row_count, float(rows.depth), dtype=np.float64),
            time=np.asarray(rows.time, dtype=np.float64),
            simulation_local_index=np.full(row_count, local_index, dtype=np.int32),
            frame_index=np.arange(row_count, dtype=np.int32),
        )
        _validate_shard(shard, simulation_count)
        for name in parts:
            parts[name].append(shard[name])

    arrays: dict[str, NDArray[Any]] = {
        "family_id": np.asarray(int(family_id), dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "parameter_group_id": np.asarray(parameter_group_ids, dtype=np.str_),
    }
    if parts["eta"]:
        arrays.update({name: np.concatenate(values) for name, values in parts.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.") as temporary:
        np.savez(temporary, **arrays)  # pyright: ignore[reportArgumentType]
        temporary.flush()
        os.link(temporary.name, path)


def load_completed_batch(path: Path) -> CompletedBatch:
    """Load and validate one completed batch."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    unexpected = arrays.keys() - {
        "family_id",
        "seed",
        "parameter_group_id",
        *_SHARD_FIELDS,
    }
    if unexpected:
        raise ValueError(
            f"completed batch contains unexpected arrays {sorted(unexpected)}"
        )
    missing_metadata = {"family_id", "seed", "parameter_group_id"}.difference(arrays)
    if missing_metadata:
        raise ValueError(
            f"completed batch is missing metadata arrays {sorted(missing_metadata)}"
        )
    family = arrays["family_id"]
    if family.dtype != np.dtype(np.int16) or family.ndim != 0:
        raise TypeError("family_id must be a scalar int16 array")
    family_id = PhysicalFamilyId(int(family.item()))

    seed = arrays["seed"]
    if seed.dtype != np.dtype(np.int64) or seed.ndim != 0:
        raise TypeError("seed must be a scalar int64 array")

    groups = arrays["parameter_group_id"]
    if groups.dtype.kind != "U" or groups.ndim != 1 or groups.size == 0:
        raise TypeError("parameter_group_id must be a nonempty string array")
    parameter_group_ids = tuple(map(str, groups))
    if any(not group_id for group_id in parameter_group_ids):
        raise ValueError("parameter_group_id entries must not be empty")

    present_shard_fields = arrays.keys() & _SHARD_FIELDS
    if present_shard_fields and present_shard_fields != _SHARD_FIELDS:
        missing = _SHARD_FIELDS - present_shard_fields
        raise ValueError(f"completed batch is missing shard arrays {sorted(missing)}")
    shard = (
        cast(DatasetShardArrays, {name: arrays[name] for name in _SHARD_FIELDS})
        if present_shard_fields
        else None
    )
    accepted_simulations = np.zeros(len(parameter_group_ids), dtype=np.bool_)
    if shard is not None:
        row_blocks = _validate_shard(shard, len(parameter_group_ids))
        accepted_simulations[list(row_blocks)] = True

    return CompletedBatch(
        family_id,
        int(seed.item()),
        parameter_group_ids,
        accepted_simulations,
        shard,
    )
