"""Build a loader-facing view from committed paper-dataset batch shards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    BatchStatus,
    inspect_batch,
)
from solver.gen_data.pipeline.artifact_io import (
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)


DATASET_VIEW_SCHEMA_VERSION = 2
TRAJECTORY_MAP_SCHEMA_VERSION = 2
_SHARED_TARGET_FIELDS = (
    "nx",
    "length",
    "gravity",
    "dno_order",
    "pad_factor",
    "maximum_wavenumber",
)
_STORED_DTYPES = {
    "eta": "float32",
    "xi": "float32",
    "gxi": "float32",
    "depth": "float64",
    "time": "float64",
}


def _same_json_value(left: object, right: object) -> bool:
    """Compare strict JSON values without Python's numeric coercions."""

    return json.dumps(
        _plain_json_value(left),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == json.dumps(
        _plain_json_value(right),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _plain_json_value(value: object) -> object:
    """Copy immutable JSON-like mappings/sequences into plain containers."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class DatasetViewPaths:
    """The two derived files consumed by the training loader."""

    manifest: Path
    trajectory_map: Path


@dataclass(frozen=True)
class _BatchContract:
    """Dataset-wide and family-specific contracts recovered from a proposal."""

    target: Mapping[str, object]
    family_execution: Mapping[str, object]
    trajectory_numerical: Mapping[str, object] | None


def _metadata_object(proposal: Mapping[str, NDArray[Any]]) -> dict[str, object]:
    encoded = proposal["metadata_json"]
    if encoded.ndim != 0 or encoded.dtype.kind not in {"U", "S"}:
        raise TypeError("proposal metadata_json must be a scalar string array")
    value = json.loads(str(encoded.item()))
    if not isinstance(value, dict):
        raise ValueError("proposal metadata_json must encode a JSON object")
    return value


def _object_field(
    value: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> dict[str, object]:
    field = value.get(name)
    if not isinstance(field, dict):
        raise ValueError(f"{context} must contain an object field {name!r}")
    return field


def _finite_positive_real(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{name} must be a finite number")
    converted = float(value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _shared_target(
    numerical: Mapping[str, object],
    *,
    role: object,
) -> dict[str, object]:
    if not isinstance(role, str) or not role:
        raise ValueError("execution contract role must be a nonempty string")
    missing = set(_SHARED_TARGET_FIELDS).difference(numerical)
    if missing:
        raise ValueError(
            f"execution contract is missing shared target fields {sorted(missing)}"
        )
    integer_values: dict[str, int] = {}
    for name in ("nx", "dno_order", "pad_factor"):
        value = numerical[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("shared target integer fields must be integers")
        integer_values[name] = value
    raw_target_nx = numerical.get("target_nx")
    target_nx = integer_values["nx"]
    if raw_target_nx is not None:
        if (
            isinstance(raw_target_nx, bool)
            or not isinstance(raw_target_nx, int)
            or raw_target_nx <= 0
            or raw_target_nx % 2
        ):
            raise ValueError("target_nx must be a positive even integer")
        target_nx = raw_target_nx
    length = _finite_positive_real(numerical["length"], name="length")
    gravity = _finite_positive_real(numerical["gravity"], name="gravity")
    evolution_maximum_wavenumber = _finite_positive_real(
        numerical["maximum_wavenumber"],
        name="maximum_wavenumber",
    )
    raw_target_maximum_wavenumber = numerical.get("target_maximum_wavenumber")
    target_maximum_wavenumber = (
        evolution_maximum_wavenumber
        if raw_target_maximum_wavenumber is None
        else _finite_positive_real(
            raw_target_maximum_wavenumber,
            name="target_maximum_wavenumber",
        )
    )
    if target_maximum_wavenumber > evolution_maximum_wavenumber:
        raise ValueError("target_maximum_wavenumber cannot exceed maximum_wavenumber")
    raw_target_dno_order = numerical.get("target_dno_order")
    target_dno_order = integer_values["dno_order"]
    if raw_target_dno_order is not None:
        if isinstance(raw_target_dno_order, bool) or not isinstance(
            raw_target_dno_order,
            int,
        ):
            raise TypeError("target_dno_order must be an integer")
        if raw_target_dno_order < 0:
            raise ValueError("target_dno_order must be nonnegative")
        target_dno_order = raw_target_dno_order
    return {
        "role": role,
        "nx": target_nx,
        "length": length,
        "gravity": gravity,
        "dno_order": target_dno_order,
        "pad_factor": integer_values["pad_factor"],
        "maximum_wavenumber": target_maximum_wavenumber,
        "dtype": "float64",
    }


def _batch_contract(
    proposal: Mapping[str, NDArray[Any]],
) -> _BatchContract | None:
    metadata = _metadata_object(proposal)
    case_kind = metadata.get("case_kind")
    if case_kind == "static":
        execution = _object_field(metadata, "contract", context="static metadata")
        target = _shared_target(execution, role=execution.get("role"))
        return _BatchContract(
            target=target,
            family_execution=execution,
            trajectory_numerical=None,
        )
    if case_kind == "trajectory":
        execution = _object_field(
            metadata,
            "trajectory_execution",
            context="trajectory metadata",
        )
        numerical = _object_field(
            execution,
            "numerical",
            context="trajectory execution",
        )
        target = _shared_target(numerical, role=execution.get("role"))
        return _BatchContract(
            target=target,
            family_execution=execution,
            trajectory_numerical=numerical,
        )
    if "contract" in metadata or "trajectory_execution" in metadata:
        raise ValueError("proposal metadata has an unknown or missing case_kind")
    return None


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def build_dataset_view(
    root: Path,
    batches: Sequence[BatchPaths],
    *,
    name: str = "paper_dataset",
    length: float = 2.0 * math.pi,
) -> DatasetViewPaths:
    """Write a training manifest and attempted-case trajectory map.

    ``batches`` defines the immutable shard order.  Every batch must already
    be committed, but a batch with no accepted cases may legitimately have no
    shard.
    """

    if not name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in name
    ):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not batches:
        raise ValueError("at least one committed batch is required")
    output_paths = DatasetViewPaths(
        manifest=root / f"{name}.dataset.json",
        trajectory_map=root / f"{name}.trajectory_map.npz",
    )
    row_trajectory_parts: list[NDArray[np.int32]] = []
    row_frame_parts: list[NDArray[np.int32]] = []
    row_shard_parts: list[NDArray[np.int32]] = []
    row_shard_row_parts: list[NDArray[np.int64]] = []
    family_ids: list[int] = []
    revision_ids: list[int] = []
    split_ids: list[int] = []
    case_ids: list[int] = []
    cell_ids: list[int] = []
    accepted_values: list[bool] = []
    required_bits_values: list[int] = []
    evaluated_bits_values: list[int] = []
    failed_bits_values: list[int] = []
    first_rows: list[int] = []
    row_counts: list[int] = []
    shard_records: list[dict[str, object]] = []
    batch_records: list[dict[str, object]] = []
    batch_contracts: list[_BatchContract | None] = []
    family_execution_contracts: dict[
        tuple[int, int],
        Mapping[str, object],
    ] = {}
    family_trajectory_numerical: dict[
        tuple[int, int],
        Mapping[str, object],
    ] = {}
    shared_target: Mapping[str, object] | None = None
    global_row_count = 0
    trajectory_offset = 0
    spatial_size: int | None = None

    for batch_index, batch in enumerate(batches):
        inspection = inspect_batch(batch)
        if inspection.status is not BatchStatus.COMMITTED:
            raise RuntimeError(
                f"dataset views require committed batches, got {inspection.status.value}"
            )
        proposal = load_npz(batch.proposal)
        contract = _batch_contract(proposal)
        batch_contracts.append(contract)
        family_id = int(proposal["family_id"])
        revision_id = int(proposal["revision_id"])

        proposed_case_ids = proposal["case_id"]
        number_of_cases = int(proposed_case_ids.size)

        shard: dict[str, NDArray[Any]] | None = None
        shard_index: int | None = None
        shard_row_count = 0
        if batch.shard.exists():
            shard = load_npz(batch.shard)
            current_spatial_size = int(shard["eta"].shape[1])
            if spatial_size is None:
                spatial_size = current_spatial_size
            elif current_spatial_size != spatial_size:
                raise ValueError("all dataset shards must use the same spatial grid")
            shard_index = len(shard_records)
            shard_row_count = int(shard["eta"].shape[0])
            shard_records.append(
                {
                    "path": _relative_path(batch.shard, output_paths.manifest.parent),
                    "n_rows": shard_row_count,
                    "batch_index": batch_index,
                }
            )
            local_index = shard["case_local_index"]
            row_trajectory_parts.append(
                np.asarray(local_index + trajectory_offset, dtype=np.int32)
            )
            row_frame_parts.append(np.asarray(shard["frame_index"], dtype=np.int32))
            row_shard_parts.append(
                np.full(shard_row_count, shard_index, dtype=np.int32)
            )
            row_shard_row_parts.append(np.arange(shard_row_count, dtype=np.int64))
        split_id = int(proposal["split_id"])
        batch_id = int(proposal["batch_id"])
        accepted_in_batch = 0
        for local_index, (proposed_case_id, case) in enumerate(
            zip(proposed_case_ids, inspection.cases)
        ):
            block = (case.first_row, case.row_count) if case.accepted else None

            family_ids.append(family_id)
            revision_ids.append(revision_id)
            split_ids.append(split_id)
            case_ids.append(int(proposed_case_id))
            cell_ids.append(int(proposal["cell_id"][local_index]))
            accepted_values.append(case.accepted)
            accepted_in_batch += int(case.accepted)
            required_bits_values.append(case.required_bits)
            evaluated_bits_values.append(case.evaluated_bits)
            failed_bits_values.append(case.failed_bits)
            if block is None:
                first_rows.append(-1)
                row_counts.append(0)
            else:
                first_rows.append(global_row_count + block[0])
                row_counts.append(block[1])

        batch_records.append(
            {
                "proposal_path": _relative_path(
                    batch.proposal,
                    output_paths.manifest.parent,
                ),
                "result_path": _relative_path(
                    batch.result,
                    output_paths.manifest.parent,
                ),
                "shard_index": shard_index,
                "family_id": family_id,
                "revision_id": revision_id,
                "split_id": split_id,
                "batch_id": batch_id,
                "n_attempted_trajectories": number_of_cases,
                "n_accepted_trajectories": accepted_in_batch,
                "n_rows": shard_row_count,
            }
        )

        if contract is not None:
            if shared_target is None:
                shared_target = contract.target
            elif contract.target != shared_target:
                raise ValueError(
                    "all batches in one dataset view must share the DNO target contract"
                )
            family_key = (family_id, revision_id)
            previous_execution = family_execution_contracts.setdefault(
                family_key,
                contract.family_execution,
            )
            if not _same_json_value(previous_execution, contract.family_execution):
                raise ValueError("one family revision cannot mix execution contracts")
            if contract.trajectory_numerical is not None:
                previous_numerical = family_trajectory_numerical.setdefault(
                    family_key,
                    contract.trajectory_numerical,
                )
                if not _same_json_value(
                    previous_numerical,
                    contract.trajectory_numerical,
                ):
                    raise ValueError(
                        "one family revision cannot mix numerical integration contracts"
                    )

        if shard is not None:
            global_row_count += int(shard["eta"].shape[0])
        trajectory_offset += number_of_cases

    has_contract = [contract is not None for contract in batch_contracts]
    if any(has_contract) and not all(has_contract):
        raise ValueError(
            "all batches must record execution contracts when any batch does"
        )
    if spatial_size is None:
        raise ValueError("a dataset view must contain at least one accepted row")
    if shared_target is not None:
        if cast(int, shared_target["nx"]) != spatial_size:
            raise ValueError("shared target nx differs from the stored spatial grid")
        if not math.isclose(
            cast(float, shared_target["length"]),
            length,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "shared target length differs from the dataset-view length"
            )
    if trajectory_offset >= 1 << 31:
        raise ValueError("trajectory count exceeds the int32 map capacity")
    if len(set(zip(family_ids, revision_ids, case_ids))) != trajectory_offset:
        raise ValueError("compound case IDs (family, revision, case) must be unique")

    row_trajectory = np.concatenate(row_trajectory_parts)
    row_frame = np.concatenate(row_frame_parts)
    row_shard = np.concatenate(row_shard_parts)
    row_shard_row = np.concatenate(row_shard_row_parts)
    trajectory_map: dict[str, NDArray[Any]] = {
        "schema_version": np.asarray(
            TRAJECTORY_MAP_SCHEMA_VERSION,
            dtype=np.int16,
        ),
        "trajectory_index": row_trajectory,
        "frame_index": row_frame,
        "shard_index": row_shard,
        "shard_row": row_shard_row,
        "trajectory_family_id": np.asarray(family_ids, dtype=np.int16),
        "trajectory_revision_id": np.asarray(revision_ids, dtype=np.int16),
        "trajectory_split_id": np.asarray(split_ids, dtype=np.uint8),
        "trajectory_case_id": np.asarray(case_ids, dtype=np.int64),
        "trajectory_cell_id": np.asarray(cell_ids, dtype=np.int32),
        "trajectory_accepted": np.asarray(accepted_values, dtype=np.bool_),
        "trajectory_required_bits": np.asarray(
            required_bits_values,
            dtype=np.uint32,
        ),
        "trajectory_evaluated_bits": np.asarray(
            evaluated_bits_values,
            dtype=np.uint32,
        ),
        "trajectory_failed_bits": np.asarray(
            failed_bits_values,
            dtype=np.uint32,
        ),
        "trajectory_first_row": np.asarray(first_rows, dtype=np.int64),
        "trajectory_row_count": np.asarray(row_counts, dtype=np.int32),
    }
    write_npz_atomic(output_paths.trajectory_map, trajectory_map)

    split_counts = {
        split_name: {
            "attempted": int(np.count_nonzero(np.asarray(split_ids) == split_id)),
            "accepted": int(
                np.count_nonzero(
                    (np.asarray(split_ids) == split_id) & np.asarray(accepted_values)
                )
            ),
        }
        for split_name, split_id in (
            ("train", 0),
            ("validation", 1),
            ("test", 2),
        )
    }
    dataset_contract: dict[str, object] | None = None
    if shared_target is not None:
        trajectory_numerical_records = [
            {
                "family_id": family_id,
                "revision_id": revision_id,
                "numerical": dict(numerical),
            }
            for (family_id, revision_id), numerical in sorted(
                family_trajectory_numerical.items()
            )
        ]
        trajectory_numerical_values = tuple(family_trajectory_numerical.values())
        common_trajectory_numerical = (
            dict(trajectory_numerical_values[0])
            if trajectory_numerical_values
            and all(
                _same_json_value(numerical, trajectory_numerical_values[0])
                for numerical in trajectory_numerical_values[1:]
            )
            else None
        )
        dataset_contract = {
            "target": dict(shared_target),
            "trajectory_numerical": common_trajectory_numerical,
            "trajectory_numerical_by_family_revision": (trajectory_numerical_records),
            "stored_dtypes": dict(_STORED_DTYPES),
            "whole_case_rows": True,
        }
    manifest: dict[str, object] = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "dataset_contract": dataset_contract,
        "dataset_batches": batch_records,
        "dataset_shards": shard_records,
        "trajectory_map_npz": output_paths.trajectory_map.name,
        "requires_trajectory_map": True,
        "n_rows": global_row_count,
        "n_trajectories": trajectory_offset,
        "n_accepted_trajectories": int(np.count_nonzero(accepted_values)),
        "n_accepted_rows": global_row_count,
        "grid": {
            "length": float(length),
            "nx": spatial_size,
        },
        "split_counts": split_counts,
    }
    write_json_atomic(output_paths.manifest, manifest)
    return output_paths
