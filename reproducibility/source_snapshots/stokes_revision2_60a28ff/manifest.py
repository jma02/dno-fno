"""Build a loader-facing view from committed paper-corpus batch shards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    BatchStatus,
    file_sha256,
    inspect_batch,
    write_json_atomic,
    write_npz_atomic,
)
from solver.gen_data.pipeline.quota_driver import canonical_json_sha256


DATASET_VIEW_SCHEMA_VERSION = 2
TRAJECTORY_MAP_SCHEMA_VERSION = 2
_SHARED_TARGET_FIELDS = (
    "nx",
    "length",
    "gravity",
    "dno_order",
    "pad_factor",
    "maximum_wavenumber",
    "dtype",
)
_STORED_DTYPES = {
    "eta": "float32",
    "xi": "float32",
    "gxi": "float32",
    "depth": "float64",
    "time": "float64",
}


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


@dataclass(frozen=True)
class _GenerationCompatibility:
    """Source and dependency identity recovered from a launcher run record."""

    dependency_environment: Mapping[str, object]
    source_sha256: Mapping[str, str]
    execution_platform: str


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
    integer_fields = ("nx", "dno_order", "pad_factor")
    if any(
        isinstance(numerical[name], bool) or not isinstance(numerical[name], int)
        for name in integer_fields
    ):
        raise TypeError("shared target integer fields must be integers")
    real_fields = ("length", "gravity", "maximum_wavenumber")
    if any(
        isinstance(numerical[name], bool)
        or not isinstance(numerical[name], (int, float))
        or not math.isfinite(float(numerical[name]))
        for name in real_fields
    ):
        raise TypeError("shared target real fields must be finite numbers")
    dtype = numerical["dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise TypeError("shared target dtype must be a nonempty string")
    return {
        "role": role,
        "nx": int(numerical["nx"]),
        "length": float(numerical["length"]),
        "gravity": float(numerical["gravity"]),
        "dno_order": int(numerical["dno_order"]),
        "pad_factor": int(numerical["pad_factor"]),
        "maximum_wavenumber": float(numerical["maximum_wavenumber"]),
        "dtype": dtype,
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


def _generation_compatibility(
    proposal: Mapping[str, NDArray[Any]],
) -> _GenerationCompatibility | None:
    metadata = _metadata_object(proposal)
    additional = metadata.get("additional_metadata")
    if additional is None:
        return None
    if not isinstance(additional, dict):
        raise ValueError("proposal additional_metadata must be a JSON object")
    run_spec = additional.get("run_spec")
    if run_spec is None:
        return None
    if not isinstance(run_spec, dict):
        raise ValueError("proposal run_spec must be a JSON object")
    configuration = _object_field(
        run_spec,
        "configuration",
        context="proposal run_spec",
    )
    if configuration.get("schema") != "paper_corpus_quota_configuration_v1":
        return None
    dependency_environment = _object_field(
        configuration,
        "dependency_environment",
        context="proposal run configuration",
    )
    raw_sources = _object_field(
        configuration,
        "source_sha256",
        context="proposal run configuration",
    )
    if not raw_sources:
        raise ValueError("proposal source_sha256 must not be empty")
    source_sha256: dict[str, str] = {}
    for path, digest in raw_sources.items():
        if not isinstance(path, str) or not path:
            raise TypeError("proposal source paths must be nonempty strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "proposal source_sha256 values must be lowercase SHA-256 digests"
            )
        source_sha256[path] = digest
    execution_platform = configuration.get("execution_platform")
    if not isinstance(execution_platform, str) or not execution_platform:
        raise ValueError(
            "proposal execution_platform must be a nonempty string"
        )
    return _GenerationCompatibility(
        dependency_environment=dependency_environment,
        source_sha256=source_sha256,
        execution_platform=execution_platform,
    )


def _validate_expected_fingerprints(
    fingerprints: Sequence[str] | None,
) -> frozenset[str] | None:
    if fingerprints is None:
        return None
    expected = frozenset(fingerprints)
    if not expected:
        raise ValueError("expected_fingerprints must not be empty")
    if any(
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        for fingerprint in expected
    ):
        raise ValueError(
            "expected_fingerprints must contain lowercase SHA-256 digests"
        )
    return expected


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def _require_case_result(
    value: object,
    *,
    expected_case_id: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("every result case must be a JSON object")
    case_id = value.get("case_id")
    if (
        not isinstance(case_id, int)
        or isinstance(case_id, bool)
        or case_id != expected_case_id
    ):
        raise ValueError("result case identity differs from its proposal")
    for name in (
        "accepted",
        "required_bits",
        "evaluated_bits",
        "failed_bits",
        "first_row",
        "row_count",
    ):
        if name not in value:
            raise ValueError(f"result case is missing {name!r}")
    if not isinstance(value["accepted"], bool):
        raise TypeError("accepted must be boolean")
    for name in (
        "required_bits",
        "evaluated_bits",
        "failed_bits",
        "first_row",
        "row_count",
    ):
        if not isinstance(value[name], int) or isinstance(value[name], bool):
            raise TypeError(f"{name} must be an integer")
    required_bits = int(value["required_bits"])
    evaluated_bits = int(value["evaluated_bits"])
    failed_bits = int(value["failed_bits"])
    if not 0 <= required_bits < 1 << 32:
        raise ValueError("required_bits must fit in uint32")
    if not 0 <= evaluated_bits < 1 << 32:
        raise ValueError("evaluated_bits must fit in uint32")
    if not 0 <= failed_bits < 1 << 32:
        raise ValueError("failed_bits must fit in uint32")
    if failed_bits & ~evaluated_bits:
        raise ValueError("failed_bits must be a subset of evaluated_bits")
    accepted = not (
        required_bits & ~evaluated_bits or required_bits & failed_bits
    )
    if bool(value["accepted"]) != accepted:
        raise ValueError("accepted disagrees with the persisted quality masks")
    return value


def _batch_case_rows(
    shard: dict[str, NDArray[Any]] | None,
    *,
    number_of_cases: int,
) -> dict[int, tuple[int, int]]:
    if shard is None:
        return {}
    local_index = shard["case_local_index"]
    if local_index.size and (
        int(local_index.min()) < 0 or int(local_index.max()) >= number_of_cases
    ):
        raise ValueError("shard contains an out-of-range case_local_index")
    return {
        int(case_index): (
            int(np.flatnonzero(local_index == case_index)[0]),
            int(np.count_nonzero(local_index == case_index)),
        )
        for case_index in np.unique(local_index)
    }


def build_dataset_view(
    root: Path,
    batches: Sequence[BatchPaths],
    *,
    name: str = "paper_corpus",
    length: float = 2.0 * math.pi,
    expected_fingerprint: str | None = None,
    expected_fingerprints: Sequence[str] | None = None,
) -> DatasetViewPaths:
    """Write a schema-v2 manifest and attempted-case trajectory map.

    ``batches`` defines the immutable shard order.  Every batch must already
    be committed, but a batch with no accepted cases may legitimately have no
    shard.  ``expected_fingerprint`` retains the single-run check.  A combined
    view may instead provide the complete set of ``expected_fingerprints``.
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
    if expected_fingerprint is not None and expected_fingerprints is not None:
        raise ValueError(
            "provide expected_fingerprint or expected_fingerprints, not both"
        )
    expected_fingerprint_set = _validate_expected_fingerprints(
        expected_fingerprints
    )

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
    fingerprints: set[str] = set()
    batch_contracts: list[_BatchContract | None] = []
    generation_compatibilities: list[_GenerationCompatibility | None] = []
    family_execution_contracts: dict[
        tuple[int, int],
        Mapping[str, object],
    ] = {}
    family_generation_compatibilities: dict[
        tuple[int, int],
        _GenerationCompatibility,
    ] = {}
    dependency_environment: Mapping[str, object] | None = None
    source_sha256: dict[str, str] = {}
    shared_target: Mapping[str, object] | None = None
    trajectory_numerical: Mapping[str, object] | None = None
    global_row_count = 0
    trajectory_offset = 0
    spatial_size: int | None = None

    for batch_index, batch in enumerate(batches):
        inspection = inspect_batch(
            batch,
            expected_fingerprint=expected_fingerprint,
        )
        if inspection.status is not BatchStatus.COMMITTED:
            raise RuntimeError(
                f"dataset views require committed batches, got {inspection.status.value}"
            )
        proposal = _read_npz(batch.proposal)
        result = _read_json_object(batch.result)
        fingerprint = str(proposal["config_fingerprint"].item())
        fingerprints.add(fingerprint)
        contract = _batch_contract(proposal)
        batch_contracts.append(contract)
        generation_compatibility = _generation_compatibility(proposal)
        generation_compatibilities.append(generation_compatibility)

        proposed_case_ids = proposal["case_id"]
        number_of_cases = int(proposed_case_ids.size)
        result_cases = result.get("cases")
        if not isinstance(result_cases, list) or len(result_cases) != number_of_cases:
            raise ValueError("result must contain every proposed case exactly once")

        shard: dict[str, NDArray[Any]] | None = None
        shard_index: int | None = None
        shard_row_count = 0
        if batch.shard.exists():
            shard = _read_npz(batch.shard)
            current_spatial_size = int(shard["eta"].shape[1])
            if spatial_size is None:
                spatial_size = current_spatial_size
            elif current_spatial_size != spatial_size:
                raise ValueError("all corpus shards must use the same spatial grid")
            shard_index = len(shard_records)
            shard_row_count = int(shard["eta"].shape[0])
            shard_records.append(
                {
                    "path": _relative_path(batch.shard, output_paths.manifest.parent),
                    "sha256": file_sha256(batch.shard),
                    "n_rows": shard_row_count,
                    "batch_index": batch_index,
                    "configuration_fingerprint": fingerprint,
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
        blocks = _batch_case_rows(shard, number_of_cases=number_of_cases)

        family_id = int(proposal["family_id"])
        revision_id = int(proposal["revision_id"])
        split_id = int(proposal["split_id"])
        batch_id = int(proposal["batch_id"])
        accepted_in_batch = 0
        for local_index, proposed_case_id in enumerate(proposed_case_ids):
            case = _require_case_result(
                result_cases[local_index],
                expected_case_id=int(proposed_case_id),
            )
            accepted = bool(case["accepted"])
            block = blocks.get(local_index)
            declared_block = (
                (int(case["first_row"]), int(case["row_count"])) if accepted else None
            )
            if declared_block != block:
                raise ValueError("result row ownership differs from its shard")

            family_ids.append(family_id)
            revision_ids.append(revision_id)
            split_ids.append(split_id)
            case_ids.append(int(proposed_case_id))
            cell_ids.append(int(proposal["cell_id"][local_index]))
            accepted_values.append(accepted)
            accepted_in_batch += int(accepted)
            required_bits_values.append(int(case["required_bits"]))
            evaluated_bits_values.append(int(case["evaluated_bits"]))
            failed_bits_values.append(int(case["failed_bits"]))
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
                "proposal_sha256": inspection.proposal_sha256,
                "result_path": _relative_path(
                    batch.result,
                    output_paths.manifest.parent,
                ),
                "result_sha256": file_sha256(batch.result),
                "shard_index": shard_index,
                "configuration_fingerprint": fingerprint,
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
            if previous_execution != contract.family_execution:
                raise ValueError(
                    "one family revision cannot mix execution contracts"
                )
            if contract.trajectory_numerical is not None:
                if trajectory_numerical is None:
                    trajectory_numerical = contract.trajectory_numerical
                elif contract.trajectory_numerical != trajectory_numerical:
                    raise ValueError(
                        "all trajectory families must share the numerical "
                        "integration contract"
                    )

        if generation_compatibility is not None:
            if dependency_environment is None:
                dependency_environment = (
                    generation_compatibility.dependency_environment
                )
            elif (
                generation_compatibility.dependency_environment
                != dependency_environment
            ):
                raise ValueError(
                    "all generation runs in one dataset view must share the "
                    "dependency environment"
                )
            for path, digest in generation_compatibility.source_sha256.items():
                previous_digest = source_sha256.setdefault(path, digest)
                if previous_digest != digest:
                    raise ValueError(
                        "generation runs disagree on the hash of shared source "
                        f"path {path!r}"
                    )
            family_key = (family_id, revision_id)
            previous_generation = family_generation_compatibilities.setdefault(
                family_key,
                generation_compatibility,
            )
            if (
                previous_generation.source_sha256
                != generation_compatibility.source_sha256
            ):
                raise ValueError(
                    "one family revision cannot mix source mappings"
                )
            if (
                previous_generation.execution_platform
                != generation_compatibility.execution_platform
            ):
                raise ValueError(
                    "one family revision cannot mix execution platforms"
                )

        if shard is not None:
            global_row_count += int(shard["eta"].shape[0])
        trajectory_offset += number_of_cases

    if expected_fingerprint_set is not None and fingerprints != set(
        expected_fingerprint_set
    ):
        raise RuntimeError(
            "dataset configuration fingerprints do not match the expected set"
        )
    has_contract = [contract is not None for contract in batch_contracts]
    if any(has_contract) and not all(has_contract):
        raise ValueError(
            "all batches must record execution contracts when any batch does"
        )
    has_generation_compatibility = [
        compatibility is not None
        for compatibility in generation_compatibilities
    ]
    if any(has_generation_compatibility) and not all(
        has_generation_compatibility
    ):
        raise ValueError(
            "all mixed generation runs must record dependency and source identity"
        )
    if len(fingerprints) != 1 and not all(has_contract):
        raise ValueError(
            "mixed configuration fingerprints require shared execution-contract "
            "metadata"
        )
    if spatial_size is None:
        raise ValueError("a dataset view must contain at least one accepted row")
    if shared_target is not None:
        if int(shared_target["nx"]) != spatial_size:
            raise ValueError("shared target nx differs from the stored spatial grid")
        if not math.isclose(
            float(shared_target["length"]),
            length,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "shared target length differs from the dataset-view length"
            )
    if trajectory_offset >= 1 << 31:
        raise ValueError("trajectory count exceeds the int32 map capacity")
    compound_ids = np.rec.fromarrays(
        (
            np.asarray(family_ids, dtype=np.int16),
            np.asarray(revision_ids, dtype=np.int16),
            np.asarray(case_ids, dtype=np.int64),
        ),
        names=("family", "revision", "case"),
    )
    if np.unique(compound_ids).size != trajectory_offset:
        raise ValueError(
            "compound case IDs (family, revision, case) must be unique"
        )

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
    configuration_fingerprints = sorted(fingerprints)
    dataset_contract: dict[str, object] | None = None
    if shared_target is not None:
        generation_identity: dict[str, object] | None = None
        if all(has_generation_compatibility):
            assert dependency_environment is not None
            family_revision_records = [
                {
                    "family_id": family_id,
                    "revision_id": revision_id,
                    "execution_platform": compatibility.execution_platform,
                    "source_sha256_fingerprint": canonical_json_sha256(
                        compatibility.source_sha256
                    ),
                }
                for (
                    family_id,
                    revision_id,
                ), compatibility in sorted(
                    family_generation_compatibilities.items()
                )
            ]
            generation_identity = {
                "dependency_environment_fingerprint": canonical_json_sha256(
                    dependency_environment
                ),
                "source_sha256_fingerprint": canonical_json_sha256(
                    source_sha256
                ),
                "family_revisions": family_revision_records,
            }
            generation_identity["compatibility_fingerprint"] = (
                canonical_json_sha256(generation_identity)
            )
        dataset_contract = {
            "target": dict(shared_target),
            "trajectory_numerical": (
                dict(trajectory_numerical)
                if trajectory_numerical is not None
                else None
            ),
            "stored_dtypes": dict(_STORED_DTYPES),
            "whole_case_rows": True,
            "generation_identity": generation_identity,
        }
    manifest: dict[str, object] = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprint": (
            configuration_fingerprints[0]
            if len(configuration_fingerprints) == 1
            else None
        ),
        "configuration_fingerprints": configuration_fingerprints,
        "dataset_contract": dataset_contract,
        "dataset_contract_fingerprint": (
            canonical_json_sha256(dataset_contract)
            if dataset_contract is not None
            else None
        ),
        "dataset_batches": batch_records,
        "dataset_shards": shard_records,
        "trajectory_map_npz": output_paths.trajectory_map.name,
        "trajectory_map_sha256": file_sha256(output_paths.trajectory_map),
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
