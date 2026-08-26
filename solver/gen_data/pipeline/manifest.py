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

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    BatchStatus,
    file_sha256,
    inspect_batch,
    write_json_atomic,
    write_npz_atomic,
)
from solver.gen_data.pipeline.valid_case_generation import canonical_json_sha256


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


def _same_json_value(left: object, right: object) -> bool:
    """Compare strict JSON values without Python's numeric coercions."""

    return canonical_json_sha256(_plain_json_value(left)) == canonical_json_sha256(
        _plain_json_value(right)
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


@dataclass(frozen=True)
class _GenerationCompatibility:
    """Source and dependency identity recovered from a launcher run record."""

    dependency_environment: Mapping[str, object]
    source_sha256: Mapping[str, str]
    execution_platform: str


@dataclass(frozen=True)
class GenerationCompatibilityVariant:
    """One explicitly audited execution/source pair for a family revision."""

    family_id: int
    revision_id: int
    execution_record_fingerprint: str
    source_sha256_fingerprint: str
    compatibility_id: str
    canonical_execution_record: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            isinstance(self.family_id, bool)
            or not isinstance(self.family_id, int)
            or self.family_id < 0
        ):
            raise ValueError("family_id must be a nonnegative integer")
        if (
            isinstance(self.revision_id, bool)
            or not isinstance(self.revision_id, int)
            or self.revision_id < 0
        ):
            raise ValueError("revision_id must be a nonnegative integer")
        for name in (
            "execution_record_fingerprint",
            "source_sha256_fingerprint",
            "compatibility_id",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        canonical = _plain_json_value(self.canonical_execution_record)
        if not isinstance(canonical, dict):
            raise TypeError("canonical_execution_record must be a JSON object")
        object.__setattr__(self, "canonical_execution_record", canonical)
        if canonical_json_sha256(canonical) != self.compatibility_id:
            raise ValueError(
                "compatibility_id must be the canonical execution-record fingerprint"
            )


@dataclass(frozen=True)
class GenerationCompatibilityResolution:
    """Canonical execution identity for one exact audited raw variant."""

    compatibility_id: str
    canonical_execution_record: Mapping[str, object]
    execution_record_fingerprint: str
    source_sha256_fingerprint: str


class GenerationCompatibilityPolicy:
    """Fail-closed lookup of explicitly audited generation variants.

    The lookup key includes the family, revision, raw execution record, and
    raw source mapping.  A policy has no effect outside the family revisions
    named by its variants.  Within those scopes, an unknown pair is rejected.
    """

    def __init__(
        self,
        variants: Sequence[GenerationCompatibilityVariant],
    ) -> None:
        selected = tuple(variants)
        if not selected:
            raise ValueError("generation compatibility variants must not be empty")
        by_key: dict[
            tuple[int, int, str, str],
            GenerationCompatibilityVariant,
        ] = {}
        canonical_by_id: dict[str, Mapping[str, object]] = {}
        for variant in selected:
            key = (
                variant.family_id,
                variant.revision_id,
                variant.execution_record_fingerprint,
                variant.source_sha256_fingerprint,
            )
            if key in by_key:
                raise ValueError("generation compatibility variants must be unique")
            previous = canonical_by_id.setdefault(
                variant.compatibility_id,
                variant.canonical_execution_record,
            )
            if previous != variant.canonical_execution_record:
                raise ValueError(
                    "one compatibility_id cannot name different canonical "
                    "execution records"
                )
            by_key[key] = variant
        self._variants = selected
        self._by_key = by_key
        self._scopes = frozenset(
            (variant.family_id, variant.revision_id) for variant in selected
        )

    @property
    def variants(self) -> tuple[GenerationCompatibilityVariant, ...]:
        return self._variants

    def applies_to(self, *, family_id: int, revision_id: int) -> bool:
        return (family_id, revision_id) in self._scopes

    def resolve(
        self,
        *,
        family_id: int,
        revision_id: int,
        execution_record: Mapping[str, object],
        source_sha256: Mapping[str, str],
    ) -> GenerationCompatibilityResolution | None:
        """Resolve one exact pair, or reject an unknown pair in policy scope."""

        if not self.applies_to(
            family_id=family_id,
            revision_id=revision_id,
        ):
            return None
        execution_fingerprint = canonical_json_sha256(
            _plain_json_value(execution_record)
        )
        source_fingerprint = canonical_json_sha256(_plain_json_value(source_sha256))
        variant = self._by_key.get(
            (
                family_id,
                revision_id,
                execution_fingerprint,
                source_fingerprint,
            )
        )
        if variant is None:
            raise ValueError(
                "generation record is not an explicitly audited compatibility variant"
            )
        return GenerationCompatibilityResolution(
            compatibility_id=variant.compatibility_id,
            canonical_execution_record=variant.canonical_execution_record,
            execution_record_fingerprint=execution_fingerprint,
            source_sha256_fingerprint=source_fingerprint,
        )


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


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
    dtype = numerical["dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise TypeError("shared target dtype must be a nonempty string")
    return {
        "role": role,
        "nx": target_nx,
        "length": length,
        "gravity": gravity,
        "dno_order": target_dno_order,
        "pad_factor": integer_values["pad_factor"],
        "maximum_wavenumber": target_maximum_wavenumber,
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
    if configuration.get("schema") != "paper_dataset_quota_configuration_v1":
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
        raise ValueError("proposal execution_platform must be a nonempty string")
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
        raise ValueError("expected_fingerprints must contain lowercase SHA-256 digests")
    return expected


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def build_dataset_view(
    root: Path,
    batches: Sequence[BatchPaths],
    *,
    name: str = "paper_dataset",
    length: float = 2.0 * math.pi,
    expected_fingerprint: str | None = None,
    expected_fingerprints: Sequence[str] | None = None,
    generation_compatibility_policy: (GenerationCompatibilityPolicy | None) = None,
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
    expected_fingerprint_set = _validate_expected_fingerprints(expected_fingerprints)

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
    family_generation_compatibility_ids: dict[
        tuple[int, int],
        str | None,
    ] = {}
    family_generation_variants: dict[
        tuple[int, int],
        dict[
            tuple[str, str],
            _GenerationCompatibility,
        ],
    ] = {}
    family_trajectory_numerical: dict[
        tuple[int, int],
        Mapping[str, object],
    ] = {}
    dependency_environment: Mapping[str, object] | None = None
    shared_target: Mapping[str, object] | None = None
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
        fingerprint = str(proposal["config_fingerprint"].item())
        fingerprints.add(fingerprint)
        contract = _batch_contract(proposal)
        batch_contracts.append(contract)
        generation_compatibility = _generation_compatibility(proposal)
        generation_compatibilities.append(generation_compatibility)
        family_id = int(proposal["family_id"])
        revision_id = int(proposal["revision_id"])
        generation_resolution: GenerationCompatibilityResolution | None = None
        if (
            generation_compatibility_policy is not None
            and generation_compatibility_policy.applies_to(
                family_id=family_id,
                revision_id=revision_id,
            )
        ):
            if contract is None or generation_compatibility is None:
                raise ValueError(
                    "an audited generation compatibility variant requires "
                    "both execution and source records"
                )
            generation_resolution = generation_compatibility_policy.resolve(
                family_id=family_id,
                revision_id=revision_id,
                execution_record=contract.family_execution,
                source_sha256=generation_compatibility.source_sha256,
            )
            assert generation_resolution is not None

        proposed_case_ids = proposal["case_id"]
        number_of_cases = int(proposed_case_ids.size)

        shard: dict[str, NDArray[Any]] | None = None
        shard_index: int | None = None
        shard_row_count = 0
        if batch.shard.exists():
            shard = _read_npz(batch.shard)
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
            effective_execution = (
                generation_resolution.canonical_execution_record
                if generation_resolution is not None
                else contract.family_execution
            )
            previous_execution = family_execution_contracts.setdefault(
                family_key,
                effective_execution,
            )
            if not _same_json_value(previous_execution, effective_execution):
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

        if generation_compatibility is not None:
            if dependency_environment is None:
                dependency_environment = generation_compatibility.dependency_environment
            elif not _same_json_value(
                generation_compatibility.dependency_environment,
                dependency_environment,
            ):
                raise ValueError(
                    "all generation runs in one dataset view must share the "
                    "dependency environment"
                )
            family_key = (family_id, revision_id)
            previous_generation = family_generation_compatibilities.setdefault(
                family_key,
                generation_compatibility,
            )
            compatibility_id = (
                generation_resolution.compatibility_id
                if generation_resolution is not None
                else None
            )
            previous_compatibility_id = family_generation_compatibility_ids.setdefault(
                family_key,
                compatibility_id,
            )
            if previous_compatibility_id != compatibility_id:
                raise ValueError(
                    "one family revision cannot mix generation compatibility identities"
                )
            if (
                previous_generation.source_sha256
                != generation_compatibility.source_sha256
                and compatibility_id is None
            ):
                raise ValueError("one family revision cannot mix source mappings")
            if (
                previous_generation.execution_platform
                != generation_compatibility.execution_platform
            ):
                raise ValueError("one family revision cannot mix execution platforms")
            execution_fingerprint = (
                canonical_json_sha256(contract.family_execution)
                if contract is not None
                else ""
            )
            source_fingerprint = canonical_json_sha256(
                generation_compatibility.source_sha256
            )
            family_generation_variants.setdefault(family_key, {})[
                (execution_fingerprint, source_fingerprint)
            ] = generation_compatibility

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
        compatibility is not None for compatibility in generation_compatibilities
    ]
    if any(has_generation_compatibility) and not all(has_generation_compatibility):
        raise ValueError(
            "all mixed generation runs must record dependency and source identity"
        )
    if generation_compatibility_policy is not None:
        for (
            family_id,
            revision_id,
        ), compatibility_id in family_generation_compatibility_ids.items():
            if (
                generation_compatibility_policy.applies_to(
                    family_id=family_id,
                    revision_id=revision_id,
                )
                and compatibility_id is None
            ):
                raise ValueError(
                    "audited generation compatibility scope was not resolved"
                )
    if len(fingerprints) != 1 and not all(has_contract):
        raise ValueError(
            "mixed configuration fingerprints require shared execution-contract "
            "metadata"
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
    configuration_fingerprints = sorted(fingerprints)
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
        generation_identity: dict[str, object] | None = None
        if all(has_generation_compatibility):
            assert dependency_environment is not None
            family_revision_records = []
            for (
                family_id,
                revision_id,
            ), compatibility in sorted(family_generation_compatibilities.items()):
                raw_variants = family_generation_variants[(family_id, revision_id)]
                variant_records = [
                    {
                        "execution_record_fingerprint": (execution_fingerprint),
                        "source_sha256_fingerprint": source_fingerprint,
                        "source_sha256": dict(
                            sorted(raw_compatibility.source_sha256.items())
                        ),
                    }
                    for (
                        execution_fingerprint,
                        source_fingerprint,
                    ), raw_compatibility in sorted(raw_variants.items())
                ]
                record: dict[str, object] = {
                    "family_id": family_id,
                    "revision_id": revision_id,
                    "execution_platform": compatibility.execution_platform,
                    "generation_variants": variant_records,
                }
                compatibility_id = family_generation_compatibility_ids[
                    (family_id, revision_id)
                ]
                if compatibility_id is not None:
                    record["generation_compatibility_id"] = compatibility_id
                if len(variant_records) == 1:
                    record["source_sha256_fingerprint"] = variant_records[0][
                        "source_sha256_fingerprint"
                    ]
                family_revision_records.append(record)
            generation_identity = {
                "dependency_environment_fingerprint": canonical_json_sha256(
                    dependency_environment
                ),
                "family_revisions": family_revision_records,
            }
            generation_identity["compatibility_fingerprint"] = canonical_json_sha256(
                generation_identity
            )
        dataset_contract = {
            "target": dict(shared_target),
            "trajectory_numerical": common_trajectory_numerical,
            "trajectory_numerical_by_family_revision": (trajectory_numerical_records),
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
