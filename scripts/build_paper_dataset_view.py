"""Preflight or build one four-family view from completed quota chunks.

Inputs are the completion summaries written by ``run_paper_dataset_quota.py``.
The default mode is read-only.  ``--execute`` builds a schema-v2 view only
after every chunk is complete, each family's cumulative intervals are nested
without gaps or overlap, all four families have the same accepted-case count
within every included split, and the input fingerprints are distinct and
exact.  Production supplies train, validation, and test chunks together;
single-split views remain available for diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.manifest import (
    DATASET_VIEW_SCHEMA_VERSION,
    TRAJECTORY_MAP_SCHEMA_VERSION,
    DatasetViewPaths,
    GenerationCompatibilityPolicy,
    GenerationCompatibilityResolution,
    GenerationCompatibilityVariant,
    build_dataset_view,
)
from solver.gen_data.pipeline.production import (
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    balanced_cell_quotas,
    paper_dataset_revision_id,
    split_code,
)
from solver.gen_data.pipeline.quota_driver import (
    AcceptedQuotaRunSpec,
    canonical_json_sha256,
    scan_quota_run,
)
from solver.gen_data.jonswap_horizon_executor import policy_record
from solver.gen_data.stokes_static_pipeline import PAPER_STATIC_STOKES_CONTRACT
from solver.gen_data.trajectory_quota_executor import TrajectoryExecutionConfig


FAMILY_ORDER = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
FAMILY_IDS = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
ROWS_PER_ACCEPTED_CASE = {
    "stokes": 1,
    "tanaka": 200,
    "benjamin_feir": 200,
    "jonswap_tma": 16,
}
TRAJECTORY_MAP_DTYPES = {
    "schema_version": np.dtype(np.int16),
    "trajectory_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
    "shard_index": np.dtype(np.int32),
    "shard_row": np.dtype(np.int64),
    "trajectory_family_id": np.dtype(np.int16),
    "trajectory_revision_id": np.dtype(np.int16),
    "trajectory_split_id": np.dtype(np.uint8),
    "trajectory_case_id": np.dtype(np.int64),
    "trajectory_cell_id": np.dtype(np.int32),
    "trajectory_accepted": np.dtype(np.bool_),
    "trajectory_required_bits": np.dtype(np.uint32),
    "trajectory_evaluated_bits": np.dtype(np.uint32),
    "trajectory_failed_bits": np.dtype(np.uint32),
    "trajectory_first_row": np.dtype(np.int64),
    "trajectory_row_count": np.dtype(np.int32),
}
SHARED_TARGET_SOURCE_PATHS = (
    "solver/gen_data/pipeline/reference.py",
    "solver/solvers/dno_series_jax.py",
)
TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT = (
    "8fbb0addacfbd0cc28ef3e5fdc5b96104f3268f9b7d173fdd41cfc2328e1341e"
)
TANAKA_REVISION3_AUDITED_REPLAY_EXECUTION_FINGERPRINT = (
    "73811acc430da1a1a3a30c18fa4b34b89e9e5756206adaf34975468f4931d963"
)
TANAKA_REVISION3_LEGACY_EXECUTION_FINGERPRINT = (
    "e8b7c0ff42240f2d0a9257fa55c71150fabc5fb66efc2704735dcf2e1e38ab4d"
)
TANAKA_REVISION3_LEGACY_SOURCE_FINGERPRINT = (
    "a1cf8a3b2649e21161a15bf26d1bb6388f2b7f93996b1c4cf7e3c297d1b98d0a"
)
# ``9d1b...`` identifies the audited historical replay.  The fresh
# ``478ecc...`` identity is computed and frozen by its preflights and audit.
TANAKA_REVISION3_AUDITED_REPLAY_SOURCE_FINGERPRINT = (
    "9d1b396cbdf9f7b4936878b602faa18c71d462478b0d7e623157b83ba2e161ee"
)


def _paper_generation_compatibility_policy() -> GenerationCompatibilityPolicy:
    """Return the exact audited Tanaka revision-3 generation bridge."""

    canonical_execution = TrajectoryExecutionConfig.paper("tanaka").to_json_record()
    variants = tuple(
        GenerationCompatibilityVariant(
            family_id=int(PhysicalFamilyId.TANAKA),
            revision_id=3,
            execution_record_fingerprint=execution_fingerprint,
            source_sha256_fingerprint=source_fingerprint,
            compatibility_id=TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
            canonical_execution_record=canonical_execution,
        )
        for execution_fingerprint, source_fingerprint in (
            (
                TANAKA_REVISION3_LEGACY_EXECUTION_FINGERPRINT,
                TANAKA_REVISION3_LEGACY_SOURCE_FINGERPRINT,
            ),
            (
                TANAKA_REVISION3_AUDITED_REPLAY_EXECUTION_FINGERPRINT,
                TANAKA_REVISION3_AUDITED_REPLAY_SOURCE_FINGERPRINT,
            ),
        )
    )
    return GenerationCompatibilityPolicy(variants)


PAPER_GENERATION_COMPATIBILITY_POLICY = _paper_generation_compatibility_policy()


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
class CompletedChunk:
    """Validated identity and durable batches from one completed quota run."""

    summary_path: Path
    summary_sha256: str
    root: Path
    family: str
    revision_id: int
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int
    accepted_after: int
    attempted_count: int
    fingerprint: str
    dependency_fingerprint: str
    execution_fingerprint: str
    generation_compatibility_id: str | None
    source_fingerprint: str
    source_sha256: Mapping[str, str]
    execution_platform: str
    batches: tuple[BatchPaths, ...]


@dataclass(frozen=True)
class CombinedDatasetPlan:
    """Ordered completed inputs and exact combined-view expectations."""

    chunks: tuple[CompletedChunk, ...]
    splits: tuple[SplitId, ...]
    accepted_cases_per_family_by_split: Mapping[str, int]
    attempted_cases_by_split: Mapping[str, int]
    attempted_cases: int
    accepted_cases: int
    expected_rows: int
    fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class CombinedViewResult:
    """Products of a completed combined-view build."""

    summary_path: Path
    view: DatasetViewPaths
    summary: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-summary",
        type=Path,
        action="append",
        required=True,
        help="Completed quota summary; repeat once for every additive chunk.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Separate directory for the derived manifest and trajectory map.",
    )
    parser.add_argument(
        "--name",
        help="View basename; defaults to paper_dataset_<split>.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Build the combined view after its read-only checks pass.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the read-only preflight explicitly (this is the default).",
    )
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _required_integer(
    mapping: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _validate_current_execution_contract(
    configuration: Mapping[str, object],
    *,
    family: str,
) -> None:
    """Require a chunk to declare the exact current paper execution contract."""

    if family == "stokes":
        key = "contract"
        expected = PAPER_STATIC_STOKES_CONTRACT.to_json_record()
    elif family in ("tanaka", "benjamin_feir", "jonswap_tma"):
        key = "trajectory_execution"
        expected = TrajectoryExecutionConfig.paper(family).to_json_record()
    else:
        raise ValueError(f"unknown paper-dataset family: {family}")
    actual = _required_mapping(configuration, key)
    if not _same_json_value(actual, expected):
        raise ValueError(
            f"{family} chunk does not use the exact current paper execution contract"
        )
    return None


def _current_execution_record(family: str) -> dict[str, object]:
    if family == "stokes":
        return PAPER_STATIC_STOKES_CONTRACT.to_json_record()
    if family in ("tanaka", "benjamin_feir", "jonswap_tma"):
        return TrajectoryExecutionConfig.paper(family).to_json_record()
    raise ValueError(f"unknown paper-dataset family: {family}")


def _resolve_generation_compatibility(
    configuration: Mapping[str, object],
    *,
    family: str,
    family_id: int,
    revision_id: int,
    execution_record: Mapping[str, object],
    source_sha256: Mapping[str, str],
    policy: GenerationCompatibilityPolicy | None,
) -> GenerationCompatibilityResolution | None:
    """Use an audited legacy bridge only when the current contract fails."""

    try:
        _validate_current_execution_contract(configuration, family=family)
    except ValueError as current_contract_error:
        if policy is None or not policy.applies_to(
            family_id=family_id,
            revision_id=revision_id,
        ):
            raise current_contract_error
        resolution = policy.resolve(
            family_id=family_id,
            revision_id=revision_id,
            execution_record=execution_record,
            source_sha256=source_sha256,
        )
        assert resolution is not None
        if not _same_json_value(
            resolution.canonical_execution_record,
            _current_execution_record(family),
        ):
            raise ValueError(
                "audited generation compatibility does not resolve to the "
                "current paper execution contract"
            )
        return resolution
    return None


def _proposal_metadata(path: Path) -> Mapping[str, object]:
    """Read the already-validated strict metadata object from one proposal."""

    with np.load(path, allow_pickle=False) as proposal:
        encoded = np.asarray(proposal["metadata_json"])
    if encoded.ndim != 0 or encoded.dtype.kind not in {"U", "S"}:
        raise TypeError("proposal metadata_json must be a scalar string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} metadata contains nonfinite JSON constant {value!r}")

    metadata = json.loads(str(encoded.item()), parse_constant=reject_constant)
    if not isinstance(metadata, dict):
        raise TypeError("proposal metadata_json must encode a JSON object")
    return metadata


def _validate_committed_proposal_contract(
    paths: BatchPaths,
    *,
    spec: AcceptedQuotaRunSpec,
) -> None:
    """Bind proposal metadata to the exact summary/run execution contract."""

    metadata = _proposal_metadata(paths.proposal)
    family = spec.family_name
    if metadata.get("family") != family:
        raise ValueError("proposal family differs from its completed chunk")
    expected_case_kind = "static" if family == "stokes" else "trajectory"
    if metadata.get("case_kind") != expected_case_kind:
        raise ValueError("proposal case kind differs from its completed chunk")
    execution_key = "contract" if family == "stokes" else "trajectory_execution"
    expected_execution = _required_mapping(spec.configuration, execution_key)
    actual_execution = _required_mapping(metadata, execution_key)
    if not _same_json_value(actual_execution, expected_execution):
        raise ValueError(f"{family} proposal execution differs from its own run spec")
    additional = _required_mapping(metadata, "additional_metadata")
    proposal_run_spec = _required_mapping(additional, "run_spec")
    if not _same_json_value(proposal_run_spec, spec.to_json_record()):
        raise ValueError("proposal run_spec differs from its completion summary")
    if family == "jonswap_tma":
        configured_policy = _required_mapping(
            spec.configuration,
            "jonswap_horizon_bucketing",
        )
        solver_batch_size = _required_integer(
            configured_policy,
            "solver_batch_size",
            minimum=1,
        )
        if solver_batch_size > spec.batch_size:
            raise ValueError(
                "JONSWAP solver batch size exceeds its proposal batch size"
            )
        adjustment_policy = TrajectoryExecutionConfig.paper(
            "jonswap_tma"
        ).jonswap_adjustment
        assert adjustment_policy is not None
        expected_policy = policy_record(
            outer_proposal_size=spec.batch_size,
            solver_batch_size=solver_batch_size,
            adjustment_policy=adjustment_policy,
        )
        if not _same_json_value(configured_policy, expected_policy):
            raise ValueError(
                "JONSWAP run does not use the exact current bucketing policy"
            )
        proposal_policy = _required_mapping(
            additional,
            "jonswap_horizon_bucketing",
        )
        if not _same_json_value(proposal_policy, configured_policy):
            raise ValueError(
                "JONSWAP proposal adjustment policy differs from its run spec"
            )
        adjustment = _required_mapping(
            configured_policy,
            "nonlinear_adjustment",
        )
        if not _same_json_value(
            adjustment,
            expected_execution["jonswap_adjustment"],
        ):
            raise ValueError(
                "JONSWAP bucketing policy differs from the paper adjustment contract"
            )


def _required_string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _required_sha256(mapping: Mapping[str, object], name: str) -> str:
    value = _required_string(mapping, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _ordered_cell_ids(
    configuration: Mapping[str, object],
) -> tuple[str, ...]:
    """Read cell order from raw JSON or an immutable run specification."""

    value = configuration.get("ordered_cell_ids")
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(cell_id, str) for cell_id in value)
    ):
        raise TypeError("chunk ordered_cell_ids must be a nonempty string sequence")
    return tuple(value)


def _configuration_record(
    spec: AcceptedQuotaRunSpec,
) -> Mapping[str, object]:
    """Return the strict JSON form of an immutable run configuration."""

    return _required_mapping(spec.to_json_record(), "configuration")


def _reconstruct_spec(
    summary: Mapping[str, object],
    *,
    root: Path,
) -> AcceptedQuotaRunSpec:
    record = _required_mapping(summary, "run_spec")
    family = _required_string(record, "family_name")
    if family not in FAMILY_IDS:
        raise ValueError(f"unknown paper-dataset family: {family}")
    family_id = PhysicalFamilyId(_required_integer(record, "family_id"))
    if family_id is not FAMILY_IDS[family]:
        raise ValueError("run summary family name and ID disagree")
    split = SplitId(_required_string(record, "split_id"))
    raw_quotas = record.get("quotas")
    if not isinstance(raw_quotas, list) or not raw_quotas:
        raise TypeError("run_spec quotas must be a nonempty list")
    quotas = tuple(
        CellQuota(
            cell_id=_required_string(quota, "cell_id"),
            target_accepted=_required_integer(quota, "target_accepted"),
        )
        for quota in raw_quotas
        if isinstance(quota, Mapping)
    )
    if len(quotas) != len(raw_quotas):
        raise TypeError("every run_spec quota must be a JSON object")
    raw_codes = _required_mapping(record, "cell_codes")
    cell_codes = {
        str(cell_id): _required_integer(raw_codes, str(cell_id))
        for cell_id in raw_codes
    }
    return AcceptedQuotaRunSpec(
        root=root,
        family_name=family,
        family_id=family_id,
        revision_id=_required_integer(record, "revision_id"),
        split_id=split,
        stream_id=_required_integer(record, "stream_id"),
        quotas=quotas,
        cell_codes=cell_codes,
        batch_size=_required_integer(record, "batch_size", minimum=1),
        first_attempt_index=_required_integer(record, "first_attempt_index"),
        maximum_attempts_per_accepted_case=_required_integer(
            record,
            "maximum_attempts_per_accepted_case",
            minimum=1,
        ),
        configuration=_required_mapping(record, "configuration"),
    )


def load_completed_chunk(
    summary_path: Path,
    *,
    generation_compatibility_policy: GenerationCompatibilityPolicy | None = None,
) -> CompletedChunk:
    """Validate one completion summary against its immutable run artifacts."""

    path = Path(summary_path).expanduser().resolve()
    summary = _read_json_object(path)
    if summary.get("schema") != "paper_dataset_quota_summary_v1":
        raise ValueError(f"{path} is not a paper-dataset quota summary")
    if summary.get("status") != "complete":
        raise RuntimeError(f"{path} does not describe a completed quota run")
    root = Path(_required_string(summary, "output_root")).expanduser().resolve()
    if path.parent != root:
        raise ValueError("chunk summary must live directly in its output root")
    spec = _reconstruct_spec(summary, root=root)
    fingerprint = _required_string(summary, "configuration_fingerprint")
    if fingerprint != spec.config_fingerprint:
        raise RuntimeError("chunk summary fingerprint disagrees with its run spec")

    configuration = _configuration_record(spec)
    if configuration.get("schema") != "paper_dataset_quota_configuration_v1":
        raise ValueError("chunk does not use the unified generation contract")
    current_revision = paper_dataset_revision_id(FAMILY_IDS[spec.family_name])
    if spec.revision_id != current_revision:
        raise ValueError(
            f"{spec.family_name} chunk revision {spec.revision_id} is not "
            f"current paper-dataset revision {current_revision}"
        )
    execution_key = (
        "contract" if spec.family_name == "stokes" else "trajectory_execution"
    )
    run_execution = _required_mapping(configuration, execution_key)
    summary_execution = _required_mapping(summary, "execution")
    if not _same_json_value(summary_execution, run_execution):
        raise ValueError("chunk summary execution differs from its own run spec")
    raw_sources = _required_mapping(configuration, "source_sha256")
    sources = {
        str(source_path): _required_sha256(raw_sources, str(source_path))
        for source_path in raw_sources
    }
    generation_resolution = _resolve_generation_compatibility(
        configuration,
        family=spec.family_name,
        family_id=int(spec.family_id),
        revision_id=spec.revision_id,
        execution_record=run_execution,
        source_sha256=sources,
        policy=generation_compatibility_policy,
    )
    accepted_count = _required_integer(
        configuration,
        "accepted_case_count",
        minimum=1,
    )
    accepted_before = _required_integer(configuration, "accepted_cases_before")
    accepted_after = _required_integer(configuration, "accepted_cases_after")
    if accepted_after != accepted_before + accepted_count:
        raise ValueError("chunk cumulative accepted interval is inconsistent")
    dependency_environment = _required_mapping(
        configuration,
        "dependency_environment",
    )
    execution_platform = _required_string(
        configuration,
        "execution_platform",
    )

    ordered_cells = _ordered_cell_ids(configuration)
    before = balanced_cell_quotas(
        ordered_cells,
        accepted_case_count=accepted_before,
    )
    after = balanced_cell_quotas(
        ordered_cells,
        accepted_case_count=accepted_after,
    )
    expected_increment = tuple(
        after_quota.target_accepted - before_quota.target_accepted
        for before_quota, after_quota in zip(before, after)
    )
    actual_increment = tuple(quota.target_accepted for quota in spec.quotas)
    if tuple(quota.cell_id for quota in spec.quotas) != tuple(ordered_cells):
        raise ValueError("chunk quota order differs from ordered_cell_ids")
    if actual_increment != expected_increment:
        raise ValueError("chunk quotas are not cumulative balanced differences")

    state = scan_quota_run(spec)
    if (
        not state.complete
        or state.pending is not None
        or state.terminal_failure is not None
        or state.attempt_limit_failure is not None
    ):
        raise RuntimeError("chunk artifacts are not in a completed quota state")
    if sum(state.accepted_by_cell.values()) != accepted_count:
        raise RuntimeError("chunk artifacts contain the wrong accepted count")
    for paths in state.committed:
        _validate_committed_proposal_contract(paths, spec=spec)
    counts = _required_mapping(summary, "counts")
    if _required_integer(counts, "accepted") != accepted_count:
        raise RuntimeError("chunk summary accepted count is inconsistent")
    attempted_count = _required_integer(counts, "attempted", minimum=accepted_count)
    if attempted_count != sum(state.attempted_by_cell.values()):
        raise RuntimeError("chunk summary attempted count is inconsistent")

    return CompletedChunk(
        summary_path=path,
        summary_sha256=file_sha256(path),
        root=root,
        family=spec.family_name,
        revision_id=spec.revision_id,
        split=spec.split_id,
        stream_id=spec.stream_id,
        accepted_before=accepted_before,
        accepted_count=accepted_count,
        accepted_after=accepted_after,
        attempted_count=attempted_count,
        fingerprint=fingerprint,
        dependency_fingerprint=canonical_json_sha256(dependency_environment),
        execution_fingerprint=canonical_json_sha256(run_execution),
        generation_compatibility_id=(
            generation_resolution.compatibility_id
            if generation_resolution is not None
            else None
        ),
        source_fingerprint=canonical_json_sha256(sources),
        source_sha256=sources,
        execution_platform=execution_platform,
        batches=state.committed,
    )


def validate_combined_plan(
    chunks: Sequence[CompletedChunk],
) -> CombinedDatasetPlan:
    """Require four equal, nested family sequences within every split."""

    values = tuple(chunks)
    if not values:
        raise ValueError("at least one completed chunk is required")
    if len({chunk.summary_path for chunk in values}) != len(values):
        raise ValueError("chunk summaries must be unique")
    split_order = tuple(
        split
        for split in (SplitId.TRAIN, SplitId.VALIDATION, SplitId.TEST)
        if any(chunk.split is split for chunk in values)
    )
    if not split_order:
        raise ValueError("combined view contains no recognized data split")
    accepted_by_split: dict[str, int] = {}
    attempted_by_split: dict[str, int] = {}
    ordered: list[CompletedChunk] = []
    for split in split_order:
        split_chunks = tuple(chunk for chunk in values if chunk.split is split)
        observed_families = {chunk.family for chunk in split_chunks}
        if observed_families != set(FAMILY_ORDER):
            raise ValueError(f"{split.value} view requires all four physical families")
        totals: dict[str, int] = {}
        for family in FAMILY_ORDER:
            family_chunks = sorted(
                (chunk for chunk in split_chunks if chunk.family == family),
                key=lambda chunk: chunk.accepted_before,
            )
            cursor = 0
            streams: set[int] = set()
            roots: set[Path] = set()
            for chunk in family_chunks:
                if chunk.accepted_before != cursor:
                    raise ValueError(
                        f"{split.value}/{family} chunks have a gap or overlap "
                        f"at cumulative count {cursor}"
                    )
                if chunk.accepted_after != cursor + chunk.accepted_count:
                    raise ValueError(
                        f"{split.value}/{family} chunk interval is inconsistent"
                    )
                if chunk.stream_id in streams:
                    raise ValueError(
                        f"{split.value}/{family} additive chunks repeat a stream ID"
                    )
                if chunk.root in roots:
                    raise ValueError(
                        f"{split.value}/{family} additive chunks repeat an output root"
                    )
                streams.add(chunk.stream_id)
                roots.add(chunk.root)
                cursor = chunk.accepted_after
                ordered.append(chunk)
            totals[family] = cursor
        if len(set(totals.values())) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted-case counts in every family"
            )
        accepted_by_split[split.value] = next(iter(totals.values()))
        attempted_by_split[split.value] = sum(
            chunk.attempted_count for chunk in split_chunks
        )

    fingerprints = tuple(sorted(chunk.fingerprint for chunk in values))
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("completed chunks must have distinct fingerprints")
    if len({chunk.dependency_fingerprint for chunk in values}) != 1:
        raise ValueError("completed chunks use different dependency environments")
    for source_path in SHARED_TARGET_SOURCE_PATHS:
        digests = {chunk.source_sha256.get(source_path) for chunk in values}
        if None in digests:
            raise ValueError(
                f"completed chunks omit shared target source {source_path}"
            )
        if len(digests) != 1:
            raise ValueError(
                f"completed chunks use different {source_path} implementations"
            )
    for chunk in values:
        current_revision = paper_dataset_revision_id(FAMILY_IDS[chunk.family])
        if chunk.revision_id != current_revision:
            raise ValueError(
                f"{chunk.family} chunk revision {chunk.revision_id} is not "
                f"current paper-dataset revision {current_revision}"
            )
    for family in FAMILY_ORDER:
        family_chunks = tuple(chunk for chunk in values if chunk.family == family)
        revisions = {chunk.revision_id for chunk in family_chunks}
        if len(revisions) != 1:
            raise ValueError(f"one paper view cannot mix {family} generator revisions")
    family_revision_keys = {(chunk.family, chunk.revision_id) for chunk in values}
    for family, revision_id in sorted(family_revision_keys):
        compatible = tuple(
            chunk
            for chunk in values
            if (chunk.family, chunk.revision_id) == (family, revision_id)
        )
        sources_match = len({chunk.source_fingerprint for chunk in compatible}) == 1
        if not sources_match:
            raise ValueError(
                f"{family} revision {revision_id} chunks use different source mappings"
            )
        if len({chunk.execution_platform for chunk in compatible}) != 1:
            raise ValueError(
                f"{family} revision {revision_id} chunks use different "
                "execution platforms"
            )
    return CombinedDatasetPlan(
        chunks=tuple(ordered),
        splits=split_order,
        accepted_cases_per_family_by_split=accepted_by_split,
        attempted_cases_by_split=attempted_by_split,
        attempted_cases=sum(attempted_by_split.values()),
        accepted_cases=(len(FAMILY_ORDER) * sum(accepted_by_split.values())),
        expected_rows=(
            sum(accepted_by_split.values()) * sum(ROWS_PER_ACCEPTED_CASE.values())
        ),
        fingerprints=fingerprints,
    )


def preflight(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None,
) -> tuple[CombinedDatasetPlan, Path, str, dict[str, object]]:
    """Load every completed chunk and return a write-free combined plan."""

    chunks = tuple(
        load_completed_chunk(
            path,
            generation_compatibility_policy=(PAPER_GENERATION_COMPATIBILITY_POLICY),
        )
        for path in summary_paths
    )
    plan = validate_combined_plan(chunks)
    root = Path(output_root).expanduser().resolve()
    if root in {chunk.root for chunk in plan.chunks}:
        raise ValueError("combined-view output root must differ from every chunk root")
    view_name = name or (
        f"paper_dataset_{plan.splits[0].value}"
        if len(plan.splits) == 1
        else "paper_dataset_all_splits"
    )
    if not view_name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in view_name
    ):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    chunk_records = [
        {
            "family": chunk.family,
            "revision_id": chunk.revision_id,
            "split": chunk.split.value,
            "stream_id": chunk.stream_id,
            "accepted_before": chunk.accepted_before,
            "accepted_count": chunk.accepted_count,
            "accepted_after": chunk.accepted_after,
            "attempted_count": chunk.attempted_count,
            "configuration_fingerprint": chunk.fingerprint,
            "dependency_fingerprint": chunk.dependency_fingerprint,
            "execution_fingerprint": chunk.execution_fingerprint,
            "generation_compatibility_id": (chunk.generation_compatibility_id),
            "source_fingerprint": chunk.source_fingerprint,
            "execution_platform": chunk.execution_platform,
            "summary_path": str(chunk.summary_path),
            "summary_sha256": chunk.summary_sha256,
            "committed_batches": len(chunk.batches),
        }
        for chunk in plan.chunks
    ]
    record: dict[str, object] = {
        "schema": "paper_dataset_combined_view_preflight_v1",
        "mode": "dry_run",
        "no_view_written": True,
        "output_root": str(root),
        "view_name": view_name,
        "splits": [split.value for split in plan.splits],
        "accepted_cases_per_family_by_split": dict(
            plan.accepted_cases_per_family_by_split
        ),
        "accepted_cases_total": plan.accepted_cases,
        "attempted_cases_total": plan.attempted_cases,
        "attempted_cases_by_split": dict(plan.attempted_cases_by_split),
        "expected_rows": plan.expected_rows,
        "expected_rows_by_split_and_family": {
            split.value: {
                family: (
                    plan.accepted_cases_per_family_by_split[split.value]
                    * ROWS_PER_ACCEPTED_CASE[family]
                )
                for family in FAMILY_ORDER
            }
            for split in plan.splits
        },
        "configuration_fingerprints": list(plan.fingerprints),
        "chunks": chunk_records,
    }
    return plan, root, view_name, record


def _result_case_vectors(
    result: Mapping[str, object],
    *,
    expected_case_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read exact trajectory decisions from one committed result record."""

    if result.get("schema") != "paper_dataset_batch_result_v1":
        raise RuntimeError("combined-view source result has an unknown schema")
    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != expected_case_ids.size:
        raise RuntimeError(
            "combined-view source result does not contain every proposed case"
        )
    accepted: list[bool] = []
    required_bits: list[int] = []
    evaluated_bits: list[int] = []
    failed_bits: list[int] = []
    first_rows: list[int] = []
    row_counts: list[int] = []
    for expected_case_id, raw_case in zip(expected_case_ids, raw_cases):
        if not isinstance(raw_case, Mapping):
            raise TypeError("combined-view source result case is not an object")
        case_id = raw_case.get("case_id")
        if (
            not isinstance(case_id, int)
            or isinstance(case_id, bool)
            or case_id != int(expected_case_id)
        ):
            raise RuntimeError("combined-view source result case identity is incorrect")
        accepted_value = raw_case.get("accepted")
        if not isinstance(accepted_value, bool):
            raise TypeError("combined-view source accepted flag is not boolean")
        integer_values: dict[str, int] = {}
        for name in (
            "required_bits",
            "evaluated_bits",
            "failed_bits",
            "first_row",
            "row_count",
        ):
            value = raw_case.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"combined-view source result {name} is not an integer")
            integer_values[name] = value
        required = integer_values["required_bits"]
        evaluated = integer_values["evaluated_bits"]
        failed = integer_values["failed_bits"]
        if any(not 0 <= value < 1 << 32 for value in (required, evaluated, failed)):
            raise ValueError("combined-view source quality mask exceeds uint32")
        if failed & ~evaluated:
            raise ValueError("combined-view source failed bits are not evaluated bits")
        mask_accepts = not (required & ~evaluated or required & failed)
        if accepted_value != mask_accepts:
            raise RuntimeError(
                "combined-view source acceptance disagrees with quality masks"
            )
        first_row = integer_values["first_row"]
        row_count = integer_values["row_count"]
        if accepted_value:
            if first_row < 0 or row_count <= 0:
                raise RuntimeError("combined-view source accepted case owns no rows")
        elif first_row != -1 or row_count != 0:
            raise RuntimeError("combined-view source rejected case owns shard rows")
        accepted.append(accepted_value)
        required_bits.append(required)
        evaluated_bits.append(evaluated)
        failed_bits.append(failed)
        first_rows.append(first_row)
        row_counts.append(row_count)
    return {
        "accepted": np.asarray(accepted, dtype=np.bool_),
        "required_bits": np.asarray(required_bits, dtype=np.uint32),
        "evaluated_bits": np.asarray(evaluated_bits, dtype=np.uint32),
        "failed_bits": np.asarray(failed_bits, dtype=np.uint32),
        "first_row": np.asarray(first_rows, dtype=np.int64),
        "row_count": np.asarray(row_counts, dtype=np.int32),
    }


def _audit_map_sources(
    arrays: Mapping[str, np.ndarray],
    *,
    plan: CombinedDatasetPlan,
) -> dict[str, int]:
    """Bind every trajectory-map entry to its immutable batch source."""

    trajectory_cursor = 0
    row_cursor = 0
    shard_cursor = 0
    batch_count = 0
    for chunk in plan.chunks:
        chunk_attempted = 0
        for paths in chunk.batches:
            with np.load(paths.proposal, allow_pickle=False) as proposal:
                required_proposal = {
                    "family_id",
                    "revision_id",
                    "split_id",
                    "batch_id",
                    "case_id",
                    "cell_id",
                    "config_fingerprint",
                }
                missing = required_proposal.difference(proposal.files)
                if missing:
                    raise RuntimeError(
                        "combined-view source proposal omits "
                        + ", ".join(sorted(missing))
                    )
                family_id = int(np.asarray(proposal["family_id"]).item())
                revision_id = int(np.asarray(proposal["revision_id"]).item())
                split_id = int(np.asarray(proposal["split_id"]).item())
                fingerprint = str(np.asarray(proposal["config_fingerprint"]).item())
                case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
                cell_ids = np.asarray(proposal["cell_id"], dtype=np.int32)
            trajectory_count = int(case_ids.size)
            if case_ids.ndim != 1 or cell_ids.shape != case_ids.shape:
                raise RuntimeError(
                    "combined-view source proposal case vectors are malformed"
                )
            if (
                family_id != int(FAMILY_IDS[chunk.family])
                or revision_id != chunk.revision_id
                or split_id != split_code(chunk.split)
                or fingerprint != chunk.fingerprint
            ):
                raise RuntimeError(
                    "combined-view source proposal differs from its chunk identity"
                )

            result = _read_json_object(paths.result)
            if result.get("config_fingerprint") != chunk.fingerprint:
                raise RuntimeError(
                    "combined-view source result has the wrong fingerprint"
                )
            if result.get("proposal_sha256") != file_sha256(paths.proposal):
                raise RuntimeError(
                    "combined-view source result has the wrong proposal hash"
                )
            decisions = _result_case_vectors(
                result,
                expected_case_ids=case_ids,
            )
            trajectory_slice = slice(
                trajectory_cursor,
                trajectory_cursor + trajectory_count,
            )
            expected_first_rows = np.where(
                decisions["accepted"],
                row_cursor + decisions["first_row"],
                -1,
            ).astype(np.int64, copy=False)
            trajectory_expectations = {
                "trajectory_family_id": np.full(
                    trajectory_count,
                    family_id,
                    dtype=np.int16,
                ),
                "trajectory_revision_id": np.full(
                    trajectory_count,
                    revision_id,
                    dtype=np.int16,
                ),
                "trajectory_split_id": np.full(
                    trajectory_count,
                    split_id,
                    dtype=np.uint8,
                ),
                "trajectory_case_id": case_ids,
                "trajectory_cell_id": cell_ids,
                "trajectory_accepted": decisions["accepted"],
                "trajectory_required_bits": decisions["required_bits"],
                "trajectory_evaluated_bits": decisions["evaluated_bits"],
                "trajectory_failed_bits": decisions["failed_bits"],
                "trajectory_first_row": expected_first_rows,
                "trajectory_row_count": decisions["row_count"],
            }
            for name, expected in trajectory_expectations.items():
                if not np.array_equal(arrays[name][trajectory_slice], expected):
                    raise RuntimeError(
                        f"combined trajectory-map field {name} differs from "
                        "its committed source"
                    )

            shard_row_count = 0
            if paths.shard.exists():
                with np.load(paths.shard, allow_pickle=False) as shard:
                    if "case_local_index" not in shard or "frame_index" not in shard:
                        raise RuntimeError(
                            "combined-view source shard omits row identity"
                        )
                    case_local_index = np.asarray(
                        shard["case_local_index"],
                        dtype=np.int32,
                    )
                    frame_index = np.asarray(shard["frame_index"], dtype=np.int32)
                if frame_index.shape != case_local_index.shape:
                    raise RuntimeError(
                        "combined-view source shard row vectors are malformed"
                    )
                shard_row_count = int(case_local_index.size)
                recorded_shard_hash = result.get("shard_sha256")
                if not isinstance(recorded_shard_hash, str):
                    raise RuntimeError(
                        "combined-view source result omits its shard hash"
                    )
                row_slice = slice(row_cursor, row_cursor + shard_row_count)
                row_expectations = {
                    "trajectory_index": case_local_index + trajectory_cursor,
                    "frame_index": frame_index,
                    "shard_index": np.full(
                        shard_row_count,
                        shard_cursor,
                        dtype=np.int32,
                    ),
                    "shard_row": np.arange(shard_row_count, dtype=np.int64),
                }
                for name, expected in row_expectations.items():
                    if not np.array_equal(arrays[name][row_slice], expected):
                        raise RuntimeError(
                            f"combined trajectory-map field {name} differs "
                            "from its committed source"
                        )
                shard_cursor += 1
            elif result.get("shard_sha256") is not None:
                raise RuntimeError(
                    "combined-view source result references a missing shard"
                )
            if int(np.sum(decisions["row_count"], dtype=np.int64)) != shard_row_count:
                raise RuntimeError(
                    "combined-view source result and shard row totals disagree"
                )
            trajectory_cursor += trajectory_count
            chunk_attempted += trajectory_count
            row_cursor += shard_row_count
            batch_count += 1
        if chunk_attempted != chunk.attempted_count:
            raise RuntimeError(
                "combined-view source batches have the wrong chunk attempt count"
            )
    if trajectory_cursor != plan.attempted_cases or row_cursor != plan.expected_rows:
        raise RuntimeError("combined-view source totals differ from its plan")
    return {
        "committed_batches": batch_count,
        "committed_shards": shard_cursor,
        "source_trajectories": trajectory_cursor,
        "source_rows": row_cursor,
    }


def _audit_manifest_sources(
    manifest: Mapping[str, object],
    *,
    plan: CombinedDatasetPlan,
    manifest_parent: Path,
) -> dict[str, int]:
    """Require exact ordered batch and shard records in the final manifest."""

    expected_batches: list[dict[str, object]] = []
    expected_shards: list[dict[str, object]] = []
    for batch_index, (chunk, paths) in enumerate(
        (chunk, paths) for chunk in plan.chunks for paths in chunk.batches
    ):
        with np.load(paths.proposal, allow_pickle=False) as proposal:
            family_id = int(np.asarray(proposal["family_id"]).item())
            revision_id = int(np.asarray(proposal["revision_id"]).item())
            split_id = int(np.asarray(proposal["split_id"]).item())
            batch_id = int(np.asarray(proposal["batch_id"]).item())
            trajectory_count = int(np.asarray(proposal["case_id"]).size)
            fingerprint = str(np.asarray(proposal["config_fingerprint"]).item())
        result = _read_json_object(paths.result)
        raw_cases = result.get("cases")
        if not isinstance(raw_cases, list):
            raise RuntimeError("combined-view source result cases are malformed")
        accepted_count = sum(
            int(isinstance(case, Mapping) and case.get("accepted") is True)
            for case in raw_cases
        )
        shard_index: int | None = None
        shard_row_count = 0
        if paths.shard.exists():
            with np.load(paths.shard, allow_pickle=False) as shard:
                shard_row_count = int(np.asarray(shard["case_local_index"]).size)
            shard_index = len(expected_shards)
            shard_hash = result.get("shard_sha256")
            if not isinstance(shard_hash, str):
                raise RuntimeError("combined-view source result omits its shard hash")
            expected_shards.append(
                {
                    "path": os.path.relpath(
                        paths.shard.resolve(),
                        start=manifest_parent.resolve(),
                    ),
                    "sha256": shard_hash,
                    "n_rows": shard_row_count,
                    "batch_index": batch_index,
                    "configuration_fingerprint": fingerprint,
                }
            )
        expected_batches.append(
            {
                "proposal_path": os.path.relpath(
                    paths.proposal.resolve(),
                    start=manifest_parent.resolve(),
                ),
                "proposal_sha256": file_sha256(paths.proposal),
                "result_path": os.path.relpath(
                    paths.result.resolve(),
                    start=manifest_parent.resolve(),
                ),
                "result_sha256": file_sha256(paths.result),
                "shard_index": shard_index,
                "configuration_fingerprint": fingerprint,
                "family_id": family_id,
                "revision_id": revision_id,
                "split_id": split_id,
                "batch_id": batch_id,
                "n_attempted_trajectories": trajectory_count,
                "n_accepted_trajectories": accepted_count,
                "n_rows": shard_row_count,
            }
        )
    raw_batches = manifest.get("dataset_batches")
    raw_shards = manifest.get("dataset_shards")
    if not _same_json_value(raw_batches, expected_batches):
        raise RuntimeError(
            "combined manifest batch records differ from ordered sources"
        )
    if not _same_json_value(raw_shards, expected_shards):
        raise RuntimeError(
            "combined manifest shard records differ from ordered sources"
        )
    return {
        "committed_batches": len(expected_batches),
        "committed_shards": len(expected_shards),
    }


def _audit_trajectory_map_against_plan(
    path: Path,
    *,
    plan: CombinedDatasetPlan,
) -> dict[str, object]:
    """Independently recover release counts and row ownership from the map."""

    required_arrays = tuple(TRAJECTORY_MAP_DTYPES)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(required_arrays):
            raise RuntimeError(
                "combined trajectory map does not have the exact schema-v2 arrays"
            )
        schema = np.asarray(stored["schema_version"])
        arrays = {
            name: np.asarray(stored[name])
            for name in required_arrays
            if name != "schema_version"
        }

    if (
        schema.ndim != 0
        or schema.dtype != TRAJECTORY_MAP_DTYPES["schema_version"]
        or int(schema) != TRAJECTORY_MAP_SCHEMA_VERSION
    ):
        raise RuntimeError("combined trajectory map has the wrong schema version")

    trajectory_fields = (
        "trajectory_family_id",
        "trajectory_revision_id",
        "trajectory_split_id",
        "trajectory_case_id",
        "trajectory_cell_id",
        "trajectory_accepted",
        "trajectory_required_bits",
        "trajectory_evaluated_bits",
        "trajectory_failed_bits",
        "trajectory_first_row",
        "trajectory_row_count",
    )
    if any(arrays[name].ndim != 1 for name in arrays):
        raise RuntimeError("combined trajectory-map arrays must be one-dimensional")
    trajectory_count = int(arrays["trajectory_family_id"].size)
    for name in trajectory_fields:
        if arrays[name].size != trajectory_count:
            raise RuntimeError(
                f"combined trajectory-map field {name} has the wrong length"
            )
    if trajectory_count != plan.attempted_cases:
        raise RuntimeError("combined trajectory map has the wrong attempted-case total")
    row_fields = (
        "trajectory_index",
        "frame_index",
        "shard_index",
        "shard_row",
    )
    for name in row_fields:
        if arrays[name].size != plan.expected_rows:
            raise RuntimeError(
                f"combined trajectory-map field {name} has the wrong length"
            )
    for name, values in arrays.items():
        if values.dtype != TRAJECTORY_MAP_DTYPES[name]:
            raise RuntimeError(
                f"combined trajectory-map field {name} has the wrong dtype"
            )

    family_ids = arrays["trajectory_family_id"]
    revision_ids = arrays["trajectory_revision_id"]
    split_ids = arrays["trajectory_split_id"]
    accepted = arrays["trajectory_accepted"]
    first_rows = arrays["trajectory_first_row"]
    row_counts = arrays["trajectory_row_count"]
    row_owners = arrays["trajectory_index"]
    if np.any(row_counts < 0):
        raise RuntimeError("combined trajectory map contains a negative row count")

    family_codes = {family: int(FAMILY_IDS[family]) for family in FAMILY_ORDER}
    split_codes = {split.value: split_code(split) for split in SplitId}
    if not np.all(np.isin(family_ids, tuple(family_codes.values()))):
        raise RuntimeError("combined trajectory map contains an unknown family ID")
    if not np.all(np.isin(split_ids, tuple(split_codes.values()))):
        raise RuntimeError("combined trajectory map contains an unknown split ID")

    expected_attempted = {
        (split.value, family): sum(
            chunk.attempted_count
            for chunk in plan.chunks
            if chunk.split is split and chunk.family == family
        )
        for split in SplitId
        for family in FAMILY_ORDER
    }
    expected_accepted = {
        (split.value, family): plan.accepted_cases_per_family_by_split.get(
            split.value,
            0,
        )
        for split in SplitId
        for family in FAMILY_ORDER
    }
    attempted_record: dict[str, dict[str, int]] = {}
    accepted_record: dict[str, dict[str, int]] = {}
    rows_record: dict[str, dict[str, int]] = {}
    for split in SplitId:
        attempted_record[split.value] = {}
        accepted_record[split.value] = {}
        rows_record[split.value] = {}
        for family in FAMILY_ORDER:
            cell = (split_ids == split_codes[split.value]) & (
                family_ids == family_codes[family]
            )
            observed_attempted = int(np.count_nonzero(cell))
            observed_accepted = int(np.count_nonzero(cell & accepted))
            observed_rows = int(np.sum(row_counts[cell & accepted], dtype=np.int64))
            expected_rows = (
                expected_accepted[(split.value, family)]
                * ROWS_PER_ACCEPTED_CASE[family]
            )
            if observed_attempted != expected_attempted[(split.value, family)]:
                raise RuntimeError(
                    f"combined trajectory map has the wrong attempted count "
                    f"for {split.value}/{family}"
                )
            if observed_accepted != expected_accepted[(split.value, family)]:
                raise RuntimeError(
                    f"combined trajectory map has the wrong accepted count "
                    f"for {split.value}/{family}"
                )
            if observed_rows != expected_rows:
                raise RuntimeError(
                    f"combined trajectory map has the wrong accepted-row sum "
                    f"for {split.value}/{family}"
                )
            attempted_record[split.value][family] = observed_attempted
            accepted_record[split.value][family] = observed_accepted
            rows_record[split.value][family] = observed_rows

    expected_family_ids = np.concatenate(
        tuple(
            np.full(
                chunk.attempted_count,
                int(FAMILY_IDS[chunk.family]),
                dtype=np.int16,
            )
            for chunk in plan.chunks
        )
    )
    expected_revision_ids = np.concatenate(
        tuple(
            np.full(
                chunk.attempted_count,
                chunk.revision_id,
                dtype=np.int16,
            )
            for chunk in plan.chunks
        )
    )
    expected_split_ids = np.concatenate(
        tuple(
            np.full(
                chunk.attempted_count,
                split_code(chunk.split),
                dtype=np.uint8,
            )
            for chunk in plan.chunks
        )
    )
    if not (
        np.array_equal(family_ids, expected_family_ids)
        and np.array_equal(revision_ids, expected_revision_ids)
        and np.array_equal(split_ids, expected_split_ids)
    ):
        raise RuntimeError(
            "combined trajectory-map identity sequence differs from its "
            "ordered completed chunks"
        )

    rows_by_family_code = {
        family_codes[family]: row_count
        for family, row_count in ROWS_PER_ACCEPTED_CASE.items()
    }
    expected_rows_per_trajectory = np.fromiter(
        (rows_by_family_code[int(family_id)] for family_id in family_ids),
        dtype=np.int32,
        count=trajectory_count,
    )
    if np.any(row_counts[accepted] != expected_rows_per_trajectory[accepted]):
        raise RuntimeError(
            "combined trajectory map gives an accepted trajectory the wrong "
            "number of rows"
        )
    if np.any(row_counts[~accepted] != 0):
        raise RuntimeError("combined trajectory map gives rejected cases rows")

    declared_rows = int(np.sum(row_counts, dtype=np.int64))
    if declared_rows != plan.expected_rows or row_owners.size != plan.expected_rows:
        raise RuntimeError("combined trajectory-map total row ownership is wrong")
    if row_owners.size and (
        np.any(row_owners < 0)
        or np.any(row_owners >= trajectory_count)
        or np.any(row_owners[1:] < row_owners[:-1])
    ):
        raise RuntimeError("combined trajectory map has invalid row owners")
    owned_rows = np.bincount(
        row_owners.astype(np.int64, copy=False),
        minlength=trajectory_count,
    )
    if not np.array_equal(owned_rows, row_counts):
        raise RuntimeError(
            "combined trajectory-map row owners disagree with trajectory counts"
        )
    expected_first_rows = np.cumsum(row_counts, dtype=np.int64) - row_counts
    expected_first_rows[~accepted] = -1
    if not np.array_equal(first_rows, expected_first_rows):
        raise RuntimeError(
            "combined trajectory-map first rows disagree with row ownership"
        )
    source_record = _audit_map_sources(arrays, plan=plan)

    return {
        "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
        "attempted_cases_by_split_and_family": attempted_record,
        "accepted_cases_by_split_and_family": accepted_record,
        "accepted_rows_by_split_and_family": rows_record,
        "total_row_ownership": {
            "attempted_trajectories": trajectory_count,
            "declared_rows": declared_rows,
            "row_owner_entries": int(row_owners.size),
            "expected_rows": plan.expected_rows,
        },
        "source_binding": source_record,
    }


def _validate_view(
    view: DatasetViewPaths,
    *,
    plan: CombinedDatasetPlan,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprint": None,
        "configuration_fingerprints": list(plan.fingerprints),
        "trajectory_map_npz": view.trajectory_map.name,
        "requires_trajectory_map": True,
        "n_rows": plan.expected_rows,
        "n_trajectories": plan.attempted_cases,
        "n_accepted_trajectories": plan.accepted_cases,
        "n_accepted_rows": plan.expected_rows,
        "grid": {
            "length": 2.0 * math.pi,
            "nx": 1024,
        },
    }
    for key, value in expected.items():
        if not _same_json_value(manifest.get(key), value):
            raise RuntimeError(
                f"combined view {key} is {manifest.get(key)!r}, expected {value!r}"
            )
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, dict):
        raise TypeError("combined view split_counts must be a JSON object")
    for split in (SplitId.TRAIN, SplitId.VALIDATION, SplitId.TEST):
        accepted_per_family = plan.accepted_cases_per_family_by_split.get(
            split.value,
            0,
        )
        expected_split = {
            "attempted": plan.attempted_cases_by_split.get(split.value, 0),
            "accepted": len(FAMILY_ORDER) * accepted_per_family,
        }
        if not _same_json_value(split_counts.get(split.value), expected_split):
            raise RuntimeError(f"combined view {split.value} split count is wrong")
    if manifest.get("trajectory_map_sha256") != file_sha256(view.trajectory_map):
        raise RuntimeError("combined trajectory-map hash differs from its manifest")
    dataset_contract = manifest.get("dataset_contract")
    if not isinstance(dataset_contract, Mapping):
        raise RuntimeError("combined view has no dataset contract")
    dataset_contract_fingerprint = _required_sha256(
        manifest,
        "dataset_contract_fingerprint",
    )
    if dataset_contract_fingerprint != canonical_json_sha256(dataset_contract):
        raise RuntimeError("combined view dataset-contract fingerprint is incorrect")
    manifest_source_audit = _audit_manifest_sources(
        manifest,
        plan=plan,
        manifest_parent=view.manifest.parent,
    )
    trajectory_map_audit = _audit_trajectory_map_against_plan(
        view.trajectory_map,
        plan=plan,
    )
    return {
        **expected,
        "dataset_contract_fingerprint": dataset_contract_fingerprint,
        "manifest_source_audit": manifest_source_audit,
        "trajectory_map_audit": trajectory_map_audit,
        "manifest": {
            "path": str(view.manifest),
            "bytes": view.manifest.stat().st_size,
            "sha256": file_sha256(view.manifest),
        },
        "trajectory_map": {
            "path": str(view.trajectory_map),
            "bytes": view.trajectory_map.stat().st_size,
            "sha256": file_sha256(view.trajectory_map),
        },
    }


def build_combined_view(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None = None,
) -> CombinedViewResult:
    """Build and validate the combined view after the same read-only preflight."""

    plan, root, view_name, preflight_record = preflight(
        summary_paths,
        output_root=output_root,
        name=name,
    )
    started_at = datetime.now().astimezone()
    started = perf_counter()
    view = build_dataset_view(
        root,
        tuple(batch for chunk in plan.chunks for batch in chunk.batches),
        name=view_name,
        length=2.0 * math.pi,
        expected_fingerprints=plan.fingerprints,
        generation_compatibility_policy=(
            PAPER_GENERATION_COMPATIBILITY_POLICY
            if any(
                chunk.generation_compatibility_id is not None for chunk in plan.chunks
            )
            else None
        ),
    )
    view_record = _validate_view(view, plan=plan)
    summary: dict[str, object] = {
        "schema": "paper_dataset_combined_view_summary_v1",
        "status": "complete",
        "preflight": preflight_record,
        "dataset_view": view_record,
        "timing_seconds": {
            "total": perf_counter() - started,
        },
        "invocation_started_at": started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }
    summary_path = root / f"{view_name}.summary.json"
    write_json_atomic(summary_path, summary)
    return CombinedViewResult(
        summary_path=summary_path,
        view=view,
        summary=summary,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert args.chunk_summary is not None
    if args.execute:
        result = build_combined_view(
            args.chunk_summary,
            output_root=args.output_root,
            name=args.name,
        )
        output: dict[str, object] = {
            "mode": "execute",
            "status": result.summary["status"],
            "summary_path": str(result.summary_path),
            "dataset_view": result.summary["dataset_view"],
            "timing_seconds": result.summary["timing_seconds"],
        }
    else:
        _, _, _, output = preflight(
            args.chunk_summary,
            output_root=args.output_root,
            name=args.name,
        )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
