"""Preflight or build one four-family view from completed quota chunks.

Inputs are the completion summaries written by ``run_paper_corpus_quota.py``.
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
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.manifest import (
    DATASET_VIEW_SCHEMA_VERSION,
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.production import (
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    balanced_cell_quotas,
)
from solver.gen_data.pipeline.quota_driver import (
    AcceptedQuotaRunSpec,
    canonical_json_sha256,
    scan_quota_run,
)


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


@dataclass(frozen=True)
class CompletedChunk:
    """Validated identity and durable batches from one completed quota run."""

    summary_path: Path
    summary_sha256: str
    root: Path
    family: str
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int
    accepted_after: int
    attempted_count: int
    fingerprint: str
    dependency_fingerprint: str
    source_fingerprint: str
    source_sha256: Mapping[str, str]
    execution_platform: str
    batches: tuple[BatchPaths, ...]


@dataclass(frozen=True)
class CombinedCorpusPlan:
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
        help="View basename; defaults to paper_corpus_<split>.",
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


def _required_string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
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
        raise TypeError(
            "chunk ordered_cell_ids must be a nonempty string sequence"
        )
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
        raise ValueError(f"unknown paper-corpus family: {family}")
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


def load_completed_chunk(summary_path: Path) -> CompletedChunk:
    """Validate one completion summary against its immutable run artifacts."""

    path = Path(summary_path).expanduser().resolve()
    summary = _read_json_object(path)
    if summary.get("schema") != "paper_corpus_quota_summary_v1":
        raise ValueError(f"{path} is not a paper-corpus quota summary")
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
    if configuration.get("schema") != "paper_corpus_quota_configuration_v1":
        raise ValueError("chunk does not use the unified generation contract")
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
    raw_sources = _required_mapping(configuration, "source_sha256")
    sources = {
        str(source_path): _required_string(raw_sources, str(source_path))
        for source_path in raw_sources
    }
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
        split=spec.split_id,
        stream_id=spec.stream_id,
        accepted_before=accepted_before,
        accepted_count=accepted_count,
        accepted_after=accepted_after,
        attempted_count=attempted_count,
        fingerprint=fingerprint,
        dependency_fingerprint=canonical_json_sha256(dependency_environment),
        source_fingerprint=canonical_json_sha256(sources),
        source_sha256=sources,
        execution_platform=execution_platform,
        batches=state.committed,
    )


def validate_combined_plan(
    chunks: Sequence[CompletedChunk],
) -> CombinedCorpusPlan:
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
            raise ValueError(
                f"{split.value} view requires all four physical families"
            )
        totals: dict[str, int] = {}
        for family in FAMILY_ORDER:
            family_chunks = sorted(
                (
                    chunk
                    for chunk in split_chunks
                    if chunk.family == family
                ),
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
                        f"{split.value}/{family} additive chunks repeat a "
                        "stream ID"
                    )
                if chunk.root in roots:
                    raise ValueError(
                        f"{split.value}/{family} additive chunks repeat an "
                        "output root"
                    )
                streams.add(chunk.stream_id)
                roots.add(chunk.root)
                cursor = chunk.accepted_after
                ordered.append(chunk)
            totals[family] = cursor
        if len(set(totals.values())) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted-case counts in every "
                "family"
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
    shared_sources: dict[str, str] = {}
    for family in FAMILY_ORDER:
        family_chunks = tuple(
            chunk for chunk in values if chunk.family == family
        )
        if len({chunk.source_fingerprint for chunk in family_chunks}) != 1:
            raise ValueError(f"{family} chunks use different source mappings")
        if len({chunk.execution_platform for chunk in family_chunks}) != 1:
            raise ValueError(f"{family} chunks use different execution platforms")
        for source_path, digest in family_chunks[0].source_sha256.items():
            previous = shared_sources.setdefault(source_path, digest)
            if previous != digest:
                raise ValueError(
                    f"families disagree on shared source {source_path!r}"
                )
    return CombinedCorpusPlan(
        chunks=tuple(ordered),
        splits=split_order,
        accepted_cases_per_family_by_split=accepted_by_split,
        attempted_cases_by_split=attempted_by_split,
        attempted_cases=sum(attempted_by_split.values()),
        accepted_cases=(
            len(FAMILY_ORDER) * sum(accepted_by_split.values())
        ),
        expected_rows=(
            sum(accepted_by_split.values())
            * sum(ROWS_PER_ACCEPTED_CASE.values())
        ),
        fingerprints=fingerprints,
    )


def preflight(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None,
) -> tuple[CombinedCorpusPlan, Path, str, dict[str, object]]:
    """Load every completed chunk and return a write-free combined plan."""

    chunks = tuple(map(load_completed_chunk, summary_paths))
    plan = validate_combined_plan(chunks)
    root = Path(output_root).expanduser().resolve()
    if root in {chunk.root for chunk in plan.chunks}:
        raise ValueError("combined-view output root must differ from every chunk root")
    view_name = name or (
        f"paper_corpus_{plan.splits[0].value}"
        if len(plan.splits) == 1
        else "paper_corpus_all_splits"
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
            "split": chunk.split.value,
            "stream_id": chunk.stream_id,
            "accepted_before": chunk.accepted_before,
            "accepted_count": chunk.accepted_count,
            "accepted_after": chunk.accepted_after,
            "attempted_count": chunk.attempted_count,
            "configuration_fingerprint": chunk.fingerprint,
            "dependency_fingerprint": chunk.dependency_fingerprint,
            "source_fingerprint": chunk.source_fingerprint,
            "execution_platform": chunk.execution_platform,
            "summary_path": str(chunk.summary_path),
            "summary_sha256": chunk.summary_sha256,
            "committed_batches": len(chunk.batches),
        }
        for chunk in plan.chunks
    ]
    record: dict[str, object] = {
        "schema": "paper_corpus_combined_view_preflight_v1",
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


def _validate_view(
    view: DatasetViewPaths,
    *,
    plan: CombinedCorpusPlan,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprint": None,
        "configuration_fingerprints": list(plan.fingerprints),
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
        if manifest.get(key) != value:
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
        if split_counts.get(split.value) != expected_split:
            raise RuntimeError(
                f"combined view {split.value} split count is wrong"
            )
    if manifest.get("trajectory_map_sha256") != file_sha256(view.trajectory_map):
        raise RuntimeError("combined trajectory-map hash differs from its manifest")
    if not isinstance(manifest.get("dataset_contract_fingerprint"), str):
        raise RuntimeError("combined view has no dataset-contract fingerprint")
    return {
        **expected,
        "dataset_contract_fingerprint": manifest[
            "dataset_contract_fingerprint"
        ],
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
        tuple(
            batch
            for chunk in plan.chunks
            for batch in chunk.batches
        ),
        name=view_name,
        length=2.0 * math.pi,
        expected_fingerprints=plan.fingerprints,
    )
    view_record = _validate_view(view, plan=plan)
    summary: dict[str, object] = {
        "schema": "paper_corpus_combined_view_summary_v1",
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
