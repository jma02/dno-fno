"""Bind the completed Stokes revision-2 corpus to the current release record.

The original Stokes completion audit already rescanned every immutable batch
and stored field.  This lightweight wrapper does not repeat that numerical
scan.  It authenticates the original audit, reloads the six completed quota
chunks through the current fail-closed corpus loader, and emits the same
root/split/source identity information used by the newer family audits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The current loader imports the numerical pipeline.  This release binding is
# deliberately CPU-only and must not contend with live generation jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stokes-audit-binding")

from scripts.build_paper_corpus_view import (  # noqa: E402
    CompletedChunk,
    load_completed_chunk,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PhysicalFamilyId,
    SplitId,
    balanced_cell_quotas,
)


AUDIT_SCHEMA = "paper_corpus_stokes_revision2_completion_binding_v1"
LEGACY_AUDIT_SCHEMA = "paper_corpus_cap4_stokes_audit_v1"
EXPECTED_LEGACY_AUDIT_SHA256 = (
    "d59d40f819a7b05f01eac90a31e74121f2ee570641d42e3cf6850b756af136aa"
)
DEFAULT_CORPUS_ROOT = ROOT / "outputs/paper_corpus_cap4_revision2_20260728"
EXPECTED_FAMILY = "stokes"
EXPECTED_FAMILY_ID = int(PhysicalFamilyId.STOKES)
EXPECTED_REVISION_ID = 2
EXPECTED_ACCEPTED_BY_SPLIT = {
    SplitId.TRAIN: 16_384,
    SplitId.VALIDATION: 1_024,
    SplitId.TEST: 1_024,
}
EXPECTED_ACCEPTED = sum(EXPECTED_ACCEPTED_BY_SPLIT.values())
EXPECTED_RETAINED_ROWS = EXPECTED_ACCEPTED
EXPECTED_CELL_IDS = (
    "finite_low",
    "finite_moderate",
    "deep_low",
    "deep_moderate",
)
EXPECTED_SOURCE_FINGERPRINT = (
    "418e799c12d08d9f79046f4d5f1c116a3c814eed15a6652d33dae8000f14ed8f"
)
HISTORICAL_SOURCE_SNAPSHOT_ROOT = (
    ROOT / "reproducibility/source_snapshots/stokes_revision2_60a28ff"
)
HISTORICAL_SOURCE_COMMIT = "60a28ffae394465c6ea295eb4ed6c075fbc756a4"
HISTORICAL_SOURCE_SNAPSHOT_MANIFEST = {
    "name": "SHA256SUMS",
    "bytes": 250,
    "sha256": "520b11b77c3fae7a74be59cc8b51520923859b7ed74865a220a119dc2772fe8e",
}
HISTORICAL_SOURCE_SNAPSHOTS = {
    "scripts/run_paper_corpus_quota.py": {
        "name": "run_paper_corpus_quota.py",
        "bytes": 33_974,
        "sha256": "ab99c067c2f22823fce861a69fd03bd8cc4524effbfd5ab1a5360bfa5e615d98",
    },
    "solver/gen_data/pipeline/manifest.py": {
        "name": "manifest.py",
        "bytes": 28_988,
        "sha256": "e2cc1a20cc01fef9dd0bb84b99485b5cf90da71f7ee035a813893f2f6a37f421",
    },
    "solver/gen_data/pipeline/production.py": {
        "name": "production.py",
        "bytes": 7_854,
        "sha256": "8ed36cf1bd57494f344096e4a508bcfb37b9103cf03ace8a63df2656c11dc735",
    },
}
EXPECTED_GENERATION_SOURCE_PATHS = frozenset(
    {
        "scripts/run_paper_corpus_quota.py",
        "solver/data/stokes_truth_jax.py",
        "solver/gen_data/generate_stokes_dataset.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/stokes_population.py",
        "solver/gen_data/stokes_quota_executor.py",
        "solver/gen_data/stokes_static_pipeline.py",
        "solver/solvers/dno_series_jax.py",
    }
)
REQUIRED_LEGACY_TRUE_CHECKS = (
    "immutable_specs_rebuilt",
    "all_transactions_rescanned",
    "proposal_hashes_verified",
    "result_hashes_verified",
    "shard_hashes_verified",
    "manifest_hashes_verified",
    "trajectory_map_hashes_verified",
    "all_fields_finite",
    "all_depths_positive",
    "all_quality_masks_accept",
)
REQUIRED_LEGACY_ZERO_CHECKS = (
    "pending_batches",
    "terminal_failures",
    "attempt_limit_failures",
)


@dataclass(frozen=True)
class ExpectedChunk:
    """Exact identity of one completed additive Stokes chunk."""

    label: str
    relative_summary: Path
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int

    @property
    def accepted_after(self) -> int:
        return self.accepted_before + self.accepted_count


EXPECTED_CHUNKS = (
    ExpectedChunk(
        "train_00000_02048",
        Path("train/stokes/chunk_00000_02048/paper_corpus_stokes_train.summary.json"),
        SplitId.TRAIN,
        0,
        0,
        2_048,
    ),
    ExpectedChunk(
        "train_02048_02048",
        Path("train/stokes/chunk_02048_02048/paper_corpus_stokes_train.summary.json"),
        SplitId.TRAIN,
        1,
        2_048,
        2_048,
    ),
    ExpectedChunk(
        "train_04096_04096",
        Path("train/stokes/chunk_04096_04096/paper_corpus_stokes_train.summary.json"),
        SplitId.TRAIN,
        2,
        4_096,
        4_096,
    ),
    ExpectedChunk(
        "train_08192_08192",
        Path("train/stokes/chunk_08192_08192/paper_corpus_stokes_train.summary.json"),
        SplitId.TRAIN,
        3,
        8_192,
        8_192,
    ),
    ExpectedChunk(
        "validation_01024",
        Path("validation/stokes/c01024/paper_corpus_stokes_validation.summary.json"),
        SplitId.VALIDATION,
        100,
        0,
        1_024,
    ),
    ExpectedChunk(
        "test_01024",
        Path("test/stokes/c01024/paper_corpus_stokes_test.summary.json"),
        SplitId.TEST,
        200,
        0,
        1_024,
    ),
)


def _read_json_object(path: Path) -> dict[str, object]:
    """Read a strict finite JSON object."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _required_mapping(mapping: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _required_integer(
    mapping: Mapping[str, object], name: str, *, minimum: int = 0
) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _required_sha256(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_lexical(path: Path, *, label: str) -> Path:
    """Return an absolute path without resolving away unsafe components."""

    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ValueError(f"{label} must not contain parent traversal")
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _require_real_directory(path: Path, *, label: str) -> Path:
    """Reject symlink components before resolving an existing directory."""

    absolute = _absolute_lexical(path, label=label)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {current}")
    if not absolute.is_dir():
        raise ValueError(f"{label} is not a real directory")
    return absolute.resolve(strict=True)


def _resolve_regular_file_inside(
    root: Path,
    relative_path: Path,
    *,
    label: str,
) -> Path:
    """Resolve a repository-relative regular file without following symlinks."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} escapes its declared root")
    current = root
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symbolic link")
    if not current.is_file():
        raise ValueError(f"{label} is not a regular file")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes its declared root")
    return resolved


def _resolve_inside(root: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} must be a nonempty relative path")
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ValueError(f"{label} must be relative to its chunk root")
    return _resolve_regular_file_inside(root, raw, label=label)


def _validate_legacy_audit(record: Mapping[str, object]) -> None:
    """Require the original expensive Stokes audit to be a complete pass."""

    if record.get("schema") != LEGACY_AUDIT_SCHEMA:
        raise ValueError("legacy Stokes audit has an unknown schema")
    if record.get("status") != "pass":
        raise RuntimeError("legacy Stokes completion audit did not pass")
    expected_counts = {
        "attempted": EXPECTED_ACCEPTED,
        "accepted": EXPECTED_ACCEPTED,
        "rejected": 0,
        "retained_rows": EXPECTED_RETAINED_ROWS,
    }
    for name, expected in expected_counts.items():
        if _required_integer(record, name) != expected:
            raise RuntimeError(f"legacy Stokes audit has wrong {name}")
    checks = _required_mapping(record, "checks")
    for name in REQUIRED_LEGACY_TRUE_CHECKS:
        if checks.get(name) is not True:
            raise RuntimeError(f"legacy Stokes audit did not prove {name}")
    for name in REQUIRED_LEGACY_ZERO_CHECKS:
        if _required_integer(checks, name) != 0:
            raise RuntimeError(f"legacy Stokes audit reports nonzero {name}")
    for name in ("run_manifest_sha256", "run_trajectory_map_sha256"):
        digests = _required_mapping(record, name)
        if set(digests) != {chunk.label for chunk in EXPECTED_CHUNKS}:
            raise RuntimeError(f"legacy Stokes audit has incomplete {name}")
        for label in digests:
            _required_sha256(digests, label)


def _load_legacy_audit(
    path: Path, *, expected_sha256: str = EXPECTED_LEGACY_AUDIT_SHA256
) -> dict[str, object]:
    """Authenticate the frozen full numerical audit before trusting it."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy Stokes audit must be a regular non-symlink file")
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("legacy Stokes audit SHA-256 differs")
    record = _read_json_object(path)
    _validate_legacy_audit(record)
    return record


def _validate_chunk_plan(chunks: Sequence[CompletedChunk]) -> None:
    """Require the six source-bound chunks to match the frozen layout."""

    if len(chunks) != len(EXPECTED_CHUNKS):
        raise RuntimeError("Stokes release requires exactly six chunks")
    for expected, chunk in zip(EXPECTED_CHUNKS, chunks):
        observed = (
            chunk.family,
            chunk.revision_id,
            chunk.split,
            chunk.stream_id,
            chunk.accepted_before,
            chunk.accepted_count,
            chunk.accepted_after,
        )
        required = (
            EXPECTED_FAMILY,
            EXPECTED_REVISION_ID,
            expected.split,
            expected.stream_id,
            expected.accepted_before,
            expected.accepted_count,
            expected.accepted_after,
        )
        if observed != required:
            raise RuntimeError(f"Stokes chunk {expected.label} has wrong identity")
        if chunk.attempted_count != chunk.accepted_count:
            raise RuntimeError(
                f"Stokes chunk {expected.label} unexpectedly rejected cases"
            )
    for attribute in (
        "dependency_fingerprint",
        "execution_fingerprint",
        "source_fingerprint",
        "execution_platform",
    ):
        if len({getattr(chunk, attribute) for chunk in chunks}) != 1:
            raise RuntimeError(f"Stokes chunks disagree on {attribute}")


def _expected_chunk_cell_counts(expected: ExpectedChunk) -> dict[str, int]:
    before = balanced_cell_quotas(
        EXPECTED_CELL_IDS,
        accepted_case_count=expected.accepted_before,
    )
    after = balanced_cell_quotas(
        EXPECTED_CELL_IDS,
        accepted_case_count=expected.accepted_after,
    )
    return {
        after_quota.cell_id: after_quota.target_accepted - before_quota.target_accepted
        for before_quota, after_quota in zip(before, after)
    }


def _validate_summary_cell_counts(
    summary: Mapping[str, object],
    *,
    expected: ExpectedChunk,
) -> dict[str, int]:
    """Require exact per-cell counts for a complete zero-rejection chunk."""

    counts = _required_mapping(summary, "counts")
    if (
        _required_integer(counts, "attempted") != expected.accepted_count
        or _required_integer(counts, "accepted") != expected.accepted_count
        or _required_integer(counts, "rejected") != 0
    ):
        raise ValueError(f"Stokes chunk {expected.label} has wrong summary counts")
    by_cell = _required_mapping(counts, "by_cell")
    expected_counts = _expected_chunk_cell_counts(expected)
    if set(by_cell) != set(EXPECTED_CELL_IDS):
        missing = sorted(set(EXPECTED_CELL_IDS) - set(by_cell))
        extra = sorted(set(by_cell) - set(EXPECTED_CELL_IDS))
        raise ValueError(
            f"Stokes chunk {expected.label} has wrong by_cell keys: "
            f"missing={missing}, extra={extra}"
        )
    for cell_id, target in expected_counts.items():
        record = _required_mapping(by_cell, cell_id)
        if set(record) != {"target_accepted", "attempted", "accepted", "rejected"}:
            raise ValueError(
                f"Stokes chunk {expected.label} has wrong fields for {cell_id}"
            )
        observed = {
            name: _required_integer(record, name)
            for name in ("target_accepted", "attempted", "accepted", "rejected")
        }
        required = {
            "target_accepted": target,
            "attempted": target,
            "accepted": target,
            "rejected": 0,
        }
        if observed != required:
            raise ValueError(
                f"Stokes chunk {expected.label} has wrong counts for {cell_id}"
            )
    return expected_counts


def _validate_view_hashes(
    root: Path,
    chunks: Sequence[CompletedChunk],
    legacy: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Bind legacy manifest/map digests to summaries and present bytes."""

    legacy_manifests = _required_mapping(legacy, "run_manifest_sha256")
    legacy_maps = _required_mapping(legacy, "run_trajectory_map_sha256")
    records: list[dict[str, object]] = []
    for expected, chunk in zip(EXPECTED_CHUNKS, chunks):
        summary = _read_json_object(chunk.summary_path)
        cell_counts = _validate_summary_cell_counts(summary, expected=expected)
        dataset_view = _required_mapping(summary, "dataset_view")
        manifest = _required_mapping(dataset_view, "manifest")
        trajectory_map = _required_mapping(dataset_view, "trajectory_map")
        manifest_sha = _required_sha256(manifest, "sha256")
        map_sha = _required_sha256(trajectory_map, "sha256")
        if manifest_sha != _required_sha256(legacy_manifests, expected.label):
            raise RuntimeError(f"legacy manifest digest disagrees for {expected.label}")
        if map_sha != _required_sha256(legacy_maps, expected.label):
            raise RuntimeError(
                f"legacy trajectory-map digest disagrees for {expected.label}"
            )
        manifest_path = _resolve_inside(
            chunk.root, manifest.get("path"), label="manifest path"
        )
        map_path = _resolve_inside(
            chunk.root, trajectory_map.get("path"), label="trajectory-map path"
        )
        if file_sha256(manifest_path) != manifest_sha:
            raise RuntimeError(f"manifest bytes changed for {expected.label}")
        if file_sha256(map_path) != map_sha:
            raise RuntimeError(f"trajectory-map bytes changed for {expected.label}")
        if not chunk.root.is_relative_to(root):
            raise RuntimeError(f"Stokes chunk {expected.label} escapes corpus root")
        records.append(
            {
                "label": expected.label,
                "split": chunk.split.value,
                "stream_id": chunk.stream_id,
                "accepted_before": chunk.accepted_before,
                "accepted_count": chunk.accepted_count,
                "accepted_after": chunk.accepted_after,
                "attempted_count": chunk.attempted_count,
                "retained_rows": chunk.accepted_count,
                "summary_path": str(chunk.summary_path),
                "summary_sha256": chunk.summary_sha256,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "trajectory_map_path": str(map_path),
                "trajectory_map_sha256": map_sha,
                "configuration_fingerprint": chunk.fingerprint,
                "accepted_by_cell": cell_counts,
                "attempted_by_cell": cell_counts,
            }
        )
    return tuple(records)


def _source_mappings(
    chunks: Sequence[CompletedChunk],
) -> tuple[dict[str, str], ...]:
    """Require one exact, canonical 14-path source map in all six chunks."""

    if len(chunks) != len(EXPECTED_CHUNKS):
        raise ValueError("Stokes source binding requires all six chunks")
    mappings: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks):
        source_mapping = dict(chunk.source_sha256)
        if set(source_mapping) != EXPECTED_GENERATION_SOURCE_PATHS:
            missing = sorted(EXPECTED_GENERATION_SOURCE_PATHS - set(source_mapping))
            extra = sorted(set(source_mapping) - EXPECTED_GENERATION_SOURCE_PATHS)
            raise ValueError(
                "Stokes source map differs from the exact frozen 14-path contract: "
                f"missing={missing}, extra={extra}"
            )
        for path, digest in source_mapping.items():
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError(f"Stokes source-map path is malformed: {path!r}")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"Stokes source-map digest is malformed for chunk {index}: {path}"
                )
        fingerprint = _canonical_sha256(source_mapping)
        if chunk.source_fingerprint != fingerprint:
            raise ValueError(
                f"Stokes source_fingerprint differs from its map in chunk {index}"
            )
        mappings.append(source_mapping)
    first = mappings[0]
    if any(mapping != first for mapping in mappings[1:]):
        raise ValueError("Stokes chunks do not contain exactly equal source maps")
    if _canonical_sha256(first) != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError("Stokes source map differs from the frozen fingerprint")
    return tuple(mappings)


def _historical_source_snapshot_record(
    source_mappings: Sequence[Mapping[str, str]],
    *,
    snapshot_root: Path = HISTORICAL_SOURCE_SNAPSHOT_ROOT,
) -> dict[str, object]:
    """Physically authenticate all three frozen Stokes generation sources."""

    if len(source_mappings) != len(EXPECTED_CHUNKS):
        raise ValueError("historical Stokes source binding requires all six chunks")
    root = _require_real_directory(snapshot_root, label="Stokes snapshot root")
    checksum_spec = HISTORICAL_SOURCE_SNAPSHOT_MANIFEST
    checksum_path = _resolve_regular_file_inside(
        root,
        Path(str(checksum_spec["name"])),
        label="Stokes snapshot SHA256SUMS",
    )
    expected_checksum_text = "".join(
        f"{spec['sha256']}  {spec['name']}\n"
        for spec in HISTORICAL_SOURCE_SNAPSHOTS.values()
    )
    if checksum_path.read_text(encoding="utf-8") != expected_checksum_text:
        raise ValueError("historical Stokes source snapshot SHA256SUMS differs")
    checksum_sha256 = file_sha256(checksum_path)
    if (
        checksum_path.stat().st_size != checksum_spec["bytes"]
        or checksum_sha256 != checksum_spec["sha256"]
    ):
        raise ValueError("historical Stokes SHA256SUMS identity differs")

    bindings: dict[str, object] = {}
    for repository_path, spec in HISTORICAL_SOURCE_SNAPSHOTS.items():
        expected_digest = str(spec["sha256"])
        if any(
            mapping.get(repository_path) != expected_digest
            for mapping in source_mappings
        ):
            raise ValueError(
                "historical Stokes snapshot differs from a chunk source map: "
                f"{repository_path}"
            )
        snapshot_path = _resolve_regular_file_inside(
            root,
            Path(str(spec["name"])),
            label=f"historical Stokes snapshot {repository_path}",
        )
        if (
            snapshot_path.stat().st_size != spec["bytes"]
            or file_sha256(snapshot_path) != expected_digest
        ):
            raise ValueError(
                f"historical Stokes snapshot bytes differ: {repository_path}"
            )
        bindings[repository_path] = {
            "snapshot_path": str(snapshot_path),
            "bytes": spec["bytes"],
            "sha256": expected_digest,
        }

    return {
        "schema": "paper_corpus_stokes_revision2_historical_source_binding_v1",
        "role": "inert_historical_byte_recovery_only",
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "snapshot_root": str(root),
        "source_map_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "source_count": len(EXPECTED_GENERATION_SOURCE_PATHS),
        "historical_source_count": len(HISTORICAL_SOURCE_SNAPSHOTS),
        "sha256sums": {
            "path": str(checksum_path),
            "bytes": checksum_spec["bytes"],
            "sha256": checksum_sha256,
        },
        "sources": bindings,
        "chunk_source_maps_checked": len(source_mappings),
    }


def _current_source_record(
    source_mappings: Sequence[Mapping[str, str]],
    *,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Physically authenticate all 11 unchanged sources in the repository."""

    if len(source_mappings) != len(EXPECTED_CHUNKS):
        raise ValueError("current Stokes source binding requires all six chunks")
    first_sources = dict(source_mappings[0])
    if any(dict(mapping) != first_sources for mapping in source_mappings[1:]):
        raise ValueError("Stokes chunks do not contain exactly equal source maps")
    if set(first_sources) != EXPECTED_GENERATION_SOURCE_PATHS:
        raise ValueError("Stokes current source binding has the wrong source set")
    root = _require_real_directory(repository_root, label="Stokes repository root")
    historical_paths = set(HISTORICAL_SOURCE_SNAPSHOTS)
    current_sources = {
        path: digest
        for path, digest in first_sources.items()
        if path not in historical_paths
    }
    if len(current_sources) != 11:
        raise ValueError("Stokes source map does not contain exactly 11 current files")
    bindings: dict[str, object] = {}
    for repository_path, expected_digest in sorted(current_sources.items()):
        source_path = _resolve_regular_file_inside(
            root,
            Path(repository_path),
            label=f"current Stokes source {repository_path}",
        )
        observed_digest = file_sha256(source_path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"current Stokes source bytes differ from generation: {repository_path}"
            )
        bindings[repository_path] = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": observed_digest,
        }
    return {
        "schema": "paper_corpus_stokes_revision2_current_source_binding_v1",
        "role": "current_repository_bytes_for_all_nonhistorical_generation_sources",
        "repository_root": str(root),
        "source_map_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "source_count": len(first_sources),
        "current_source_count": len(bindings),
        "historical_snapshot_source_paths": sorted(historical_paths),
        "sources": bindings,
        "chunk_source_maps_checked": len(source_mappings),
    }


def _source_identity_record(chunks: Sequence[CompletedChunk]) -> dict[str, object]:
    """Return strict current and historical identities for one frozen map."""

    source_mappings = _source_mappings(chunks)
    return {
        "source_sha256": dict(source_mappings[0]),
        "historical_generation_source_snapshot_binding": (
            _historical_source_snapshot_record(source_mappings)
        ),
        "nonhistorical_generation_source_binding": _current_source_record(
            source_mappings
        ),
    }


def audit_corpus(root: Path, legacy_audit_path: Path) -> dict[str, object]:
    """Authenticate the completed Stokes corpus and return its binding."""

    started = perf_counter()
    corpus_root = _require_real_directory(root, label="Stokes corpus root")
    requested_legacy = _absolute_lexical(
        legacy_audit_path,
        label="legacy Stokes audit",
    )
    if requested_legacy.parent != corpus_root:
        raise ValueError("legacy Stokes audit must live in the corpus root")
    legacy_path = _resolve_regular_file_inside(
        corpus_root,
        Path(requested_legacy.name),
        label="legacy Stokes audit",
    )
    legacy = _load_legacy_audit(legacy_path)
    chunks = tuple(
        load_completed_chunk(corpus_root / expected.relative_summary)
        for expected in EXPECTED_CHUNKS
    )
    _validate_chunk_plan(chunks)
    chunk_records = _validate_view_hashes(corpus_root, chunks, legacy)
    accepted_by_split = {
        split.value: sum(
            chunk.accepted_count for chunk in chunks if chunk.split is split
        )
        for split in EXPECTED_ACCEPTED_BY_SPLIT
    }
    if accepted_by_split != {
        split.value: count for split, count in EXPECTED_ACCEPTED_BY_SPLIT.items()
    }:
        raise RuntimeError("Stokes split counts differ from the frozen layout")
    accepted_by_split_and_cell = {
        split.value: {
            cell_id: sum(
                _expected_chunk_cell_counts(expected)[cell_id]
                for expected in EXPECTED_CHUNKS
                if expected.split is split
            )
            for cell_id in EXPECTED_CELL_IDS
        }
        for split in EXPECTED_ACCEPTED_BY_SPLIT
    }
    source_identity = _source_identity_record(chunks)
    return {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "generated_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": perf_counter() - started,
        "root": str(corpus_root),
        "family": EXPECTED_FAMILY,
        "family_id": EXPECTED_FAMILY_ID,
        "revision_id": EXPECTED_REVISION_ID,
        "counts": {
            "attempted": EXPECTED_ACCEPTED,
            "accepted": EXPECTED_ACCEPTED,
            "rejected": 0,
            "retained_rows": EXPECTED_RETAINED_ROWS,
            "accepted_by_split": accepted_by_split,
            "accepted_by_split_and_cell": accepted_by_split_and_cell,
        },
        "identity": {
            "dependency_fingerprint": chunks[0].dependency_fingerprint,
            "execution_fingerprint": chunks[0].execution_fingerprint,
            "source_fingerprint": chunks[0].source_fingerprint,
            "execution_platform": chunks[0].execution_platform,
            "configuration_fingerprints": [chunk.fingerprint for chunk in chunks],
            **source_identity,
        },
        "legacy_numerical_audit": {
            "path": str(legacy_path),
            "sha256": file_sha256(legacy_path),
            "schema": LEGACY_AUDIT_SCHEMA,
            "status": "pass",
            "finite_float_values_checked": _required_integer(
                legacy, "finite_float_values_checked", minimum=1
            ),
            "transaction_artifact_bytes_checked": _required_integer(
                legacy, "transaction_artifact_bytes_checked", minimum=1
            ),
        },
        "chunks": list(chunk_records),
        "checks": {
            "legacy_numerical_audit_authenticated": True,
            "current_chunk_loader_passed": True,
            "exact_six_chunk_layout": True,
            "exact_split_counts": True,
            "summary_cell_counts_match_completed_quota_state": True,
            "manifest_and_trajectory_map_hashes_bound": True,
            "common_source_execution_and_dependency_identity": True,
            "exact_equal_14_path_source_maps": True,
            "source_map_fingerprint_verified": True,
            "historical_generation_source_snapshots_bound_to_all_chunks": True,
            "all_nonhistorical_generation_sources_match_current_bytes": True,
            "no_rejected_or_unowned_rows": True,
        },
    }


def _validate_output_path(output: Path, *, root: Path, legacy: Path) -> None:
    """Prevent a diagnostic invocation from replacing corpus evidence."""

    corpus_root = _require_real_directory(root, label="Stokes corpus root")
    requested = _absolute_lexical(output, label="binding output")
    legacy_path = _absolute_lexical(legacy, label="legacy Stokes audit")
    if requested == legacy_path:
        raise ValueError("binding output must not overwrite the legacy audit")
    if requested.name != "stokes_completion_binding.json":
        raise ValueError("binding output must use the exact release basename")
    if requested.is_symlink():
        raise FileExistsError("refusing symbolic-link binding output")
    output_parent = _require_real_directory(
        requested.parent,
        label="binding output parent",
    )
    if requested.parent != corpus_root or output_parent != corpus_root:
        raise ValueError("binding output must live directly in the corpus root")
    if requested.exists():
        if not requested.is_file():
            raise FileExistsError("binding output is not a regular file")
        existing = _read_json_object(requested)
        if existing.get("schema") != AUDIT_SCHEMA:
            raise FileExistsError("refusing to overwrite a non-binding corpus artifact")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Completed Stokes revision-2 corpus root.",
    )
    parser.add_argument(
        "--legacy-audit",
        type=Path,
        help="Original full numerical audit; defaults to <root>/stokes_completion_audit.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Binding artifact; defaults to <root>/stokes_completion_binding.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = _require_real_directory(args.root, label="Stokes corpus root")
    legacy = (
        _absolute_lexical(args.legacy_audit, label="legacy Stokes audit")
        if args.legacy_audit is not None
        else root / "stokes_completion_audit.json"
    )
    output = (
        _absolute_lexical(args.output, label="binding output")
        if args.output is not None
        else root / "stokes_completion_binding.json"
    )
    _validate_output_path(output, root=root, legacy=legacy)
    record = audit_corpus(root, legacy)
    write_json_atomic(output, record)
    print(json.dumps({"status": "pass", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
