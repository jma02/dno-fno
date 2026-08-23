"""Audit the final paper-dataset view and stream its training normalization.

This is a read-only, CPU-only handoff audit.  It authenticates the completed
combined view down to its source summaries, maps, and shards; reconstructs the
trainer's seed-dependent training-row selection; and computes normalization
extrema without concatenating the dataset fields in memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault(
    "MPLCONFIGDIR",
    "/tmp/mpl-paper-dataset-training-handoff",
)

import numpy as np

from solver.gen_data.pipeline.manifest import (
    DATASET_VIEW_SCHEMA_VERSION,
    TRAJECTORY_MAP_SCHEMA_VERSION,
)
from solver.gen_data.pipeline.production import (
    PAPER_DATASET_REVISION_BY_FAMILY,
    PhysicalFamilyId,
    SplitId,
    split_code,
)
from solver.gen_data.pipeline.quota_driver import canonical_json_sha256


AUDIT_SCHEMA = "paper_dataset_training_handoff_audit_v1"
TRAINING_IMPLEMENTATION_SCHEMA = "paper_dataset_training_implementation_v1"
COMBINED_SUMMARY_SCHEMA = "paper_dataset_combined_view_summary_v1"
COMBINED_PREFLIGHT_SCHEMA = "paper_dataset_combined_view_preflight_v1"
SOURCE_SUMMARY_SCHEMA = "paper_dataset_quota_summary_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FAMILY_IDS = {
    "stokes": int(PhysicalFamilyId.STOKES),
    "tanaka": int(PhysicalFamilyId.TANAKA),
    "benjamin_feir": int(PhysicalFamilyId.BENJAMIN_FEIR),
    "jonswap_tma": int(PhysicalFamilyId.JONSWAP_TMA),
}
SPLIT_IDS = {split.value: split_code(split) for split in SplitId}
MAP_DTYPES = {
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
TRAJECTORY_FIELDS = tuple(
    name
    for name in MAP_DTYPES
    if name.startswith("trajectory_") and name != "trajectory_index"
)
ROW_FIELDS = ("trajectory_index", "frame_index", "shard_index", "shard_row")
REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
TRAINING_IMPLEMENTATION_FILES = (
    (
        "scripts/audit_paper_dataset_training_handoff.py",
        "independent_cpu_streaming_handoff_audit",
    ),
    (
        "train-jax-10m/util.py",
        "trainer_dataset_loading_split_selection_and_normalization",
    ),
    (
        "train-jax-10m/1d_dno_fno_jax.py",
        "canonical_trainer_entrypoint_for_handoff_command_template",
    ),
    (
        "scripts/test_audit_paper_dataset_training_handoff.py",
        "semantic_regression_evidence",
    ),
)
TRAINING_IMPLEMENTATION_RELATIONSHIP = {
    "audit_strategy": "independent_numpy_cpu_streaming_reimplementation",
    "trainer_helpers_compared_by_regression_test": [
        "load_dataset_arrays",
        "build_dataset_split_indices",
        "load_or_compute_stats",
    ],
    "regression_test_path": "scripts/test_audit_paper_dataset_training_handoff.py",
    "claim_scope": (
        "The audit independently reconstructs the selected training rows and "
        "normalization statistics, and the named regression test compares those "
        "results with the trainer helpers. This is evidence for the tested "
        "dataset-handoff and statistics semantics plus the canonical command "
        "entrypoint bytes, not a proof that the full trainer is equivalent; it "
        "does not authenticate the model or transitive training stack."
    ),
    "formal_equivalence_claim": False,
}


@dataclass(frozen=True)
class DatasetContract:
    """Exact source and population contract for one training handoff."""

    accepted_cases_per_family_by_split: Mapping[str, int]
    rows_per_accepted_case: Mapping[str, int]
    source_count_by_family: Mapping[str, int]
    chunk_layout_by_family_and_split: Mapping[
        str, Mapping[str, tuple[tuple[int, int, int], ...]]
    ]
    revision_by_family: Mapping[str, int]
    nx: int
    length: float

    @property
    def accepted_cases(self) -> int:
        return len(FAMILY_IDS) * sum(self.accepted_cases_per_family_by_split.values())

    @property
    def accepted_rows(self) -> int:
        return sum(
            self.accepted_cases_per_family_by_split[split]
            * self.rows_per_accepted_case[family]
            for split in SPLIT_IDS
            for family in FAMILY_IDS
        )

    @property
    def source_count(self) -> int:
        return sum(self.source_count_by_family.values())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "families": list(FAMILY_IDS),
            "family_ids": dict(FAMILY_IDS),
            "revision_by_family": dict(self.revision_by_family),
            "source_count_by_family": dict(self.source_count_by_family),
            "chunk_layout_by_family_and_split": {
                split: {
                    family: [list(chunk) for chunk in chunks]
                    for family, chunks in families.items()
                }
                for split, families in self.chunk_layout_by_family_and_split.items()
            },
            "source_count": self.source_count,
            "accepted_cases_per_family_by_split": dict(
                self.accepted_cases_per_family_by_split
            ),
            "rows_per_accepted_case": dict(self.rows_per_accepted_case),
            "accepted_cases": self.accepted_cases,
            "accepted_rows": self.accepted_rows,
            "grid": {"nx": self.nx, "length": self.length},
        }
        record["fingerprint"] = canonical_json_sha256(record)
        return record


FINAL_CONTRACT = DatasetContract(
    accepted_cases_per_family_by_split={
        "train": 16_384,
        "validation": 1_024,
        "test": 1_024,
    },
    rows_per_accepted_case={
        "stokes": 1,
        "tanaka": 200,
        "benjamin_feir": 200,
        "jonswap_tma": 16,
    },
    source_count_by_family={
        "stokes": 6,
        "tanaka": 6,
        "benjamin_feir": 6,
        "jonswap_tma": 8,
    },
    chunk_layout_by_family_and_split={
        "train": {
            "stokes": (
                (0, 2_048, 0),
                (2_048, 2_048, 1),
                (4_096, 4_096, 2),
                (8_192, 8_192, 3),
            ),
            "tanaka": (
                (0, 2_048, 0),
                (2_048, 2_048, 1),
                (4_096, 4_096, 2),
                (8_192, 8_192, 3),
            ),
            "benjamin_feir": (
                (0, 2_048, 0),
                (2_048, 2_048, 1),
                (4_096, 4_096, 2),
                (8_192, 8_192, 3),
            ),
            "jonswap_tma": (
                (0, 2_048, 0),
                (2_048, 2_048, 1),
                (4_096, 4_096, 2),
                (8_192, 4_096, 3),
                (12_288, 2_048, 4),
                (14_336, 2_048, 5),
            ),
        },
        "validation": {family: ((0, 1_024, 100),) for family in FAMILY_IDS},
        "test": {family: ((0, 1_024, 200),) for family in FAMILY_IDS},
    },
    revision_by_family={
        family: int(PAPER_DATASET_REVISION_BY_FAMILY[PhysicalFamilyId(family_id)])
        for family, family_id in FAMILY_IDS.items()
    },
    nx=1024,
    length=2.0 * math.pi,
)


@dataclass(frozen=True)
class Artifact:
    """Authenticated identity of one immutable input file."""

    path: Path
    bytes: int
    sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class Shard:
    """One combined-view shard and its declared identity."""

    index: int
    path: Path
    sha256: str
    rows: int
    configuration_fingerprint: str


@dataclass(frozen=True)
class LoadedInput:
    """Authenticated combined view needed by the streaming pass."""

    summary: Artifact
    manifest: Artifact
    trajectory_map: Artifact
    manifest_record: Mapping[str, Any]
    map_arrays: Mapping[str, np.ndarray]
    shards: tuple[Shard, ...]
    chunks: tuple[dict[str, object], ...]
    source_artifacts: tuple[dict[str, object], ...]
    attempted_cases: int


class FileAuthenticator:
    """Hash each physical file once and retain its authenticated identity."""

    def __init__(self) -> None:
        self._artifacts: dict[Path, Artifact] = {}
        self._identities: dict[Path, tuple[int, int, int, int, int]] = {}

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def authenticate(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        context: str,
    ) -> Artifact:
        resolved = path.expanduser().resolve(strict=True)
        artifact = self._artifacts.get(resolved)
        if artifact is None:
            before = self._identity(resolved)
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            after = self._identity(resolved)
            if before != after:
                raise RuntimeError(f"{context} changed while being hashed")
            artifact = Artifact(
                path=resolved,
                bytes=after[2],
                sha256=digest.hexdigest(),
            )
            self._artifacts[resolved] = artifact
            self._identities[resolved] = after
        else:
            self.assert_unchanged(artifact, context=context)
        if expected_sha256 is not None:
            _require_sha256(expected_sha256, context=f"{context} SHA-256")
            if artifact.sha256 != expected_sha256:
                raise RuntimeError(f"{context} SHA-256 differs")
        if expected_bytes is not None:
            if expected_bytes < 0 or artifact.bytes != expected_bytes:
                raise RuntimeError(f"{context} byte count differs")
        return artifact

    def assert_unchanged(self, artifact: Artifact, *, context: str) -> None:
        expected = self._identities.get(artifact.path)
        if expected is None or self._identity(artifact.path) != expected:
            raise RuntimeError(f"{context} changed after authentication")

    @property
    def unique_bytes(self) -> int:
        return sum(artifact.bytes for artifact in self._artifacts.values())

    @property
    def unique_files(self) -> int:
        return len(self._artifacts)


def _repository_file(
    relative: str,
    *,
    repository_root: Path | None = None,
    context: str,
) -> Path:
    """Resolve one fixed repository file without following any symlink."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{context} path must be repository-relative")
    requested_root = (repository_root or REPOSITORY_ROOT).expanduser()
    absolute_root = Path(os.path.abspath(requested_root))
    current = Path(absolute_root.anchor)
    for component in absolute_root.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{context} repository root contains a symbolic link")
    root = absolute_root.resolve(strict=True)
    current = root
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{context} contains a symbolic link")
    if not current.exists() or not stat.S_ISREG(current.stat().st_mode):
        raise ValueError(f"{context} is not a regular file")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{context} escapes the repository root")
    return resolved


def _training_implementation_record(
    *,
    authenticator: FileAuthenticator | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Authenticate the handoff implementation and cited semantic evidence."""

    files = authenticator or FileAuthenticator()
    records = []
    for relative, role in TRAINING_IMPLEMENTATION_FILES:
        artifact = files.authenticate(
            _repository_file(
                relative,
                repository_root=repository_root,
                context=f"training implementation {relative}",
            ),
            context=f"training implementation {relative}",
        )
        records.append(
            {
                "path": relative,
                "role": role,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            }
        )
    payload: dict[str, object] = {
        "schema": TRAINING_IMPLEMENTATION_SCHEMA,
        "files": records,
        "semantic_relationship": dict(TRAINING_IMPLEMENTATION_RELATIONSHIP),
    }
    return {**payload, "fingerprint": canonical_json_sha256(payload)}


def _authenticate_training_implementation_record(
    value: object,
    *,
    repository_root: Path | None = None,
) -> tuple[Artifact, ...]:
    """Validate an exact implementation record and rehash its physical files."""

    record = _mapping(value, context="training implementation")
    expected_keys = {"schema", "files", "semantic_relationship", "fingerprint"}
    if set(record) != expected_keys:
        raise RuntimeError("training implementation fields differ")
    if record.get("schema") != TRAINING_IMPLEMENTATION_SCHEMA:
        raise RuntimeError("training implementation schema differs")
    relationship = _mapping(
        record.get("semantic_relationship"),
        context="training implementation semantic relationship",
    )
    if canonical_json_sha256(relationship) != canonical_json_sha256(
        TRAINING_IMPLEMENTATION_RELATIONSHIP
    ):
        raise RuntimeError("training implementation semantic relationship differs")
    raw_files = _sequence(record.get("files"), context="training implementation files")
    if len(raw_files) != len(TRAINING_IMPLEMENTATION_FILES):
        raise RuntimeError("training implementation file set differs")
    authenticator = FileAuthenticator()
    artifacts = []
    canonical_files = []
    for index, ((expected_path, expected_role), raw_file) in enumerate(
        zip(TRAINING_IMPLEMENTATION_FILES, raw_files, strict=True)
    ):
        context = f"training implementation file {index}"
        file_record = _mapping(raw_file, context=context)
        if set(file_record) != {"path", "role", "bytes", "sha256"}:
            raise RuntimeError(f"{context} fields differ")
        if (
            file_record.get("path") != expected_path
            or file_record.get("role") != expected_role
        ):
            raise RuntimeError(f"{context} path or role differs")
        expected_bytes = _integer(file_record.get("bytes"), context=f"{context} bytes")
        expected_sha256 = _require_sha256(
            file_record.get("sha256"), context=f"{context} SHA-256"
        )
        artifact = authenticator.authenticate(
            _repository_file(
                expected_path,
                repository_root=repository_root,
                context=context,
            ),
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            context=context,
        )
        artifacts.append(artifact)
        canonical_files.append(dict(file_record))
    for index, (artifact, (expected_path, _)) in enumerate(
        zip(artifacts, TRAINING_IMPLEMENTATION_FILES, strict=True)
    ):
        current_path = _repository_file(
            expected_path,
            repository_root=repository_root,
            context=f"training implementation file {index}",
        )
        if current_path != artifact.path:
            raise RuntimeError(f"training implementation file {index} path changed")
        authenticator.assert_unchanged(
            artifact,
            context=f"training implementation file {index}",
        )
    payload = {
        "schema": TRAINING_IMPLEMENTATION_SCHEMA,
        "files": canonical_files,
        "semantic_relationship": dict(relationship),
    }
    if record.get("fingerprint") != canonical_json_sha256(payload):
        raise RuntimeError("training implementation fingerprint differs")
    return tuple(artifacts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the final-audit command line."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The JSON is an audit artifact, not the trainer's .stats.json cache. "
            "It records the exact cache path, canonical trainer command template, "
            "and a canonical fingerprint for post-startup comparison."
        ),
    )
    parser.add_argument("--combined-summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: a training_handoff_audit JSON beside the combined summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Trainer split seed; the training default is 0.",
    )
    return parser.parse_args(argv)


def _read_authenticated_json(
    authenticator: FileAuthenticator,
    artifact: Artifact,
    *,
    context: str,
) -> dict[str, Any]:
    authenticator.assert_unchanged(artifact, context=context)
    encoded = artifact.path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != artifact.sha256:
        raise RuntimeError(f"{context} bytes differ after authentication")
    authenticator.assert_unchanged(artifact, context=context)
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {artifact.path}")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _resolve_path(base: Path, value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path must be a nonempty string")
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else base / raw).resolve(strict=True)


def _require_within(path: Path, root: Path, *, context: str) -> None:
    if not path.is_relative_to(root):
        raise RuntimeError(f"{context} escapes its artifact root")


def _artifact_from_record(
    authenticator: FileAuthenticator,
    *,
    base: Path,
    value: object,
    context: str,
) -> Artifact:
    record = _mapping(value, context=context)
    path = _resolve_path(base, record.get("path"), context=context)
    expected_bytes = _integer(record.get("bytes"), context=f"{context} bytes")
    expected_sha256 = _require_sha256(
        record.get("sha256"), context=f"{context} SHA-256"
    )
    return authenticator.authenticate(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        context=context,
    )


def _load_map(
    artifact: Artifact,
    *,
    authenticator: FileAuthenticator,
    context: str,
) -> dict[str, np.ndarray]:
    authenticator.assert_unchanged(artifact, context=context)
    with np.load(artifact.path, allow_pickle=False) as stored:
        if set(stored.files) != set(MAP_DTYPES):
            raise RuntimeError(f"{context} does not have the exact schema-v2 arrays")
        arrays = {name: np.asarray(stored[name]) for name in MAP_DTYPES}
    authenticator.assert_unchanged(artifact, context=context)
    schema = arrays.pop("schema_version")
    if (
        schema.ndim != 0
        or schema.dtype != MAP_DTYPES["schema_version"]
        or int(schema) != TRAJECTORY_MAP_SCHEMA_VERSION
    ):
        raise RuntimeError(f"{context} has the wrong schema version")
    for name, array in arrays.items():
        if array.ndim != 1 or array.dtype != MAP_DTYPES[name]:
            raise RuntimeError(f"{context} field {name} has the wrong shape or dtype")
    return arrays


def _same_json(left: object, right: object) -> bool:
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _expected_rows_by_split_and_family(
    contract: DatasetContract,
) -> dict[str, dict[str, int]]:
    return {
        split: {
            family: contract.accepted_cases_per_family_by_split[split]
            * contract.rows_per_accepted_case[family]
            for family in FAMILY_IDS
        }
        for split in SPLIT_IDS
    }


def _validate_preflight(
    preflight: Mapping[str, Any],
    *,
    contract: DatasetContract,
) -> tuple[dict[str, object], ...]:
    if preflight.get("schema") != COMBINED_PREFLIGHT_SCHEMA:
        raise ValueError("combined preflight has the wrong schema")
    expected_fields: dict[str, object] = {
        "splits": list(SPLIT_IDS),
        "accepted_cases_per_family_by_split": dict(
            contract.accepted_cases_per_family_by_split
        ),
        "accepted_cases_total": contract.accepted_cases,
        "expected_rows": contract.accepted_rows,
        "expected_rows_by_split_and_family": (
            _expected_rows_by_split_and_family(contract)
        ),
    }
    for name, expected in expected_fields.items():
        if not _same_json(preflight.get(name), expected):
            raise RuntimeError(f"combined preflight {name} differs from contract")
    raw_chunks = _sequence(preflight.get("chunks"), context="preflight chunks")
    if len(raw_chunks) != contract.source_count:
        raise RuntimeError("combined preflight source count differs from contract")

    chunks: list[dict[str, object]] = []
    source_counts = {family: 0 for family in FAMILY_IDS}
    accepted_totals = {
        (family, split): 0 for split in SPLIT_IDS for family in FAMILY_IDS
    }
    attempted_by_split = {split: 0 for split in SPLIT_IDS}
    interval_ends = {(family, split): 0 for split in SPLIT_IDS for family in FAMILY_IDS}
    observed_layout: dict[tuple[str, str], list[tuple[int, int, int]]] = {
        (family, split): [] for split in SPLIT_IDS for family in FAMILY_IDS
    }
    summary_paths: set[Path] = set()
    configuration_fingerprints: list[str] = []
    for index, raw_chunk in enumerate(raw_chunks):
        chunk = dict(_mapping(raw_chunk, context=f"preflight chunk {index}"))
        family = chunk.get("family")
        split = chunk.get("split")
        if family not in FAMILY_IDS or split not in SPLIT_IDS:
            raise RuntimeError(f"preflight chunk {index} has unknown family or split")
        assert isinstance(family, str) and isinstance(split, str)
        revision = _integer(chunk.get("revision_id"), context=f"chunk {index} revision")
        if revision != contract.revision_by_family[family]:
            raise RuntimeError(f"preflight chunk {index} has the wrong revision")
        accepted_before = _integer(
            chunk.get("accepted_before"), context=f"chunk {index} accepted_before"
        )
        accepted_count = _integer(
            chunk.get("accepted_count"), context=f"chunk {index} accepted_count"
        )
        accepted_after = _integer(
            chunk.get("accepted_after"), context=f"chunk {index} accepted_after"
        )
        attempted_count = _integer(
            chunk.get("attempted_count"), context=f"chunk {index} attempted_count"
        )
        stream_id = _integer(chunk.get("stream_id"), context=f"chunk {index} stream_id")
        key = (family, split)
        if (
            accepted_before != interval_ends[key]
            or accepted_after != accepted_before + accepted_count
        ):
            raise RuntimeError(
                f"preflight chunk {index} has a noncontiguous quota interval"
            )
        interval_ends[key] = accepted_after
        accepted_totals[key] += accepted_count
        attempted_by_split[split] += attempted_count
        source_counts[family] += 1
        observed_layout[key].append((accepted_before, accepted_count, stream_id))
        raw_summary_path = chunk.get("summary_path")
        if (
            not isinstance(raw_summary_path, str)
            or not Path(raw_summary_path).expanduser().is_absolute()
        ):
            raise RuntimeError(f"preflight chunk {index} summary path must be absolute")
        summary_path = _resolve_path(
            Path.cwd(), raw_summary_path, context=f"chunk {index} summary"
        )
        if summary_path in summary_paths:
            raise RuntimeError("combined preflight repeats a source summary")
        summary_paths.add(summary_path)
        _require_sha256(
            chunk.get("summary_sha256"), context=f"chunk {index} summary SHA-256"
        )
        configuration_fingerprints.append(
            _require_sha256(
                chunk.get("configuration_fingerprint"),
                context=f"chunk {index} configuration fingerprint",
            )
        )
        chunks.append(chunk)

    expected_accepted = {
        (family, split): contract.accepted_cases_per_family_by_split[split]
        for split in SPLIT_IDS
        for family in FAMILY_IDS
    }
    if accepted_totals != expected_accepted:
        raise RuntimeError(
            "preflight family/split accepted counts differ from contract"
        )
    if source_counts != dict(contract.source_count_by_family):
        raise RuntimeError("preflight family source counts differ from contract")
    expected_layout = {
        (family, split): list(contract.chunk_layout_by_family_and_split[split][family])
        for split in SPLIT_IDS
        for family in FAMILY_IDS
    }
    if observed_layout != expected_layout:
        raise RuntimeError("preflight frozen chunk layout differs from contract")
    if not _same_json(preflight.get("attempted_cases_by_split"), attempted_by_split):
        raise RuntimeError("preflight attempted split counts disagree with chunks")
    if preflight.get("attempted_cases_total") != sum(attempted_by_split.values()):
        raise RuntimeError("preflight attempted total disagrees with chunks")
    expected_fingerprints = sorted(configuration_fingerprints)
    if len(set(expected_fingerprints)) != len(expected_fingerprints) or not _same_json(
        preflight.get("configuration_fingerprints"), expected_fingerprints
    ):
        raise RuntimeError(
            "preflight configuration fingerprints differ from source chunks"
        )
    return tuple(chunks)


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    view_record: Mapping[str, Any],
    contract: DatasetContract,
) -> str:
    expected: dict[str, object] = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprints": preflight.get("configuration_fingerprints"),
        "requires_trajectory_map": True,
        "n_rows": contract.accepted_rows,
        "n_accepted_trajectories": contract.accepted_cases,
        "n_accepted_rows": contract.accepted_rows,
        "grid": {"nx": contract.nx, "length": contract.length},
    }
    attempted = _integer(
        preflight.get("attempted_cases_total"), context="preflight attempted total"
    )
    expected["n_trajectories"] = attempted
    expected_split_counts = {
        split: {
            "attempted": preflight["attempted_cases_by_split"][split],
            "accepted": len(FAMILY_IDS)
            * contract.accepted_cases_per_family_by_split[split],
        }
        for split in SPLIT_IDS
    }
    expected["split_counts"] = expected_split_counts
    for name, value in expected.items():
        if not _same_json(manifest.get(name), value):
            raise RuntimeError(f"combined manifest {name} differs from contract")
        if name in view_record and not _same_json(view_record.get(name), value):
            raise RuntimeError(f"combined summary dataset_view {name} differs")
    dataset_contract = _mapping(
        manifest.get("dataset_contract"), context="combined dataset contract"
    )
    fingerprint = _require_sha256(
        manifest.get("dataset_contract_fingerprint"),
        context="combined dataset-contract fingerprint",
    )
    if canonical_json_sha256(dataset_contract) != fingerprint:
        raise RuntimeError("combined dataset-contract fingerprint is incorrect")
    if view_record.get("dataset_contract_fingerprint") != fingerprint:
        raise RuntimeError("combined summary records a different dataset contract")
    return fingerprint


def _validate_map_population(
    arrays: Mapping[str, np.ndarray],
    *,
    manifest: Mapping[str, Any],
    contract: DatasetContract,
) -> dict[str, object]:
    trajectory_count = int(arrays["trajectory_family_id"].size)
    row_count = int(arrays["trajectory_index"].size)
    if any(arrays[name].size != trajectory_count for name in TRAJECTORY_FIELDS):
        raise RuntimeError("combined map trajectory arrays have inconsistent lengths")
    if any(arrays[name].size != row_count for name in ROW_FIELDS):
        raise RuntimeError("combined map row arrays have inconsistent lengths")
    if (
        trajectory_count != manifest["n_trajectories"]
        or row_count != contract.accepted_rows
    ):
        raise RuntimeError("combined map population differs from manifest")

    family_ids = arrays["trajectory_family_id"]
    revision_ids = arrays["trajectory_revision_id"]
    split_ids = arrays["trajectory_split_id"]
    accepted = arrays["trajectory_accepted"]
    row_counts = arrays["trajectory_row_count"]
    first_rows = arrays["trajectory_first_row"]
    required = arrays["trajectory_required_bits"]
    evaluated = arrays["trajectory_evaluated_bits"]
    failed = arrays["trajectory_failed_bits"]
    if np.any(failed & ~evaluated):
        raise RuntimeError("combined map failed bits are not evaluated bits")
    mask_accepts = ((required & ~evaluated) == 0) & ((required & failed) == 0)
    if not np.array_equal(accepted, mask_accepts):
        raise RuntimeError("combined map acceptance disagrees with quality masks")
    if np.any(row_counts[accepted] <= 0) or np.any(row_counts[~accepted] != 0):
        raise RuntimeError("combined map row ownership disagrees with acceptance")

    expected_first = np.cumsum(row_counts, dtype=np.int64) - row_counts
    expected_first[~accepted] = -1
    if not np.array_equal(first_rows, expected_first):
        raise RuntimeError("combined map first rows are not canonical")
    row_owner = arrays["trajectory_index"]
    cursor = 0
    for trajectory, count_value in enumerate(row_counts):
        count = int(count_value)
        if count and not np.all(row_owner[cursor : cursor + count] == trajectory):
            raise RuntimeError("combined map row owners are not canonical")
        cursor += count
    if cursor != row_count:
        raise RuntimeError("combined map row count sum is incorrect")

    accepted_cases: dict[str, dict[str, int]] = {split: {} for split in SPLIT_IDS}
    accepted_rows: dict[str, dict[str, int]] = {split: {} for split in SPLIT_IDS}
    for split, split_id in SPLIT_IDS.items():
        for family, family_id in FAMILY_IDS.items():
            expected_revision = contract.revision_by_family[family]
            cell = (split_ids == split_id) & (family_ids == family_id)
            if np.any(revision_ids[cell] != expected_revision):
                raise RuntimeError(f"combined map has the wrong {family} revision")
            observed_cases = int(np.count_nonzero(cell & accepted))
            observed_rows = int(np.sum(row_counts[cell & accepted], dtype=np.int64))
            expected_cases = contract.accepted_cases_per_family_by_split[split]
            expected_rows = expected_cases * contract.rows_per_accepted_case[family]
            if observed_cases != expected_cases or observed_rows != expected_rows:
                raise RuntimeError(
                    f"combined map {split}/{family} population differs from contract"
                )
            if np.any(
                row_counts[cell & accepted] != contract.rows_per_accepted_case[family]
            ):
                raise RuntimeError(
                    f"combined map {split}/{family} has a wrong trajectory length"
                )
            accepted_cases[split][family] = observed_cases
            accepted_rows[split][family] = observed_rows
    if not np.all(np.isin(family_ids, tuple(FAMILY_IDS.values()))):
        raise RuntimeError("combined map contains an unknown family")
    if not np.all(np.isin(split_ids, tuple(SPLIT_IDS.values()))):
        raise RuntimeError("combined map contains an unknown split")
    return {
        "attempted_cases": trajectory_count,
        "accepted_cases": int(np.count_nonzero(accepted)),
        "accepted_rows": row_count,
        "accepted_cases_by_split_and_family": accepted_cases,
        "accepted_rows_by_split_and_family": accepted_rows,
    }


def _resolve_manifest_shards(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[Shard, ...]:
    records = _sequence(manifest.get("dataset_shards"), context="dataset shards")
    shards: list[Shard] = []
    paths: set[Path] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, context=f"dataset shard {index}")
        path = _resolve_path(
            manifest_path.parent, record.get("path"), context=f"dataset shard {index}"
        )
        if path in paths:
            raise RuntimeError("combined manifest repeats a shard path")
        paths.add(path)
        shards.append(
            Shard(
                index=index,
                path=path,
                sha256=_require_sha256(
                    record.get("sha256"), context=f"dataset shard {index} SHA-256"
                ),
                rows=_integer(
                    record.get("n_rows"),
                    context=f"dataset shard {index} rows",
                    minimum=1,
                ),
                configuration_fingerprint=_require_sha256(
                    record.get("configuration_fingerprint"),
                    context=f"dataset shard {index} configuration fingerprint",
                ),
            )
        )
    return tuple(shards)


def _validate_map_shard_coordinates(
    arrays: Mapping[str, np.ndarray], shards: Sequence[Shard]
) -> None:
    row_shards = arrays["shard_index"]
    shard_rows = arrays["shard_row"]
    cursor = 0
    for shard in shards:
        stop = cursor + shard.rows
        if not np.all(row_shards[cursor:stop] == shard.index) or not np.array_equal(
            shard_rows[cursor:stop], np.arange(shard.rows, dtype=np.int64)
        ):
            raise RuntimeError(
                "combined map shard coordinates differ from manifest order"
            )
        cursor = stop
    if cursor != row_shards.size:
        raise RuntimeError("combined shard row total differs from trajectory map")


def _source_artifacts(
    chunks: Sequence[dict[str, object]],
    *,
    combined_manifest: Mapping[str, Any],
    combined_manifest_path: Path,
    combined_map: Mapping[str, np.ndarray],
    authenticator: FileAuthenticator,
) -> tuple[tuple[dict[str, object], ...], tuple[Shard, ...]]:
    combined_shards = _resolve_manifest_shards(
        combined_manifest, manifest_path=combined_manifest_path
    )
    combined_batches = _sequence(
        combined_manifest.get("dataset_batches"), context="combined dataset batches"
    )
    trajectory_offset = 0
    row_offset = 0
    shard_offset = 0
    batch_offset = 0
    source_records: list[dict[str, object]] = []

    for chunk_index, chunk in enumerate(chunks):
        summary_path = _resolve_path(
            Path.cwd(),
            chunk.get("summary_path"),
            context=f"chunk {chunk_index} summary",
        )
        summary_artifact = authenticator.authenticate(
            summary_path,
            expected_sha256=str(chunk["summary_sha256"]),
            context=f"source summary {chunk_index}",
        )
        summary = _read_authenticated_json(
            authenticator,
            summary_artifact,
            context=f"source summary {chunk_index}",
        )
        if (
            summary.get("schema") != SOURCE_SUMMARY_SCHEMA
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(
                f"source summary {chunk_index} is not complete schema v1"
            )
        raw_output_root = summary.get("output_root")
        if (
            not isinstance(raw_output_root, str)
            or Path(raw_output_root).expanduser().resolve() != summary_path.parent
        ):
            raise RuntimeError(
                f"source summary {chunk_index} has the wrong output root"
            )
        if (
            summary.get("configuration_fingerprint")
            != chunk["configuration_fingerprint"]
        ):
            raise RuntimeError(
                f"source summary {chunk_index} configuration fingerprint differs"
            )
        counts = _mapping(
            summary.get("counts"), context=f"source summary {chunk_index} counts"
        )
        if (
            counts.get("accepted") != chunk["accepted_count"]
            or counts.get("attempted") != chunk["attempted_count"]
        ):
            raise RuntimeError(f"source summary {chunk_index} counts differ")
        run_spec = _mapping(
            summary.get("run_spec"),
            context=f"source summary {chunk_index} run_spec",
        )
        expected_run_identity = {
            "family_name": chunk["family"],
            "family_id": FAMILY_IDS[str(chunk["family"])],
            "revision_id": chunk["revision_id"],
            "split_id": chunk["split"],
            "stream_id": chunk["stream_id"],
        }
        if any(
            not _same_json(run_spec.get(name), expected)
            for name, expected in expected_run_identity.items()
        ):
            raise RuntimeError(f"source summary {chunk_index} run identity differs")
        if canonical_json_sha256(run_spec) != chunk["configuration_fingerprint"]:
            raise RuntimeError(
                f"source summary {chunk_index} run-spec fingerprint differs"
            )
        run_configuration = _mapping(
            run_spec.get("configuration"),
            context=f"source summary {chunk_index} run configuration",
        )
        expected_quota_interval = {
            "accepted_case_count": chunk["accepted_count"],
            "accepted_cases_before": chunk["accepted_before"],
            "accepted_cases_after": chunk["accepted_after"],
        }
        if any(
            run_configuration.get(name) != expected
            for name, expected in expected_quota_interval.items()
        ):
            raise RuntimeError(f"source summary {chunk_index} quota interval differs")
        view = _mapping(
            summary.get("dataset_view"), context=f"source view {chunk_index}"
        )
        source_manifest_artifact = _artifact_from_record(
            authenticator,
            base=summary_path.parent,
            value=view.get("manifest"),
            context=f"source manifest {chunk_index}",
        )
        source_map_artifact = _artifact_from_record(
            authenticator,
            base=summary_path.parent,
            value=view.get("trajectory_map"),
            context=f"source map {chunk_index}",
        )
        _require_within(
            source_manifest_artifact.path,
            summary_path.parent,
            context=f"source manifest {chunk_index}",
        )
        _require_within(
            source_map_artifact.path,
            summary_path.parent,
            context=f"source map {chunk_index}",
        )
        source_manifest = _read_authenticated_json(
            authenticator,
            source_manifest_artifact,
            context=f"source manifest {chunk_index}",
        )
        source_map_name = source_manifest.get("trajectory_map_npz")
        if (
            _resolve_path(
                source_manifest_artifact.path.parent,
                source_map_name,
                context=f"source manifest {chunk_index} map",
            )
            != source_map_artifact.path
        ):
            raise RuntimeError(f"source manifest {chunk_index} names a different map")
        if source_manifest.get("trajectory_map_sha256") != source_map_artifact.sha256:
            raise RuntimeError(f"source manifest {chunk_index} map SHA-256 differs")
        source_map = _load_map(
            source_map_artifact,
            authenticator=authenticator,
            context=f"source map {chunk_index}",
        )
        source_trajectories = int(source_map["trajectory_family_id"].size)
        source_rows = int(source_map["trajectory_index"].size)
        expected_family_id = FAMILY_IDS[str(chunk["family"])]
        expected_revision_id = int(chunk["revision_id"])
        expected_split_id = SPLIT_IDS[str(chunk["split"])]
        if not (
            np.all(source_map["trajectory_family_id"] == expected_family_id)
            and np.all(source_map["trajectory_revision_id"] == expected_revision_id)
            and np.all(source_map["trajectory_split_id"] == expected_split_id)
        ):
            raise RuntimeError(
                f"source map {chunk_index} family/revision/split identity differs"
            )
        source_fingerprints = source_manifest.get("configuration_fingerprints")
        if not _same_json(
            source_fingerprints,
            [chunk["configuration_fingerprint"]],
        ):
            raise RuntimeError(
                f"source manifest {chunk_index} configuration fingerprint differs"
            )
        if source_trajectories != int(chunk["attempted_count"]):
            raise RuntimeError(f"source map {chunk_index} attempted count differs")
        if int(np.count_nonzero(source_map["trajectory_accepted"])) != int(
            chunk["accepted_count"]
        ):
            raise RuntimeError(f"source map {chunk_index} accepted count differs")
        if (
            source_manifest.get("n_trajectories") != source_trajectories
            or source_manifest.get("n_accepted_trajectories")
            != int(chunk["accepted_count"])
            or source_manifest.get("n_rows") != source_rows
            or source_manifest.get("n_accepted_rows") != source_rows
        ):
            raise RuntimeError(f"source manifest {chunk_index} population differs")

        trajectory_slice = slice(
            trajectory_offset, trajectory_offset + source_trajectories
        )
        for name in TRAJECTORY_FIELDS:
            expected = source_map[name]
            if name == "trajectory_first_row":
                expected = np.where(expected >= 0, expected + row_offset, -1).astype(
                    np.int64, copy=False
                )
            if not np.array_equal(combined_map[name][trajectory_slice], expected):
                raise RuntimeError(
                    f"combined map field {name} differs from source {chunk_index}"
                )
        row_slice = slice(row_offset, row_offset + source_rows)
        row_expectations = {
            "trajectory_index": source_map["trajectory_index"] + trajectory_offset,
            "frame_index": source_map["frame_index"],
            "shard_index": source_map["shard_index"] + shard_offset,
            "shard_row": source_map["shard_row"],
        }
        for name, expected in row_expectations.items():
            if not np.array_equal(combined_map[name][row_slice], expected):
                raise RuntimeError(
                    f"combined map row field {name} differs from source {chunk_index}"
                )

        source_shards_raw = _sequence(
            source_manifest.get("dataset_shards"),
            context=f"source {chunk_index} shards",
        )
        source_batches = _sequence(
            source_manifest.get("dataset_batches"),
            context=f"source {chunk_index} batches",
        )
        if batch_offset + len(source_batches) > len(combined_batches):
            raise RuntimeError("combined manifest omits source batches")
        batch_artifacts: list[dict[str, object]] = []
        for local_batch_index, raw_source_batch in enumerate(source_batches):
            source_batch = _mapping(
                raw_source_batch,
                context=f"source {chunk_index} batch {local_batch_index}",
            )
            combined_batch = _mapping(
                combined_batches[batch_offset + local_batch_index],
                context="combined dataset batch",
            )
            if set(source_batch) != set(combined_batch):
                raise RuntimeError("combined batch schema differs from source")
            for path_field in ("proposal_path", "result_path"):
                source_path = _resolve_path(
                    source_manifest_artifact.path.parent,
                    source_batch.get(path_field),
                    context=f"source batch {path_field}",
                )
                _require_within(
                    source_path,
                    summary_path.parent,
                    context=f"source batch {path_field}",
                )
                combined_path = _resolve_path(
                    combined_manifest_path.parent,
                    combined_batch.get(path_field),
                    context=f"combined batch {path_field}",
                )
                if source_path != combined_path:
                    raise RuntimeError("combined batch path differs from source")
                hash_field = (
                    "proposal_sha256"
                    if path_field == "proposal_path"
                    else "result_sha256"
                )
                physical_artifact = authenticator.authenticate(
                    source_path,
                    expected_sha256=_require_sha256(
                        source_batch.get(hash_field),
                        context=f"source batch {hash_field}",
                    ),
                    context=f"source {chunk_index} batch {local_batch_index} {path_field}",
                )
                batch_artifacts.append(
                    {
                        "batch_index": local_batch_index,
                        "kind": path_field.removesuffix("_path"),
                        **physical_artifact.to_record(),
                    }
                )
            source_shard_index = source_batch.get("shard_index")
            expected_shard_index = (
                None
                if source_shard_index is None
                else int(source_shard_index) + shard_offset
            )
            for field in source_batch:
                if field in ("proposal_path", "result_path", "shard_index"):
                    continue
                if not _same_json(combined_batch.get(field), source_batch.get(field)):
                    raise RuntimeError(
                        f"combined batch field {field} differs from source"
                    )
            if combined_batch.get("shard_index") != expected_shard_index:
                raise RuntimeError("combined batch shard index differs from source")

        for local_shard_index, raw_source_shard in enumerate(source_shards_raw):
            source_shard = _mapping(
                raw_source_shard,
                context=f"source {chunk_index} shard {local_shard_index}",
            )
            global_shard_index = shard_offset + local_shard_index
            if global_shard_index >= len(combined_shards):
                raise RuntimeError("combined manifest omits a source shard")
            combined_shard = combined_shards[global_shard_index]
            combined_shard_record = _mapping(
                combined_manifest["dataset_shards"][global_shard_index],
                context=f"combined shard {global_shard_index}",
            )
            if set(source_shard) != set(combined_shard_record):
                raise RuntimeError("combined shard schema differs from source")
            source_path = _resolve_path(
                source_manifest_artifact.path.parent,
                source_shard.get("path"),
                context=f"source shard {chunk_index}/{local_shard_index}",
            )
            _require_within(
                source_path,
                summary_path.parent,
                context=f"source shard {chunk_index}/{local_shard_index}",
            )
            if (
                combined_shard.path != source_path
                or combined_shard.sha256 != source_shard.get("sha256")
                or combined_shard.rows != source_shard.get("n_rows")
                or combined_shard.configuration_fingerprint
                != source_shard.get("configuration_fingerprint")
                or source_shard.get("batch_index") is None
                or combined_shard_record.get("batch_index")
                != int(source_shard["batch_index"]) + batch_offset
            ):
                raise RuntimeError("combined shard identity differs from source")

        source_records.append(
            {
                "index": chunk_index,
                "family": chunk["family"],
                "split": chunk["split"],
                "revision_id": chunk["revision_id"],
                "attempted_cases": source_trajectories,
                "accepted_cases": int(chunk["accepted_count"]),
                "accepted_rows": source_rows,
                "summary": summary_artifact.to_record(),
                "manifest": source_manifest_artifact.to_record(),
                "trajectory_map": source_map_artifact.to_record(),
                "batch_artifacts": batch_artifacts,
                "shards": len(source_shards_raw),
            }
        )
        trajectory_offset += source_trajectories
        row_offset += source_rows
        shard_offset += len(source_shards_raw)
        batch_offset += len(source_batches)

    if (
        trajectory_offset != combined_map["trajectory_family_id"].size
        or row_offset != combined_map["trajectory_index"].size
        or shard_offset != len(combined_shards)
        or batch_offset != len(combined_batches)
    ):
        raise RuntimeError("combined source-binding totals are incomplete")
    return tuple(source_records), combined_shards


def load_and_authenticate(
    combined_summary_path: Path,
    *,
    contract: DatasetContract = FINAL_CONTRACT,
    authenticator: FileAuthenticator | None = None,
) -> tuple[LoadedInput, FileAuthenticator, str, dict[str, object]]:
    """Authenticate one completed combined view and its exact source population."""

    files = authenticator or FileAuthenticator()
    summary_artifact = files.authenticate(
        combined_summary_path, context="combined summary"
    )
    summary = _read_authenticated_json(
        files,
        summary_artifact,
        context="combined summary",
    )
    if summary.get("schema") != COMBINED_SUMMARY_SCHEMA:
        raise ValueError("combined summary has the wrong schema")
    if summary.get("status") != "complete":
        raise ValueError("combined summary is not complete")
    preflight = _mapping(summary.get("preflight"), context="combined preflight")
    chunks = _validate_preflight(preflight, contract=contract)
    view = _mapping(summary.get("dataset_view"), context="combined dataset_view")
    manifest_artifact = _artifact_from_record(
        files,
        base=summary_artifact.path.parent,
        value=view.get("manifest"),
        context="combined manifest",
    )
    map_artifact = _artifact_from_record(
        files,
        base=summary_artifact.path.parent,
        value=view.get("trajectory_map"),
        context="combined trajectory map",
    )
    _require_within(
        manifest_artifact.path,
        summary_artifact.path.parent,
        context="combined manifest",
    )
    _require_within(
        map_artifact.path,
        summary_artifact.path.parent,
        context="combined trajectory map",
    )
    manifest = _read_authenticated_json(
        files,
        manifest_artifact,
        context="combined manifest",
    )
    map_path = _resolve_path(
        manifest_artifact.path.parent,
        manifest.get("trajectory_map_npz"),
        context="combined manifest trajectory map",
    )
    if (
        map_path != map_artifact.path
        or manifest.get("trajectory_map_sha256") != map_artifact.sha256
    ):
        raise RuntimeError("combined manifest names a different trajectory map")
    contract_fingerprint = _validate_manifest(
        manifest, preflight=preflight, view_record=view, contract=contract
    )
    map_arrays = _load_map(
        map_artifact,
        authenticator=files,
        context="combined trajectory map",
    )
    population = _validate_map_population(
        map_arrays, manifest=manifest, contract=contract
    )
    source_artifacts, shards = _source_artifacts(
        chunks,
        combined_manifest=manifest,
        combined_manifest_path=manifest_artifact.path,
        combined_map=map_arrays,
        authenticator=files,
    )
    _validate_map_shard_coordinates(map_arrays, shards)
    return (
        LoadedInput(
            summary=summary_artifact,
            manifest=manifest_artifact,
            trajectory_map=map_artifact,
            manifest_record=manifest,
            map_arrays=map_arrays,
            shards=shards,
            chunks=chunks,
            source_artifacts=source_artifacts,
            attempted_cases=int(population["attempted_cases"]),
        ),
        files,
        contract_fingerprint,
        population,
    )


@dataclass
class Extrema:
    """Streaming minimum, maximum, and absolute maximum."""

    minimum: float = math.inf
    maximum: float = -math.inf
    absolute_maximum: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise RuntimeError("normalization input contains a nonfinite value")
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.absolute_maximum = max(
            self.absolute_maximum, float(np.max(np.abs(values)))
        )

    def require_nonempty(self, *, context: str) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise RuntimeError(f"no selected values were observed for {context}")


def _selection(indices: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(indices, dtype="<i8")
    return {
        "count": int(canonical.shape[0]),
        "sha256": hashlib.sha256(memoryview(canonical).cast("B")).hexdigest(),
    }


def training_row_selection(
    arrays: Mapping[str, np.ndarray], *, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Reproduce build_dataset_split_indices(...)[0] and _index_selection."""

    indices = split_row_selections(arrays, seed=seed)["train"]
    return indices, _selection(indices)


def split_row_selections(
    arrays: Mapping[str, np.ndarray], *, seed: int
) -> dict[str, np.ndarray]:
    """Reproduce all preassigned split selections used by the trainer."""

    owners = arrays["trajectory_index"]
    accepted = arrays["trajectory_accepted"][owners]
    row_split = arrays["trajectory_split_id"][owners]
    rng = np.random.default_rng(seed)
    return {
        split: rng.permutation(
            np.flatnonzero(accepted & (row_split == split_id))
        )
        for split, split_id in SPLIT_IDS.items()
    }


def full_row_selection_record(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: DatasetContract,
    seed: int,
) -> dict[str, object]:
    """Audit that every retained row belongs to exactly one fixed split."""

    selections = split_row_selections(arrays, seed=seed)
    owners = arrays["trajectory_index"]
    trajectory_families = arrays["trajectory_family_id"]
    trajectory_splits = arrays["trajectory_split_id"]
    expected_counts = {
        split: sum(
            contract.accepted_cases_per_family_by_split[split]
            * contract.rows_per_accepted_case[family]
            for family in FAMILY_IDS
        )
        for split in SPLIT_IDS
    }
    selection_records: dict[str, object] = {}
    for split, selected in selections.items():
        selected_owners = owners[selected]
        candidates = np.flatnonzero(
            arrays["trajectory_accepted"][owners]
            & (trajectory_splits[owners] == SPLIT_IDS[split])
        )
        if (
            selected.size != expected_counts[split]
            or np.unique(selected).size != selected.size
            or not np.array_equal(np.sort(selected), candidates)
            or np.any(trajectory_splits[selected_owners] != SPLIT_IDS[split])
        ):
            raise RuntimeError(
                f"{split} selection does not contain every retained row exactly once"
            )
        per_family = {
            family: int(
                np.count_nonzero(trajectory_families[selected_owners] == family_id)
            )
            for family, family_id in FAMILY_IDS.items()
        }
        expected_per_family = {
            family: contract.accepted_cases_per_family_by_split[split]
            * contract.rows_per_accepted_case[family]
            for family in FAMILY_IDS
        }
        if per_family != expected_per_family:
            raise RuntimeError(f"{split} row counts differ from contract")
        selection_records[split] = {
            "policy": "all_retained_rows",
            "per_family_row_count": per_family,
            "all_retained_rows": True,
            "dtype": "<i8",
            **_selection(selected),
        }
    train = selections["train"]
    validation = selections["validation"]
    test_leakage = int(
        np.count_nonzero(
            trajectory_splits[owners[np.concatenate((train, validation))]]
            == SPLIT_IDS["test"]
        )
    )
    if test_leakage != 0:
        raise RuntimeError("train/validation selections leak test rows")
    return {
        "schema": "paper_dataset_full_row_selection_v1",
        "seed": seed,
        "splits": selection_records,
        "test_rows_in_train_or_validation": test_leakage,
    }


def compute_streaming_stats(
    loaded: LoadedInput,
    *,
    authenticator: FileAuthenticator,
    seed: int,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Hash and stream all shards, reducing only accepted training rows."""

    train_indices, selection = training_row_selection(loaded.map_arrays, seed=seed)
    selected_mask = np.zeros(loaded.map_arrays["trajectory_index"].size, dtype=np.bool_)
    selected_mask[train_indices] = True
    eta_extrema = Extrema()
    xi_extrema = Extrema()
    gxi_extrema = Extrema()
    depth_extrema = Extrema()
    log_depth_extrema = Extrema()
    shard_records: list[dict[str, object]] = []
    cursor = 0
    for shard in loaded.shards:
        stop = cursor + shard.rows
        local_mask = selected_mask[cursor:stop]
        before = shard.path.stat()
        artifact = authenticator.authenticate(
            shard.path,
            expected_sha256=shard.sha256,
            context=f"combined shard {shard.index}",
        )
        with np.load(shard.path, allow_pickle=False) as archive:
            required = {"eta", "xi", "gxi", "depth", "time"}
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(
                    f"combined shard {shard.index} omits {sorted(missing)}"
                )
            eta = np.asarray(archive["eta"], dtype=np.float32)
            if eta.shape != (shard.rows, int(loaded.manifest_record["grid"]["nx"])):
                raise RuntimeError(f"combined shard {shard.index} eta shape differs")
            eta_extrema.update(eta[local_mask])
            del eta
            xi = np.asarray(archive["xi"], dtype=np.float32)
            if xi.shape != (shard.rows, int(loaded.manifest_record["grid"]["nx"])):
                raise RuntimeError(f"combined shard {shard.index} xi shape differs")
            xi = xi - xi.mean(axis=1, keepdims=True)
            xi_extrema.update(xi[local_mask])
            del xi
            gxi = np.asarray(archive["gxi"], dtype=np.float32)
            if gxi.shape != (shard.rows, int(loaded.manifest_record["grid"]["nx"])):
                raise RuntimeError(f"combined shard {shard.index} gxi shape differs")
            gxi_extrema.update(gxi[local_mask])
            del gxi
            depth32 = np.asarray(archive["depth"], dtype=np.float32)
            time32 = np.asarray(archive["time"], dtype=np.float32)
            if depth32.shape != (shard.rows,) or time32.shape != (shard.rows,):
                raise RuntimeError(f"combined shard {shard.index} scalar shape differs")
            depth64 = np.asarray(depth32, dtype=np.float64)
            selected_depth = depth64[local_mask]
            if np.any(selected_depth <= 0.0):
                raise RuntimeError("selected training depth must be strictly positive")
            depth_extrema.update(selected_depth)
            log_depth_extrema.update(np.log(np.clip(selected_depth, 1.0e-12, None)))
        after = shard.path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise RuntimeError(f"combined shard {shard.index} changed during audit")
        authenticator.assert_unchanged(
            artifact, context=f"combined shard {shard.index}"
        )
        shard_records.append(
            {
                "index": shard.index,
                "rows": shard.rows,
                "selected_training_rows": int(np.count_nonzero(local_mask)),
                **artifact.to_record(),
            }
        )
        cursor = stop
    if cursor != selected_mask.size:
        raise RuntimeError("streamed shard rows do not cover the combined map")
    for name, extrema in (
        ("eta", eta_extrema),
        ("xi", xi_extrema),
        ("gxi", gxi_extrema),
        ("depth", depth_extrema),
        ("log_depth", log_depth_extrema),
    ):
        extrema.require_nonempty(context=name)
    stats: dict[str, object] = {
        "dataset": str(loaded.manifest.path),
        "num_examples": int(train_indices.size),
        "storage_num_examples": int(selected_mask.size),
        "feature_min": [eta_extrema.minimum, xi_extrema.minimum],
        "feature_max": [eta_extrema.maximum, xi_extrema.maximum],
        "feature_absmax": [
            eta_extrema.absolute_maximum,
            xi_extrema.absolute_maximum,
        ],
        "target_min": gxi_extrema.minimum,
        "target_max": gxi_extrema.maximum,
        "target_absmax": gxi_extrema.absolute_maximum,
        "depth_min": depth_extrema.minimum,
        "depth_max": depth_extrema.maximum,
        "log_depth_min": log_depth_extrema.minimum,
        "log_depth_max": log_depth_extrema.maximum,
        "domain_length": float(loaded.manifest_record["grid"]["length"]),
        "index_selection": selection,
        "dataset_identity": {
            "schema_version": loaded.manifest_record["schema_version"],
            "manifest_sha256": loaded.manifest.sha256,
            "dataset_contract_fingerprint": loaded.manifest_record[
                "dataset_contract_fingerprint"
            ],
            "trajectory_map_sha256": loaded.manifest_record["trajectory_map_sha256"],
        },
    }
    return stats, tuple(shard_records)


def audit_training_handoff(
    combined_summary_path: Path,
    *,
    seed: int = 0,
    contract: DatasetContract = FINAL_CONTRACT,
) -> dict[str, object]:
    """Return a complete, strict JSON record for one authenticated handoff."""

    authenticator = FileAuthenticator()
    training_implementation = _training_implementation_record(
        authenticator=authenticator
    )
    loaded, authenticator, dataset_contract_fingerprint, population = (
        load_and_authenticate(
            combined_summary_path,
            contract=contract,
            authenticator=authenticator,
        )
    )
    stats, shard_records = compute_streaming_stats(
        loaded, authenticator=authenticator, seed=seed
    )
    full_row_selection = full_row_selection_record(
        loaded.map_arrays,
        contract=contract,
        seed=seed,
    )
    expected_training_rows = sum(
        contract.accepted_cases_per_family_by_split["train"]
        * contract.rows_per_accepted_case[family]
        for family in FAMILY_IDS
    )
    if stats["num_examples"] != expected_training_rows:
        raise RuntimeError("training-row selection count differs from contract")
    stats_fingerprint = canonical_json_sha256(stats)
    dataset_argument = str(loaded.manifest.path)
    source_binding_payload = {
        "chunks": list(loaded.source_artifacts),
        "source_count": len(loaded.source_artifacts),
    }
    source_binding_fingerprint = canonical_json_sha256(source_binding_payload)
    shard_identity_payload = [
        {key: record[key] for key in ("index", "path", "bytes", "sha256", "rows")}
        for record in shard_records
    ]
    record: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "status": "complete",
        "purpose": "training_handoff_normalization_audit",
        "audit_only": True,
        "acceptance_or_generation_effect": False,
        "execution": {
            "backend": "numpy_cpu_streaming",
            "gpu_used": False,
            "training_split_seed": seed,
            "xi_projection": "float32 row mean subtraction",
        },
        "training_implementation": training_implementation,
        "combined_summary": loaded.summary.to_record(),
        "dataset_view": {
            "manifest": loaded.manifest.to_record(),
            "trajectory_map": loaded.trajectory_map.to_record(),
            "dataset_contract_fingerprint": dataset_contract_fingerprint,
            "shard_count": len(shard_records),
            "shard_rows": sum(int(record["rows"]) for record in shard_records),
            "shard_bytes": sum(int(record["bytes"]) for record in shard_records),
            "shard_identity_fingerprint": canonical_json_sha256(shard_identity_payload),
            "shards": list(shard_records),
        },
        "source_binding": {
            **source_binding_payload,
            "fingerprint": source_binding_fingerprint,
        },
        "final_contract": {
            "expected": contract.to_record(),
            "observed": population,
        },
        "training_normalization": {
            "trainer_semantics": (
                "load_dataset_arrays + build_dataset_split_indices + "
                "load_or_compute_stats"
            ),
            "accepted_training_row_selection": {
                "split": "train",
                "split_id": SPLIT_IDS["train"],
                "seed": seed,
                "dtype": "<i8",
                **stats["index_selection"],
            },
            "full_row_selection": full_row_selection,
            "stats": stats,
            "stats_fingerprint": stats_fingerprint,
            "handoff": {
                "audit_artifact_is_trainer_stats_cache": False,
                "stats_cache_written_by_audit": False,
                "expected_trainer_stats_cache": str(
                    loaded.manifest.path.with_suffix(".stats.json")
                ),
                "comparison": (
                    "After trainer startup, require its stats cache to equal "
                    "training_normalization.stats and its canonical JSON SHA-256 "
                    "to equal training_normalization.stats_fingerprint."
                ),
                "installation": (
                    "No cache is installed by this read-only audit; the canonical "
                    "trainer invocation computes or reuses the adjacent cache."
                ),
                "canonical_command_template": [
                    "uv",
                    "run",
                    "python",
                    "train-jax-10m/1d_dno_fno_jax.py",
                    "--dataset",
                    dataset_argument,
                    "--seed",
                    str(seed),
                    "--norm",
                    "scale",
                    "<model-and-training-arguments>",
                ],
            },
        },
        "integrity": {
            "unique_files_hashed": authenticator.unique_files,
            "unique_bytes_hashed": authenticator.unique_bytes,
        },
    }
    json.dumps(record, allow_nan=False)
    return record


def _default_output(combined_summary: Path) -> Path:
    name = combined_summary.name
    stem = name[: -len(".summary.json")] if name.endswith(".summary.json") else name
    return combined_summary.with_name(f"{stem}.training_handoff_audit.json")


def write_audit(path: Path, record: Mapping[str, object]) -> None:
    """Atomically publish only a complete audit record."""

    if record.get("schema") != AUDIT_SCHEMA or record.get("status") != "complete":
        raise ValueError("only a complete training-handoff audit may be written")
    _authenticate_training_implementation_record(record.get("training_implementation"))
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise FileExistsError("audit output must not be a symbolic link")
    destination = requested.resolve()
    input_paths: set[Path] = set()

    def add_artifact(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            input_paths.add(Path(raw_path).expanduser().resolve())

    add_artifact(record.get("combined_summary"))
    dataset_view = record.get("dataset_view")
    if isinstance(dataset_view, Mapping):
        add_artifact(dataset_view.get("manifest"))
        add_artifact(dataset_view.get("trajectory_map"))
        raw_shards = dataset_view.get("shards")
        if isinstance(raw_shards, Sequence) and not isinstance(
            raw_shards, (str, bytes)
        ):
            for shard in raw_shards:
                add_artifact(shard)
    source_binding = record.get("source_binding")
    if isinstance(source_binding, Mapping):
        raw_chunks = source_binding.get("chunks")
        if isinstance(raw_chunks, Sequence) and not isinstance(
            raw_chunks, (str, bytes)
        ):
            for raw_chunk in raw_chunks:
                if isinstance(raw_chunk, Mapping):
                    for name in ("summary", "manifest", "trajectory_map"):
                        add_artifact(raw_chunk.get(name))
                    raw_batches = raw_chunk.get("batch_artifacts")
                    if isinstance(raw_batches, Sequence) and not isinstance(
                        raw_batches, (str, bytes)
                    ):
                        for batch_artifact in raw_batches:
                            add_artifact(batch_artifact)
    if destination in input_paths:
        raise FileExistsError("audit output would overwrite an authenticated input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing: object = None
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass
        if not isinstance(existing, Mapping) or existing.get("schema") != AUDIT_SCHEMA:
            raise FileExistsError(
                "existing audit output is not a training-handoff audit"
            )
        if canonical_json_sha256(existing) != canonical_json_sha256(record):
            raise FileExistsError(
                "existing training-handoff audit has different content"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Rehash the implementation/evidence after computation and serialization,
        # as close as possible to the immutable publication boundary.
        _authenticate_training_implementation_record(
            record.get("training_implementation")
        )
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    record = audit_training_handoff(args.combined_summary, seed=args.seed)
    output = args.output or _default_output(args.combined_summary)
    write_audit(output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "output": str(output.expanduser().resolve()),
                "training_normalization": record["training_normalization"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
