"""Diagnose Craig--Sulem order convergence on the final JONSWAP tail.

This program is deliberately CPU-only and read-only with respect to the
dataset.  It requires the passed revision-4 completion audit, independently
rescans every accepted trajectory to find the three largest predefined
``G(eta)xi`` modes-96:128 fractions, and recomputes those trajectories at
orders four through eight in float64 with the frozen P128/pad-eight target.

The resulting JSON is diagnostic evidence only.  Neither its values nor its
status participate in dataset acceptance or release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These must precede every import that can initialize JAX.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-jonswap-order-diagnostic")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.reference import project_fixed_band  # noqa: E402
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
)


AUDIT_SCHEMA = "paper_dataset_jonswap_tma_revision4_completion_audit_v1"
DIAGNOSTIC_SCHEMA = "paper_dataset_jonswap_revision4_order_convergence_v1"
RENDERER_SCHEMA = "paper_dataset_all_family_diagnostic_tail_v3"
COMBINED_SUMMARY_SCHEMA = "paper_dataset_combined_view_summary_v1"
DEFAULT_DATASET_ROOT = ROOT / "outputs/paper_dataset_jonswap_revision4_relative_band_v1"
EXPECTED_ACCEPTED = 18_432
EXPECTED_CHUNKS = 8
ROWS_PER_TRAJECTORY = 16
NX = 1024
LENGTH = 2.0 * math.pi
P_MAX = 128.0
PAD_FACTOR = 8
ORDERS = (4, 5, 6, 7, 8)
TOP_COUNT = 3
HIGH_BAND = (96.0, 128.0)
IMPLEMENTATION_SCHEMA = "paper_dataset_order_diagnostic_implementation_v1"
IMPLEMENTATION_FILES = (
    (
        "scripts/diagnose_final_jonswap_order_convergence.py",
        "order_diagnostic_scan_recomputation_and_record_producer",
    ),
    (
        "scripts/audit_completed_jonswap_revision4.py",
        "completion_audit_schema_and_tail_metric_producer",
    ),
    (
        "solver/gen_data/pipeline/reference.py",
        "fixed_band_projection_implementation",
    ),
    (
        "solver/solvers/dno_series_jax.py",
        "craig_sulem_order_evaluator",
    ),
)
IMPLEMENTATION_RELATIONSHIP = {
    "scope": "direct_repository_implementation_bytes",
    "external_runtime_is_not_transitively_authenticated": True,
}
SELECTION_DEFINITION = (
    "For every accepted trajectory, maximize over its 16 stored frames the "
    "ratio sum_{96<=|k|<=128}|Gxi_hat_k|^2 / "
    "sum_{1<=|k|<=128}|Gxi_hat_k|^2, then take the three largest "
    "trajectories; ties retain completion-audit scan order."
)
RELATIVE_L2_DEFINITION = (
    "sqrt(mean(|a-b|^2))/sqrt(mean(|b|^2)); each saved frame is evaluated separately"
)
INTERPRETATION_SCOPE = (
    "Formal Craig--Sulem order stability of the declared discrete P128/pad8 "
    "target on the final three largest stored JONSWAP tails; this is not an "
    "exact-DNO or continuum-convergence claim and cannot reject a case."
)

JsonObject = Mapping[str, object]
OrderEvaluator = Callable[[np.ndarray, np.ndarray, float], Mapping[int, "OrderFields"]]


@dataclass(frozen=True)
class BatchBinding:
    """Paths and attempt interval for one immutable transaction."""

    batch_id: int
    shard_index: int
    attempt_start: int
    attempt_stop: int
    proposal_path: Path
    proposal_sha256: str
    result_path: Path
    result_sha256: str


@dataclass(frozen=True)
class Candidate:
    """One accepted trajectory and its independently measured peak tail."""

    scan_index: int
    peak_fraction: float
    peak_frame_index: int
    case_id: int
    chunk_label: str
    split: str
    trajectory_index: int
    cell_id: str
    batch: BatchBinding
    local_index: int
    shard_path: Path
    shard_relative_path: str
    shard_sha256: str
    shard_first_row: int
    summary_path: Path
    summary_sha256: str
    manifest_path: Path
    manifest_sha256: str
    trajectory_map_path: Path
    trajectory_map_sha256: str

    def audit_identity(self) -> dict[str, object]:
        """Return the identity representation used by the completion audit."""

        return {
            "case_id": self.case_id,
            "chunk_label": self.chunk_label,
            "split": self.split,
            "batch_id": self.batch.batch_id,
            "local_index": self.local_index,
            "frame_index": self.peak_frame_index,
            "cell_id": self.cell_id,
            "shard_path": self.shard_relative_path,
        }


@dataclass(frozen=True)
class OrderFields:
    """Raw and P128-projected cumulative DNO fields at one order."""

    raw: np.ndarray
    projected: np.ndarray


@dataclass(frozen=True)
class ScanResult:
    """Authenticated result of the complete tail-ranking scan."""

    candidates: tuple[Candidate, ...]
    accepted_scanned: int
    rows_scanned: int
    shards_scanned: int
    chunk_bindings: tuple[dict[str, object], ...]
    completion_audit_maximum_value: float
    independent_maximum_value: float
    completion_audit_maximum_absolute_difference: float


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def diagnostic_implementation_record(
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Bind implementation bytes at process start with within-set stability."""

    root = repository_root.expanduser().resolve(strict=True)
    files: list[dict[str, object]] = []
    identities: list[tuple[Path, tuple[int, ...]]] = []
    for relative, role in IMPLEMENTATION_FILES:
        requested = root / relative
        if requested.is_symlink():
            raise ValueError(f"implementation file must not be a symlink: {requested}")
        path = requested.resolve(strict=True)
        if path != requested or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"implementation file is not canonical: {requested}")
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"implementation file changed while hashed: {relative}")
        identities.append((path, after_identity))
        files.append(
            {
                "path": relative,
                "role": role,
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    for (relative, _), (path, expected_identity) in zip(
        IMPLEMENTATION_FILES, identities, strict=True
    ):
        current = path.stat()
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if current_identity != expected_identity:
            raise RuntimeError(f"implementation file changed after hashing: {relative}")
    payload: dict[str, object] = {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": files,
        "semantic_relationship": dict(IMPLEMENTATION_RELATIONSHIP),
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def _require_diagnostic_implementation_current(
    expected: Mapping[str, object],
) -> None:
    if diagnostic_implementation_record() != dict(expected):
        raise RuntimeError("order diagnostic implementation changed during execution")


def validated_output_path(path: Path) -> Path:
    """Return a safe output path without clobbering another artifact type."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return resolved
    if not resolved.is_file():
        raise FileExistsError(f"diagnostic output path is not a file: {resolved}")
    try:
        existing = _strict_json_object(resolved)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise FileExistsError(
            f"refusing to replace non-diagnostic artifact: {resolved}"
        ) from error
    if existing.get("schema") != DIAGNOSTIC_SCHEMA:
        raise FileExistsError(
            f"refusing to replace non-diagnostic artifact: {resolved}"
        )
    return resolved


def _mapping(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty string")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _contained_path(root: Path, value: object, *, context: str) -> Path:
    raw = Path(_string(value, context=context)).expanduser()
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{context} escapes the dataset root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    return resolved


def _artifact_path(root: Path, value: object, *, context: str) -> Path:
    raw = Path(_string(value, context=context))
    if raw.is_absolute():
        raise ValueError(f"{context} must be relative to its chunk root")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{context} escapes its chunk root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    return resolved


def _verified_hash(path: Path, expected: object, *, context: str) -> str:
    expected_hash = _string(expected, context=f"{context}.sha256")
    actual = file_sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{context} hash mismatch")
    return actual


def high_band_fraction(fields: np.ndarray) -> np.ndarray:
    """Return the completion-audit P96:128/P1:128 energy fraction."""

    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("high-band input must have shape (frames, nx)")
    nx = values.shape[1]
    modes = 2.0 * np.pi * np.fft.rfftfreq(nx, d=LENGTH / nx)
    coefficients = np.fft.rfft(values, axis=1)
    multiplicity = np.full(modes.size, 2.0, dtype=np.float64)
    multiplicity[0] = 1.0
    if nx % 2 == 0:
        multiplicity[-1] = 1.0
    energy = multiplicity[None, :] * np.square(np.abs(coefficients))
    delivered = (modes >= 1.0) & (modes <= HIGH_BAND[1])
    high = (modes >= HIGH_BAND[0]) & (modes <= HIGH_BAND[1])
    denominator = np.sum(energy[:, delivered], axis=1)
    return np.divide(
        np.sum(energy[:, high], axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def _load_map(
    path: Path, *, attempted: int, retained_rows: int
) -> dict[str, np.ndarray]:
    expected_dtypes = {
        "frame_index": np.dtype(np.int32),
        "shard_index": np.dtype(np.int32),
        "shard_row": np.dtype(np.int64),
        "trajectory_accepted": np.dtype(np.bool_),
        "trajectory_case_id": np.dtype(np.int64),
        "trajectory_cell_id": np.dtype(np.int32),
        "trajectory_family_id": np.dtype(np.int16),
        "trajectory_first_row": np.dtype(np.int64),
        "trajectory_index": np.dtype(np.int32),
        "trajectory_revision_id": np.dtype(np.int16),
        "trajectory_row_count": np.dtype(np.int32),
        "trajectory_split_id": np.dtype(np.uint8),
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(expected_dtypes) - set(archive.files))
        if missing:
            raise ValueError(f"trajectory map omits arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in expected_dtypes}
        if "schema_version" not in archive.files or int(archive["schema_version"]) != 2:
            raise ValueError("trajectory map must use schema version 2")
    trajectory_names = {
        "trajectory_accepted",
        "trajectory_case_id",
        "trajectory_cell_id",
        "trajectory_family_id",
        "trajectory_first_row",
        "trajectory_revision_id",
        "trajectory_row_count",
        "trajectory_split_id",
    }
    for name, dtype in expected_dtypes.items():
        array = arrays[name]
        if array.dtype != dtype:
            raise TypeError(
                f"trajectory map {name} has dtype {array.dtype}, expected {dtype}"
            )
        expected_shape = (attempted,) if name in trajectory_names else (retained_rows,)
        if array.shape != expected_shape:
            raise ValueError(
                f"trajectory map {name} has shape {array.shape}, expected {expected_shape}"
            )
    return arrays


def _batch_bindings(
    manifest: JsonObject,
    *,
    chunk_root: Path,
    attempted: int,
) -> tuple[tuple[BatchBinding, ...], dict[int, JsonObject]]:
    raw_batches = _list(
        manifest.get("dataset_batches"), context="manifest.dataset_batches"
    )
    raw_shards = _list(
        manifest.get("dataset_shards"), context="manifest.dataset_shards"
    )
    shards_by_index: dict[int, JsonObject] = {}
    for raw in raw_shards:
        shard = _mapping(raw, context="manifest shard")
        index = _integer(shard.get("batch_index"), context="manifest shard.batch_index")
        if index in shards_by_index:
            raise ValueError("manifest repeats a shard batch index")
        shards_by_index[index] = shard

    cursor = 0
    bindings: list[BatchBinding] = []
    for position, raw in enumerate(raw_batches):
        batch = _mapping(raw, context=f"manifest.dataset_batches[{position}]")
        batch_id = _integer(batch.get("batch_id"), context="dataset batch.batch_id")
        shard_index = _integer(
            batch.get("shard_index"), context="dataset batch.shard_index"
        )
        attempted_count = _integer(
            batch.get("n_attempted_trajectories"),
            context="dataset batch.n_attempted_trajectories",
        )
        if shard_index not in shards_by_index:
            raise ValueError("dataset batch references a missing shard")
        proposal_path = _artifact_path(
            chunk_root,
            batch.get("proposal_path"),
            context="dataset batch.proposal_path",
        )
        result_path = _artifact_path(
            chunk_root, batch.get("result_path"), context="dataset batch.result_path"
        )
        bindings.append(
            BatchBinding(
                batch_id=batch_id,
                shard_index=shard_index,
                attempt_start=cursor,
                attempt_stop=cursor + attempted_count,
                proposal_path=proposal_path,
                proposal_sha256=_string(
                    batch.get("proposal_sha256"),
                    context="dataset batch.proposal_sha256",
                ),
                result_path=result_path,
                result_sha256=_string(
                    batch.get("result_sha256"), context="dataset batch.result_sha256"
                ),
            )
        )
        cursor += attempted_count
    if cursor != attempted:
        raise ValueError("manifest batch attempts do not cover the trajectory map")
    if len({binding.batch_id for binding in bindings}) != len(bindings):
        raise ValueError("manifest repeats a batch id")
    if len({binding.shard_index for binding in bindings}) != len(bindings):
        raise ValueError("manifest repeats a shard index")
    return tuple(bindings), shards_by_index


def _validate_trajectory_map(
    arrays: Mapping[str, np.ndarray],
    *,
    bindings: Sequence[BatchBinding],
    accepted: int,
    split_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted_mask = arrays["trajectory_accepted"]
    accepted_indices = np.flatnonzero(accepted_mask)
    if accepted_indices.size != accepted:
        raise ValueError("trajectory map accepted count differs from completion audit")
    if np.unique(arrays["trajectory_case_id"]).size != accepted_mask.size:
        raise ValueError("trajectory map repeats case IDs within a chunk")
    if not np.all(arrays["trajectory_family_id"] == 4):
        raise ValueError("trajectory map contains a non-JONSWAP family")
    if not np.all(arrays["trajectory_revision_id"] == 4):
        raise ValueError("trajectory map contains a non-revision-4 trajectory")
    if not np.all(arrays["trajectory_split_id"] == split_code):
        raise ValueError("trajectory map split differs from the audited chunk")
    if not np.array_equal(
        arrays["trajectory_first_row"][accepted_indices],
        np.arange(accepted, dtype=np.int64) * ROWS_PER_TRAJECTORY,
    ):
        raise ValueError("accepted trajectory first rows are not contiguous")
    if not np.all(
        arrays["trajectory_row_count"][accepted_indices] == ROWS_PER_TRAJECTORY
    ):
        raise ValueError("accepted trajectory row count differs from 16")
    rejected = np.flatnonzero(~accepted_mask)
    if not np.all(arrays["trajectory_first_row"][rejected] == -1) or not np.all(
        arrays["trajectory_row_count"][rejected] == 0
    ):
        raise ValueError("rejected trajectories own rows")
    if not np.array_equal(
        arrays["trajectory_index"], np.repeat(accepted_indices, ROWS_PER_TRAJECTORY)
    ):
        raise ValueError("row-to-trajectory ownership is not canonical")
    if not np.array_equal(
        arrays["frame_index"], np.tile(np.arange(ROWS_PER_TRAJECTORY), accepted)
    ):
        raise ValueError("trajectory-map frame indices are not canonical")

    batch_for_attempt = np.empty(accepted_mask.size, dtype=np.int64)
    local_for_attempt = np.empty(accepted_mask.size, dtype=np.int64)
    shard_for_attempt = np.empty(accepted_mask.size, dtype=np.int64)
    for batch_position, binding in enumerate(bindings):
        selected = slice(binding.attempt_start, binding.attempt_stop)
        batch_for_attempt[selected] = batch_position
        local_for_attempt[selected] = np.arange(
            binding.attempt_stop - binding.attempt_start, dtype=np.int64
        )
        shard_for_attempt[selected] = binding.shard_index
    if not np.array_equal(
        arrays["shard_index"],
        np.repeat(shard_for_attempt[accepted_indices], ROWS_PER_TRAJECTORY),
    ):
        raise ValueError("row map assigns a trajectory to the wrong batch shard")
    return batch_for_attempt, local_for_attempt, accepted_indices


def _split_code(split: str) -> int:
    codes = {"train": 0, "validation": 1, "test": 2}
    if split not in codes:
        raise ValueError(f"unsupported JONSWAP split {split!r}")
    return codes[split]


def _scan_chunk(
    dataset_root: Path,
    audit_chunk: JsonObject,
    *,
    scan_start: int,
) -> tuple[list[Candidate], int, int, dict[str, object]]:
    label = _string(audit_chunk.get("label"), context="audit chunk.label")
    split = _string(audit_chunk.get("split"), context="audit chunk.split")
    accepted = _integer(
        audit_chunk.get("accepted_count"), context="audit chunk.accepted_count"
    )
    attempted = _integer(
        audit_chunk.get("attempted_count"), context="audit chunk.attempted_count"
    )
    retained_rows = _integer(
        audit_chunk.get("retained_rows"), context="audit chunk.retained_rows"
    )
    if retained_rows != accepted * ROWS_PER_TRAJECTORY:
        raise ValueError("audit chunk retained-row count is inconsistent")
    summary_path = _contained_path(
        dataset_root, audit_chunk.get("summary_path"), context="audit chunk.summary_path"
    )
    summary_sha256 = _verified_hash(
        summary_path, audit_chunk.get("summary_sha256"), context="chunk summary"
    )
    chunk_root = summary_path.parent
    summary = _strict_json_object(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("generation summary is not complete")
    dataset_view = _mapping(summary.get("dataset_view"), context="summary.dataset_view")
    manifest_record = _mapping(
        dataset_view.get("manifest"), context="summary.dataset_view.manifest"
    )
    map_record = _mapping(
        dataset_view.get("trajectory_map"),
        context="summary.dataset_view.trajectory_map",
    )
    manifest_path = _artifact_path(
        chunk_root, manifest_record.get("path"), context="dataset view manifest.path"
    )
    map_path = _artifact_path(
        chunk_root, map_record.get("path"), context="dataset view trajectory_map.path"
    )
    manifest_sha256 = _verified_hash(
        manifest_path, manifest_record.get("sha256"), context="dataset manifest"
    )
    map_sha256 = _verified_hash(
        map_path, map_record.get("sha256"), context="trajectory map"
    )
    if (
        Path(
            _string(audit_chunk.get("manifest_path"), context="audit manifest path")
        ).resolve()
        != manifest_path
    ):
        raise ValueError("completion audit and summary name different manifests")
    if audit_chunk.get("manifest_sha256") != manifest_sha256:
        raise ValueError("completion audit and summary have different manifest hashes")
    if (
        Path(
            _string(audit_chunk.get("trajectory_map_path"), context="audit map path")
        ).resolve()
        != map_path
    ):
        raise ValueError("completion audit and summary name different trajectory maps")
    if audit_chunk.get("trajectory_map_sha256") != map_sha256:
        raise ValueError("completion audit and summary have different map hashes")

    manifest = _strict_json_object(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("dataset manifest must use schema version 2")
    grid = _mapping(manifest.get("grid"), context="manifest.grid")
    if (
        grid.get("nx") != NX
        or _finite(grid.get("length"), context="grid.length") != LENGTH
    ):
        raise ValueError("dataset manifest differs from the exact P128 target grid")
    if manifest.get("n_accepted_trajectories") != accepted:
        raise ValueError("manifest accepted count differs from completion audit")
    if manifest.get("n_accepted_rows") != retained_rows:
        raise ValueError("manifest row count differs from completion audit")
    if manifest.get("n_trajectories") != attempted:
        raise ValueError("manifest attempt count differs from completion audit")

    bindings, shards_by_index = _batch_bindings(
        manifest, chunk_root=chunk_root, attempted=attempted
    )
    arrays = _load_map(map_path, attempted=attempted, retained_rows=retained_rows)
    batch_for_attempt, local_for_attempt, accepted_indices = _validate_trajectory_map(
        arrays,
        bindings=bindings,
        accepted=accepted,
        split_code=_split_code(split),
    )
    run_spec = _mapping(summary.get("run_spec"), context="summary.run_spec")
    configuration = _mapping(
        run_spec.get("configuration"), context="summary.run_spec.configuration"
    )
    generation_sources = _mapping(
        configuration.get("source_sha256"),
        context="summary.run_spec.configuration.source_sha256",
    )
    target_source_sha256: dict[str, str] = {}
    for relative_path in (
        "solver/gen_data/pipeline/reference.py",
        "solver/solvers/dno_series_jax.py",
    ):
        expected_hash = _string(
            generation_sources.get(relative_path),
            context=f"generation source hash {relative_path}",
        )
        current_hash = file_sha256(ROOT / relative_path)
        if current_hash != expected_hash:
            raise ValueError(
                f"current target implementation differs from generation: {relative_path}"
            )
        target_source_sha256[relative_path] = expected_hash
    raw_cell_codes = _mapping(run_spec.get("cell_codes"), context="run_spec.cell_codes")
    cell_ids = {
        _integer(code, context=f"cell code {cell_id}"): str(cell_id)
        for cell_id, code in raw_cell_codes.items()
    }
    unknown_codes = sorted(set(map(int, arrays["trajectory_cell_id"])) - set(cell_ids))
    if unknown_codes:
        raise ValueError(f"trajectory map contains unknown cell codes: {unknown_codes}")

    candidates: list[Candidate] = []
    row_coverage = np.zeros(retained_rows, dtype=np.bool_)
    for binding in bindings:
        raw_shard = shards_by_index[binding.shard_index]
        if (
            _integer(raw_shard.get("batch_index"), context="shard.batch_index")
            != binding.batch_id
        ):
            raise ValueError(
                "manifest shard index is not bound to its transaction batch"
            )
        shard_path = _artifact_path(
            chunk_root, raw_shard.get("path"), context="manifest shard.path"
        )
        shard_sha256 = _verified_hash(
            shard_path, raw_shard.get("sha256"), context="dataset shard"
        )
        rows_in_shard = _integer(
            raw_shard.get("n_rows"), context="manifest shard.n_rows"
        )
        attempt_indices = accepted_indices[
            (accepted_indices >= binding.attempt_start)
            & (accepted_indices < binding.attempt_stop)
        ]
        with np.load(shard_path, allow_pickle=False) as shard:
            required = {"eta", "xi", "gxi", "depth", "case_local_index", "frame_index"}
            missing = sorted(required - set(shard.files))
            if missing:
                raise ValueError(f"dataset shard omits arrays: {missing}")
            eta = np.asarray(shard["eta"])
            xi = np.asarray(shard["xi"])
            gxi = np.asarray(shard["gxi"])
            depth = np.asarray(shard["depth"])
            case_local = np.asarray(shard["case_local_index"])
            frame_index = np.asarray(shard["frame_index"])
        if (
            eta.shape != (rows_in_shard, NX)
            or xi.shape != eta.shape
            or gxi.shape != eta.shape
        ):
            raise ValueError("dataset shard state/target shapes are incorrect")
        if eta.dtype != np.float32 or xi.dtype != np.float32 or gxi.dtype != np.float32:
            raise TypeError("dataset shard state/target fields must be float32")
        if depth.shape != (rows_in_shard,) or depth.dtype != np.float64:
            raise TypeError("dataset shard depth field must be float64")
        if case_local.dtype != np.int32 or frame_index.dtype != np.int32:
            raise TypeError("dataset shard ownership arrays must be int32")
        if not all(np.isfinite(field).all() for field in (eta, xi, gxi, depth)):
            raise ValueError("dataset shard contains a nonfinite selected field")
        mapped_rows = np.flatnonzero(arrays["shard_index"] == binding.shard_index)
        if mapped_rows.size != rows_in_shard:
            raise ValueError("trajectory map and shard disagree on row count")
        mapped_shard_rows = arrays["shard_row"][mapped_rows]
        if not np.array_equal(np.sort(mapped_shard_rows), np.arange(rows_in_shard)):
            raise ValueError(
                "trajectory map does not cover each shard row exactly once"
            )
        row_coverage[mapped_rows] = True

        for attempt_index in attempt_indices:
            global_first = int(arrays["trajectory_first_row"][attempt_index])
            global_rows = np.arange(global_first, global_first + ROWS_PER_TRAJECTORY)
            shard_rows = arrays["shard_row"][global_rows]
            if not np.array_equal(
                shard_rows,
                np.arange(shard_rows[0], shard_rows[0] + ROWS_PER_TRAJECTORY),
            ):
                raise ValueError("one trajectory is not contiguous inside its shard")
            local_index = int(local_for_attempt[attempt_index])
            if not np.all(case_local[shard_rows] == local_index):
                raise ValueError("shard local case index differs from the transaction")
            if not np.array_equal(
                frame_index[shard_rows], np.arange(ROWS_PER_TRAJECTORY)
            ):
                raise ValueError("shard frame order differs from the trajectory map")
            fractions = high_band_fraction(gxi[shard_rows])
            peak_frame = int(np.argmax(fractions))
            batch_position = int(batch_for_attempt[attempt_index])
            if bindings[batch_position] != binding:
                raise RuntimeError("internal batch binding is inconsistent")
            candidates.append(
                Candidate(
                    scan_index=scan_start + len(candidates),
                    peak_fraction=float(fractions[peak_frame]),
                    peak_frame_index=peak_frame,
                    case_id=int(arrays["trajectory_case_id"][attempt_index]),
                    chunk_label=label,
                    split=split,
                    trajectory_index=int(attempt_index),
                    cell_id=cell_ids[int(arrays["trajectory_cell_id"][attempt_index])],
                    batch=binding,
                    local_index=local_index,
                    shard_path=shard_path,
                    shard_relative_path=str(shard_path.relative_to(chunk_root)),
                    shard_sha256=shard_sha256,
                    shard_first_row=int(shard_rows[0]),
                    summary_path=summary_path,
                    summary_sha256=summary_sha256,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                    trajectory_map_path=map_path,
                    trajectory_map_sha256=map_sha256,
                )
            )
    if not np.all(row_coverage):
        raise ValueError("not every retained row was scanned")
    if len(candidates) != accepted:
        raise RuntimeError("not every accepted trajectory was ranked")
    chunk_binding = {
        "label": label,
        "split": split,
        "accepted_scanned": accepted,
        "rows_scanned": retained_rows,
        "shards_scanned": len(bindings),
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "trajectory_map_path": str(map_path),
        "trajectory_map_sha256": map_sha256,
        "generation_target_source_sha256": target_source_sha256,
    }
    return candidates, retained_rows, len(bindings), chunk_binding


def _audit_maximum(audit: JsonObject) -> tuple[float, JsonObject]:
    diagnostics = _mapping(
        audit.get("trajectory_quality_diagnostics"),
        context="audit.trajectory_quality_diagnostics",
    )
    if diagnostics.get("metric_values_affect_acceptance_or_audit_status") is not False:
        raise ValueError(
            "completion audit does not mark morphology values as non-gating"
        )
    if diagnostics.get("accepted_trajectories_checked") != audit.get("accepted"):
        raise ValueError("completion-audit morphology coverage is incomplete")
    metrics = _mapping(diagnostics.get("metrics"), context="diagnostics.metrics")
    metric = _mapping(
        metrics.get("gxi_high_band_energy_fraction_maximum"),
        context="gxi maximum high-band metric",
    )
    maximum = _mapping(metric.get("maximum"), context="gxi maximum observation")
    return (
        _finite(maximum.get("value"), context="gxi maximum value"),
        _mapping(maximum.get("identity"), context="gxi maximum identity"),
    )


def scan_final_tail(
    audit_path: Path,
    *,
    expected_accepted: int = EXPECTED_ACCEPTED,
    expected_chunks: int = EXPECTED_CHUNKS,
    top_count: int = TOP_COUNT,
) -> tuple[JsonObject, ScanResult]:
    """Authenticate a passed completion audit and independently rank its tail."""

    resolved_audit = Path(audit_path).expanduser().resolve()
    audit = _strict_json_object(resolved_audit)
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "pass":
        raise ValueError("a passed revision-4 JONSWAP completion audit is required")
    if audit.get("accepted") != expected_accepted:
        raise ValueError("completion audit does not contain the exact final case count")
    if audit.get("retained_rows") != expected_accepted * ROWS_PER_TRAJECTORY:
        raise ValueError("completion audit does not contain the exact final row count")
    dataset_root = (
        Path(_string(audit.get("dataset_root"), context="audit.dataset_root"))
        .expanduser()
        .resolve()
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"completion-audit dataset root does not exist: {dataset_root}"
        )
    raw_chunks = _list(audit.get("chunks"), context="audit.chunks")
    if len(raw_chunks) != expected_chunks:
        raise ValueError("completion audit does not contain the exact chunk plan")
    all_candidates: list[Candidate] = []
    rows_scanned = 0
    shards_scanned = 0
    chunk_bindings: list[dict[str, object]] = []
    for raw_chunk in raw_chunks:
        chunk = _mapping(raw_chunk, context="audit chunk")
        candidates, rows, shards, binding = _scan_chunk(
            dataset_root, chunk, scan_start=len(all_candidates)
        )
        all_candidates.extend(candidates)
        rows_scanned += rows
        shards_scanned += shards
        chunk_bindings.append(binding)
    if len(all_candidates) != expected_accepted:
        raise RuntimeError("chunk scan does not cover the complete accepted dataset")
    target_source_records = {
        json.dumps(
            binding["generation_target_source_sha256"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for binding in chunk_bindings
    }
    if len(target_source_records) != 1:
        raise ValueError("JONSWAP chunks name different target source hashes")
    if top_count <= 0 or top_count > len(all_candidates):
        raise ValueError("top_count is outside the accepted trajectory count")
    ranked = tuple(
        sorted(all_candidates, key=lambda item: (-item.peak_fraction, item.scan_index))[
            :top_count
        ]
    )
    audit_value, audit_identity = _audit_maximum(audit)
    # The completion audit intentionally computes this descriptive quantity
    # directly from stored float32 arrays, whereas this independent scan first
    # promotes them to float64.  The identity must match exactly; the scalar is
    # compared at a tolerance far below float32 storage precision.
    maximum_difference = abs(ranked[0].peak_fraction - audit_value)
    if maximum_difference > 1e-7:
        raise ValueError("independent maximum value differs from the completion audit")
    if ranked[0].audit_identity() != dict(audit_identity):
        raise ValueError(
            "independent maximum identity differs from the completion audit"
        )
    return audit, ScanResult(
        candidates=ranked,
        accepted_scanned=len(all_candidates),
        rows_scanned=rows_scanned,
        shards_scanned=shards_scanned,
        chunk_bindings=tuple(chunk_bindings),
        completion_audit_maximum_value=audit_value,
        independent_maximum_value=ranked[0].peak_fraction,
        completion_audit_maximum_absolute_difference=maximum_difference,
    )


def authenticate_renderer_selection(
    renderer_summary_path: Path,
    scan: ScanResult,
    *,
    expected_accepted: int = EXPECTED_ACCEPTED,
) -> dict[str, object]:
    """Cross-check the independently selected tail against the final renderer."""

    path = Path(renderer_summary_path).expanduser().resolve()
    renderer = _strict_json_object(path)
    if (
        renderer.get("schema") != RENDERER_SCHEMA
        or renderer.get("status") != "complete"
    ):
        raise ValueError("a completed final diagnostic-renderer summary is required")
    parameters = _mapping(renderer.get("parameters"), context="renderer.parameters")
    if parameters.get("final_paper_dataset_contract_required") is not True:
        raise ValueError("renderer did not enforce the final paper-dataset contract")
    source_binding = _mapping(
        renderer.get("source_binding"), context="renderer.source_binding"
    )
    if source_binding.get("mode") != "combined_summary":
        raise ValueError("renderer summary is not bound to a combined dataset summary")
    combined_path = (
        Path(
            _string(
                source_binding.get("combined_summary_path"),
                context="renderer combined_summary_path",
            )
        )
        .expanduser()
        .resolve()
    )
    if not combined_path.is_file():
        raise FileNotFoundError(
            f"renderer combined summary does not exist: {combined_path}"
        )
    combined_sha256 = _verified_hash(
        combined_path,
        source_binding.get("combined_summary_sha256"),
        context="renderer combined summary",
    )
    combined = _strict_json_object(combined_path)
    if (
        combined.get("schema") != COMBINED_SUMMARY_SCHEMA
        or combined.get("status") != "complete"
    ):
        raise ValueError("renderer names a non-complete combined dataset summary")
    if source_binding.get("expected_sources") != 26:
        raise ValueError("renderer is not bound to the exact 26 final source chunks")
    if source_binding.get("expected_accepted_cases") != 73_728:
        raise ValueError("renderer is not bound to the exact final case population")
    if source_binding.get("expected_retained_rows") != 7_686_144:
        raise ValueError("renderer is not bound to the exact final row population")
    population = _mapping(renderer.get("population"), context="renderer.population")
    if (
        population.get("sources"),
        population.get("accepted_cases"),
        population.get("retained_rows"),
    ) != (26, 73_728, 7_686_144):
        raise ValueError("renderer did not scan the exact final paper population")
    families = _mapping(renderer.get("families"), context="renderer.families")
    jonswap = _mapping(families.get("jonswap_tma"), context="renderer JONSWAP family")
    if jonswap.get("accepted_cases") != expected_accepted:
        raise ValueError(
            "renderer JONSWAP case count differs from its completion audit"
        )
    if jonswap.get("retained_rows") != expected_accepted * ROWS_PER_TRAJECTORY:
        raise ValueError("renderer JONSWAP row count differs from its completion audit")
    rankings = _mapping(jonswap.get("rankings"), context="renderer JONSWAP rankings")
    selected = _list(
        rankings.get("gxi_high_band"), context="renderer JONSWAP gxi_high_band"
    )
    if len(selected) < len(scan.candidates):
        raise ValueError("renderer summary omits independently selected JONSWAP cases")
    for rank, (raw_case, candidate) in enumerate(
        zip(selected, scan.candidates), start=1
    ):
        case = _mapping(raw_case, context=f"renderer JONSWAP rank {rank}")
        expected = {
            "family": "jonswap_tma",
            "split": candidate.split,
            "case_id": candidate.case_id,
            "category": candidate.cell_id,
            "trajectory_index": candidate.trajectory_index,
            "shard_index": candidate.batch.shard_index,
            "first_shard_row": candidate.shard_first_row,
            "row_count": ROWS_PER_TRAJECTORY,
            "maximum_gxi_high_band_fraction_frame": candidate.peak_frame_index,
            "source_root": str(candidate.summary_path.parent),
        }
        if any(case.get(name) != value for name, value in expected.items()):
            raise ValueError(f"renderer JONSWAP rank {rank} identity differs")
        observed_fraction = _finite(
            case.get("maximum_gxi_high_band_fraction"),
            context=f"renderer JONSWAP rank {rank} fraction",
        )
        if not math.isclose(
            observed_fraction,
            candidate.peak_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"renderer JONSWAP rank {rank} metric differs")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": RENDERER_SCHEMA,
        "status": "complete",
        "combined_summary_path": str(combined_path),
        "combined_summary_sha256": combined_sha256,
        "exact_final_population_verified": True,
        "independent_top_selection_reproduced": True,
    }


def evaluate_orders(
    eta: np.ndarray,
    xi: np.ndarray,
    depth: float,
) -> Mapping[int, OrderFields]:
    """Evaluate cumulative Craig--Sulem sums M=4,...,8 on CPU in float64."""

    if not jax.config.x64_enabled:
        raise RuntimeError("order-convergence diagnostic requires JAX float64")
    platforms = {device.platform for device in jax.devices()}
    if platforms != {"cpu"}:
        raise RuntimeError(
            f"order-convergence diagnostic requires CPU only, got {platforms}"
        )
    eta64 = jnp.asarray(eta, dtype=jnp.float64)
    xi64 = jnp.asarray(xi, dtype=jnp.float64)
    if eta64.shape != (ROWS_PER_TRAJECTORY, NX) or xi64.shape != eta64.shape:
        raise ValueError("selected state must contain 16 frames on the 1024-point grid")
    _, modes = build_grid(NX, LENGTH)
    modes = jnp.asarray(modes, dtype=jnp.float64)
    eta_projected = project_fixed_band(eta64, modes, maximum_wavenumber=P_MAX)
    xi_projected = project_fixed_band(xi64, modes, maximum_wavenumber=P_MAX)
    fields: dict[int, OrderFields] = {}
    for order in ORDERS:
        raw = dno_series_eval(
            eta_projected,
            xi_projected,
            modes,
            jnp.asarray(depth, dtype=jnp.float64),
            order,
            pad_factor=PAD_FACTOR,
        )
        projected = project_fixed_band(
            raw,
            modes,
            maximum_wavenumber=P_MAX,
            remove_mean=True,
        )
        raw.block_until_ready()
        projected.block_until_ready()
        fields[order] = OrderFields(
            raw=np.asarray(raw, dtype=np.float64),
            projected=np.asarray(projected, dtype=np.float64),
        )
    return fields


def _l2_rows(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)), axis=1))


def _relative_rows(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator_norm = _l2_rows(denominator)
    if np.any(denominator_norm <= np.finfo(np.float64).tiny):
        raise ValueError("relative order diagnostic has a zero denominator")
    return _l2_rows(numerator) / denominator_norm


def _finite_list(values: np.ndarray) -> list[float]:
    result = [float(value) for value in np.asarray(values).ravel()]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("order diagnostic produced a nonfinite value")
    return result


def _validate_selected_sources(
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    _verified_hash(
        candidate.shard_path, candidate.shard_sha256, context="selected shard"
    )
    proposal_sha256 = _verified_hash(
        candidate.batch.proposal_path,
        candidate.batch.proposal_sha256,
        context="selected proposal",
    )
    result_sha256 = _verified_hash(
        candidate.batch.result_path,
        candidate.batch.result_sha256,
        context="selected result",
    )
    with np.load(candidate.batch.proposal_path, allow_pickle=False) as proposal:
        if int(proposal["batch_id"]) != candidate.batch.batch_id:
            raise ValueError("selected proposal has the wrong batch id")
        if int(proposal["case_id"][candidate.local_index]) != candidate.case_id:
            raise ValueError("selected proposal has the wrong case id")
        specification = json.loads(
            str(proposal["case_spec_json"][candidate.local_index])
        )
        if not isinstance(specification, dict):
            raise TypeError("selected proposal case specification must be an object")
        if specification.get("cell_id") != candidate.cell_id:
            raise ValueError("selected proposal has the wrong cell id")
    result = _strict_json_object(candidate.batch.result_path)
    cases = _list(result.get("cases"), context="selected result.cases")
    case = _mapping(cases[candidate.local_index], context="selected result case")
    expected_case = {
        "accepted": True,
        "case_id": candidate.case_id,
        "first_row": candidate.shard_first_row,
        "row_count": ROWS_PER_TRAJECTORY,
    }
    if any(case.get(name) != value for name, value in expected_case.items()):
        raise ValueError("selected result case differs from map/shard ownership")
    if result.get("proposal_sha256") != proposal_sha256:
        raise ValueError("selected result does not bind the proposal hash")
    if result.get("shard_sha256") != candidate.shard_sha256:
        raise ValueError("selected result does not bind the shard hash")
    rows = slice(
        candidate.shard_first_row,
        candidate.shard_first_row + ROWS_PER_TRAJECTORY,
    )
    with np.load(candidate.shard_path, allow_pickle=False) as shard:
        eta = np.asarray(shard["eta"][rows], dtype=np.float64)
        xi = np.asarray(shard["xi"][rows], dtype=np.float64)
        stored = np.asarray(shard["gxi"][rows], dtype=np.float64)
        depths = np.asarray(shard["depth"][rows], dtype=np.float64)
    if not np.all(depths == depths[0]):
        raise ValueError("selected trajectory depth varies between frames")
    if not math.isclose(
        float(depths[0]), float(specification["depth"]), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("selected proposal and shard depths differ")
    stored_fractions = high_band_fraction(stored)
    stored_peak_frame = int(np.argmax(stored_fractions))
    if stored_peak_frame != candidate.peak_frame_index or not math.isclose(
        float(stored_fractions[stored_peak_frame]),
        candidate.peak_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("selected shard no longer reproduces the ranking metric")
    source_record = {
        "summary_path": str(candidate.summary_path),
        "summary_sha256": candidate.summary_sha256,
        "manifest_path": str(candidate.manifest_path),
        "manifest_sha256": candidate.manifest_sha256,
        "trajectory_map_path": str(candidate.trajectory_map_path),
        "trajectory_map_sha256": candidate.trajectory_map_sha256,
        "proposal_path": str(candidate.batch.proposal_path),
        "proposal_sha256": proposal_sha256,
        "result_path": str(candidate.batch.result_path),
        "result_sha256": result_sha256,
        "shard_path": str(candidate.shard_path),
        "shard_sha256": candidate.shard_sha256,
    }
    return eta, xi, stored, float(depths[0]), source_record


def case_order_record(
    candidate: Candidate,
    *,
    evaluator: OrderEvaluator = evaluate_orders,
) -> dict[str, object]:
    """Recompute and summarize all declared orders for one selected case."""

    eta, xi, stored, depth, sources = _validate_selected_sources(candidate)
    fields = dict(evaluator(eta, xi, depth))
    if set(fields) != set(ORDERS):
        raise ValueError("order evaluator did not return exactly orders 4 through 8")
    per_order: dict[str, object] = {}
    for order in ORDERS:
        projected = np.asarray(fields[order].projected, dtype=np.float64)
        raw = np.asarray(fields[order].raw, dtype=np.float64)
        if projected.shape != eta.shape or raw.shape != eta.shape:
            raise ValueError("order evaluator returned the wrong field shape")
        fractions = high_band_fraction(projected)
        norms = _l2_rows(projected)
        per_order[str(order)] = {
            "projected_l2_norm_by_frame": _finite_list(norms),
            "projected_high_band_energy_fraction_by_frame": _finite_list(fractions),
            "maximum_projected_high_band_energy_fraction": float(np.max(fractions)),
            "maximum_projected_l2_norm": float(np.max(norms)),
        }
    transitions: list[dict[str, object]] = []
    for previous, current in zip(ORDERS, ORDERS[1:]):
        projected_relative = _relative_rows(
            fields[current].projected - fields[previous].projected,
            fields[current].projected,
        )
        raw_relative = _relative_rows(
            fields[current].raw - fields[previous].raw,
            fields[current].raw,
        )
        fraction_change = np.abs(
            high_band_fraction(fields[current].projected)
            - high_band_fraction(fields[previous].projected)
        )
        transitions.append(
            {
                "from_order": previous,
                "to_order": current,
                "projected_relative_l2_difference_by_frame": _finite_list(
                    projected_relative
                ),
                "maximum_projected_relative_l2_difference": float(
                    np.max(projected_relative)
                ),
                "raw_preprojection_relative_l2_difference_by_frame": _finite_list(
                    raw_relative
                ),
                "maximum_raw_preprojection_relative_l2_difference": float(
                    np.max(raw_relative)
                ),
                "absolute_high_band_fraction_change_by_frame": _finite_list(
                    fraction_change
                ),
                "maximum_absolute_high_band_fraction_change": float(
                    np.max(fraction_change)
                ),
            }
        )
    stored_relative = _relative_rows(fields[6].projected - stored, stored)
    six_to_eight = _relative_rows(
        fields[8].projected - fields[6].projected, fields[8].projected
    )
    return {
        "rank_metric": candidate.peak_fraction,
        "identity": candidate.audit_identity(),
        "trajectory_index_within_chunk": candidate.trajectory_index,
        "depth": depth,
        "sources": sources,
        "orders": per_order,
        "successive_order_changes": transitions,
        "stored_float32_vs_recomputed_order6": {
            "relative_l2_difference_by_frame": _finite_list(stored_relative),
            "maximum_relative_l2_difference": float(np.max(stored_relative)),
        },
        "order6_to_order8": {
            "projected_relative_l2_difference_by_frame": _finite_list(six_to_eight),
            "maximum_projected_relative_l2_difference": float(np.max(six_to_eight)),
        },
    }


def run_diagnostic(
    audit_path: Path,
    *,
    renderer_summary_path: Path | None = None,
    evaluator: OrderEvaluator = evaluate_orders,
    expected_accepted: int = EXPECTED_ACCEPTED,
    expected_chunks: int = EXPECTED_CHUNKS,
    top_count: int = TOP_COUNT,
) -> dict[str, object]:
    """Return the complete non-gating order-convergence record."""

    implementation = diagnostic_implementation_record()
    started_at = datetime.now().astimezone()
    started = perf_counter()
    resolved_audit = Path(audit_path).expanduser().resolve()
    audit, scan = scan_final_tail(
        resolved_audit,
        expected_accepted=expected_accepted,
        expected_chunks=expected_chunks,
        top_count=top_count,
    )
    renderer_binding = (
        authenticate_renderer_selection(
            renderer_summary_path,
            scan,
            expected_accepted=expected_accepted,
        )
        if renderer_summary_path is not None
        else None
    )
    records = [case_order_record(case, evaluator=evaluator) for case in scan.candidates]
    identity = _mapping(audit.get("identity"), context="audit.identity")
    generation_source_fingerprint = _string(
        identity.get("source_sha256_fingerprint"),
        context="audit identity source fingerprint",
    )
    generation_dependency_fingerprint = _string(
        identity.get("dependency_environment_fingerprint"),
        context="audit identity dependency fingerprint",
    )
    generation_execution_fingerprint = _string(
        identity.get("execution_record_fingerprint"),
        context="audit identity execution fingerprint",
    )
    generation_support_sources = dict(
        _mapping(
            identity.get("current_support_source_sha256"),
            context="audit identity support source hashes",
        )
    )
    _require_diagnostic_implementation_current(implementation)
    diagnostic_source_sha256 = {
        str(item["path"]): str(item["sha256"]) for item in implementation["files"]
    }
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "pass",
        "diagnostic_only": True,
        "affects_dataset_acceptance": False,
        "affects_dataset_release": False,
        "completion_audit": {
            "path": str(resolved_audit),
            "sha256": file_sha256(resolved_audit),
            "schema": AUDIT_SCHEMA,
            "accepted": expected_accepted,
            "retained_rows": expected_accepted * ROWS_PER_TRAJECTORY,
        },
        "selection": {
            "definition": SELECTION_DEFINITION,
            "independently_rescanned": True,
            "completion_audit_global_maximum_reproduced": True,
            "completion_audit_global_maximum_value": (
                scan.completion_audit_maximum_value
            ),
            "independent_global_maximum_value": scan.independent_maximum_value,
            "maximum_value_absolute_difference": (
                scan.completion_audit_maximum_absolute_difference
            ),
            "maximum_value_comparison_tolerance": 1e-7,
            "accepted_trajectories_scanned": scan.accepted_scanned,
            "stored_rows_scanned": scan.rows_scanned,
            "shards_scanned": scan.shards_scanned,
            "selected_count": len(records),
            "final_renderer_cross_check": renderer_binding,
        },
        "numerical_definition": {
            "execution_platform": "cpu",
            "dtype": "float64",
            "nx": NX,
            "length": LENGTH,
            "input_and_output_projection": "sharp |k| <= 128",
            "zero_output_mean": True,
            "pad_factor": PAD_FACTOR,
            "cumulative_orders": list(ORDERS),
            "relative_l2_definition": RELATIVE_L2_DEFINITION,
        },
        "identity": {
            "generation_source_sha256_fingerprint": generation_source_fingerprint,
            "generation_dependency_environment_fingerprint": generation_dependency_fingerprint,
            "generation_execution_record_fingerprint": generation_execution_fingerprint,
            "generation_support_source_sha256": generation_support_sources,
            "diagnostic_source_sha256": diagnostic_source_sha256,
        },
        "diagnostic_implementation": implementation,
        "chunk_bindings": list(scan.chunk_bindings),
        "cases": records,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "timing_seconds": perf_counter() - started,
        "invocation_started_at": started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "jonswap_tma_completion_audit.json",
        help="Passed final revision-4 JONSWAP completion-audit JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON; defaults beside the audit as "
            "jonswap_tma_order_convergence_diagnostic.json."
        ),
    )
    parser.add_argument(
        "--renderer-summary",
        type=Path,
        help=(
            "Optional completed all-family renderer summary. When supplied, "
            "authenticate its combined-view hash and require its JONSWAP top "
            "three to equal the independent full-dataset rescan."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit_path = args.audit.expanduser().resolve()
    output = validated_output_path(
        args.output
        if args.output is not None
        else audit_path.parent / "jonswap_tma_order_convergence_diagnostic.json"
    )
    record = run_diagnostic(
        audit_path,
        renderer_summary_path=args.renderer_summary,
    )
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "diagnostic_only": record["diagnostic_only"],
                "selected_count": len(record["cases"]),
                "output": str(output),
                "sha256": file_sha256(output),
                "timing_seconds": record["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
