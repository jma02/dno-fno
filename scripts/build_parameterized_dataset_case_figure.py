"""Build deterministic family illustrations from the completed paper dataset.

The four profiles are accepted validation cases, not manufactured states and
not empirical medoids.  For each family, the script selects the lower median
case ID within one fixed central parameter category and plots that case's first
owned stored row.  The completed combined-view summary is the sole dataset
input and binds every source summary, trajectory map, and selected shard.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Final

import numpy as np

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / f"dno-fno-matplotlib-{os.getuid()}"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_paper_dataset_worst_cases import (  # noqa: E402
    CombinedSummaryBinding,
    DatasetSource,
    TrajectoryIndex,
    load_combined_summary_binding,
    load_source_summary,
    read_json,
    sha256,
    validate_bound_sources,
    validate_final_paper_dataset,
    validate_scanned_population,
)
from scripts.build_paper_dataset_view import (  # noqa: E402
    PAPER_GENERATION_COMPATIBILITY_POLICY,
    TRAJECTORY_MAP_DTYPES,
    CombinedDatasetPlan,
    CompletedChunk,
    _validate_view,
    load_completed_chunk,
    validate_combined_plan,
)
from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_SAMPLE_CELL_IDS,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)
from solver.gen_data.pipeline.manifest import (  # noqa: E402
    TRAJECTORY_MAP_SCHEMA_VERSION,
    DatasetViewPaths,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PhysicalFamilyId,
    SplitId,
    balanced_valid_case_targets,
    split_code,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    STOKES_SAMPLE_CELL_IDS,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_SAMPLE_CELL_IDS,
)


SIDECAR_SCHEMA = "paper_dataset_deterministic_family_illustrations_v1"
IMPLEMENTATION_SCHEMA = "paper_dataset_family_figure_implementation_v1"
IMPLEMENTATION_FILES: Final = (
    (
        "scripts/build_parameterized_dataset_case_figure.py",
        "family_figure_selection_rendering_and_sidecar_producer",
    ),
    (
        "scripts/render_paper_dataset_worst_cases.py",
        "source_summary_and_combined_binding_helpers",
    ),
    (
        "scripts/build_paper_dataset_view.py",
        "canonical_combined_view_validator",
    ),
    (
        "solver/gen_data/stokes_sampling.py",
        "stokes_sample_cell_definition",
    ),
    (
        "solver/gen_data/tanaka_sampling.py",
        "tanaka_sample_cell_definition",
    ),
    (
        "solver/gen_data/benjamin_feir_sampling.py",
        "benjamin_feir_sample_cell_definition",
    ),
    (
        "solver/gen_data/jonswap_tma_sampling.py",
        "jonswap_tma_sample_cell_definition",
    ),
    (
        "solver/gen_data/pipeline/manifest.py",
        "trajectory_map_schema_and_dataset_view_contract",
    ),
    (
        "solver/gen_data/pipeline/production.py",
        "family_split_and_balanced_quota_contract",
    ),
)
IMPLEMENTATION_RELATIONSHIP: Final = {
    "scope": "direct_repository_implementation_bytes",
    "external_runtime_is_not_transitively_authenticated": True,
}
SIDECAR_DESCRIPTION = (
    "Deterministic accepted validation illustrations; these are lower-median "
    "case IDs in fixed central cells, not medoids."
)
SELECTION_RULE = "sort accepted validation case IDs; choose explicit lower median"
DIMENSIONLESS_VARIABLES: Final = {
    "horizontal": "x/L",
    "surface_elevation": "eta/h",
    "surface_potential": "xi/(h*sqrt(g*h))",
}
PUBLICATION_RULE = "PDF and PNG first; complete JSON sidecar last"
FAMILY_ORDER: Final = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
FAMILY_LABELS: Final = {
    "stokes": "Finite-depth Stokes",
    "tanaka": "Tanaka",
    "benjamin_feir": "Benjamin--Feir",
    "jonswap_tma": "JONSWAP/TMA",
}
FAMILY_IDS: Final = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
EXPECTED_REVISIONS: Final = {
    "stokes": 2,
    "tanaka": 3,
    "benjamin_feir": 4,
    "jonswap_tma": 4,
}
CENTRAL_VALIDATION_CATEGORIES: Final = {
    "stokes": "finite_moderate",
    "tanaka": "main_m2_q1",
    "benjamin_feir": "n_c_09__delta_n_02",
    "jonswap_tma": "finite__gamma_3p3__right_0p5",
}
EXPECTED_CELL_IDS: Final = {
    "stokes": STOKES_SAMPLE_CELL_IDS,
    "tanaka": TANAKA_SAMPLE_CELL_IDS,
    "benjamin_feir": BENJAMIN_FEIR_SAMPLE_CELL_IDS,
    "jonswap_tma": JONSWAP_TMA_SAMPLE_CELL_IDS,
}
EXPECTED_SOURCE_COUNT_BY_FAMILY: Final = {
    "stokes": 6,
    "tanaka": 6,
    "benjamin_feir": 6,
    "jonswap_tma": 8,
}
STANDARD_TRAIN_INTERVALS: Final = (
    (0, 2_048),
    (2_048, 4_096),
    (4_096, 8_192),
    (8_192, 16_384),
)
JONSWAP_TRAIN_INTERVALS: Final = (
    (0, 2_048),
    (2_048, 4_096),
    (4_096, 8_192),
    (8_192, 12_288),
    (12_288, 14_336),
    (14_336, 16_384),
)
EXPECTED_ACCEPTED_CASES_BY_SPLIT: Final = {
    "train": 16_384,
    "validation": 1_024,
    "test": 1_024,
}


@dataclass(frozen=True)
class ArtifactIdentity:
    """Authenticated identity of one immutable file."""

    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ExpectedChunkLayout:
    """One source interval in the frozen final combined-view order."""

    family: str
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_after: int


@dataclass(frozen=True)
class CombinedViewIdentity:
    """Authenticated manifest and trajectory map of the combined view."""

    manifest: ArtifactIdentity
    trajectory_map: ArtifactIdentity


@dataclass(frozen=True)
class SourceIdentity:
    """One authenticated source plus the revision and numerical scales used."""

    source: DatasetSource
    revision_id: int
    summary: ArtifactIdentity
    manifest: ArtifactIdentity
    trajectory_map: ArtifactIdentity
    shard_sha256: Mapping[int, str]
    length: float
    gravity: float
    stored_nx: int


@dataclass(frozen=True)
class SelectedTrajectory:
    """Deterministically selected accepted validation trajectory."""

    identity: SourceIdentity
    trajectory: TrajectoryIndex
    candidate_count: int
    lower_median_index: int


@dataclass(frozen=True)
class DimensionlessProfile:
    """Dimensionless variables shown in one figure row."""

    x_over_length: np.ndarray
    eta_over_depth: np.ndarray
    xi_over_depth_speed: np.ndarray


@dataclass(frozen=True)
class Illustration:
    """One selected first-row state and its dimensionless representation."""

    selected: SelectedTrajectory
    eta: np.ndarray
    xi: np.ndarray
    depth: float
    time: float
    frame_index: int
    profile: DimensionlessProfile


def _canonical_sha256(value: object) -> str:
    """Return the canonical JSON fingerprint of one proof payload."""

    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def figure_implementation_record(
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Bind the exact direct repository implementation used by the figure."""

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
        digest = sha256(path)
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


def _require_figure_implementation_current(
    expected: Mapping[str, object],
) -> None:
    if figure_implementation_record() != dict(expected):
        raise RuntimeError("figure implementation changed during execution")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-summary",
        type=Path,
        required=True,
        help=(
            "Completed c16384 all-split combined-view summary for the exact "
            "final paper dataset."
        ),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "notes/figures/parameterized_dataset_case_examples",
        help=(
            "Output basename; files are staged, then the complete JSON commit "
            "marker is published last."
        ),
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _positive_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be finite and positive")
    return result


def _artifact_identity(
    path: Path,
    *,
    expected_bytes: object,
    expected_sha256: object,
    context: str,
) -> ArtifactIdentity:
    """Authenticate one file against an explicit byte count and digest."""

    resolved = path.expanduser().resolve(strict=True)
    byte_count = _nonnegative_integer(expected_bytes, context=f"{context} bytes")
    if resolved.stat().st_size != byte_count:
        raise ValueError(f"{context} byte count differs")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"{context} has an invalid SHA-256 digest")
    observed_sha256 = sha256(resolved)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{context} SHA-256 differs")
    return ArtifactIdentity(
        path=resolved,
        bytes=byte_count,
        sha256=observed_sha256,
    )


def _bound_combined_artifact(
    combined_root: Path,
    record: Mapping[str, Any],
    *,
    context: str,
) -> ArtifactIdentity:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context} path must be a nonempty string")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = combined_root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(combined_root):
        raise ValueError(f"{context} path escapes the combined-view root")
    return _artifact_identity(
        resolved,
        expected_bytes=record.get("bytes"),
        expected_sha256=record.get("sha256"),
        context=context,
    )


def load_combined_view_identity(
    binding: CombinedSummaryBinding,
) -> CombinedViewIdentity:
    """Validate the saved view named by the completed summary."""

    summary = read_json(binding.path)
    view = _mapping(summary.get("dataset_view"), context="combined dataset_view")
    manifest = _bound_combined_artifact(
        binding.path.parent,
        _mapping(view.get("manifest"), context="combined manifest record"),
        context="combined manifest",
    )
    trajectory_map = _bound_combined_artifact(
        binding.path.parent,
        _mapping(
            view.get("trajectory_map"),
            context="combined trajectory-map record",
        ),
        context="combined trajectory map",
    )
    manifest_record = read_json(manifest.path)
    if manifest_record.get("trajectory_map_sha256") != trajectory_map.sha256:
        raise ValueError("combined manifest trajectory-map SHA-256 differs")
    raw_map_path = manifest_record.get("trajectory_map_npz")
    if not isinstance(raw_map_path, str) or not raw_map_path:
        raise ValueError("combined manifest has no trajectory-map path")
    manifest_map_path = (manifest.path.parent / raw_map_path).resolve(strict=True)
    if manifest_map_path != trajectory_map.path:
        raise ValueError("combined manifest names a different trajectory map")
    accepted_cases = _nonnegative_integer(
        view.get("n_accepted_trajectories"),
        context="combined accepted trajectories",
    )
    retained_rows = _nonnegative_integer(
        view.get("n_accepted_rows"),
        context="combined accepted rows",
    )
    if accepted_cases != binding.expected_accepted_cases:
        raise ValueError("combined-view accepted trajectories differ from preflight")
    if retained_rows != binding.expected_retained_rows:
        raise ValueError("combined-view accepted rows differ from preflight")
    return CombinedViewIdentity(manifest=manifest, trajectory_map=trajectory_map)


def _expected_chunk_layout() -> tuple[ExpectedChunkLayout, ...]:
    """Return the exact 26-source order used by the final c16384 view."""

    train = tuple(
        ExpectedChunkLayout(
            family=family,
            split=SplitId.TRAIN,
            stream_id=stream_id,
            accepted_before=before,
            accepted_after=after,
        )
        for family in FAMILY_ORDER
        for stream_id, (before, after) in enumerate(
            JONSWAP_TRAIN_INTERVALS
            if family == "jonswap_tma"
            else STANDARD_TRAIN_INTERVALS
        )
    )
    held_out = tuple(
        ExpectedChunkLayout(
            family=family,
            split=split,
            stream_id=100 if split is SplitId.VALIDATION else 200,
            accepted_before=0,
            accepted_after=1_024,
        )
        for split in (SplitId.VALIDATION, SplitId.TEST)
        for family in FAMILY_ORDER
    )
    return train + held_out


def _expected_incremental_valid_case_targets(
    family: str,
    *,
    accepted_before: int,
    accepted_after: int,
) -> tuple[dict[str, object], ...]:
    """Return exact balanced-cell increments for one cumulative interval."""

    cells = EXPECTED_CELL_IDS[family]
    before = balanced_valid_case_targets(
        cells,
        case_count=accepted_before,
    )
    after = balanced_valid_case_targets(
        cells,
        case_count=accepted_after,
    )
    return tuple(
        {
            "cell_id": after_target.cell_id,
            "target_accepted": (after_target.case_count - before_target.case_count),
        }
        for before_target, after_target in zip(before, after)
    )


def _validate_chunk_taxonomy(chunk: CompletedChunk) -> None:
    """Require the current cell map and exact balanced valid-case targets."""

    summary = read_json(chunk.summary_path)
    run_spec = _mapping(summary.get("run_spec"), context="source run_spec")
    configuration = _mapping(
        run_spec.get("configuration"),
        context="source configuration",
    )
    expected_cells = EXPECTED_CELL_IDS[chunk.family]
    raw_cells = configuration.get("ordered_cell_ids")
    if not isinstance(raw_cells, list) or tuple(raw_cells) != expected_cells:
        raise ValueError(f"{chunk.family} source has the wrong ordered cell taxonomy")
    expected_codes = {
        cell_id: cell_code for cell_code, cell_id in enumerate(expected_cells)
    }
    if run_spec.get("cell_codes") != expected_codes:
        raise ValueError(f"{chunk.family} source has the wrong cell-code map")
    expected_targets = _expected_incremental_valid_case_targets(
        chunk.family,
        accepted_before=chunk.accepted_before,
        accepted_after=chunk.accepted_after,
    )
    raw_targets = run_spec.get("quotas")
    if not isinstance(raw_targets, list) or tuple(raw_targets) != expected_targets:
        raise ValueError(
            f"{chunk.family} source has the wrong balanced valid-case targets"
        )
    interval = {
        "accepted_case_count": chunk.accepted_count,
        "accepted_cases_before": chunk.accepted_before,
        "accepted_cases_after": chunk.accepted_after,
    }
    for name, expected in interval.items():
        if configuration.get(name) != expected:
            raise ValueError(f"{chunk.family} source {name} differs from its interval")


def _validate_preflight_against_plan(
    binding: CombinedSummaryBinding,
    plan: CombinedDatasetPlan,
) -> None:
    """Bind every recorded preflight field to the reloaded source plan."""

    summary = read_json(binding.path)
    preflight = _mapping(summary.get("preflight"), context="combined preflight")
    if preflight.get("splits") != [split.value for split in SplitId]:
        raise ValueError("combined preflight does not contain all splits in order")
    expected_top_level: dict[str, object] = {
        "accepted_cases_per_family_by_split": EXPECTED_ACCEPTED_CASES_BY_SPLIT,
        "accepted_cases_total": 73_728,
        "attempted_cases_total": plan.attempted_cases,
        "attempted_cases_by_split": dict(plan.attempted_cases_by_split),
        "expected_rows": 7_686_144,
        "expected_rows_by_split_and_family": {
            split.value: {
                "stokes": EXPECTED_ACCEPTED_CASES_BY_SPLIT[split.value],
                "tanaka": 200 * EXPECTED_ACCEPTED_CASES_BY_SPLIT[split.value],
                "benjamin_feir": (200 * EXPECTED_ACCEPTED_CASES_BY_SPLIT[split.value]),
                "jonswap_tma": (16 * EXPECTED_ACCEPTED_CASES_BY_SPLIT[split.value]),
            }
            for split in SplitId
        },
        "configuration_fingerprints": list(plan.fingerprints),
    }
    for name, expected in expected_top_level.items():
        if preflight.get(name) != expected:
            raise ValueError(f"combined preflight {name} differs from the final plan")

    raw_chunks = preflight.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(plan.chunks):
        raise ValueError("combined preflight has the wrong source count")
    for index, (raw_chunk, chunk) in enumerate(zip(raw_chunks, plan.chunks)):
        record = _mapping(raw_chunk, context=f"combined preflight chunk {index}")
        expected_record = {
            "family": chunk.family,
            "revision_id": chunk.revision_id,
            "split": chunk.split.value,
            "stream_id": chunk.stream_id,
            "accepted_before": chunk.accepted_before,
            "accepted_count": chunk.accepted_count,
            "accepted_after": chunk.accepted_after,
            "attempted_count": chunk.attempted_count,
            "configuration_fingerprint": chunk.fingerprint,
            "execution_fingerprint": chunk.execution_fingerprint,
            "generation_compatibility_id": chunk.generation_compatibility_id,
            "source_fingerprint": chunk.source_fingerprint,
            "execution_platform": chunk.execution_platform,
            "summary_path": str(chunk.summary_path),
            "summary_sha256": chunk.summary_sha256,
            "committed_batches": len(chunk.batches),
        }
        for name, expected in expected_record.items():
            if record.get(name) != expected:
                raise ValueError(
                    f"combined preflight chunk {index} {name} differs from its source"
                )

    if tuple(chunk.summary_path for chunk in plan.chunks) != (
        binding.source_summary_paths
    ):
        raise ValueError("combined preflight source order differs from the final plan")
    if tuple(chunk.summary_sha256 for chunk in plan.chunks) != (
        binding.source_summary_sha256
    ):
        raise ValueError("combined preflight source hashes differ from the final plan")


def _validate_final_plan_layout(plan: CombinedDatasetPlan) -> None:
    """Require the frozen per-family source counts and cumulative intervals."""

    expected_layout = _expected_chunk_layout()
    observed_layout = tuple(
        ExpectedChunkLayout(
            family=chunk.family,
            split=chunk.split,
            stream_id=chunk.stream_id,
            accepted_before=chunk.accepted_before,
            accepted_after=chunk.accepted_after,
        )
        for chunk in plan.chunks
    )
    if observed_layout != expected_layout:
        raise ValueError("combined view has the wrong 26-source interval layout")
    source_counts = {
        family: sum(chunk.family == family for chunk in plan.chunks)
        for family in FAMILY_ORDER
    }
    if source_counts != EXPECTED_SOURCE_COUNT_BY_FAMILY:
        raise ValueError("combined view has the wrong per-family source counts")
    if (
        plan.splits != tuple(SplitId)
        or dict(plan.accepted_cases_per_family_by_split)
        != EXPECTED_ACCEPTED_CASES_BY_SPLIT
        or plan.accepted_cases != 73_728
        or plan.expected_rows != 7_686_144
    ):
        raise ValueError("combined source plan differs from the final population")


def validate_final_combined_view(
    binding: CombinedSummaryBinding,
    combined_view: CombinedViewIdentity,
) -> CombinedDatasetPlan:
    """Rebuild and authenticate the exact final source/view relationship."""

    if sha256(binding.path) != binding.sha256:
        raise ValueError("combined summary changed after it was loaded")
    chunks = tuple(
        load_completed_chunk(
            path,
            generation_compatibility_policy=PAPER_GENERATION_COMPATIBILITY_POLICY,
        )
        for path in binding.source_summary_paths
    )
    plan = validate_combined_plan(chunks)
    _validate_final_plan_layout(plan)
    for chunk in plan.chunks:
        _validate_chunk_taxonomy(chunk)
    _validate_preflight_against_plan(binding, plan)

    observed_view = _validate_view(
        DatasetViewPaths(
            manifest=combined_view.manifest.path,
            trajectory_map=combined_view.trajectory_map.path,
        ),
        plan=plan,
    )
    recorded_view = _mapping(
        read_json(binding.path).get("dataset_view"),
        context="combined dataset_view",
    )
    if observed_view != recorded_view:
        raise ValueError(
            "combined dataset view differs from its exact source-bound audit"
        )
    return plan


def _load_exact_trajectory_map(path: Path, *, context: str) -> dict[str, np.ndarray]:
    """Load one exact schema-v2 trajectory map with dtype checks."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(TRAJECTORY_MAP_DTYPES):
            raise ValueError(f"{context} has the wrong trajectory-map fields")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    schema = arrays.pop("schema_version")
    if (
        schema.ndim != 0
        or schema.dtype != TRAJECTORY_MAP_DTYPES["schema_version"]
        or int(schema) != TRAJECTORY_MAP_SCHEMA_VERSION
    ):
        raise ValueError(f"{context} has the wrong trajectory-map schema")
    for name, values in arrays.items():
        if values.ndim != 1 or values.dtype != TRAJECTORY_MAP_DTYPES[name]:
            raise ValueError(f"{context} field {name} has the wrong shape or dtype")
    return arrays


def validate_source_map_binding(
    sources: Sequence[DatasetSource],
    combined_view: CombinedViewIdentity,
) -> None:
    """Require the combined map to be the exact ordered union of source maps."""

    if sha256(combined_view.trajectory_map.path) != combined_view.trajectory_map.sha256:
        raise ValueError("combined trajectory map changed after authentication")
    combined = _load_exact_trajectory_map(
        combined_view.trajectory_map.path,
        context="combined trajectory map",
    )
    trajectory_fields = tuple(
        name
        for name in TRAJECTORY_MAP_DTYPES
        if name.startswith("trajectory_") and name != "trajectory_index"
    )
    row_fields = ("trajectory_index", "frame_index", "shard_index", "shard_row")
    trajectory_offset = 0
    row_offset = 0
    shard_offset = 0
    for source_index, source in enumerate(sources):
        source_map = _load_exact_trajectory_map(
            source.map_path,
            context=f"source trajectory map {source_index}",
        )
        trajectory_count = source_map["trajectory_family_id"].size
        row_count = source_map["trajectory_index"].size
        trajectory_slice = slice(
            trajectory_offset,
            trajectory_offset + trajectory_count,
        )
        for name in trajectory_fields:
            expected = source_map[name]
            if name == "trajectory_first_row":
                expected = np.where(expected >= 0, expected + row_offset, -1).astype(
                    np.int64,
                    copy=False,
                )
            if not np.array_equal(combined[name][trajectory_slice], expected):
                raise ValueError(
                    f"combined trajectory-map field {name} differs from source "
                    f"{source_index}"
                )
        row_slice = slice(row_offset, row_offset + row_count)
        expected_rows = {
            "trajectory_index": source_map["trajectory_index"] + trajectory_offset,
            "frame_index": source_map["frame_index"],
            "shard_index": source_map["shard_index"] + shard_offset,
            "shard_row": source_map["shard_row"],
        }
        for name in row_fields:
            if not np.array_equal(combined[name][row_slice], expected_rows[name]):
                raise ValueError(
                    f"combined trajectory-map row field {name} differs from source "
                    f"{source_index}"
                )
        trajectory_offset += trajectory_count
        row_offset += row_count
        shard_offset += len(source.shard_paths)
    if (
        trajectory_offset != combined["trajectory_family_id"].size
        or row_offset != combined["trajectory_index"].size
    ):
        raise ValueError("combined trajectory map has entries outside its sources")


def _source_numerical_scales(
    summary: Mapping[str, Any],
    *,
    family: str,
) -> tuple[float, float, int]:
    run_spec = _mapping(summary.get("run_spec"), context="source run_spec")
    configuration = _mapping(
        run_spec.get("configuration"),
        context="source configuration",
    )
    if family == "stokes":
        numerical = _mapping(
            configuration.get("contract"),
            context="Stokes numerical contract",
        )
    else:
        execution = _mapping(
            configuration.get("trajectory_execution"),
            context=f"{family} trajectory execution",
        )
        numerical = _mapping(
            execution.get("numerical"),
            context=f"{family} numerical contract",
        )
    view = _mapping(summary.get("dataset_view"), context="source dataset_view")
    grid = _mapping(view.get("grid"), context="source stored grid")
    length = _positive_float(numerical.get("length"), context="domain length")
    gravity = _positive_float(numerical.get("gravity"), context="gravity")
    stored_nx = _nonnegative_integer(grid.get("nx"), context="stored grid nx")
    if stored_nx < 2:
        raise ValueError("stored grid nx must be at least two")
    grid_length = _positive_float(grid.get("length"), context="stored grid length")
    if not math.isclose(length, grid_length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("stored-grid length differs from numerical contract")
    return length, gravity, stored_nx


def _validate_source_map_identity(
    source: DatasetSource,
    *,
    revision_id: int,
) -> None:
    """Require every attempted trajectory to declare one exact identity."""

    with np.load(source.map_path, allow_pickle=False) as archive:
        required = {
            "trajectory_accepted",
            "trajectory_family_id",
            "trajectory_revision_id",
            "trajectory_split_id",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"source trajectory map lacks identity fields: {sorted(missing)}"
            )
        accepted = np.asarray(archive["trajectory_accepted"], dtype=np.bool_)
        family_ids = np.asarray(archive["trajectory_family_id"], dtype=np.int64)
        revisions = np.asarray(archive["trajectory_revision_id"], dtype=np.int64)
        split_ids = np.asarray(archive["trajectory_split_id"], dtype=np.int64)
    expected_shape = accepted.shape
    if accepted.ndim != 1 or any(
        values.shape != expected_shape for values in (family_ids, revisions, split_ids)
    ):
        raise ValueError("source trajectory identity arrays have inconsistent shapes")
    expected_family = int(FAMILY_IDS[source.family])
    expected_split = split_code(SplitId(source.split))
    if not np.all(family_ids == expected_family):
        raise ValueError(f"{source.family} trajectory map has a wrong family ID")
    if not np.all(revisions == revision_id):
        raise ValueError(f"{source.family} trajectory map has a wrong revision")
    if not np.all(split_ids == expected_split):
        raise ValueError(f"{source.family} trajectory map has a wrong split ID")


def load_source_identity(source: DatasetSource) -> SourceIdentity:
    """Recover and authenticate the revision and artifacts of one source."""

    summary = read_json(source.summary_path)
    run_spec = _mapping(summary.get("run_spec"), context="source run_spec")
    if run_spec.get("family_name") != source.family:
        raise ValueError("source run-spec family differs from its summary name")
    if run_spec.get("split_id") != source.split:
        raise ValueError("source run-spec split differs from its summary name")
    revision_id = _nonnegative_integer(
        run_spec.get("revision_id"),
        context="source revision_id",
    )
    expected_revision = EXPECTED_REVISIONS.get(source.family)
    if expected_revision is None or revision_id != expected_revision:
        raise ValueError(
            f"{source.family} source revision {revision_id} differs from the "
            f"required revision {expected_revision}"
        )

    view = _mapping(summary.get("dataset_view"), context="source dataset_view")
    manifest_record = _mapping(
        view.get("manifest"),
        context="source manifest record",
    )
    map_record = _mapping(
        view.get("trajectory_map"),
        context="source trajectory-map record",
    )
    summary_identity = ArtifactIdentity(
        path=source.summary_path,
        bytes=source.summary_path.stat().st_size,
        sha256=sha256(source.summary_path),
    )
    manifest_identity = _artifact_identity(
        source.manifest_path,
        expected_bytes=manifest_record.get("bytes"),
        expected_sha256=manifest_record.get("sha256"),
        context="source manifest",
    )
    map_identity = _artifact_identity(
        source.map_path,
        expected_bytes=map_record.get("bytes"),
        expected_sha256=map_record.get("sha256"),
        context="source trajectory map",
    )

    manifest = read_json(source.manifest_path)
    raw_shards = manifest.get("dataset_shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes)):
        raise TypeError("source manifest shards must be a JSON array")
    if len(raw_shards) != len(source.shard_paths):
        raise ValueError("source manifest and authenticated shard count differ")
    shard_hashes: dict[int, str] = {}
    for shard_index, raw_record in enumerate(raw_shards):
        record = _mapping(raw_record, context=f"source shard {shard_index}")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(
                f"source shard {shard_index} has an invalid SHA-256 digest"
            )
        path = source.shard_paths.get(shard_index)
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"source shard {shard_index} has no path")
        declared_path = (source.root / raw_path).resolve(strict=True)
        if not declared_path.is_relative_to(source.root):
            raise ValueError(f"source shard {shard_index} path escapes its root")
        if path is None or path != declared_path:
            raise ValueError(f"source shard {shard_index} path differs")
        # ``load_source_summary`` authenticated every shard against these
        # exact manifest digests immediately before this identity is built.
        shard_hashes[shard_index] = expected_sha256

    _validate_source_map_identity(source, revision_id=revision_id)
    length, gravity, stored_nx = _source_numerical_scales(
        summary,
        family=source.family,
    )
    return SourceIdentity(
        source=source,
        revision_id=revision_id,
        summary=summary_identity,
        manifest=manifest_identity,
        trajectory_map=map_identity,
        shard_sha256=shard_hashes,
        length=length,
        gravity=gravity,
        stored_nx=stored_nx,
    )


def validate_release_population(
    binding: CombinedSummaryBinding,
    sources: Sequence[DatasetSource],
) -> tuple[SourceIdentity, ...]:
    """Require the exact 26-source final revision-(2,3,4,4) population."""

    validate_bound_sources(binding, sources)
    accepted_cases = sum(len(source.trajectories) for source in sources)
    retained_rows = sum(
        trajectory.row_count for source in sources for trajectory in source.trajectories
    )
    validate_scanned_population(
        binding,
        source_count=len(sources),
        accepted_cases=accepted_cases,
        retained_rows=retained_rows,
    )
    validate_final_paper_dataset(sources, retained_rows=retained_rows)
    identities = tuple(load_source_identity(source) for source in sources)
    observed_revisions = {
        family: {
            identity.revision_id
            for identity in identities
            if identity.source.family == family
        }
        for family in FAMILY_ORDER
    }
    expected_revisions = {
        family: {revision} for family, revision in EXPECTED_REVISIONS.items()
    }
    if observed_revisions != expected_revisions:
        raise ValueError(
            "final paper-dataset revisions differ: "
            f"observed={observed_revisions}, expected={expected_revisions}"
        )
    return identities


def select_validation_trajectories(
    identities: Sequence[SourceIdentity],
) -> tuple[SelectedTrajectory, ...]:
    """Choose the explicitly defined lower-median case ID in each fixed cell."""

    selected: list[SelectedTrajectory] = []
    for family in FAMILY_ORDER:
        category = CENTRAL_VALIDATION_CATEGORIES[family]
        candidates = sorted(
            (
                (identity, trajectory)
                for identity in identities
                if identity.source.family == family
                and identity.source.split == SplitId.VALIDATION.value
                for trajectory in identity.source.trajectories
                if trajectory.category == category
            ),
            key=lambda pair: pair[1].case_id,
        )
        if not candidates:
            raise ValueError(
                f"no accepted validation cases found for {family}/{category}"
            )
        case_ids = [trajectory.case_id for _, trajectory in candidates]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate accepted case IDs in {family}/{category}")
        lower_median_index = (len(candidates) - 1) // 2
        identity, trajectory = candidates[lower_median_index]
        selected.append(
            SelectedTrajectory(
                identity=identity,
                trajectory=trajectory,
                candidate_count=len(candidates),
                lower_median_index=lower_median_index,
            )
        )
    return tuple(selected)


def dimensionless_profile(
    eta: np.ndarray,
    xi: np.ndarray,
    *,
    depth: float,
    gravity: float,
) -> DimensionlessProfile:
    """Return the exact dimensionless variables used in the figure."""

    eta_values = np.asarray(eta, dtype=np.float64)
    xi_values = np.asarray(xi, dtype=np.float64)
    if eta_values.ndim != 1 or xi_values.shape != eta_values.shape:
        raise ValueError("eta and xi must be one-dimensional arrays of equal shape")
    if eta_values.size < 2:
        raise ValueError("a profile must contain at least two grid points")
    if not np.all(np.isfinite(eta_values)) or not np.all(np.isfinite(xi_values)):
        raise ValueError("profile fields must be finite")
    positive_depth = _positive_float(depth, context="profile depth")
    positive_gravity = _positive_float(gravity, context="profile gravity")
    return DimensionlessProfile(
        x_over_length=np.arange(eta_values.size, dtype=np.float64) / eta_values.size,
        eta_over_depth=eta_values / positive_depth,
        xi_over_depth_speed=xi_values
        / (positive_depth * math.sqrt(positive_gravity * positive_depth)),
    )


def _validate_selected_row_ownership(selected: SelectedTrajectory) -> int:
    """Bind the chosen shard row to frame zero in the authenticated source map."""

    identity = selected.identity
    trajectory = selected.trajectory
    if sha256(identity.source.map_path) != identity.trajectory_map.sha256:
        raise ValueError("selected source trajectory-map SHA-256 differs")
    with np.load(identity.source.map_path, allow_pickle=False) as archive:
        first_rows = np.asarray(archive["trajectory_first_row"], dtype=np.int64)
        row_counts = np.asarray(archive["trajectory_row_count"], dtype=np.int64)
        row_trajectories = np.asarray(archive["trajectory_index"], dtype=np.int64)
        row_shards = np.asarray(archive["shard_index"], dtype=np.int64)
        shard_rows = np.asarray(archive["shard_row"], dtype=np.int64)
        frame_indices = np.asarray(archive["frame_index"], dtype=np.int64)
    trajectory_index = trajectory.trajectory_index
    if not 0 <= trajectory_index < first_rows.size:
        raise ValueError("selected trajectory index is outside its source map")
    first = int(first_rows[trajectory_index])
    count = int(row_counts[trajectory_index])
    if first < 0 or count != trajectory.row_count:
        raise ValueError("selected trajectory ownership differs from its source map")
    if not 0 <= first < row_trajectories.size:
        raise ValueError("selected first row is outside its source map")
    observed = (
        int(row_trajectories[first]),
        int(row_shards[first]),
        int(shard_rows[first]),
    )
    expected = (
        trajectory_index,
        trajectory.shard_index,
        trajectory.first_shard_row,
    )
    if observed != expected:
        raise ValueError("selected first-row ownership differs from its source map")
    frame_index = int(frame_indices[first])
    if frame_index != 0:
        raise ValueError("selected trajectory's first owned row is not frame zero")
    return frame_index


def load_illustration(selected: SelectedTrajectory) -> Illustration:
    """Load and nondimensionalize the selected trajectory's first owned row."""

    identity = selected.identity
    trajectory = selected.trajectory
    frame_index = _validate_selected_row_ownership(selected)
    shard_path = identity.source.shard_paths[trajectory.shard_index]
    expected_shard_sha256 = identity.shard_sha256[trajectory.shard_index]
    if sha256(shard_path) != expected_shard_sha256:
        raise ValueError("selected shard SHA-256 differs")
    with np.load(shard_path, allow_pickle=False) as archive:
        missing = {"eta", "xi", "depth", "time"}.difference(archive.files)
        if missing:
            raise ValueError(f"selected shard lacks fields: {sorted(missing)}")
        eta_all = np.asarray(archive["eta"])
        xi_all = np.asarray(archive["xi"])
        depths = np.asarray(archive["depth"], dtype=np.float64)
        times = np.asarray(archive["time"], dtype=np.float64)
    if eta_all.ndim != 2 or xi_all.shape != eta_all.shape:
        raise ValueError("selected shard eta/xi shapes differ")
    if eta_all.shape[1] != identity.stored_nx:
        raise ValueError("selected shard width differs from the stored-grid contract")
    if depths.shape != (eta_all.shape[0],) or times.shape != (eta_all.shape[0],):
        raise ValueError("selected shard scalar-row shapes differ")
    row = trajectory.first_shard_row
    if not 0 <= row < eta_all.shape[0]:
        raise ValueError("selected shard row is outside the shard")
    eta = np.asarray(eta_all[row], dtype=np.float64)
    xi = np.asarray(xi_all[row], dtype=np.float64)
    depth = float(depths[row])
    time = float(times[row])
    if not math.isfinite(time):
        raise ValueError("selected row time must be finite")
    if time != 0.0:
        raise ValueError("selected frame-zero row must have time zero")
    profile = dimensionless_profile(
        eta,
        xi,
        depth=depth,
        gravity=identity.gravity,
    )
    return Illustration(
        selected=selected,
        eta=eta,
        xi=xi,
        depth=depth,
        time=time,
        frame_index=frame_index,
        profile=profile,
    )


def _render_figure(
    illustrations: Sequence[Illustration],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Render four deterministic validation illustrations."""

    if tuple(item.selected.identity.source.family for item in illustrations) != (
        FAMILY_ORDER
    ):
        raise ValueError("illustrations must follow the fixed four-family order")
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(8.3, 8.8),
        sharex="col",
        constrained_layout=True,
    )
    colors = ("#3366a6", "#b54a3a", "#6b4c9a", "#2a8c6a")
    for row, (illustration, color) in enumerate(zip(illustrations, colors)):
        source = illustration.selected.identity.source
        trajectory = illustration.selected.trajectory
        profile = illustration.profile
        axes[row, 0].plot(
            profile.x_over_length,
            profile.eta_over_depth,
            color=color,
            linewidth=1.05,
        )
        axes[row, 1].plot(
            profile.x_over_length,
            profile.xi_over_depth_speed,
            color=color,
            linewidth=1.05,
        )
        axes[row, 0].set_ylabel(r"$\eta/h$")
        axes[row, 1].set_ylabel(r"$\xi/(h\sqrt{gh})$")
        axes[row, 0].text(
            0.02,
            0.92,
            (
                f"{FAMILY_LABELS[source.family]}\n"
                f"{trajectory.category}; case {trajectory.case_id}"
            ),
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=8.1,
        )
        for axis in axes[row]:
            axis.grid(alpha=0.22, linewidth=0.5)
    axes[0, 0].set_title("surface elevation")
    axes[0, 1].set_title("surface potential")
    axes[-1, 0].set_xlabel(r"$x/L$")
    axes[-1, 1].set_xlabel(r"$x/L$")
    figure.suptitle(
        "Deterministic validation illustrations (not empirical medoids)",
        fontsize=10,
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "build_parameterized_dataset_case_figure.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "build_parameterized_dataset_case_figure.py"},
    )
    plt.close(figure)


def _artifact_record(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": str(identity.path),
        "bytes": identity.bytes,
        "sha256": identity.sha256,
    }


def _case_record(illustration: Illustration) -> dict[str, object]:
    selected = illustration.selected
    identity = selected.identity
    source = identity.source
    trajectory = selected.trajectory
    shard_path = source.shard_paths[trajectory.shard_index]
    return {
        "family": source.family,
        "revision_id": identity.revision_id,
        "split": source.split,
        "category": trajectory.category,
        "case_id": trajectory.case_id,
        "time": illustration.time,
        "depth": illustration.depth,
        "gravity": identity.gravity,
        "domain_length": identity.length,
        "stored_nx": identity.stored_nx,
        "selection": {
            "rule": SELECTION_RULE,
            "candidate_count": selected.candidate_count,
            "lower_median_index_zero_based": selected.lower_median_index,
            "is_empirical_medoid": False,
        },
        "owned_row": {
            "accepted_index": trajectory.accepted_index,
            "trajectory_index": trajectory.trajectory_index,
            "frame_index": illustration.frame_index,
            "shard_index": trajectory.shard_index,
            "shard_row": trajectory.first_shard_row,
        },
        "dimensionless_variables": dict(DIMENSIONLESS_VARIABLES),
        "source_summary": _artifact_record(identity.summary),
        "source_manifest": _artifact_record(identity.manifest),
        "source_trajectory_map": _artifact_record(identity.trajectory_map),
        "selected_shard": {
            "path": str(shard_path),
            "bytes": shard_path.stat().st_size,
            "sha256": identity.shard_sha256[trajectory.shard_index],
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def publish_outputs(
    illustrations: Sequence[Illustration],
    *,
    output_stem: Path,
    binding: CombinedSummaryBinding,
    combined_view: CombinedViewIdentity,
    implementation: Mapping[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Stage outputs and publish JSON last as the set's completion marker."""

    bound_implementation = (
        figure_implementation_record()
        if implementation is None
        else dict(implementation)
    )
    resolved_stem = output_stem.expanduser().resolve()
    resolved_stem.parent.mkdir(parents=True, exist_ok=True)
    final_pdf = resolved_stem.with_suffix(".pdf")
    final_png = resolved_stem.with_suffix(".png")
    final_json = resolved_stem.with_suffix(".json")
    lock_path = resolved_stem.parent / f".{resolved_stem.name}.publication.lock"
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_stem.name}.staging-",
            dir=resolved_stem.parent,
        )
    )
    try:
        staged_pdf = staging_root / final_pdf.name
        staged_png = staging_root / final_png.name
        staged_json = staging_root / final_json.name
        _render_figure(
            illustrations,
            pdf_path=staged_pdf,
            png_path=staged_png,
        )
        pdf_identity = ArtifactIdentity(
            path=final_pdf,
            bytes=staged_pdf.stat().st_size,
            sha256=sha256(staged_pdf),
        )
        png_identity = ArtifactIdentity(
            path=final_png,
            bytes=staged_png.stat().st_size,
            sha256=sha256(staged_png),
        )
        _require_figure_implementation_current(bound_implementation)
        sidecar: dict[str, Any] = {
            "schema": SIDECAR_SCHEMA,
            "status": "complete",
            "description": SIDECAR_DESCRIPTION,
            "combined_summary": {
                "path": str(binding.path),
                "bytes": binding.path.stat().st_size,
                "sha256": binding.sha256,
            },
            "combined_view": {
                "manifest": _artifact_record(combined_view.manifest),
                "trajectory_map": _artifact_record(combined_view.trajectory_map),
            },
            "release_contract": {
                "source_count": binding.expected_source_count,
                "source_count_by_family": EXPECTED_SOURCE_COUNT_BY_FAMILY,
                "accepted_cases": binding.expected_accepted_cases,
                "retained_rows": binding.expected_retained_rows,
                "family_revisions": EXPECTED_REVISIONS,
                "ordered_cell_ids": EXPECTED_CELL_IDS,
                "central_validation_categories": CENTRAL_VALIDATION_CATEGORIES,
                "source_intervals": [
                    {
                        "family": item.family,
                        "split": item.split.value,
                        "stream_id": item.stream_id,
                        "accepted_before": item.accepted_before,
                        "accepted_after": item.accepted_after,
                    }
                    for item in _expected_chunk_layout()
                ],
            },
            "publication": {
                "commit_marker": str(final_json),
                "rule": PUBLICATION_RULE,
            },
            "cases": [_case_record(item) for item in illustrations],
            "figure_implementation": bound_implementation,
            "artifacts": {
                "pdf": _artifact_record(pdf_identity),
                "png": _artifact_record(png_identity),
            },
        }
        _write_json(staged_json, sidecar)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if final_json.exists() or final_json.is_symlink():
                final_json.unlink()
            os.replace(staged_pdf, final_pdf)
            os.replace(staged_png, final_png)
            os.replace(staged_json, final_json)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return final_pdf, final_png, final_json


def build_figure(
    combined_summary: Path,
    *,
    output_stem: Path,
) -> tuple[Path, Path, Path]:
    """Validate the final dataset, select four cases, and publish the figure."""

    implementation = figure_implementation_record()
    binding = load_combined_summary_binding(combined_summary)
    combined_view = load_combined_view_identity(binding)
    validate_final_combined_view(binding, combined_view)
    sources = tuple(load_source_summary(path) for path in binding.source_summary_paths)
    identities = validate_release_population(binding, sources)
    validate_source_map_binding(sources, combined_view)
    selected = select_validation_trajectories(identities)
    illustrations = tuple(load_illustration(item) for item in selected)
    return publish_outputs(
        illustrations,
        output_stem=output_stem,
        binding=binding,
        combined_view=combined_view,
        implementation=implementation,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Build the source-bound deterministic family figure."""

    args = parse_args(argv)
    pdf_path, png_path, json_path = build_figure(
        args.combined_summary,
        output_stem=args.output_stem,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pdf": str(pdf_path),
                "png": str(png_path),
                "sidecar": str(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
