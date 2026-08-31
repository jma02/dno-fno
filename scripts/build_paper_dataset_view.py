"""Build one four-family training view from completed dataset chunks.

The default mode checks the inputs without writing anything. ``--execute``
builds the manifest and trajectory map after every chunk is complete, chunk
intervals are contiguous, and all four physical families contribute the same
number of accepted simulations to each included split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from solver.gen_data.jonswap_horizon_executor import (
    BUCKETING_CONFIG_KEY,
    BucketingConfig,
)
from solver.gen_data.pipeline.artifact_io import write_json_atomic
from solver.gen_data.pipeline.batch_artifacts import parse_batch_plan_metadata
from solver.gen_data.pipeline.batch_storage import BatchPaths
from solver.gen_data.pipeline.build_dataset_view import (
    DATASET_VIEW_SCHEMA_VERSION,
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.simulation_allocation import (
    DATASET_REVISION_BY_FAMILY,
    PhysicalFamilyId,
    SampleCellTarget,
    SplitId,
    balanced_simulation_targets,
)
from solver.gen_data.pipeline.dataset_generation import (
    DatasetChunkConfig,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.trajectory_subsampling import TrajectoryFamily
from solver.gen_data.stokes_static_pipeline import PAPER_STATIC_STOKES_CONTRACT
from solver.gen_data.trajectory_batch_executor import paper_trajectory_execution


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
ROWS_PER_ACCEPTED_SIMULATION = {
    "stokes": 1,
    "tanaka": 200,
    "benjamin_feir": 200,
    "jonswap_tma": 16,
}


@dataclass(frozen=True)
class CompletedChunk:
    """One completed generator invocation and its committed batches."""

    summary_path: Path
    root: Path
    family: str
    revision_id: int
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int
    accepted_after: int
    attempted_count: int
    batches: tuple[BatchPaths, ...]


@dataclass(frozen=True)
class CombinedDatasetPlan:
    """Ordered chunks and expected combined-view sizes."""

    chunks: tuple[CompletedChunk, ...]
    splits: tuple[SplitId, ...]
    accepted_simulations_per_family_by_split: Mapping[str, int]
    attempted_simulations_by_split: Mapping[str, int]
    attempted_simulations: int
    accepted_simulations: int
    expected_rows: int


@dataclass(frozen=True)
class CombinedViewResult:
    """Files and summary produced by a combined-view build."""

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
        help="Completed generation summary; repeat for every additive chunk.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for the derived manifest and trajectory map.",
    )
    parser.add_argument("--name", help="Output basename.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Build the view.")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and print the plan without writing (the default).",
    )
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
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


def _ordered_cell_ids(configuration: Mapping[str, object]) -> tuple[str, ...]:
    value = configuration.get("ordered_cell_ids")
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(cell_id, str) for cell_id in value)
    ):
        raise TypeError("ordered_cell_ids must be a nonempty string sequence")
    return tuple(value)


def _reconstruct_chunk_config(
    summary: Mapping[str, object],
    *,
    root: Path,
) -> DatasetChunkConfig:
    record = _required_mapping(summary, "run_spec")
    family = _required_string(record, "family_name")
    if family not in FAMILY_IDS:
        raise ValueError(f"unknown dataset family: {family}")
    family_id = PhysicalFamilyId(_required_integer(record, "family_id"))
    if family_id is not FAMILY_IDS[family]:
        raise ValueError("family name and ID disagree")
    raw_targets = record.get("quotas")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise TypeError("run_spec quotas must be a nonempty list")
    if any(not isinstance(target, Mapping) for target in raw_targets):
        raise TypeError("every run_spec quota must be a JSON object")
    targets = tuple(
        SampleCellTarget(
            cell_id=_required_string(target, "cell_id"),
            simulation_count=_required_integer(target, "target_accepted"),
        )
        for target in raw_targets
        if isinstance(target, Mapping)
    )
    raw_codes = _required_mapping(record, "cell_codes")
    return DatasetChunkConfig(
        root=root,
        family_name=family,
        family_id=family_id,
        revision_id=_required_integer(record, "revision_id"),
        split_id=SplitId(_required_string(record, "split_id")),
        stream_id=_required_integer(record, "stream_id"),
        simulation_targets=targets,
        cell_codes={
            str(cell_id): _required_integer(raw_codes, str(cell_id))
            for cell_id in raw_codes
        },
        batch_size=_required_integer(record, "batch_size", minimum=1),
        first_attempt_index=_required_integer(record, "first_attempt_index"),
        configuration=_required_mapping(record, "configuration"),
    )


def _validate_current_execution_contract(chunk_config: DatasetChunkConfig) -> None:
    configuration = chunk_config.configuration
    if chunk_config.family_name == "stokes":
        name = "contract"
        expected = PAPER_STATIC_STOKES_CONTRACT.to_json_record()
    else:
        name = "trajectory_execution"
        family = cast(TrajectoryFamily, chunk_config.family_name)
        expected = paper_trajectory_execution(family).to_json_record()
    if _required_mapping(configuration, name) != expected:
        raise ValueError(
            f"{chunk_config.family_name} chunk does not use the current numerical contract"
        )

    if chunk_config.family_name != "jonswap_tma":
        return
    record = _required_mapping(configuration, BUCKETING_CONFIG_KEY)
    solver_batch_size = _required_integer(record, "solver_batch_size", minimum=1)
    if solver_batch_size > chunk_config.batch_size:
        raise ValueError("JONSWAP solver batch size exceeds dataset batch size")
    adjustment = paper_trajectory_execution("jonswap_tma").jonswap_adjustment
    assert adjustment is not None
    expected_config = BucketingConfig(
        batch_size=chunk_config.batch_size,
        solver_batch_size=solver_batch_size,
        adjustment=adjustment,
    )
    if not expected_config.matches_record(record):
        raise ValueError("JONSWAP chunk does not use the current bucketing config")


def _validate_batch_plan_metadata(
    paths: BatchPaths,
    *,
    chunk_config: DatasetChunkConfig,
) -> None:
    with np.load(paths.batch_plan, allow_pickle=False) as batch_plan:
        encoded = np.asarray(batch_plan["metadata_json"])
    metadata = parse_batch_plan_metadata(encoded)
    if metadata.get("family") != chunk_config.family_name:
        raise ValueError("batch-plan family differs from its chunk")
    expected_type = "static" if chunk_config.family_name == "stokes" else "trajectory"
    if metadata.get("simulation_type") != expected_type:
        raise ValueError("batch-plan simulation type differs from its chunk")
    execution_name = "contract" if expected_type == "static" else "trajectory_execution"
    if _required_mapping(metadata, execution_name) != _required_mapping(
        chunk_config.configuration, execution_name
    ):
        raise ValueError("batch-plan numerical contract differs from its chunk")


def load_completed_chunk(summary_path: Path) -> CompletedChunk:
    """Load one summary and verify that all of its batches are complete."""

    path = Path(summary_path).expanduser().resolve()
    summary = _read_json_object(path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"{path} does not describe completed generation")
    root = Path(_required_string(summary, "output_root")).expanduser().resolve()
    if path.parent != root:
        raise ValueError("chunk summary must live directly in its output root")
    chunk_config = _reconstruct_chunk_config(summary, root=root)
    current_revision = DATASET_REVISION_BY_FAMILY[FAMILY_IDS[chunk_config.family_name]]
    if chunk_config.revision_id != current_revision:
        raise ValueError(
            f"{chunk_config.family_name} revision {chunk_config.revision_id} is not current "
            f"revision {current_revision}"
        )
    _validate_current_execution_contract(chunk_config)

    configuration = chunk_config.configuration
    accepted_count = _required_integer(
        configuration,
        "accepted_simulation_count",
        minimum=1,
    )
    accepted_before = _required_integer(configuration, "accepted_simulations_before")
    accepted_after = _required_integer(configuration, "accepted_simulations_after")
    if accepted_after != accepted_before + accepted_count:
        raise ValueError("chunk accepted interval is inconsistent")

    ordered_cells = _ordered_cell_ids(configuration)
    before = balanced_simulation_targets(
        ordered_cells, simulation_count=accepted_before
    )
    after = balanced_simulation_targets(ordered_cells, simulation_count=accepted_after)
    expected_increment = tuple(
        end.simulation_count - start.simulation_count
        for start, end in zip(before, after)
    )
    if (
        tuple(target.cell_id for target in chunk_config.simulation_targets)
        != ordered_cells
    ):
        raise ValueError("chunk quota order differs from ordered_cell_ids")
    if (
        tuple(target.simulation_count for target in chunk_config.simulation_targets)
        != expected_increment
    ):
        raise ValueError("chunk quotas do not match its accepted interval")

    state = scan_dataset_generation(chunk_config)
    if not state.complete or state.pending_batch is not None:
        raise RuntimeError("chunk batches are not complete")
    if sum(state.accepted_simulation_counts.values()) != accepted_count:
        raise RuntimeError("chunk batches contain the wrong accepted count")
    for paths in state.committed:
        _validate_batch_plan_metadata(paths, chunk_config=chunk_config)

    counts = _required_mapping(summary, "counts")
    if _required_integer(counts, "accepted") != accepted_count:
        raise RuntimeError("chunk summary accepted count is inconsistent")
    attempted_count = _required_integer(counts, "attempted", minimum=accepted_count)
    if attempted_count != sum(state.simulation_attempt_counts.values()):
        raise RuntimeError("chunk summary attempted count is inconsistent")

    return CompletedChunk(
        summary_path=path,
        root=root,
        family=chunk_config.family_name,
        revision_id=chunk_config.revision_id,
        split=chunk_config.split_id,
        stream_id=chunk_config.stream_id,
        accepted_before=accepted_before,
        accepted_count=accepted_count,
        accepted_after=accepted_after,
        attempted_count=attempted_count,
        batches=state.committed,
    )


def validate_combined_plan(
    chunks: Sequence[CompletedChunk],
) -> CombinedDatasetPlan:
    """Order contiguous chunks and require equal family totals per split."""

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
    accepted_by_split: dict[str, int] = {}
    attempted_by_split: dict[str, int] = {}
    ordered: list[CompletedChunk] = []
    for split in split_order:
        split_chunks = tuple(chunk for chunk in values if chunk.split is split)
        if {chunk.family for chunk in split_chunks} != set(FAMILY_ORDER):
            raise ValueError(f"{split.value} requires all four physical families")
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
                        f"{split.value}/{family} chunks have a gap or overlap at {cursor}"
                    )
                if chunk.stream_id in streams or chunk.root in roots:
                    raise ValueError(
                        f"{split.value}/{family} chunks must use distinct streams and roots"
                    )
                streams.add(chunk.stream_id)
                roots.add(chunk.root)
                cursor = chunk.accepted_after
                ordered.append(chunk)
            totals[family] = cursor
        if len(set(totals.values())) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        accepted_by_split[split.value] = next(iter(totals.values()))
        attempted_by_split[split.value] = sum(
            chunk.attempted_count for chunk in split_chunks
        )

    accepted_simulations = len(FAMILY_ORDER) * sum(accepted_by_split.values())
    return CombinedDatasetPlan(
        chunks=tuple(ordered),
        splits=split_order,
        accepted_simulations_per_family_by_split=accepted_by_split,
        attempted_simulations_by_split=attempted_by_split,
        attempted_simulations=sum(attempted_by_split.values()),
        accepted_simulations=accepted_simulations,
        expected_rows=(
            sum(accepted_by_split.values()) * sum(ROWS_PER_ACCEPTED_SIMULATION.values())
        ),
    )


def preflight(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None,
) -> tuple[CombinedDatasetPlan, Path, str, dict[str, object]]:
    """Load all chunks and return a write-free build plan."""

    plan = validate_combined_plan(
        tuple(load_completed_chunk(path) for path in summary_paths)
    )
    root = Path(output_root).expanduser().resolve()
    if root in {chunk.root for chunk in plan.chunks}:
        raise ValueError("view output root must differ from every chunk root")
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
    record: dict[str, object] = {
        "mode": "dry_run",
        "no_view_written": True,
        "output_root": str(root),
        "view_name": view_name,
        "splits": [split.value for split in plan.splits],
        "accepted_simulations_per_family_by_split": dict(
            plan.accepted_simulations_per_family_by_split
        ),
        "accepted_simulations_total": plan.accepted_simulations,
        "attempted_simulations_total": plan.attempted_simulations,
        "expected_rows": plan.expected_rows,
        "chunks": [
            {
                "family": chunk.family,
                "revision_id": chunk.revision_id,
                "split": chunk.split.value,
                "stream_id": chunk.stream_id,
                "accepted_before": chunk.accepted_before,
                "accepted_after": chunk.accepted_after,
                "attempted_count": chunk.attempted_count,
                "summary_path": str(chunk.summary_path),
                "committed_batches": len(chunk.batches),
            }
            for chunk in plan.chunks
        ],
    }
    return plan, root, view_name, record


def _validate_view(
    view: DatasetViewPaths,
    *,
    plan: CombinedDatasetPlan,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "trajectory_map_npz": view.trajectory_map.name,
        "requires_trajectory_map": True,
        "n_rows": plan.expected_rows,
        "n_trajectories": plan.attempted_simulations,
        "n_accepted_trajectories": plan.accepted_simulations,
        "n_accepted_rows": plan.expected_rows,
        "grid": {"length": 2.0 * math.pi, "nx": 1024},
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"combined view {name} is {manifest.get(name)!r}, expected {value!r}"
            )
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise TypeError("combined view split_counts must be an object")
    for split in (SplitId.TRAIN, SplitId.VALIDATION, SplitId.TEST):
        per_family = plan.accepted_simulations_per_family_by_split.get(split.value, 0)
        expected_split = {
            "attempted": plan.attempted_simulations_by_split.get(split.value, 0),
            "accepted": len(FAMILY_ORDER) * per_family,
        }
        if split_counts.get(split.value) != expected_split:
            raise RuntimeError(f"combined view {split.value} split count is wrong")

    with np.load(view.trajectory_map, allow_pickle=False) as stored:
        accepted = np.asarray(stored["trajectory_accepted"], dtype=np.bool_)
        row_count = np.asarray(stored["trajectory_row_count"], dtype=np.int64)
        row_owner = np.asarray(stored["trajectory_index"], dtype=np.int64)
    if accepted.size != plan.attempted_simulations:
        raise RuntimeError("trajectory map has the wrong simulation count")
    if int(np.count_nonzero(accepted)) != plan.accepted_simulations:
        raise RuntimeError("trajectory map has the wrong accepted count")
    if int(np.sum(row_count, dtype=np.int64)) != plan.expected_rows:
        raise RuntimeError("trajectory map declares the wrong row count")
    if row_owner.size != plan.expected_rows:
        raise RuntimeError("trajectory map has the wrong number of row owners")
    return {
        **expected,
        "manifest": {
            "path": str(view.manifest),
            "bytes": view.manifest.stat().st_size,
        },
        "trajectory_map": {
            "path": str(view.trajectory_map),
            "bytes": view.trajectory_map.stat().st_size,
        },
    }


def build_combined_view(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None = None,
) -> CombinedViewResult:
    """Build and validate one combined training view."""

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
    )
    summary: dict[str, object] = {
        "status": "complete",
        "preflight": preflight_record,
        "dataset_view": _validate_view(view, plan=plan),
        "timing_seconds": {"total": perf_counter() - started},
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
    print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
