"""Combine completed family/split runs into one loader-ready dataset view."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from solver.gen_data.pipeline.artifact_io import write_json_atomic
from solver.gen_data.pipeline.build_dataset_view import (
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.simulation_allocation import (
    DatasetSplit,
    PhysicalFamilyId,
)


FAMILY_ORDER = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
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
_VIEW_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class CompletedRun:
    """The batch files produced by one family/split generation run."""

    summary_path: Path
    root: Path
    family: str
    split: DatasetSplit
    accepted_count: int
    attempted_count: int
    batches: tuple[Path, ...]


@dataclass(frozen=True)
class CombinedDatasetPlan:
    """Validated run order and expected combined-dataset sizes."""

    runs: tuple[CompletedRun, ...]
    splits: tuple[DatasetSplit, ...]
    accepted_per_family_by_split: Mapping[str, int]
    attempted_by_split: Mapping[str, int]
    attempted_count: int
    accepted_count: int
    expected_rows: int


@dataclass(frozen=True)
class CombinedViewResult:
    summary_path: Path
    view: DatasetViewPaths
    summary: dict[str, object]


def _mapping(record: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = record.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _integer(
    record: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _string(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def load_completed_run(summary_path: Path) -> CompletedRun:
    """Read one new-format summary and its ordered completed batches."""

    path = Path(summary_path).expanduser().resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if summary.get("status") != "complete":
        raise RuntimeError(f"{path} does not describe completed generation")

    root = Path(_string(summary, "output_root")).expanduser().resolve()
    if path.parent != root:
        raise ValueError("run summary must live directly in its output root")
    run_spec = _mapping(summary, "run_spec")
    family = _string(run_spec, "family_name")
    if family not in FAMILY_ORDER:
        raise ValueError(f"unknown dataset family: {family}")
    split = DatasetSplit(_string(run_spec, "dataset_split"))
    accepted_count = _integer(run_spec, "accepted_simulation_count", minimum=1)
    counts = _mapping(summary, "counts")
    if _integer(counts, "accepted") != accepted_count:
        raise RuntimeError("summary accepted count disagrees with its run spec")
    attempted_count = _integer(counts, "attempted", minimum=accepted_count)

    raw_batch_paths = summary.get("batch_paths")
    if (
        not isinstance(raw_batch_paths, list)
        or not raw_batch_paths
        or any(not isinstance(value, str) or not value for value in raw_batch_paths)
    ):
        raise TypeError("batch_paths must be a nonempty list of relative paths")
    batches = tuple((root / value).resolve() for value in raw_batch_paths)
    if any(not path.is_relative_to(root) for path in batches):
        raise ValueError("batch paths must stay inside the run output root")
    missing = tuple(path for path in batches if not path.is_file())
    if missing:
        raise FileNotFoundError(f"completed batch does not exist: {missing[0]}")

    return CompletedRun(
        summary_path=path,
        root=root,
        family=family,
        split=split,
        accepted_count=accepted_count,
        attempted_count=attempted_count,
        batches=batches,
    )


def validate_combined_plan(runs: Sequence[CompletedRun]) -> CombinedDatasetPlan:
    """Require one equally sized run per family in every included split."""

    values = tuple(runs)
    if not values:
        raise ValueError("at least one completed run is required")
    identities = tuple((run.split, run.family) for run in values)
    if len(set(identities)) != len(identities):
        raise ValueError("each family/split may have only one generation run")
    batch_paths = tuple(batch for run in values for batch in run.batches)
    if len(set(batch_paths)) != len(batch_paths):
        raise ValueError("generation runs must not repeat a completed batch")

    splits = tuple(
        split for split in DatasetSplit if any(run.split is split for run in values)
    )
    accepted_by_split: dict[str, int] = {}
    attempted_by_split: dict[str, int] = {}
    ordered_runs: list[CompletedRun] = []
    for split in splits:
        by_family = {run.family: run for run in values if run.split is split}
        if set(by_family) != set(FAMILY_ORDER):
            raise ValueError(f"{split.value} requires all four physical families")
        family_counts = {run.accepted_count for run in by_family.values()}
        if len(family_counts) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        accepted_by_split[split.value] = family_counts.pop()
        attempted_by_split[split.value] = sum(
            run.attempted_count for run in by_family.values()
        )
        ordered_runs.extend(by_family[family] for family in FAMILY_ORDER)

    accepted_count = len(FAMILY_ORDER) * sum(accepted_by_split.values())
    return CombinedDatasetPlan(
        runs=tuple(ordered_runs),
        splits=splits,
        accepted_per_family_by_split=accepted_by_split,
        attempted_by_split=attempted_by_split,
        attempted_count=sum(attempted_by_split.values()),
        accepted_count=accepted_count,
        expected_rows=sum(accepted_by_split.values())
        * sum(ROWS_PER_ACCEPTED_SIMULATION.values()),
    )


def preflight(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None,
) -> tuple[CombinedDatasetPlan, Path, str, dict[str, object]]:
    plan = validate_combined_plan(tuple(map(load_completed_run, summary_paths)))
    root = Path(output_root).expanduser().resolve()
    if root in {run.root for run in plan.runs}:
        raise ValueError("view output root must differ from every generation root")
    view_name = name or (
        f"paper_dataset_{plan.splits[0].value}"
        if len(plan.splits) == 1
        else "paper_dataset_all_splits"
    )
    if _VIEW_NAME.fullmatch(view_name) is None:
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    record: dict[str, object] = {
        "mode": "dry_run",
        "output_root": str(root),
        "view_name": view_name,
        "splits": [split.value for split in plan.splits],
        "attempted_simulations": plan.attempted_count,
        "accepted_simulations": plan.accepted_count,
        "expected_rows": plan.expected_rows,
        "run_summaries": [str(run.summary_path) for run in plan.runs],
    }
    return plan, root, view_name, record


def build_combined_view(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None = None,
) -> CombinedViewResult:
    """Build one combined view from the validated family/split runs."""

    plan, root, view_name, _ = preflight(
        summary_paths,
        output_root=output_root,
        name=name,
    )
    started = perf_counter()
    view = build_dataset_view(
        root,
        tuple(batch for run in plan.runs for batch in run.batches),
        name=view_name,
        length=2.0 * math.pi,
    )
    with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
        family_ids = np.asarray(trajectory_map["trajectory_family_id"])
        dataset_splits = np.asarray(trajectory_map["trajectory_dataset_split"])
        accepted = np.asarray(trajectory_map["trajectory_accepted"], dtype=np.bool_)
        row_counts = np.asarray(trajectory_map["trajectory_row_count"], dtype=np.int64)
    for run in plan.runs:
        belongs_to_run = (family_ids == int(FAMILY_IDS[run.family])) & (
            dataset_splits == run.split.value
        )
        actual_attempted = int(np.count_nonzero(belongs_to_run))
        actual_accepted = int(np.count_nonzero(belongs_to_run & accepted))
        actual_rows = int(np.sum(row_counts[belongs_to_run], dtype=np.int64))
        expected_rows = run.accepted_count * ROWS_PER_ACCEPTED_SIMULATION[run.family]
        if (actual_attempted, actual_accepted, actual_rows) != (
            run.attempted_count,
            run.accepted_count,
            expected_rows,
        ):
            raise RuntimeError(
                f"built view counts disagree with {run.split.value}/{run.family} summary"
            )
    summary: dict[str, object] = {
        "status": "complete",
        "attempted_simulations": plan.attempted_count,
        "accepted_simulations": plan.accepted_count,
        "rows": plan.expected_rows,
        "run_summaries": [str(run.summary_path) for run in plan.runs],
        "dataset_view": {
            "manifest": str(view.manifest),
            "trajectory_map": str(view.trajectory_map),
        },
        "timing_seconds": perf_counter() - started,
    }
    summary_path = root / f"{view_name}.summary.json"
    write_json_atomic(summary_path, summary)
    return CombinedViewResult(summary_path, view, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-summary",
        type=Path,
        action="append",
        required=True,
        help="Completed family/split summary; repeat for every run.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", help="Output basename.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Build the view; without this flag, print the validated plan.",
    )
    args = parser.parse_args()
    summaries = tuple(args.run_summary)
    if args.execute:
        result = build_combined_view(
            summaries,
            output_root=args.output_root,
            name=args.name,
        )
        output: dict[str, object] = {
            "mode": "execute",
            "summary_path": str(result.summary_path),
            **result.summary,
        }
    else:
        _, _, _, output = preflight(
            summaries,
            output_root=args.output_root,
            name=args.name,
        )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
