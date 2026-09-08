"""Combine completed family/split runs into one loader-ready dataset view."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from solver.gen_data.pipeline.artifact_io import write_json_atomic
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


FAMILY_IDS = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
CompletedRun = NamedTuple(
    "CompletedRun",
    [
        ("summary_path", Path),
        ("accepted_count", int),
        ("attempted_count", int),
        ("batches", tuple[Path, ...]),
    ],
)


def build_combined_view(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    name: str | None = None,
) -> Path:
    """Build equally sized family populations and return their combined summary."""

    root = output_root.expanduser().resolve()
    runs: dict[tuple[DatasetSplit, str], CompletedRun] = {}
    batch_paths: set[Path] = set()
    for summary_path in summary_paths:
        path = summary_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            raise ValueError(f"{path} must describe completed generation")
        run_root = Path(summary["output_root"]).expanduser().resolve()
        if path.parent != run_root:
            raise ValueError("run summary must live directly in its output root")
        if root == run_root:
            raise ValueError("view output root must differ from every generation root")

        run_spec = summary["run_spec"]
        family = run_spec["family_name"]
        if family not in FAMILY_IDS:
            raise ValueError(f"unknown dataset family: {family}")
        split = DatasetSplit(run_spec["dataset_split"])
        identity = (split, family)
        if identity in runs:
            raise ValueError("each family/split may have only one generation run")
        accepted_count = run_spec["accepted_simulation_count"]
        attempted_count = summary["counts"]["attempted"]
        if (
            type(accepted_count) is not int
            or accepted_count <= 0
            or type(attempted_count) is not int
            or attempted_count < accepted_count
        ):
            raise ValueError(
                "run counts must be positive integers with attempted >= accepted"
            )
        if summary["counts"]["accepted"] != accepted_count:
            raise ValueError("summary accepted count disagrees with its run spec")

        raw_paths = summary["batch_paths"]
        if (
            not isinstance(raw_paths, list)
            or not raw_paths
            or any(not isinstance(value, str) or not value for value in raw_paths)
        ):
            raise ValueError("batch_paths must be a nonempty list of paths")
        batches = tuple((run_root / value).resolve() for value in raw_paths)
        if any(not batch.is_relative_to(run_root) for batch in batches):
            raise ValueError("batch paths must stay inside the run output root")
        if len(set(batches)) != len(batches) or batch_paths.intersection(batches):
            raise ValueError("generation runs must not repeat a completed batch")
        batch_paths.update(batches)
        runs[identity] = CompletedRun(path, accepted_count, attempted_count, batches)

    ordered_runs: dict[tuple[DatasetSplit, str], CompletedRun] = {}
    splits = tuple(
        split for split in DatasetSplit if any(key[0] == split for key in runs)
    )
    for split in splits:
        by_family = {
            family: run
            for (run_split, family), run in runs.items()
            if run_split == split
        }
        if by_family.keys() != FAMILY_IDS.keys():
            raise ValueError(f"{split.value} requires all four physical families")
        if len({run.accepted_count for run in by_family.values()}) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        ordered_runs.update(
            ((split, family), by_family[family]) for family in FAMILY_IDS
        )

    view_name = name or (
        f"paper_dataset_{splits[0].value}"
        if len(splits) == 1
        else "paper_dataset_all_splits"
    )
    view = build_dataset_view(
        root,
        tuple(batch for run in ordered_runs.values() for batch in run.batches),
        name=view_name,
    )
    with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
        family_ids = trajectory_map["trajectory_family_id"]
        dataset_splits = trajectory_map["trajectory_dataset_split"]
        accepted = trajectory_map["trajectory_accepted"]
        row_counts = trajectory_map["trajectory_row_count"]
    offset = 0
    for (split, family), run in ordered_runs.items():
        run_slice = slice(offset, offset + run.attempted_count)
        if (
            accepted[run_slice].size != run.attempted_count
            or not np.all(family_ids[run_slice] == int(FAMILY_IDS[family]))
            or not np.all(dataset_splits[run_slice] == split.value)
            or np.count_nonzero(accepted[run_slice]) != run.accepted_count
        ):
            raise RuntimeError(
                f"built view counts disagree with {split.value}/{family} summary"
            )
        offset += run.attempted_count
    if offset != accepted.size:
        raise RuntimeError("built view contains simulations absent from run summaries")
    summary = {
        "status": "complete",
        "attempted_simulations": int(accepted.size),
        "accepted_simulations": int(np.count_nonzero(accepted)),
        "rows": int(row_counts.sum()),
        "run_summaries": [str(run.summary_path) for run in ordered_runs.values()],
        "dataset_view": {
            "manifest": str(view.manifest),
            "trajectory_map": str(view.trajectory_map),
        },
    }
    summary_path = root / f"{view_name}.summary.json"
    write_json_atomic(summary_path, summary)
    return summary_path


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
    args = parser.parse_args()
    if args.name and any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in args.name
    ):
        parser.error("--name must contain only letters, digits, '_' or '-'")
    summary_path = build_combined_view(
        args.run_summary,
        output_root=args.output_root,
        name=args.name,
    )
    print(json.dumps({"summary_path": str(summary_path)}))
