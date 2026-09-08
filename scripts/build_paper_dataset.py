"""Combine completed family/split runs into NumPy arrays for training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


def build_paper_dataset(summary_paths: Sequence[Path], *, output_root: Path) -> Path:
    """Combine equally sized family populations, preserving their assigned splits."""
    batches_by_run: dict[tuple[DatasetSplit, PhysicalFamilyId], tuple[Path, ...]] = {}
    simulations_per_run: dict[tuple[DatasetSplit, PhysicalFamilyId], int] = {}
    batch_paths: set[Path] = set()
    for summary_path in summary_paths:
        path = summary_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_spec = summary["run_spec"]
        family = PhysicalFamilyId[run_spec["family_name"].upper()]
        split = DatasetSplit(run_spec["dataset_split"])
        if (split, family) in batches_by_run:
            raise ValueError("each family/split may have only one generation run")
        batches = tuple(
            (path.parent / value).resolve() for value in summary["batch_paths"]
        )

        # Check the summaries against the batch metadata before exporting any data.
        attempted = accepted = 0
        for batch_path in batches:
            if batch_path in batch_paths:
                raise ValueError("generation runs must not repeat a completed batch")
            batch_paths.add(batch_path)
            with np.load(batch_path, allow_pickle=False) as batch:
                if (
                    int(batch["family_id"]) != family
                    or str(batch["dataset_split"]) != split.value
                ):
                    raise ValueError(f"batch family/split disagrees with {path}")
                attempted += batch["parameter_group_id"].size
                if "simulation_local_index" in batch:
                    accepted += np.unique(batch["simulation_local_index"]).size
        if (
            attempted != summary["counts"]["attempted"]
            or accepted != summary["counts"]["accepted"]
        ):
            raise ValueError(f"batch counts disagree with {path}")
        batches_by_run[split, family] = batches
        simulations_per_run[split, family] = accepted

    ordered_batches: list[Path] = []
    for split in DatasetSplit:
        families = {
            family for run_split, family in batches_by_run if run_split == split
        }
        if not families:
            continue
        if families != set(PhysicalFamilyId):
            raise ValueError(f"{split.value} requires all four physical families")
        if len({simulations_per_run[split, family] for family in families}) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        for family in PhysicalFamilyId:
            ordered_batches.extend(batches_by_run[split, family])
    return build_dataset(output_root, ordered_batches)


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
    args = parser.parse_args()
    dataset = build_paper_dataset(args.run_summary, output_root=args.output_root)
    print(json.dumps({"dataset": str(dataset)}))
