"""Combine completed family/split runs into NumPy arrays for training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


def build_paper_dataset(summary_paths: Sequence[Path], *, output_root: Path) -> Path:
    """Combine equally sized family populations, preserving their assigned splits."""
    batches_by_run: dict[tuple[DatasetSplit, PhysicalFamilyId], tuple[Path, ...]] = {}
    simulations_per_run: dict[tuple[DatasetSplit, PhysicalFamilyId], int] = {}
    for summary_path in summary_paths:
        path = summary_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_spec = summary["run_spec"]
        family = PhysicalFamilyId[run_spec["family_name"].upper()]
        split = DatasetSplit(run_spec["dataset_split"])
        if (split, family) in batches_by_run:
            raise ValueError("each family/split may have only one generation run")
        batches_by_run[split, family] = tuple(
            (path.parent / value).resolve() for value in summary["batch_paths"]
        )
        simulations_per_run[split, family] = summary["counts"]["accepted"]

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
    if len(set(ordered_batches)) != len(ordered_batches):
        raise ValueError("generation runs must not repeat a completed batch")
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
