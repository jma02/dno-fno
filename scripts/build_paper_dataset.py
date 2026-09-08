"""Combine all 12 family/split runs into NumPy arrays for training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


def build_paper_dataset(summary_paths: Sequence[Path], *, output_root: Path) -> Path:
    """Build train, validation, and test with equal family populations in each."""
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

    if len(batches_by_run) != 12:
        raise ValueError("provide all 12 family/split runs")
    batch_paths: list[Path] = []
    for split in DatasetSplit:
        if (
            len({simulations_per_run[split, family] for family in PhysicalFamilyId})
            != 1
        ):
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        for family in PhysicalFamilyId:
            batch_paths.extend(batches_by_run[split, family])
    if len(set(batch_paths)) != len(batch_paths):
        raise ValueError("generation runs must not repeat a completed batch")
    return build_dataset(output_root, batch_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-summary",
        type=Path,
        action="append",
        required=True,
        help="Repeat for all 12 runs: four families, each with train/validation/test.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_paper_dataset(args.run_summary, output_root=args.output_root)
    print(json.dumps({"dataset": str(dataset)}))
