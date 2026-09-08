"""Combine completed family/split runs into NumPy arrays for training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


FAMILY_IDS = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}


def build_paper_dataset(summary_paths: Sequence[Path], *, output_root: Path) -> Path:
    """Combine equally sized family populations, preserving their assigned splits."""
    runs: dict[tuple[DatasetSplit, str], tuple[Path, ...]] = {}
    accepted_counts: dict[tuple[DatasetSplit, str], int] = {}
    batch_paths: set[Path] = set()
    for summary_path in summary_paths:
        path = summary_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise ValueError(f"{path} must describe completed generation")
        run_root = Path(summary["output_root"]).expanduser().resolve()
        if path.parent != run_root:
            raise ValueError("run summary must live directly in its output root")
        run_spec = summary["run_spec"]
        family = run_spec["family_name"]
        split = DatasetSplit(run_spec["dataset_split"])
        identity = (split, family)
        if identity in runs:
            raise ValueError("each family/split may have only one generation run")
        batches = tuple(
            (run_root / value).resolve() for value in summary["batch_paths"]
        )
        if any(not batch.is_relative_to(run_root) for batch in batches):
            raise ValueError("batch paths must stay inside the run output root")
        if len(set(batches)) != len(batches) or batch_paths.intersection(batches):
            raise ValueError("generation runs must not repeat a completed batch")
        batch_paths.update(batches)

        # Check the summaries against the batch metadata before exporting any data.
        attempted = accepted = 0
        for batch_path in batches:
            with np.load(batch_path, allow_pickle=False) as batch:
                if (
                    int(batch["family_id"]) != int(FAMILY_IDS[family])
                    or str(batch["dataset_split"]) != split.value
                ):
                    raise ValueError(f"batch family/split disagrees with {path}")
                attempted += batch["parameter_group_id"].size
                if "simulation_local_index" in batch:
                    accepted += np.unique(batch["simulation_local_index"]).size
        if (
            attempted != summary["counts"]["attempted"]
            or accepted != summary["counts"]["accepted"]
            or accepted != run_spec["accepted_simulation_count"]
        ):
            raise ValueError(f"batch counts disagree with {path}")
        runs[identity] = batches
        accepted_counts[identity] = accepted

    ordered_batches: list[Path] = []
    for split in DatasetSplit:
        families = {family for run_split, family in runs if run_split == split}
        if not families:
            continue
        if families != FAMILY_IDS.keys():
            raise ValueError(f"{split.value} requires all four physical families")
        if len({accepted_counts[split, family] for family in families}) != 1:
            raise ValueError(
                f"{split.value} requires equal accepted counts in every family"
            )
        ordered_batches.extend(
            batch for family in FAMILY_IDS for batch in runs[split, family]
        )
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
