"""Tests for ordering completed generation runs into one training view."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.build_paper_dataset_view import (
    FAMILY_IDS,
    FAMILY_ORDER,
    ROWS_PER_ACCEPTED_SIMULATION,
    CompletedRun,
    build_combined_view,
    load_completed_run,
    preflight,
    validate_combined_plan,
)
from solver.gen_data.pipeline.types import DatasetSplit
from solver.gen_data.pipeline.build_dataset_view import DatasetViewPaths


def _run(
    root: Path,
    *,
    family: str,
    split: DatasetSplit,
    accepted_count: int = 8,
) -> CompletedRun:
    return CompletedRun(
        summary_path=(root / f"{split.value}_{family}.summary.json"),
        root=(root / f"{split.value}_{family}"),
        family=family,
        split=split,
        accepted_count=accepted_count,
        attempted_count=accepted_count + 1,
        batches=(),
    )


def _one_split(root: Path, *, count: int = 8) -> tuple[CompletedRun, ...]:
    return tuple(
        _run(root, family=family, split=DatasetSplit.TRAIN, accepted_count=count)
        for family in FAMILY_ORDER
    )


def _write_trajectory_map(path: Path, runs: tuple[CompletedRun, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    family_ids: list[int] = []
    splits: list[str] = []
    accepted: list[bool] = []
    row_counts: list[int] = []
    for run in runs:
        family_ids.extend([int(FAMILY_IDS[run.family])] * run.attempted_count)
        splits.extend([run.split.value] * run.attempted_count)
        accepted.extend(
            [True] * run.accepted_count
            + [False] * (run.attempted_count - run.accepted_count)
        )
        row_counts.extend(
            [ROWS_PER_ACCEPTED_SIMULATION[run.family]] * run.accepted_count
            + [0] * (run.attempted_count - run.accepted_count)
        )
    np.savez(
        path,
        trajectory_family_id=np.asarray(family_ids, dtype=np.int16),
        trajectory_dataset_split=np.asarray(splits),
        trajectory_accepted=np.asarray(accepted, dtype=np.bool_),
        trajectory_row_count=np.asarray(row_counts, dtype=np.int32),
    )


class CombinedPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_equal_family_runs_form_one_ordered_plan(self) -> None:
        runs = tuple(reversed(_one_split(self.root)))
        plan = validate_combined_plan(runs)

        self.assertEqual(
            tuple(run.family for run in plan.runs),
            FAMILY_ORDER,
        )
        self.assertEqual(plan.splits, (DatasetSplit.TRAIN,))
        self.assertEqual(plan.accepted_per_family_by_split, {"train": 8})
        self.assertEqual(plan.accepted_count, 32)
        self.assertEqual(plan.attempted_count, 36)
        self.assertEqual(plan.expected_rows, 8 * (1 + 200 + 200 + 16))

    def test_each_split_requires_all_families_with_equal_totals(self) -> None:
        with self.assertRaisesRegex(ValueError, "all four"):
            validate_combined_plan(_one_split(self.root)[:-1])

        runs = list(_one_split(self.root))
        runs[-1] = replace(
            runs[-1],
            accepted_count=7,
            attempted_count=8,
        )
        with self.assertRaisesRegex(ValueError, "equal accepted counts"):
            validate_combined_plan(runs)

    def test_each_family_and_split_requires_exactly_one_run(self) -> None:
        runs = list(_one_split(self.root))
        first = runs[0]
        second = replace(
            first,
            summary_path=self.root / "second.summary.json",
            root=self.root / "second",
        )
        with self.assertRaisesRegex(ValueError, "only one generation run"):
            validate_combined_plan((*runs, second))

    def test_generation_runs_must_not_repeat_batches(self) -> None:
        runs = list(_one_split(self.root))
        repeated_batch = self.root / "batch_000000.npz"
        runs[0] = replace(runs[0], batches=(repeated_batch,))
        runs[1] = replace(runs[1], batches=(repeated_batch,))

        with self.assertRaisesRegex(ValueError, "must not repeat"):
            validate_combined_plan(runs)

    def test_preflight_is_write_free(self) -> None:
        runs = _one_split(self.root)
        output = self.root / "view"
        with mock.patch(
            "scripts.build_paper_dataset_view.load_completed_run",
            side_effect=runs,
        ):
            plan, resolved, name, record = preflight(
                [run.summary_path for run in runs],
                output_root=output,
                name="combined",
            )

        self.assertEqual(plan.accepted_count, 32)
        self.assertEqual(resolved, output.resolve())
        self.assertEqual(name, "combined")
        self.assertEqual(record["accepted_simulations"], 32)
        self.assertFalse(output.exists())

    def test_completed_run_loads_batch_paths_from_summary(self) -> None:
        root = self.root / "tanaka"
        batch = root / "batches" / "tanaka" / "train" / "batch_000000.npz"
        batch.parent.mkdir(parents=True)
        batch.touch()
        summary_path = root / "paper_dataset_tanaka_train.summary.json"
        summary = {
            "status": "complete",
            "output_root": str(root),
            "run_spec": {
                "family_name": "tanaka",
                "dataset_split": "train",
                "accepted_simulation_count": 8,
            },
            "counts": {"accepted": 8, "attempted": 9},
            "batch_paths": [str(batch.relative_to(root))],
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        run = load_completed_run(summary_path)

        self.assertEqual(run.family, "tanaka")
        self.assertEqual(run.split, DatasetSplit.TRAIN)
        self.assertEqual(run.accepted_count, 8)
        self.assertEqual(run.attempted_count, 9)
        self.assertEqual(run.batches, (batch.resolve(),))

    def test_completed_summary_lists_its_generation_runs(self) -> None:
        runs = _one_split(self.root)
        output = self.root / "view"
        view = DatasetViewPaths(
            manifest=output / "combined.dataset.json",
            trajectory_map=output / "combined.trajectory_map.npz",
        )
        _write_trajectory_map(view.trajectory_map, runs)
        with (
            mock.patch(
                "scripts.build_paper_dataset_view.load_completed_run",
                side_effect=runs,
            ),
            mock.patch(
                "scripts.build_paper_dataset_view.build_dataset_view",
                return_value=view,
            ),
        ):
            result = build_combined_view(
                [run.summary_path for run in runs],
                output_root=output,
                name="combined",
            )

        self.assertEqual(
            result.summary["run_summaries"],
            [str(run.summary_path) for run in runs],
        )

    def test_built_view_must_match_declared_run_counts(self) -> None:
        actual_runs = _one_split(self.root)
        declared_runs = (
            replace(actual_runs[0], attempted_count=10),
            *actual_runs[1:],
        )
        output = self.root / "view"
        view = DatasetViewPaths(
            manifest=output / "combined.dataset.json",
            trajectory_map=output / "combined.trajectory_map.npz",
        )
        _write_trajectory_map(view.trajectory_map, actual_runs)
        with (
            mock.patch(
                "scripts.build_paper_dataset_view.load_completed_run",
                side_effect=declared_runs,
            ),
            mock.patch(
                "scripts.build_paper_dataset_view.build_dataset_view",
                return_value=view,
            ),
            self.assertRaisesRegex(RuntimeError, "view counts disagree"),
        ):
            build_combined_view(
                [run.summary_path for run in declared_runs],
                output_root=output,
                name="combined",
            )


if __name__ == "__main__":
    unittest.main()
