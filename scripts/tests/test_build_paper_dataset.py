"""Combine real batch artifacts and reject invalid family populations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.build_paper_dataset import build_paper_dataset
from solver.gen_data.pipeline.artifact_io import write_json_atomic
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)


def _write_run(
    root: Path,
    family: str,
    split: DatasetSplit,
    *,
    accepted_count: int = 1,
) -> Path:
    run_root = root / f"{split.value}_{family}"
    batch = run_root / "batches" / "batch_000000.npz"
    frame_count = int(PhysicalFamilyId[family.upper()])
    eta = np.arange(frame_count * 4, dtype=np.float64).reshape(frame_count, 4)
    rows = SimulationRows(
        eta, eta + 1.0, eta - 1.0, 1.0, np.arange(frame_count, dtype=np.float64)
    )
    save_completed_batch(
        batch,
        ("group",) * (accepted_count + 1),
        (None, *((rows,) * accepted_count)),
        family_id=PhysicalFamilyId[family.upper()],
        dataset_split=split,
    )
    summary_path = run_root / f"paper_dataset_{family}_{split.value}.summary.json"
    write_json_atomic(
        summary_path,
        {
            "status": "complete",
            "output_root": str(run_root),
            "run_spec": {
                "family_name": family,
                "dataset_split": split.value,
                "accepted_simulation_count": accepted_count,
            },
            "counts": {"accepted": accepted_count, "attempted": accepted_count + 1},
            "batch_paths": [str(batch.relative_to(run_root))],
        },
    )
    return summary_path


def _write_runs(root: Path) -> tuple[Path, ...]:
    return tuple(
        _write_run(root, family.name.lower(), split)
        for split in DatasetSplit
        for family in PhysicalFamilyId
    )


class PaperDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_combines_sparse_batches_in_split_and_family_order(self) -> None:
        paths = _write_runs(self.root)
        dataset = build_paper_dataset(
            tuple(reversed(paths)), output_root=self.root / "dataset"
        )
        arrays = {path.stem: np.load(path) for path in dataset.glob("*.npy")}
        np.testing.assert_array_equal(
            arrays["family_id"], [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] * 3
        )
        np.testing.assert_array_equal(
            arrays["simulation_id"], np.ones(30, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            arrays["dataset_split"],
            np.repeat([split.value for split in DatasetSplit], 10),
        )
        np.testing.assert_array_equal(
            arrays["frame_index"], [0, 0, 1, 0, 1, 2, 0, 1, 2, 3] * 3
        )
        self.assertEqual(arrays["eta"].shape, (30, 4))
        self.assertEqual(np.count_nonzero(arrays["frame_index"] == 0), 12)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_paper_dataset",
                *(f"--run-summary={path}" for path in paths),
                f"--output-root={self.root / 'cli_dataset'}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_dataset = Path(json.loads(result.stdout)["dataset"])
        for name, array in arrays.items():
            np.testing.assert_array_equal(np.load(cli_dataset / f"{name}.npy"), array)

    def test_requires_all_runs_and_equal_accepted_counts(self) -> None:
        paths = _write_runs(self.root)
        for incomplete in (paths[:-1], paths[:-4], paths[:4]):
            with self.subTest(run_count=len(incomplete)):
                with self.assertRaisesRegex(ValueError, "all 12 family/split runs"):
                    build_paper_dataset(incomplete, output_root=self.root / "missing")
                self.assertFalse((self.root / "missing").exists())
        unequal = _write_run(
            self.root / "unequal", "stokes", DatasetSplit.TRAIN, accepted_count=2
        )
        with self.assertRaisesRegex(ValueError, "equal accepted counts"):
            build_paper_dataset(
                (unequal, *paths[1:]), output_root=self.root / "dataset"
            )

    def test_rejects_duplicate_runs_and_batches(self) -> None:
        paths = _write_runs(self.root)
        duplicate = _write_run(self.root / "duplicate", "stokes", DatasetSplit.TRAIN)
        with self.assertRaisesRegex(ValueError, "only one generation run"):
            build_paper_dataset((*paths, duplicate), output_root=self.root / "dataset")
        summary = json.loads(paths[0].read_text())
        alias = paths[0].parent / "alias.npz"
        alias.symlink_to(paths[0].parent / summary["batch_paths"][0])
        summary["batch_paths"].append(alias.name)
        write_json_atomic(paths[0], summary)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            build_paper_dataset(paths, output_root=self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
