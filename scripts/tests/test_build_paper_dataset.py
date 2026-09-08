"""Pool generation runs and split whole simulations without family quotas."""

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
    PhysicalFamilyId,
    SimulationRows,
)


def _write_run(
    root: Path,
    family: str,
    *,
    seed: int = 2026072210,
    accepted_count: int = 1,
) -> Path:
    run_root = root / f"{family}_{seed}"
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
        seed=seed,
    )
    summary_path = run_root / f"paper_dataset_{family}.summary.json"
    write_json_atomic(
        summary_path,
        {
            "status": "complete",
            "output_root": str(run_root),
            "run_spec": {
                "family_name": family,
                "seed": seed,
                "accepted_simulation_count": accepted_count,
            },
            "counts": {"accepted": accepted_count, "attempted": accepted_count + 1},
            "batch_paths": [str(batch.relative_to(run_root))],
        },
    )
    return summary_path


class PaperDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_pools_unequal_family_populations_and_splits_simulations(self) -> None:
        paths = (
            _write_run(self.root, "stokes", accepted_count=3),
            _write_run(self.root, "tanaka", accepted_count=7),
        )
        dataset = build_paper_dataset(
            tuple(reversed(paths)), output_root=self.root / "dataset"
        )
        arrays = {path.stem: np.load(path) for path in dataset.glob("*.npy")}
        np.testing.assert_array_equal(arrays["family_id"], [1] * 3 + [2] * 14)
        np.testing.assert_array_equal(
            arrays["simulation_id"], [0, 1, 2, *np.repeat(np.arange(3, 10), 2)]
        )
        self.assertEqual(
            {
                split: np.unique(
                    arrays["simulation_id"][arrays["dataset_split"] == split]
                ).size
                for split in ("train", "validation", "test")
            },
            {"train": 8, "validation": 1, "test": 1},
        )
        np.testing.assert_array_equal(arrays["frame_index"], [0] * 3 + [0, 1] * 7)
        self.assertEqual(arrays["eta"].shape, (17, 4))
        self.assertEqual(np.count_nonzero(arrays["frame_index"] == 0), 10)
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

    def test_accepts_one_family_and_rejects_invalid_split_fractions(self) -> None:
        path = _write_run(self.root, "stokes", accepted_count=10)
        dataset = build_paper_dataset((path,), output_root=self.root / "dataset")
        np.testing.assert_array_equal(np.load(dataset / "family_id.npy"), np.ones(10))
        for validation, test in (
            (-0.1, 0.1),
            (0.1, -0.1),
            (0.6, 0.4),
            (float("nan"), 0.1),
        ):
            with self.subTest(validation=validation, test=test):
                with self.assertRaisesRegex(ValueError, "fractions"):
                    build_paper_dataset(
                        (path,),
                        output_root=self.root / "invalid",
                        validation_fraction=validation,
                        test_fraction=test,
                    )
        self.assertFalse((self.root / "invalid").exists())

    def test_rejects_duplicate_runs_and_batches(self) -> None:
        paths = (_write_run(self.root, "stokes"),)
        duplicate = _write_run(self.root / "duplicate", "stokes")
        with self.assertRaisesRegex(ValueError, "distinct family/seed pairs"):
            build_paper_dataset((*paths, duplicate), output_root=self.root / "dataset")
        summary = json.loads(paths[0].read_text())
        alias = paths[0].parent / "alias.npz"
        alias.symlink_to(paths[0].parent / summary["batch_paths"][0])
        summary["batch_paths"].append(alias.name)
        write_json_atomic(paths[0], summary)
        with self.assertRaisesRegex(ValueError, "must not appear twice"):
            build_paper_dataset(paths, output_root=self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
