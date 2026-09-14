"""Build a dataset directly from saved NPZ files, without run summaries."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.build_paper_dataset import build_dataset, select_simulation_rows
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
    frame_count: int | None = None,
) -> Path:
    run_root = root / f"{family}_{seed}"
    batch = run_root / "batches" / "batch_000000.npz"
    if frame_count is None:
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
    return batch


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
        dataset = build_dataset(self.root / "dataset", tuple(reversed(paths)))
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
                *(f"--input-root={path.parent.parent}" for path in paths),
                f"--output-root={self.root / 'cli_dataset'}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_dataset = Path(result.stdout.strip())
        for name, array in arrays.items():
            np.testing.assert_array_equal(np.load(cli_dataset / f"{name}.npy"), array)

    def test_accepts_one_family_and_rejects_invalid_split_fractions(self) -> None:
        path = _write_run(self.root, "stokes", accepted_count=10)
        dataset = build_dataset(self.root / "dataset", (path,))
        np.testing.assert_array_equal(np.load(dataset / "family_id.npy"), np.ones(10))
        for validation, test in (
            (-0.1, 0.1),
            (0.1, -0.1),
            (0.6, 0.5),
            (float("nan"), 0.1),
        ):
            with self.subTest(validation=validation, test=test):
                with self.assertRaisesRegex(ValueError, "fractions"):
                    build_dataset(
                        self.root / "invalid",
                        (path,),
                        validation_fraction=validation,
                        test_fraction=test,
                    )
        self.assertFalse((self.root / "invalid").exists())

    def test_rejects_duplicate_runs_and_batches(self) -> None:
        paths = (_write_run(self.root, "stokes"),)
        duplicate = _write_run(self.root / "duplicate", "stokes")
        with self.assertRaisesRegex(ValueError, "distinct family/seed pairs"):
            build_dataset(self.root / "dataset", (*paths, duplicate))
        alias = paths[0].with_name("batch_alias.npz")
        alias.symlink_to(paths[0])
        with self.assertRaisesRegex(ValueError, "must not appear twice"):
            build_dataset(self.root / "dataset", (*paths, alias))
        self.assertFalse((self.root / "dataset").exists())

    def test_balances_raw_families_without_changing_selected_fields_or_splits(
        self,
    ) -> None:
        paths = tuple(
            _write_run(self.root, family, accepted_count=5, frame_count=frames)
            for family, frames in (
                ("stokes", 1),
                ("tanaka", 4),
                ("benjamin_feir", 4),
                ("jonswap_tma", 2),
            )
        )
        paths += (_write_run(self.root, "stokes", seed=19, accepted_count=5),)
        source = build_dataset(self.root / "uncapped", paths)
        original = {path.stem: np.load(path) for path in source.glob("*.npy")}
        output = build_dataset(
            self.root / "balanced",
            paths,
            max_snapshots_per_simulation=2,
        )
        arrays = {path.stem: np.load(path) for path in output.glob("*.npy")}
        np.testing.assert_array_equal(
            np.unique(arrays["family_id"], return_counts=True)[1], [10] * 4
        )
        np.testing.assert_array_equal(np.unique(arrays["simulation_id"]), np.arange(25))
        np.testing.assert_array_equal(arrays["x"], original["x"])
        for simulation_id in range(25):
            with self.subTest(simulation_id=simulation_id):
                rows = arrays["simulation_id"] == simulation_id
                self.assertEqual(np.unique(arrays["dataset_split"][rows]).size, 1)
                np.testing.assert_array_equal(
                    arrays["frame_index"][rows], np.arange(np.count_nonzero(rows))
                )
                source_rows = np.flatnonzero(original["simulation_id"] == simulation_id)
                selected = source_rows[
                    np.linspace(
                        0, source_rows.size - 1, min(source_rows.size, 2), dtype=int
                    )
                ]
                for name in arrays.keys() - {"x", "frame_index"}:
                    np.testing.assert_array_equal(
                        arrays[name][rows], original[name][selected]
                    )
        with self.assertRaises(FileExistsError):
            build_dataset(output, paths)

    def test_snapshot_cap_selects_spaced_frames_including_endpoints(self) -> None:
        path = _write_run(self.root, "tanaka", accepted_count=10, frame_count=5)
        raw = build_dataset(self.root / "raw", (path,), max_snapshots_per_simulation=3)
        np.testing.assert_array_equal(np.load(raw / "time.npy"), [0, 2, 4] * 10)
        np.testing.assert_array_equal(np.load(raw / "frame_index.npy"), [0, 1, 2] * 10)
        np.testing.assert_array_equal(
            np.load(raw / "eta.npy"),
            np.tile(np.arange(20).reshape(5, 4)[[0, 2, 4]], (10, 1)),
        )

    def test_keep_all_rows_skips_per_simulation_selection(self) -> None:
        counts = [1, 5, 2, 200]
        ids = np.repeat([1, 4, 5, 9], counts)
        expected_frames = np.concatenate([np.arange(count) for count in counts])
        for maximum in (None, 200, 300):
            with (
                self.subTest(maximum=maximum),
                patch("numpy.unique", side_effect=AssertionError("unexpected sorting")),
                patch(
                    "numpy.linspace", side_effect=AssertionError("unexpected sampling")
                ),
            ):
                rows, frames = select_simulation_rows(ids, maximum)
            np.testing.assert_array_equal(rows, np.arange(ids.size))
            np.testing.assert_array_equal(frames, expected_frames)


if __name__ == "__main__":
    unittest.main()
