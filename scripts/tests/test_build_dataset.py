"""CPU tests for exporting accepted simulations into training arrays."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solver.gen_data.pipeline.batch_storage import save_completed_batch
from scripts.build_paper_dataset import build_dataset
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)


def _write_batch(
    path: Path,
    *,
    family_id: PhysicalFamilyId,
    simulation_count: int,
    accepted_local_indices: tuple[int, ...],
    frames_per_simulation: int,
    spatial_size: int = 4,
    seed: int = 2026072210,
) -> None:
    accepted = set(accepted_local_indices)
    save_completed_batch(
        path,
        tuple(f"group_{index}" for index in range(simulation_count)),
        tuple(
            SimulationRows(
                eta=np.full(
                    (frames_per_simulation, spatial_size),
                    local_index + 1.0,
                ),
                xi=np.full(
                    (frames_per_simulation, spatial_size),
                    local_index + 1.1,
                ),
                gxi=np.full(
                    (frames_per_simulation, spatial_size),
                    local_index + 0.9,
                ),
                depth=1.0,
                time=np.arange(frames_per_simulation, dtype=np.float64),
            )
            if local_index in accepted
            else None
            for local_index in range(simulation_count)
        ),
        family_id=family_id,
        seed=seed,
    )


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_export_preserves_rows_and_simulation_splits_without_batch_files(
        self,
    ) -> None:
        batches: list[Path] = []
        for batch_id, (family, count, accepted, frames) in enumerate(
            (
                (PhysicalFamilyId.STOKES, 2, (), 1),
                (PhysicalFamilyId.TANAKA, 3, (0, 2), 1),
                (PhysicalFamilyId.STOKES, 1, (0,), 1),
                (PhysicalFamilyId.STOKES, 2, (1,), 2),
            )
        ):
            path = self.root / f"batch_{batch_id:06d}.npz"
            _write_batch(
                path,
                family_id=family,
                simulation_count=count,
                accepted_local_indices=accepted,
                frames_per_simulation=frames,
            )
            batches.append(path)

        with self.assertRaisesRegex(ValueError, "at least one accepted row"):
            build_dataset(self.root / "empty", batches[:1])
        self.assertFalse((self.root / "empty").exists())

        with patch("numpy.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                build_dataset(self.root / "dataset", batches)
        self.assertFalse((self.root / "dataset").exists())
        self.assertEqual(list(self.root.glob(".dataset-*")), [])

        dataset = build_dataset(
            self.root / "dataset", batches, validation_fraction=0.25, test_fraction=0.25
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_paper_dataset",
                f"--input-root={self.root}",
                f"--output-root={self.root / 'cli_dataset'}",
                "--validation-fraction=0.25",
                "--test-fraction=0.25",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_dataset = Path(result.stdout.strip())
        self.assertEqual(cli_dataset, self.root / "cli_dataset")
        for batch in batches:
            batch.unlink()
        arrays = {
            path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
            for path in dataset.glob("*.npy")
        }
        for name, array in arrays.items():
            np.testing.assert_array_equal(np.load(cli_dataset / f"{name}.npy"), array)
        self.assertEqual(len(tuple(dataset.iterdir())), 11)
        self.assertTrue(all(isinstance(array, np.memmap) for array in arrays.values()))
        np.testing.assert_array_equal(arrays["family_id"], [2, 2, 1, 1, 1])
        np.testing.assert_array_equal(arrays["simulation_id"], [0, 1, 2, 3, 3])
        np.testing.assert_array_equal(arrays["frame_index"], [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(
            arrays["dataset_split"], ["test", "validation", "train", "train", "train"]
        )
        np.testing.assert_array_equal(
            arrays["parameter_group_id"],
            ["group_0", "group_2", "group_0", "group_1", "group_1"],
        )
        np.testing.assert_array_equal(
            arrays["eta"], np.repeat([[1], [3], [1], [2], [2]], 4, axis=1)
        )
        np.testing.assert_array_equal(arrays["time"], [0, 0, 0, 0, 1])
        np.testing.assert_allclose(arrays["x"], np.arange(4) * np.pi / 2)
        with self.assertRaises(FileExistsError):
            build_dataset(dataset, batches)

    def test_rejects_invalid_split_fractions_and_duplicate_inputs(self) -> None:
        first = self.root / "first/batch_000000.npz"
        second = self.root / "second/batch_000000.npz"
        for path in (first, second):
            _write_batch(
                path,
                family_id=PhysicalFamilyId.STOKES,
                simulation_count=1,
                accepted_local_indices=(0,),
                frames_per_simulation=1,
            )
        for validation, test in ((-0.1, 0.1), (0.1, -0.1), (0.6, 0.5)):
            with self.subTest(validation=validation, test=test):
                with self.assertRaisesRegex(ValueError, "fractions"):
                    build_dataset(
                        self.root / "invalid",
                        (first,),
                        validation_fraction=validation,
                        test_fraction=test,
                    )
        with self.assertRaisesRegex(ValueError, "must not appear twice"):
            build_dataset(self.root / "duplicate_batch", (first, first))
        with self.assertRaisesRegex(ValueError, "distinct family/seed pairs"):
            build_dataset(self.root / "duplicate_run", (first, second))

    def test_mixed_spatial_grids_are_rejected_before_publication(self) -> None:
        first = self.root / "batches/stokes/batch_000000.npz"
        second = self.root / "batches/stokes/batch_000001.npz"
        _write_batch(
            first,
            family_id=PhysicalFamilyId.STOKES,
            simulation_count=1,
            accepted_local_indices=(0,),
            frames_per_simulation=1,
            spatial_size=4,
        )
        _write_batch(
            second,
            family_id=PhysicalFamilyId.STOKES,
            simulation_count=1,
            accepted_local_indices=(0,),
            frames_per_simulation=1,
            spatial_size=6,
        )

        with self.assertRaisesRegex(ValueError, "same spatial grid"):
            build_dataset(self.root / "dataset", (first, second))
        self.assertFalse((self.root / "dataset").exists())

    def test_split_seed_changes_only_labels_and_input_order_does_not_matter(
        self,
    ) -> None:
        batches = [self.root / f"batch_{i}.npz" for i in range(2)]
        for i, path in enumerate(batches):
            _write_batch(
                path,
                family_id=PhysicalFamilyId.STOKES,
                simulation_count=5,
                accepted_local_indices=tuple(range(5)),
                frames_per_simulation=i + 1,
            )
        outputs = [
            build_dataset(self.root / "first", batches),
            build_dataset(self.root / "reordered", tuple(reversed(batches))),
            build_dataset(self.root / "reseeded", batches, seed=7),
        ]
        for path in outputs[0].glob("*.npy"):
            expected = np.load(path)
            np.testing.assert_array_equal(np.load(outputs[1] / path.name), expected)
            if path.stem != "dataset_split":
                np.testing.assert_array_equal(np.load(outputs[2] / path.name), expected)
        first_labels, _, new_labels = [
            np.load(output / "dataset_split.npy") for output in outputs
        ]
        self.assertFalse(np.array_equal(first_labels, new_labels))
        ids = np.load(outputs[0] / "simulation_id.npy")
        for labels in (first_labels, new_labels):
            for simulation_id in range(10):
                self.assertEqual(np.unique(labels[ids == simulation_id]).size, 1)
            self.assertEqual(
                {
                    label: np.unique(ids[labels == label]).size
                    for label in np.unique(labels)
                },
                {"train": 8, "validation": 1, "test": 1},
            )


if __name__ == "__main__":
    unittest.main()
