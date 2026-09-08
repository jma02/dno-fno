"""CPU tests for exporting accepted simulations into training arrays."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    save_completed_batch,
)
from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)


def _write_batch(
    path: Path,
    *,
    family_id: PhysicalFamilyId,
    dataset_split: DatasetSplit,
    simulation_count: int,
    accepted_local_indices: tuple[int, ...],
    frames_per_simulation: int,
    spatial_size: int = 4,
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
        dataset_split=dataset_split,
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
        for batch_id, (family, split, count, accepted, frames) in enumerate(
            (
                (PhysicalFamilyId.STOKES, DatasetSplit.TRAIN, 2, (), 1),
                (PhysicalFamilyId.TANAKA, DatasetSplit.TRAIN, 3, (0, 2), 1),
                (PhysicalFamilyId.STOKES, DatasetSplit.TEST, 1, (0,), 1),
                (PhysicalFamilyId.STOKES, DatasetSplit.TRAIN, 2, (1,), 2),
            )
        ):
            path = self.root / f"batch_{batch_id:06d}.npz"
            _write_batch(
                path,
                family_id=family,
                dataset_split=split,
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

        dataset = build_dataset(self.root / "dataset", batches)
        for batch in batches:
            batch.unlink()
        arrays = {
            path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
            for path in dataset.glob("*.npy")
        }
        self.assertEqual(len(tuple(dataset.iterdir())), 11)
        self.assertTrue(all(isinstance(array, np.memmap) for array in arrays.values()))
        np.testing.assert_array_equal(arrays["family_id"], [2, 2, 1, 1, 1])
        np.testing.assert_array_equal(arrays["simulation_id"], [0, 2, 0, 3, 3])
        np.testing.assert_array_equal(arrays["frame_index"], [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(
            arrays["dataset_split"], ["train", "train", "test", "train", "train"]
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

    def test_mixed_spatial_grids_are_rejected_before_publication(self) -> None:
        first = batch_path(self.root, family="stokes", split="test", batch_id=0)
        second = batch_path(self.root, family="stokes", split="test", batch_id=1)
        _write_batch(
            first,
            family_id=PhysicalFamilyId.STOKES,
            dataset_split=DatasetSplit.TEST,
            simulation_count=1,
            accepted_local_indices=(0,),
            frames_per_simulation=1,
            spatial_size=4,
        )
        _write_batch(
            second,
            family_id=PhysicalFamilyId.STOKES,
            dataset_split=DatasetSplit.TEST,
            simulation_count=1,
            accepted_local_indices=(0,),
            frames_per_simulation=1,
            spatial_size=6,
        )

        with self.assertRaisesRegex(ValueError, "same spatial grid"):
            build_dataset(self.root / "dataset", (first, second))
        self.assertFalse((self.root / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
