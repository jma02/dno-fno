"""CPU tests for building a training view from committed batches."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    save_completed_batch,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
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


class DatasetViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_view_preserves_sparse_simulation_ids_and_row_ownership(self) -> None:
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
            build_dataset_view(self.root, batches[:1])
        self.assertFalse((self.root / "paper_dataset.dataset.json").exists())
        self.assertFalse((self.root / "paper_dataset.trajectory_map.npz").exists())

        view = build_dataset_view(self.root, batches)
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["n_rows"], 5)
        self.assertEqual(manifest["n_trajectories"], 8)
        self.assertEqual(
            [record["n_rows"] for record in manifest["dataset_shards"]],
            [2, 1, 2],
        )
        self.assertEqual(manifest["grid"]["nx"], 4)
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            np.testing.assert_array_equal(
                trajectory_map["trajectory_accepted"],
                np.asarray([False, False, True, False, True, True, False, True]),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_first_row"],
                np.asarray([-1, -1, 0, -1, 1, 2, -1, 3], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_index"],
                np.asarray([2, 4, 5, 7, 7], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_simulation_id"],
                np.asarray([0, 1, 0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_row_count"],
                np.asarray([0, 0, 1, 0, 1, 1, 0, 2], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                trajectory_map["shard_index"],
                np.asarray([0, 0, 1, 2, 2], dtype=np.int32),
            )

    def test_aliased_batch_paths_are_rejected_before_publication(self) -> None:
        path = batch_path(self.root, family="stokes", split="train", batch_id=0)
        _write_batch(
            path,
            family_id=PhysicalFamilyId.STOKES,
            dataset_split=DatasetSplit.TRAIN,
            simulation_count=2,
            accepted_local_indices=(0, 1),
            frames_per_simulation=1,
        )
        alias = batch_path(self.root, family="stokes", split="train", batch_id=1)
        alias.symlink_to(path)

        with self.assertRaisesRegex(ValueError, "batch paths must be unique"):
            build_dataset_view(self.root, (path, alias))

        self.assertFalse((self.root / "paper_dataset.dataset.json").exists())
        self.assertFalse((self.root / "paper_dataset.trajectory_map.npz").exists())

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
            build_dataset_view(self.root, (first, second))
        self.assertFalse((self.root / "paper_dataset.dataset.json").exists())
        self.assertFalse((self.root / "paper_dataset.trajectory_map.npz").exists())


if __name__ == "__main__":
    unittest.main()
