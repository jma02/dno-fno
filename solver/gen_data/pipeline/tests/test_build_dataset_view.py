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
from solver.gen_data.pipeline.batch_artifacts import SimulationResult
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.types import (
    BatchPlanArrays,
    DatasetShardArrays,
    DatasetSplit,
)


def _proposal(
    *,
    family_id: int,
    dataset_split: DatasetSplit,
    attempt_numbers: tuple[int, ...],
) -> BatchPlanArrays:
    return {
        "family_id": np.asarray(family_id, dtype=np.int16),
        "dataset_split": np.asarray(dataset_split.value),
        "parameter_group_id": np.asarray(
            [f"group_{index}" for index in range(len(attempt_numbers))]
        ),
        "simulation_spec_json": np.asarray(
            [
                json.dumps({"attempt_number": attempt_number})
                for attempt_number in attempt_numbers
            ]
        ),
    }


def _write_batch(
    path: Path,
    *,
    proposal: BatchPlanArrays,
    accepted_local_indices: tuple[int, ...],
    frames_per_simulation: int,
    spatial_size: int = 4,
) -> None:
    simulation_local_index = np.repeat(
        np.asarray(accepted_local_indices, dtype=np.int32),
        frames_per_simulation,
    )
    frame_index = np.tile(
        np.arange(frames_per_simulation, dtype=np.int32),
        len(accepted_local_indices),
    )
    row_count = simulation_local_index.size
    field = np.arange(row_count * spatial_size, dtype=np.float32).reshape(
        row_count, spatial_size
    )
    shard = DatasetShardArrays(
        eta=field,
        xi=field + np.float32(0.1),
        gxi=field - np.float32(0.1),
        depth=np.ones(row_count, dtype=np.float64),
        time=frame_index.astype(np.float64),
        simulation_local_index=simulation_local_index,
        frame_index=frame_index,
    )
    blocks = {
        local_index: (position * frames_per_simulation, frames_per_simulation)
        for position, local_index in enumerate(accepted_local_indices)
    }
    save_completed_batch(
        path,
        plan=proposal,
        shard=shard,
        simulations=tuple(
            SimulationResult(
                accepted=local_index in blocks,
                failed_checks=() if local_index in blocks else ("nonfinite_state",),
                metrics={},
            )
            for local_index in range(int(proposal["simulation_spec_json"].size))
        ),
    )


class DatasetViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_view_preserves_rejected_simulations_and_row_ownership(self) -> None:
        first = batch_path(self.root, family="stokes", split="train", batch_id=0)
        second = batch_path(self.root, family="tanaka", split="validation", batch_id=0)
        _write_batch(
            first,
            proposal=_proposal(
                family_id=0,
                dataset_split=DatasetSplit.TRAIN,
                attempt_numbers=(10, 11),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=1,
        )
        _write_batch(
            second,
            proposal=_proposal(
                family_id=1,
                dataset_split=DatasetSplit.VALIDATION,
                attempt_numbers=(20,),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=2,
        )

        view = build_dataset_view(self.root, (first, second))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["n_rows"], 3)
        self.assertEqual(manifest["n_trajectories"], 3)
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            np.testing.assert_array_equal(
                trajectory_map["trajectory_accepted"],
                np.asarray([True, False, True]),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_first_row"],
                np.asarray([0, -1, 1], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_index"],
                np.asarray([0, 2, 2], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_simulation_id"],
                np.asarray([0, 1, 0], dtype=np.int64),
            )

    def test_simulation_ids_continue_across_batches_in_the_same_split(self) -> None:
        first = batch_path(self.root, family="stokes", split="train", batch_id=0)
        second = batch_path(self.root, family="stokes", split="train", batch_id=1)
        for path, attempt_numbers in ((first, (10, 11)), (second, (20,))):
            _write_batch(
                path,
                proposal=_proposal(
                    family_id=1,
                    dataset_split=DatasetSplit.TRAIN,
                    attempt_numbers=attempt_numbers,
                ),
                accepted_local_indices=tuple(range(len(attempt_numbers))),
                frames_per_simulation=1,
            )

        view = build_dataset_view(self.root, (first, second))
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            np.testing.assert_array_equal(
                trajectory_map["trajectory_simulation_id"],
                np.asarray([0, 1, 2], dtype=np.int64),
            )

    def test_duplicate_batch_path_is_rejected(self) -> None:
        path = batch_path(self.root, family="stokes", split="train", batch_id=0)
        _write_batch(
            path,
            proposal=_proposal(
                family_id=0,
                dataset_split=DatasetSplit.TRAIN,
                attempt_numbers=(10, 11),
            ),
            accepted_local_indices=(0, 1),
            frames_per_simulation=1,
        )

        with self.assertRaisesRegex(ValueError, "batch paths must be unique"):
            build_dataset_view(self.root, (path, path))

        self.assertFalse((self.root / "paper_dataset.dataset.json").exists())
        self.assertFalse((self.root / "paper_dataset.trajectory_map.npz").exists())

    def test_dual_grid_view_uses_target_grid(self) -> None:
        paths = batch_path(self.root, family="jonswap_tma", split="test", batch_id=0)
        _write_batch(
            paths,
            proposal=_proposal(
                family_id=3,
                dataset_split=DatasetSplit.TEST,
                attempt_numbers=(30,),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=2,
            spatial_size=4,
        )
        view = build_dataset_view(self.root, (paths,))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["grid"]["nx"], 4)

    def test_missing_batch_cannot_enter_view(self) -> None:
        path = batch_path(self.root, family="stokes", split="test", batch_id=0)
        with self.assertRaises(FileNotFoundError):
            build_dataset_view(self.root, (path,))


if __name__ == "__main__":
    unittest.main()
