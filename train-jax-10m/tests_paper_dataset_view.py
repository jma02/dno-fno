"""CPU integration test for schema-v2 paper-dataset views."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    save_completed_batch,
)
from solver.gen_data.pipeline.batch_artifacts import (
    SimulationCommitRecord,
    compute_batch_simulation_ids,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit
from solver.gen_data.pipeline.types import BatchPlanArrays, DatasetShardArrays

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    build_dataset_split_indices,
    load_dataset_arrays,
    load_or_compute_stats,
)


def _write_batch(
    root: Path,
    *,
    family: str,
    family_id: int,
    split: str,
    dataset_split: DatasetSplit,
    attempt_indices: tuple[int, ...],
    accepted_local_indices: tuple[int, ...],
) -> Path:
    path = batch_path(
        root,
        family=family,
        split=split,
        batch_id=0,
    )
    proposal = BatchPlanArrays(
        family_id=np.asarray(family_id, dtype=np.int16),
        dataset_split=np.asarray(dataset_split.value),
        parameter_group_id=np.asarray(
            [f"group_{index}" for index in range(len(attempt_indices))]
        ),
        worker_stream_id=np.zeros(len(attempt_indices), dtype=np.uint32),
        attempt_index=np.asarray(attempt_indices, dtype=np.uint64),
        simulation_spec_json=np.asarray(
            [
                json.dumps({"attempt_index": attempt_index})
                for attempt_index in attempt_indices
            ]
        ),
        metadata_json=np.asarray("{}"),
    )
    simulation_ids = compute_batch_simulation_ids(proposal)
    frames_per_simulation = 2
    simulation_local_index = np.repeat(
        np.asarray(accepted_local_indices, dtype=np.int32),
        frames_per_simulation,
    )
    frame_index = np.tile(
        np.arange(frames_per_simulation, dtype=np.int32),
        len(accepted_local_indices),
    )
    rows = simulation_local_index.size
    field = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 10.0
    shard = DatasetShardArrays(
        eta=field,
        xi=field + np.float32(0.1),
        gxi=field - np.float32(0.1),
        depth=np.repeat(
            np.arange(1, len(accepted_local_indices) + 1, dtype=np.float64),
            frames_per_simulation,
        ),
        time=frame_index.astype(np.float64),
        simulation_local_index=simulation_local_index,
        frame_index=frame_index,
    )
    blocks = {
        local_index: (position * frames_per_simulation, frames_per_simulation)
        for position, local_index in enumerate(accepted_local_indices)
    }
    records = tuple(
        SimulationCommitRecord(
            simulation_id=int(simulation_id),
            accepted=local_index in blocks,
            required_bits=63 if local_index in blocks else 32,
            evaluated_bits=63,
            failed_bits=0 if local_index in blocks else 32,
            first_row=blocks.get(local_index, (-1, 0))[0],
            row_count=blocks.get(local_index, (-1, 0))[1],
            metrics={},
        )
        for local_index, simulation_id in enumerate(simulation_ids)
    )
    save_completed_batch(
        path,
        plan=proposal,
        shard=shard,
        simulations=records,
        metadata={},
    )
    return path


def _build_single_family_view(root: Path) -> Path:
    batch = _write_batch(
        root,
        family="stokes",
        family_id=0,
        split="train",
        dataset_split=DatasetSplit.TRAIN,
        attempt_indices=(10,),
        accepted_local_indices=(0,),
    )
    return build_dataset_view(
        root,
        (batch,),
    ).manifest


def test_schema_v2_loads_shards_and_uses_preassigned_splits() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        train = _write_batch(
            root,
            family="stokes",
            family_id=0,
            split="train",
            dataset_split=DatasetSplit.TRAIN,
            attempt_indices=(10, 11),
            accepted_local_indices=(0,),
        )
        validation = _write_batch(
            root,
            family="tanaka",
            family_id=1,
            split="validation",
            dataset_split=DatasetSplit.VALIDATION,
            attempt_indices=(20,),
            accepted_local_indices=(0,),
        )
        paths = build_dataset_view(
            root,
            (train, validation),
        )

        dataset = load_dataset_arrays(paths.manifest)
        assert dataset["eta"].shape == (4, 4)
        assert np.array_equal(
            dataset["dataset_split"],
            np.asarray(
                [
                    DatasetSplit.TRAIN.value,
                    DatasetSplit.TRAIN.value,
                    DatasetSplit.VALIDATION.value,
                    DatasetSplit.VALIDATION.value,
                ]
            ),
        )
        assert np.array_equal(
            dataset["accepted_mask"],
            np.ones(4, dtype=np.bool_),
        )
        assert "trajectory_index" not in dataset
        assert "family_id" not in dataset
        assert "quality_required_bits" not in dataset

        train_rows, validation_rows, test_rows = build_dataset_split_indices(
            dataset,
            seed=17,
        )
        assert set(train_rows.tolist()) == {0, 1}
        assert set(validation_rows.tolist()) == {2, 3}
        assert test_rows.size == 0
        repeated = build_dataset_split_indices(dataset, seed=17)
        assert all(
            np.array_equal(first, second)
            for first, second in zip(
                (train_rows, validation_rows, test_rows),
                repeated,
            )
        )


def test_stats_cache_is_refreshed_when_dataset_inputs_change() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        manifest_path = _build_single_family_view(root)
        dataset = load_dataset_arrays(manifest_path)
        indices = np.arange(dataset["eta"].shape[0], dtype=np.int64)
        first = load_or_compute_stats(
            manifest_path,
            dataset=dataset,
            indices=indices,
        )

        cache_path = manifest_path.with_suffix(".stats.json")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["target_absmax"] = 123456.0
        cache_path.write_text(json.dumps(cached), encoding="utf-8")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["view_note"] = "changed"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        second = load_or_compute_stats(
            manifest_path,
            dataset=dataset,
            indices=indices,
        )
        assert second["target_absmax"] == first["target_absmax"]
        assert second["target_absmax"] != 123456.0


def test_direct_npz_dataset_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        legacy_path = Path(raw_directory) / "legacy.npz"
        try:
            load_dataset_arrays(legacy_path)
        except ValueError as error:
            assert "schema-v2" in str(error)
        else:
            raise AssertionError("direct legacy NPZ dataset did not fail closed")


def main() -> int:
    test_schema_v2_loads_shards_and_uses_preassigned_splits()
    test_stats_cache_is_refreshed_when_dataset_inputs_change()
    test_direct_npz_dataset_is_rejected()
    print("[PASS] schema-v2 paper-dataset view and preassigned splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
