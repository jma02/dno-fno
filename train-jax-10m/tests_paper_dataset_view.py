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
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)

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
    family_id: PhysicalFamilyId,
    split: str,
    dataset_split: DatasetSplit,
    simulation_count: int,
    accepted_local_indices: tuple[int, ...],
) -> Path:
    path = batch_path(
        root,
        family=family,
        split=split,
        batch_id=0,
    )
    frames_per_simulation = 2
    accepted = set(accepted_local_indices)
    save_completed_batch(
        path,
        tuple(f"group_{index}" for index in range(simulation_count)),
        tuple(
            SimulationRows(
                eta=np.full((frames_per_simulation, 4), local_index + 1.0),
                xi=np.full((frames_per_simulation, 4), local_index + 1.1),
                gxi=np.full((frames_per_simulation, 4), local_index + 0.9),
                depth=float(local_index + 1),
                time=np.arange(frames_per_simulation, dtype=np.float64),
            )
            if local_index in accepted
            else None
            for local_index in range(simulation_count)
        ),
        family_id=family_id,
        dataset_split=dataset_split,
    )
    return path


def test_schema_v2_loads_shards_and_uses_preassigned_splits() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        train = _write_batch(
            root,
            family="stokes",
            family_id=PhysicalFamilyId.STOKES,
            split="train",
            dataset_split=DatasetSplit.TRAIN,
            simulation_count=2,
            accepted_local_indices=(0,),
        )
        validation = _write_batch(
            root,
            family="tanaka",
            family_id=PhysicalFamilyId.TANAKA,
            split="validation",
            dataset_split=DatasetSplit.VALIDATION,
            simulation_count=1,
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
        batch = _write_batch(
            root,
            family="stokes",
            family_id=PhysicalFamilyId.STOKES,
            split="train",
            dataset_split=DatasetSplit.TRAIN,
            simulation_count=1,
            accepted_local_indices=(0,),
        )
        manifest_path = build_dataset_view(root, (batch,)).manifest
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
        flat_dataset_path = Path(raw_directory) / "flat_dataset.npz"
        try:
            load_dataset_arrays(flat_dataset_path)
        except ValueError as error:
            assert "schema-v2" in str(error)
        else:
            raise AssertionError("direct NPZ dataset did not fail closed")


def main() -> int:
    test_schema_v2_loads_shards_and_uses_preassigned_splits()
    test_stats_cache_is_refreshed_when_dataset_inputs_change()
    test_direct_npz_dataset_is_rejected()
    print("[PASS] schema-v2 paper-dataset view and preassigned splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
