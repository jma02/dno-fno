"""CPU integration tests for saved dataset arrays."""

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
from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    build_dataset_split_indices,
    get_batches,
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
                xi=np.full((frames_per_simulation, 4), local_index + 1.1)
                + np.linspace(-0.5, 0.5, 4),
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


def test_loads_readonly_arrays_and_keeps_simulation_splits() -> None:
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
        dataset_path = build_dataset(
            root / "dataset",
            (train, validation),
        )

        dataset = load_dataset_arrays(dataset_path)
        assert dataset["eta"].shape == (4, 4)
        assert all(
            isinstance(dataset[name], np.memmap) and not dataset[name].flags.writeable
            for name in ("eta", "xi", "gxi", "depth", "time", "dataset_split", "x")
        )
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
        stored_xi = np.array(dataset["xi"])
        batches = list(
            get_batches(
                dataset["eta"],
                dataset["xi"],
                dataset["gxi"],
                dataset["depth"],
                train_rows,
                batch_size=1,
                rng=np.random.default_rng(17),
                drop_last=False,
            )
        )
        batch_indices = np.concatenate([batch[-1] for batch in batches])
        assert set(batch_indices) == set(train_rows)
        np.testing.assert_array_equal(
            np.concatenate([batch[1] for batch in batches]),
            (stored_xi - stored_xi.mean(axis=1, keepdims=True))[batch_indices],
        )
        stats = load_or_compute_stats(dataset_path, dataset, indices=train_rows)
        centered_xi = stored_xi[train_rows] - stored_xi[train_rows].mean(
            axis=1, keepdims=True
        )
        assert np.asarray(stats["feature_absmax"])[1] == np.abs(centered_xi).max()
        np.testing.assert_array_equal(dataset["xi"], stored_xi)


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
        dataset_path = build_dataset(root / "dataset", (batch,))
        dataset = load_dataset_arrays(dataset_path)
        indices = np.arange(dataset["eta"].shape[0], dtype=np.int64)
        first = load_or_compute_stats(
            dataset_path,
            dataset=dataset,
            indices=indices,
        )

        cache_path = dataset_path / "stats.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["target_absmax"] = 123456.0
        cache_path.write_text(json.dumps(cached), encoding="utf-8")

        target_path = dataset_path / "gxi.npy"
        targets = np.load(target_path, mmap_mode="r+")
        targets *= 2
        targets.flush()
        second = load_or_compute_stats(
            dataset_path,
            dataset=dataset,
            indices=indices,
        )
        assert second["target_absmax"] == 2 * np.asarray(first["target_absmax"])
        assert second["target_absmax"] != 123456.0


def test_manifest_is_not_a_dataset_directory() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        manifest_path = Path(raw_directory) / "old.dataset.json"
        manifest_path.write_text("{}", encoding="utf-8")
        try:
            load_dataset_arrays(manifest_path)
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("manifest was accepted as a dataset directory")


def main() -> int:
    test_loads_readonly_arrays_and_keeps_simulation_splits()
    test_stats_cache_is_refreshed_when_dataset_inputs_change()
    test_manifest_is_not_a_dataset_directory()
    print("[PASS] dataset arrays, preassigned splits, and read-only preprocessing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
