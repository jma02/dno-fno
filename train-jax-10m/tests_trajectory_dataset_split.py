"""CPU regression tests for trajectory-mapped dataset views and splits.

Run with::

    JAX_PLATFORMS=cpu uv run python train-jax-10m/tests_trajectory_dataset_split.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    build_dataset_split_indices,
    build_split_indices,
    load_dataset_arrays,
    load_or_compute_stats,
)


def _write_fixture(directory: Path) -> tuple[Path, Path]:
    dataset_path = directory / "fixture.npz"
    manifest_path = directory / "fixture.dataset.json"
    trajectory_map_path = directory / "fixture.trajectory_map.npz"

    num_trajectories = 22
    frames_per_trajectory = 2
    trajectory_index = np.repeat(
        np.arange(num_trajectories, dtype=np.int32),
        frames_per_trajectory,
    )
    num_rows = trajectory_index.shape[0]
    frame_index = np.tile(
        np.arange(frames_per_trajectory, dtype=np.int32),
        num_trajectories,
    )
    family = np.repeat(np.arange(2, dtype=np.int16), 11)
    case = np.tile(np.arange(11, dtype=np.int64), 2)
    accepted = np.ones(num_trajectories, dtype=np.bool_)
    accepted[[10, 21]] = False
    failed = np.zeros(num_trajectories, dtype=np.uint32)
    failed[~accepted] = np.uint32(1)

    eta = np.arange(num_rows * 4, dtype=np.float32).reshape(num_rows, 4)
    eta[~accepted[trajectory_index]] = 1.0e6
    np.savez(
        dataset_path,
        eta=eta,
        xi=eta * 0.5,
        gxi=eta * 0.25,
        depth=np.ones(num_rows, dtype=np.float32),
        time=frame_index.astype(np.float32),
        source=family[trajectory_index].astype(np.int8),
        x=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False, dtype=np.float32),
    )
    dataset_path.with_suffix(".meta.json").write_text(
        json.dumps({"length": 2.0 * np.pi}),
        encoding="utf-8",
    )
    np.savez(
        trajectory_map_path,
        schema_version=np.asarray(1, dtype=np.int16),
        trajectory_index=trajectory_index,
        frame_index=frame_index,
        trajectory_family_id=family,
        trajectory_revision_id=np.zeros(num_trajectories, dtype=np.int16),
        trajectory_case_id=case,
        trajectory_accepted=accepted,
        trajectory_evaluated_bits=np.ones(num_trajectories, dtype=np.uint32),
        trajectory_failed_bits=failed,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_npz": dataset_path.name,
                "trajectory_map_npz": trajectory_map_path.name,
                "requires_trajectory_map": True,
                "n_rows": num_rows,
                "n_trajectories": num_trajectories,
                "n_accepted_rows": int(frames_per_trajectory * accepted.sum()),
            }
        ),
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def test_manifest_view_loads_trajectory_map_without_changing_direct_npz() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        dataset_path, manifest_path = _write_fixture(Path(raw_directory))
        legacy = load_dataset_arrays(dataset_path)
        filtered = load_dataset_arrays(manifest_path)

        assert "trajectory_index" not in legacy
        assert np.array_equal(legacy["case_id"], np.arange(44, dtype=np.int64))
        assert all(
            np.array_equal(left, right)
            for left, right in zip(
                build_dataset_split_indices(legacy, seed=17),
                build_split_indices(44, seed=17),
            )
        )
        assert np.array_equal(filtered["family_id"], filtered["source"])
        assert int(np.count_nonzero(filtered["accepted_mask"])) == 40
        assert np.array_equal(
            filtered["case_id"],
            np.tile(np.repeat(np.arange(11, dtype=np.int64), 2), 2),
        )


def test_split_is_deterministic_stratified_and_trajectory_disjoint() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        _, manifest_path = _write_fixture(Path(raw_directory))
        dataset = load_dataset_arrays(manifest_path)
        splits = build_dataset_split_indices(dataset, seed=17)
        repeated = build_dataset_split_indices(dataset, seed=17)
        assert all(np.array_equal(left, right) for left, right in zip(splits, repeated))

        assigned: dict[int, int] = {}
        for split_id, row_indices in enumerate(splits):
            assert np.all(dataset["accepted_mask"][row_indices])
            for trajectory in np.unique(dataset["trajectory_index"][row_indices]):
                assert int(trajectory) not in assigned
                assigned[int(trajectory)] = split_id
        assert len(assigned) == 20

        for family in (0, 1):
            trajectory_counts = []
            for row_indices in splits:
                family_rows = row_indices[dataset["family_id"][row_indices] == family]
                trajectory_counts.append(
                    np.unique(dataset["trajectory_index"][family_rows]).shape[0]
                )
            assert trajectory_counts == [8, 1, 1]


def test_training_stats_exclude_rejected_and_nontraining_rows() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        _, manifest_path = _write_fixture(Path(raw_directory))
        dataset = load_dataset_arrays(manifest_path)
        train_indices, _, _ = build_dataset_split_indices(dataset, seed=17)
        stats = load_or_compute_stats(
            manifest_path,
            dataset=dataset,
            indices=train_indices,
        )

        assert stats["feature_max"][0] < 1.0e6
        assert stats["index_selection"]["count"] == train_indices.shape[0]
        cached = load_or_compute_stats(
            manifest_path,
            dataset=dataset,
            indices=train_indices,
        )
        assert cached == stats


def test_required_trajectory_map_cannot_silently_fall_back() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        _, manifest_path = _write_fixture(directory)
        (directory / "fixture.trajectory_map.npz").unlink()
        try:
            load_dataset_arrays(manifest_path)
        except FileNotFoundError as error:
            assert "requires a trajectory map" in str(error)
        else:
            raise AssertionError("Missing required trajectory map did not raise")


def main() -> int:
    test_manifest_view_loads_trajectory_map_without_changing_direct_npz()
    test_split_is_deterministic_stratified_and_trajectory_disjoint()
    test_training_stats_exclude_rejected_and_nontraining_rows()
    test_required_trajectory_map_cannot_silently_fall_back()
    print("[PASS] trajectory-map loading and trajectory-grouped splitting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
