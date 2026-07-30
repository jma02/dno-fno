"""CPU integration test for schema-v2 paper-corpus views."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    CaseCommitRecord,
    commit_batch,
    ensure_proposal,
    ensure_shard,
)
from solver.gen_data.pipeline.manifest import build_dataset_view

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    build_dataset_split_indices,
    build_hierarchical_epoch_indices,
    load_dataset_arrays,
    load_dataset_identity,
    load_dataset_meta,
    load_or_compute_stats,
)


FINGERPRINT = "d" * 64


def _write_batch(
    root: Path,
    *,
    family: str,
    family_id: int,
    split: str,
    split_id: int,
    case_ids: tuple[int, ...],
    accepted_local_indices: tuple[int, ...],
) -> BatchPaths:
    paths = BatchPaths.under(
        root,
        family=family,
        split=split,
        batch_id=0,
    )
    proposal = {
        "config_fingerprint": np.asarray(FINGERPRINT),
        "family_id": np.asarray(family_id, dtype=np.int16),
        "revision_id": np.asarray(1, dtype=np.int16),
        "split_id": np.asarray(split_id, dtype=np.uint8),
        "batch_id": np.asarray(0, dtype=np.int64),
        "case_id": np.asarray(case_ids, dtype=np.int64),
        "cell_id": np.arange(len(case_ids), dtype=np.int32),
        "case_spec_json": np.asarray(
            [json.dumps({"case_id": case_id}) for case_id in case_ids]
        ),
        "metadata_json": np.asarray("{}"),
    }
    proposal_sha256 = ensure_proposal(paths, proposal)
    frames_per_case = 2
    case_local_index = np.repeat(
        np.asarray(accepted_local_indices, dtype=np.int32),
        frames_per_case,
    )
    frame_index = np.tile(
        np.arange(frames_per_case, dtype=np.int32),
        len(accepted_local_indices),
    )
    rows = case_local_index.size
    field = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 10.0
    shard = {
        "eta": field,
        "xi": field + np.float32(0.1),
        "gxi": field - np.float32(0.1),
        "depth": np.repeat(
            np.arange(1, len(accepted_local_indices) + 1, dtype=np.float64),
            frames_per_case,
        ),
        "time": frame_index.astype(np.float64),
        "case_local_index": case_local_index,
        "frame_index": frame_index,
        "selected_dense_index": frame_index * np.int32(4),
        "config_fingerprint": np.asarray(FINGERPRINT),
        "proposal_sha256": np.asarray(proposal_sha256),
    }
    ensure_shard(paths, shard)
    blocks = {
        local_index: (position * frames_per_case, frames_per_case)
        for position, local_index in enumerate(accepted_local_indices)
    }
    records = tuple(
        CaseCommitRecord(
            case_id=case_id,
            accepted=local_index in blocks,
            required_bits=63 if local_index in blocks else 32,
            evaluated_bits=63,
            failed_bits=0 if local_index in blocks else 32,
            first_row=blocks.get(local_index, (-1, 0))[0],
            row_count=blocks.get(local_index, (-1, 0))[1],
            metrics={},
        )
        for local_index, case_id in enumerate(case_ids)
    )
    commit_batch(paths, cases=records, metadata={})
    return paths


def _build_single_family_view(root: Path) -> Path:
    batch = _write_batch(
        root,
        family="stokes",
        family_id=0,
        split="train",
        split_id=0,
        case_ids=(10,),
        accepted_local_indices=(0,),
    )
    return build_dataset_view(
        root,
        (batch,),
        expected_fingerprint=FINGERPRINT,
    ).manifest


def test_schema_v2_loads_shards_and_uses_preassigned_splits() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        train = _write_batch(
            root,
            family="stokes",
            family_id=0,
            split="train",
            split_id=0,
            case_ids=(10, 11),
            accepted_local_indices=(0,),
        )
        validation = _write_batch(
            root,
            family="tanaka",
            family_id=1,
            split="validation",
            split_id=1,
            case_ids=(20,),
            accepted_local_indices=(0,),
        )
        paths = build_dataset_view(
            root,
            (train, validation),
            expected_fingerprint=FINGERPRINT,
        )

        dataset = load_dataset_arrays(paths.manifest)
        assert dataset["eta"].shape == (4, 4)
        assert np.array_equal(
            dataset["case_id"],
            np.asarray([10, 10, 20, 20], dtype=np.int64),
        )
        assert np.array_equal(
            dataset["family_id"],
            np.asarray([0, 0, 1, 1], dtype=np.int16),
        )
        assert np.array_equal(dataset["source"], dataset["family_id"])
        assert np.array_equal(
            dataset["quality_required_bits"],
            np.full(4, 63, dtype=np.uint32),
        )
        assert np.array_equal(
            dataset["split_id"],
            np.asarray([0, 0, 1, 1], dtype=np.uint8),
        )
        assert np.array_equal(
            dataset["frame_index"],
            np.asarray([0, 1, 0, 1], dtype=np.int32),
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

        metadata = load_dataset_meta(paths.manifest)
        assert metadata["dataset_kind"] == "paper_corpus_shards"
        assert metadata["nx"] == 4
        assert np.isclose(metadata["length"], 2.0 * np.pi)


def test_hierarchical_epoch_uses_every_case_once_and_one_time_each() -> None:
    trajectory_index = np.repeat(np.arange(4, dtype=np.int32), 3)
    family_id = np.repeat(np.asarray([0, 0, 1, 1], dtype=np.int16), 3)
    dataset = {
        "eta": np.zeros((12, 4), dtype=np.float32),
        "trajectory_index": trajectory_index,
        "family_id": family_id,
        "split_id": np.zeros(12, dtype=np.uint8),
        "accepted_mask": np.ones(12, dtype=np.bool_),
    }
    selected = build_hierarchical_epoch_indices(
        dataset,
        split_id=0,
        seed=19,
        epoch=4,
    )
    repeated = build_hierarchical_epoch_indices(
        dataset,
        split_id=0,
        seed=19,
        epoch=4,
    )
    assert np.array_equal(selected, repeated)
    assert selected.shape == (4,)
    selected_trajectories = trajectory_index[selected]
    assert np.array_equal(
        np.sort(selected_trajectories),
        np.arange(4, dtype=np.int32),
    )
    assert np.bincount(family_id[selected], minlength=2).tolist() == [2, 2]

    dataset["accepted_mask"] = dataset["accepted_mask"].copy()
    dataset["accepted_mask"][9:] = False
    try:
        build_hierarchical_epoch_indices(
            dataset,
            split_id=0,
            seed=19,
            epoch=4,
        )
    except ValueError as error:
        assert "equal accepted case counts" in str(error)
    else:
        raise AssertionError("unequal family case counts did not fail closed")


def test_schema_v2_rejects_changed_trajectory_map() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        manifest_path = _build_single_family_view(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trajectory_map_path = root / manifest["trajectory_map_npz"]
        contents = bytearray(trajectory_map_path.read_bytes())
        contents[-1] ^= 1
        trajectory_map_path.write_bytes(contents)

        try:
            load_dataset_arrays(manifest_path)
        except RuntimeError as error:
            assert "Trajectory map hash does not match" in str(error)
        else:
            raise AssertionError("changed trajectory map did not fail closed")


def test_schema_v2_checks_declared_trajectory_map_size() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        manifest_path = _build_single_family_view(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trajectory_map_path = root / manifest["trajectory_map_npz"]
        manifest["trajectory_map_bytes"] = trajectory_map_path.stat().st_size + 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        try:
            load_dataset_arrays(manifest_path)
        except RuntimeError as error:
            assert "Trajectory map byte count does not match" in str(error)
        else:
            raise AssertionError("trajectory-map size mismatch did not fail closed")


def test_stats_cache_is_bound_to_manifest_identity() -> None:
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
        first_identity = first["dataset_identity"]
        assert first_identity == load_dataset_identity(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["view_note"] = "identity changed"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        second = load_or_compute_stats(
            manifest_path,
            dataset=dataset,
            indices=indices,
        )
        second_identity = second["dataset_identity"]

        assert second_identity == load_dataset_identity(manifest_path)
        assert second_identity != first_identity
        cache_path = manifest_path.with_suffix(".stats.json")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["dataset_identity"] == second_identity


def test_schema_v2_rejects_changed_dataset_contract() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        manifest_path = _build_single_family_view(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_contract"] = {"target": {"nx": 4}}
        manifest["dataset_contract_fingerprint"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        try:
            load_dataset_arrays(manifest_path)
        except RuntimeError as error:
            assert "Dataset contract fingerprint does not match" in str(error)
        else:
            raise AssertionError("changed dataset contract did not fail closed")


def main() -> int:
    test_schema_v2_loads_shards_and_uses_preassigned_splits()
    test_hierarchical_epoch_uses_every_case_once_and_one_time_each()
    test_schema_v2_rejects_changed_trajectory_map()
    test_schema_v2_checks_declared_trajectory_map_size()
    test_stats_cache_is_bound_to_manifest_identity()
    test_schema_v2_rejects_changed_dataset_contract()
    print("[PASS] schema-v2 paper-corpus view and preassigned splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
