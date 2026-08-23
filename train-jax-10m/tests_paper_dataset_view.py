"""CPU integration test for schema-v2 paper-dataset views."""
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
    load_dataset_arrays,
    load_dataset_identity,
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
            dataset["split_id"],
            np.asarray([0, 0, 1, 1], dtype=np.uint8),
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
    test_stats_cache_is_bound_to_manifest_identity()
    test_direct_npz_dataset_is_rejected()
    print("[PASS] schema-v2 paper-dataset view and preassigned splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
