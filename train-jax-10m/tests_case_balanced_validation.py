"""CPU tests for schema-v2 case-balanced validation sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import build_case_balanced_validation_indices  # noqa: E402


def _schema_v2_fixture() -> dict[str, np.ndarray]:
    rows_per_case = np.asarray([1, 200, 200, 16], dtype=np.int64)
    trajectory_index = np.repeat(
        np.arange(rows_per_case.size, dtype=np.int32),
        rows_per_case,
    )
    family_by_case = np.arange(rows_per_case.size, dtype=np.int16)
    return {
        "eta": np.zeros((trajectory_index.size, 4), dtype=np.float32),
        "trajectory_index": trajectory_index,
        "family_id": family_by_case[trajectory_index],
        "split_id": np.ones(trajectory_index.size, dtype=np.uint8),
        "accepted_mask": np.ones(trajectory_index.size, dtype=np.bool_),
    }


def test_schema_v2_validation_selects_one_time_per_case() -> None:
    dataset = _schema_v2_fixture()
    stored_rows = np.arange(dataset["eta"].shape[0], dtype=np.int64)

    selected = build_case_balanced_validation_indices(
        dataset,
        stored_rows,
        seed=23,
    )
    repeated = build_case_balanced_validation_indices(
        dataset,
        stored_rows,
        seed=23,
    )

    assert np.array_equal(selected, repeated)
    assert selected.shape == (4,)
    assert np.array_equal(
        np.sort(dataset["trajectory_index"][selected]),
        np.arange(4, dtype=np.int32),
    )
    assert np.bincount(dataset["family_id"][selected], minlength=4).tolist() == [
        1,
        1,
        1,
        1,
    ]
    assert np.bincount(dataset["family_id"], minlength=4).tolist() == [
        1,
        200,
        200,
        16,
    ]


def test_schema_v1_validation_preserves_stored_row_indices() -> None:
    dataset = _schema_v2_fixture()
    del dataset["split_id"]
    stored_rows = np.asarray([9, 2, 6], dtype=np.int64)

    selected = build_case_balanced_validation_indices(
        dataset,
        stored_rows,
        seed=23,
    )

    assert selected is stored_rows


def main() -> int:
    test_schema_v2_validation_selects_one_time_per_case()
    test_schema_v1_validation_preserves_stored_row_indices()
    print("[PASS] schema-v2 validation is case balanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
