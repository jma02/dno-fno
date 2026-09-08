"""CPU tests for complete dataset batch artifacts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.artifact_io import load_npz, write_npz_atomic
from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    load_completed_batch,
    save_completed_batch,
    simulation_row_blocks,
)
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)


def _simulation_rows(
    *,
    frame_count: int = 2,
    depth: float = 1.0,
    offset: float = 0.0,
) -> SimulationRows:
    eta = np.arange(frame_count * 4, dtype=np.float64).reshape(frame_count, 4)
    eta = eta / 100.0 + offset
    return SimulationRows(
        eta,
        eta + 0.1,
        eta - 0.1,
        depth,
        np.linspace(0.0, 1.0, frame_count, dtype=np.float64),
    )


class BatchStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = batch_path(
            Path(self.temporary.name),
            family="tanaka",
            batch_id=7,
        )

    def test_sparse_completed_batch_round_trip_and_no_overwrite(self) -> None:
        rows = (
            _simulation_rows(),
            None,
            _simulation_rows(frame_count=3, depth=2.0, offset=1.0),
        )
        save_completed_batch(
            self.path,
            ("low", "high", "low"),
            rows,
            family_id=PhysicalFamilyId.TANAKA,
            seed=2026072210,
        )

        batch = load_completed_batch(self.path)
        self.assertEqual(batch.family_id, PhysicalFamilyId.TANAKA)
        self.assertEqual(batch.seed, 2026072210)
        self.assertEqual(load_npz(self.path)["seed"].dtype, np.dtype(np.int64))
        self.assertEqual(batch.parameter_group_ids, ("low", "high", "low"))
        np.testing.assert_array_equal(
            batch.accepted_simulations,
            np.asarray([True, False, True]),
        )
        assert batch.shard is not None
        np.testing.assert_array_equal(
            batch.shard["simulation_local_index"],
            np.asarray([0, 0, 2, 2, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            batch.shard["frame_index"],
            np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
        )
        self.assertEqual(batch.shard["eta"].dtype, np.dtype(np.float32))

        with self.assertRaises(FileExistsError):
            save_completed_batch(
                self.path,
                ("low",),
                (_simulation_rows(offset=9.0),),
                family_id=PhysicalFamilyId.TANAKA,
                seed=2026072210,
            )
        np.testing.assert_array_equal(
            load_completed_batch(self.path).accepted_simulations,
            np.asarray([True, False, True]),
        )

    def test_all_rejected_batch_has_no_shard(self) -> None:
        save_completed_batch(
            self.path,
            ("a", "b", "c"),
            (None, None, None),
            family_id=PhysicalFamilyId.JONSWAP_TMA,
            seed=2026072205,
        )

        batch = load_completed_batch(self.path)
        self.assertIsNone(batch.shard)
        np.testing.assert_array_equal(
            batch.accepted_simulations,
            np.zeros(3, dtype=np.bool_),
        )

    def test_save_rejects_invalid_stored_rows(self) -> None:
        valid = _simulation_rows()
        invalid_rows = (
            valid._replace(xi=np.zeros((2, 3), dtype=np.float64)),
            valid._replace(time=np.asarray([1.0, 0.0])),
            valid._replace(depth=0.0),
            valid._replace(
                eta=np.full((2, 4), np.finfo(np.float64).max, dtype=np.float64)
            ),
        )
        for index, rows in enumerate(invalid_rows):
            with self.subTest(index=index), self.assertRaises(ValueError):
                save_completed_batch(
                    self.path,
                    ("group",),
                    (rows,),
                    family_id=PhysicalFamilyId.STOKES,
                    seed=2026072204,
                )
            self.assertFalse(self.path.exists())

    def test_load_validates_frame_time_depth_and_complete_shard(self) -> None:
        save_completed_batch(
            self.path,
            ("group",),
            (_simulation_rows(frame_count=3),),
            family_id=PhysicalFamilyId.BENJAMIN_FEIR,
            seed=2026072210,
        )
        valid = load_npz(self.path)
        corruptions = (
            ("frame", "frame_index", np.asarray([0, 2, 1], dtype=np.int32)),
            ("time", "time", np.asarray([0.0, 1.0, 0.5], dtype=np.float64)),
            ("depth", "depth", np.asarray([1.0, 2.0, 1.0], dtype=np.float64)),
        )
        for label, field, replacement in corruptions:
            with self.subTest(field=field):
                arrays = dict(valid)
                arrays[field] = replacement
                corrupt_path = self.path.with_name(f"{label}.npz")
                write_npz_atomic(corrupt_path, arrays)
                with self.assertRaises(ValueError):
                    load_completed_batch(corrupt_path)

        arrays = dict(valid)
        del arrays["gxi"]
        incomplete_path = self.path.with_name("incomplete.npz")
        write_npz_atomic(incomplete_path, arrays)
        with self.assertRaisesRegex(ValueError, "missing shard arrays"):
            load_completed_batch(incomplete_path)

        arrays = dict(valid)
        arrays["simulation_results_json"] = np.asarray("legacy bookkeeping")
        unexpected_path = self.path.with_name("unexpected.npz")
        write_npz_atomic(unexpected_path, arrays)
        with self.assertRaisesRegex(ValueError, "unexpected arrays"):
            load_completed_batch(unexpected_path)

    def test_simulation_row_blocks_requires_sparse_ordered_blocks(self) -> None:
        self.assertEqual(
            simulation_row_blocks(
                np.asarray([0, 0, 2, 2, 2], dtype=np.int32),
                number_of_simulations=3,
            ),
            {0: (0, 2), 2: (2, 3)},
        )
        with self.assertRaisesRegex(ValueError, "ordered block"):
            simulation_row_blocks(
                np.asarray([0, 2, 1], dtype=np.int32),
                number_of_simulations=3,
            )
        with self.assertRaisesRegex(ValueError, "absent from the batch"):
            simulation_row_blocks(
                np.asarray([3], dtype=np.int32),
                number_of_simulations=3,
            )


if __name__ == "__main__":
    unittest.main()
