"""CPU tests for complete dataset batch artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_artifacts import SimulationCommitRecord
from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    load_completed_batch,
    save_completed_batch,
)
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit, SimulationKey
from solver.gen_data.pipeline.types import BatchPlanArrays, DatasetShardArrays


def _plan() -> BatchPlanArrays:
    return {
        "family_id": np.asarray(2, dtype=np.int16),
        "dataset_split": np.asarray(DatasetSplit.TRAIN.value),
        "parameter_group_id": np.asarray(["low", "high", "low"]),
        "worker_stream_id": np.asarray([2, 2, 2], dtype=np.uint32),
        "attempt_index": np.asarray([0, 1, 2], dtype=np.uint64),
        "simulation_spec_json": np.asarray(
            [json.dumps({"amplitude": value}) for value in (0.1, 0.2, 0.3)]
        ),
        "metadata_json": np.asarray(json.dumps({"seed": 11})),
    }


def _simulation_id(attempt_index: int) -> int:
    return SimulationKey(
        family_id=2,
        dataset_split=DatasetSplit.TRAIN,
        worker_stream_id=2,
        attempt_index=attempt_index,
    ).simulation_id


def _shard() -> DatasetShardArrays:
    simulation_local_index = np.asarray([0, 0, 2, 2, 2], dtype=np.int32)
    frame_index = np.asarray([0, 1, 0, 1, 2], dtype=np.int32)
    field = np.arange(20, dtype=np.float32).reshape(5, 4) / 100.0
    return {
        "eta": field,
        "xi": field + np.float32(0.1),
        "gxi": field - np.float32(0.1),
        "depth": np.asarray([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64),
        "time": np.asarray([0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float64),
        "simulation_local_index": simulation_local_index,
        "frame_index": frame_index,
    }


def _records() -> tuple[SimulationCommitRecord, ...]:
    return (
        SimulationCommitRecord(
            simulation_id=_simulation_id(0),
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=0,
            row_count=2,
            metrics={"maximum_error": 1e-5},
        ),
        SimulationCommitRecord(
            simulation_id=_simulation_id(1),
            accepted=False,
            required_bits=32,
            evaluated_bits=32,
            failed_bits=32,
            first_row=-1,
            row_count=0,
            metrics={"maximum_error": None},
        ),
        SimulationCommitRecord(
            simulation_id=_simulation_id(2),
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=2,
            row_count=3,
            metrics={"maximum_error": 2e-5},
        ),
    )


class BatchStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = batch_path(
            Path(self.temporary.name),
            family="tanaka",
            split="train",
            batch_id=7,
        )

    def test_completed_batch_round_trip(self) -> None:
        save_completed_batch(
            self.path,
            plan=_plan(),
            shard=_shard(),
            simulations=_records(),
            metadata={"elapsed_seconds": 12.5},
        )

        batch = load_completed_batch(self.path)
        self.assertEqual(batch.simulations, _records())
        self.assertEqual(batch.metadata, {"elapsed_seconds": 12.5})
        self.assertIsNotNone(batch.shard)
        assert batch.shard is not None
        np.testing.assert_array_equal(batch.shard["eta"], _shard()["eta"])
        with self.assertRaises(FileExistsError):
            save_completed_batch(
                self.path,
                plan=_plan(),
                shard=_shard(),
                simulations=_records(),
                metadata={},
            )

    def test_all_rejected_batch_needs_no_row_arrays(self) -> None:
        simulations = tuple(
            SimulationCommitRecord(
                simulation_id=_simulation_id(attempt_index),
                accepted=False,
                required_bits=32,
                evaluated_bits=32,
                failed_bits=32,
                first_row=-1,
                row_count=0,
                metrics={},
            )
            for attempt_index in range(3)
        )
        save_completed_batch(
            self.path,
            plan=_plan(),
            shard=None,
            simulations=simulations,
            metadata={},
        )
        self.assertIsNone(load_completed_batch(self.path).shard)

    def test_inconsistent_rows_are_rejected_before_writing(self) -> None:
        wrong = list(_records())
        wrong[1] = SimulationCommitRecord(
            simulation_id=_simulation_id(1),
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=2,
            row_count=1,
            metrics={},
        )
        with self.assertRaisesRegex(ValueError, "rows do not match"):
            save_completed_batch(
                self.path,
                plan=_plan(),
                shard=_shard(),
                simulations=wrong,
                metadata={},
            )
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
