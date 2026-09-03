"""CPU tests for complete dataset batch artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_artifacts import SimulationResult
from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    load_completed_batch,
    save_completed_batch,
)
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit
from solver.gen_data.pipeline.types import BatchPlanArrays, DatasetShardArrays


def _plan() -> BatchPlanArrays:
    return {
        "family_id": np.asarray(2, dtype=np.int16),
        "dataset_split": np.asarray(DatasetSplit.TRAIN.value),
        "parameter_group_id": np.asarray(["low", "high", "low"]),
        "simulation_spec_json": np.asarray(
            [json.dumps({"amplitude": value}) for value in (0.1, 0.2, 0.3)]
        ),
    }


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


def _records() -> tuple[SimulationResult, ...]:
    return (
        SimulationResult(
            accepted=True,
            failed_checks=(),
            metrics={"maximum_error": 1e-5},
        ),
        SimulationResult(
            accepted=False,
            failed_checks=("integration_failure",),
            metrics={"maximum_error": None},
        ),
        SimulationResult(
            accepted=True,
            failed_checks=(),
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
        )

        batch = load_completed_batch(self.path)
        self.assertEqual(batch.simulations, _records())
        self.assertIsNotNone(batch.shard)
        assert batch.shard is not None
        np.testing.assert_array_equal(batch.shard["eta"], _shard()["eta"])
        with self.assertRaises(FileExistsError):
            save_completed_batch(
                self.path,
                plan=_plan(),
                shard=_shard(),
                simulations=_records(),
            )

    def test_all_rejected_batch_needs_no_row_arrays(self) -> None:
        simulations = tuple(
            SimulationResult(
                accepted=False,
                failed_checks=("integration_failure",),
                metrics={},
            )
            for _ in range(3)
        )
        save_completed_batch(
            self.path,
            plan=_plan(),
            shard=None,
            simulations=simulations,
        )
        self.assertIsNone(load_completed_batch(self.path).shard)

    def test_failed_check_names_are_validated_and_ordered(self) -> None:
        result = SimulationResult(
            accepted=False,
            failed_checks=("outside_support", "nonfinite_state"),
            metrics={},
        )

        self.assertEqual(
            result.failed_checks,
            ("nonfinite_state", "outside_support"),
        )
        with self.assertRaisesRegex(ValueError, "unknown failed checks"):
            SimulationResult(
                accepted=False,
                failed_checks=("made_up_failure",),
                metrics={},
            )

    def test_inconsistent_rows_are_rejected_before_writing(self) -> None:
        wrong = list(_records())
        wrong[1] = SimulationResult(
            accepted=True,
            failed_checks=(),
            metrics={},
        )
        with self.assertRaisesRegex(ValueError, "rows do not match"):
            save_completed_batch(
                self.path,
                plan=_plan(),
                shard=_shard(),
                simulations=wrong,
            )
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
