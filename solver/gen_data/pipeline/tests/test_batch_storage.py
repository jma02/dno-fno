"""CPU tests for transactional dataset batch storage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    BatchStatus,
    SimulationCommitRecord,
    commit_batch,
    save_batch_plan,
    save_shard,
    inspect_batch,
    record_fatal_failure,
)


def _batch_plan() -> dict[str, np.ndarray]:
    specifications = [
        json.dumps({"amplitude": value}, sort_keys=True) for value in (0.1, 0.2, 0.3)
    ]
    return {
        "family_id": np.asarray(2, dtype=np.int16),
        "revision_id": np.asarray(1, dtype=np.int16),
        "split_id": np.asarray(0, dtype=np.uint8),
        "batch_id": np.asarray(7, dtype=np.int64),
        "simulation_id": np.asarray([70, 71, 72], dtype=np.int64),
        "cell_id": np.asarray([0, 1, 0], dtype=np.int32),
        "root_seed": np.asarray([11, 11, 11], dtype=np.uint64),
        "stream_id": np.asarray([2, 2, 2], dtype=np.uint32),
        "attempt_index": np.asarray([0, 1, 2], dtype=np.uint64),
        "simulation_spec_json": np.asarray(specifications),
        "metadata_json": np.asarray(json.dumps({"seed": 11}, sort_keys=True)),
        "phase_right": np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=np.float64,
        ),
    }


def _shard_arrays() -> dict[str, np.ndarray]:
    simulation_local_index = np.asarray([0, 0, 2, 2, 2], dtype=np.int32)
    frame_index = np.asarray([0, 1, 0, 1, 2], dtype=np.int32)
    selected_dense_index = np.asarray([0, 10, 0, 5, 10], dtype=np.int32)
    rows = simulation_local_index.size
    field = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 100.0
    return {
        "eta": field,
        "xi": field + np.float32(0.1),
        "gxi": field - np.float32(0.1),
        "depth": np.asarray([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64),
        "time": np.asarray([0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float64),
        "simulation_local_index": simulation_local_index,
        "frame_index": frame_index,
        "selected_dense_index": selected_dense_index,
    }


def _simulation_records() -> tuple[SimulationCommitRecord, ...]:
    return (
        SimulationCommitRecord(
            simulation_id=70,
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=0,
            row_count=2,
            metrics={"maximum_error": 1e-5},
        ),
        SimulationCommitRecord(
            simulation_id=71,
            accepted=False,
            required_bits=32,
            evaluated_bits=32,
            failed_bits=32,
            first_row=-1,
            row_count=0,
            metrics={"maximum_error": None},
        ),
        SimulationCommitRecord(
            simulation_id=72,
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
        self.root = Path(self.temporary.name)
        self.paths = BatchPaths.for_batch(
            self.root,
            family="tanaka",
            split="train",
            batch_id=7,
        )

    def test_interruption_boundaries_replay_to_one_commit(self) -> None:
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.EMPTY)

        proposal = _batch_plan()
        save_batch_plan(self.paths, proposal)
        proposal_bytes = self.paths.batch_plan.read_bytes()
        self.assertEqual(
            inspect_batch(self.paths).status,
            BatchStatus.PLAN_SAVED,
        )
        save_batch_plan(self.paths, proposal)
        self.assertEqual(self.paths.batch_plan.read_bytes(), proposal_bytes)

        shard = _shard_arrays()
        save_shard(self.paths, shard)
        shard_bytes = self.paths.shard.read_bytes()
        inspection = inspect_batch(self.paths)
        self.assertEqual(inspection.status, BatchStatus.SHARD_WRITTEN)
        save_shard(self.paths, shard)
        self.assertEqual(self.paths.shard.read_bytes(), shard_bytes)

        commit_batch(
            self.paths,
            simulations=_simulation_records(),
            metadata={"elapsed_seconds": 12.5},
        )
        inspection = inspect_batch(self.paths)
        self.assertEqual(inspection.status, BatchStatus.COMMITTED)
        self.assertEqual(inspection.simulations, _simulation_records())
        with self.assertRaisesRegex(RuntimeError, "already committed"):
            commit_batch(
                self.paths,
                simulations=_simulation_records(),
                metadata={"elapsed_seconds": 12.5},
            )

    def test_rejected_simulations_never_own_partial_rows(self) -> None:
        save_batch_plan(self.paths, _batch_plan())
        save_shard(self.paths, _shard_arrays())

        wrong = list(_simulation_records())
        wrong[1] = SimulationCommitRecord(
            simulation_id=71,
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=2,
            row_count=1,
            metrics={},
        )
        with self.assertRaisesRegex(ValueError, "row ownership"):
            commit_batch(self.paths, simulations=wrong, metadata={})

        shard = _shard_arrays()
        shard["simulation_local_index"] = np.asarray(
            [0, 2, 0, 2, 2],
            dtype=np.int32,
        )
        different_paths = BatchPaths.for_batch(
            self.root,
            family="tanaka",
            split="train",
            batch_id=8,
        )
        proposal = _batch_plan()
        proposal["batch_id"] = np.asarray(8, dtype=np.int64)
        save_batch_plan(different_paths, proposal)
        with self.assertRaisesRegex(ValueError, "ordered block"):
            save_shard(different_paths, shard)

    def test_all_rejected_batch_commits_without_a_shard(self) -> None:
        save_batch_plan(self.paths, _batch_plan())
        simulations = tuple(
            SimulationCommitRecord(
                simulation_id=simulation_id,
                accepted=False,
                required_bits=32,
                evaluated_bits=32,
                failed_bits=32,
                first_row=-1,
                row_count=0,
                metrics={"maximum_error": None},
            )
            for simulation_id in (70, 71, 72)
        )
        commit_batch(self.paths, simulations=simulations, metadata={"accepted": 0})
        self.assertFalse(self.paths.shard.exists())
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.COMMITTED)

    def test_existing_proposal_or_shard_must_replay_exactly(self) -> None:
        proposal = _batch_plan()
        save_batch_plan(self.paths, proposal)
        changed = _batch_plan()
        changed["cell_id"] = np.asarray([1, 1, 0], dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "differs from replayed arrays"):
            save_batch_plan(self.paths, changed)

        shard = _shard_arrays()
        save_shard(self.paths, shard)
        changed_shard = _shard_arrays()
        changed_shard["eta"] = changed_shard["eta"].copy()
        changed_shard["eta"][0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(RuntimeError, "differs from replayed arrays"):
            save_shard(self.paths, changed_shard)

    def test_orphan_and_malformed_results_are_detected(self) -> None:
        self.paths.result.parent.mkdir(parents=True, exist_ok=True)
        self.paths.result.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "without its batch plan"):
            inspect_batch(self.paths)

        self.paths.result.unlink()
        save_batch_plan(self.paths, _batch_plan())
        save_shard(self.paths, _shard_arrays())
        commit_batch(self.paths, simulations=_simulation_records(), metadata={})
        payload = json.loads(self.paths.result.read_text(encoding="utf-8"))
        payload["simulations"] = payload["simulations"][:-1]
        self.paths.result.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "every planned simulation"):
            inspect_batch(self.paths)

    def test_fatal_failure_is_terminal_and_bound_to_proposal(self) -> None:
        save_batch_plan(self.paths, _batch_plan())
        record_fatal_failure(
            self.paths,
            phase="integrate",
            exception_type="RuntimeError",
            message="injected",
            telemetry={"completed_steps": 3},
        )
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.FAILED)
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            save_batch_plan(self.paths, _batch_plan())
        with self.assertRaisesRegex(RuntimeError, "failed batch"):
            commit_batch(self.paths, simulations=_simulation_records(), metadata={})

    def test_nonfinite_json_and_invalid_selected_times_are_rejected(self) -> None:
        save_batch_plan(self.paths, _batch_plan())
        shard = _shard_arrays()
        shard["selected_dense_index"] = np.asarray(
            [0, 0, 0, 5, 10],
            dtype=np.int32,
        )
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            save_shard(self.paths, shard)

        with self.assertRaises(ValueError):
            record_fatal_failure(
                self.paths,
                phase="integrate",
                exception_type="RuntimeError",
                message="bad telemetry",
                telemetry={"residual": float("nan")},
            )


if __name__ == "__main__":
    unittest.main()
