"""CPU tests for transactional paper-corpus batch storage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    BatchStatus,
    CaseCommitRecord,
    commit_batch,
    ensure_proposal,
    ensure_shard,
    file_sha256,
    inspect_batch,
    record_fatal_failure,
)


FINGERPRINT = "a" * 64


def _proposal_arrays() -> dict[str, np.ndarray]:
    specifications = [
        json.dumps({"amplitude": value}, sort_keys=True) for value in (0.1, 0.2, 0.3)
    ]
    return {
        "config_fingerprint": np.asarray(FINGERPRINT),
        "family_id": np.asarray(2, dtype=np.int16),
        "revision_id": np.asarray(1, dtype=np.int16),
        "split_id": np.asarray(0, dtype=np.uint8),
        "batch_id": np.asarray(7, dtype=np.int64),
        "case_id": np.asarray([70, 71, 72], dtype=np.int64),
        "cell_id": np.asarray([0, 1, 0], dtype=np.int32),
        "case_spec_json": np.asarray(specifications),
        "metadata_json": np.asarray(json.dumps({"seed": 11}, sort_keys=True)),
        "phase_right": np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=np.float64,
        ),
    }


def _shard_arrays(proposal_sha256: str) -> dict[str, np.ndarray]:
    case_local_index = np.asarray([0, 0, 2, 2, 2], dtype=np.int32)
    frame_index = np.asarray([0, 1, 0, 1, 2], dtype=np.int32)
    selected_dense_index = np.asarray([0, 10, 0, 5, 10], dtype=np.int32)
    rows = case_local_index.size
    field = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 100.0
    return {
        "eta": field,
        "xi": field + np.float32(0.1),
        "gxi": field - np.float32(0.1),
        "depth": np.asarray([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64),
        "time": np.asarray([0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float64),
        "case_local_index": case_local_index,
        "frame_index": frame_index,
        "selected_dense_index": selected_dense_index,
        "config_fingerprint": np.asarray(FINGERPRINT),
        "proposal_sha256": np.asarray(proposal_sha256),
    }


def _case_records() -> tuple[CaseCommitRecord, ...]:
    return (
        CaseCommitRecord(
            case_id=70,
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=0,
            row_count=2,
            metrics={"maximum_error": 1e-5},
        ),
        CaseCommitRecord(
            case_id=71,
            accepted=False,
            required_bits=32,
            evaluated_bits=32,
            failed_bits=32,
            first_row=-1,
            row_count=0,
            metrics={"maximum_error": None},
        ),
        CaseCommitRecord(
            case_id=72,
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=2,
            row_count=3,
            metrics={"maximum_error": 2e-5},
        ),
    )


class ArchiveTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = BatchPaths.under(
            self.root,
            family="tanaka",
            split="train",
            batch_id=7,
        )

    def test_interruption_boundaries_replay_to_one_commit(self) -> None:
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.EMPTY)

        proposal = _proposal_arrays()
        proposal_hash = ensure_proposal(self.paths, proposal)
        proposal_bytes = self.paths.proposal.read_bytes()
        self.assertEqual(
            inspect_batch(
                self.paths,
                expected_fingerprint=FINGERPRINT,
            ).status,
            BatchStatus.PROPOSED,
        )
        self.assertEqual(ensure_proposal(self.paths, proposal), proposal_hash)
        self.assertEqual(self.paths.proposal.read_bytes(), proposal_bytes)

        shard = _shard_arrays(proposal_hash)
        shard_hash = ensure_shard(self.paths, shard)
        shard_bytes = self.paths.shard.read_bytes()
        inspection = inspect_batch(self.paths)
        self.assertEqual(inspection.status, BatchStatus.SHARD_WRITTEN)
        self.assertEqual(inspection.shard_sha256, shard_hash)
        self.assertEqual(ensure_shard(self.paths, shard), shard_hash)
        self.assertEqual(self.paths.shard.read_bytes(), shard_bytes)

        result_hash = commit_batch(
            self.paths,
            cases=_case_records(),
            metadata={"elapsed_seconds": 12.5},
        )
        result_bytes = self.paths.result.read_bytes()
        inspection = inspect_batch(self.paths)
        self.assertEqual(inspection.status, BatchStatus.COMMITTED)
        self.assertEqual(inspection.proposal_sha256, proposal_hash)
        self.assertEqual(inspection.shard_sha256, shard_hash)
        self.assertEqual(
            commit_batch(
                self.paths,
                cases=_case_records(),
                metadata={"elapsed_seconds": 12.5},
            ),
            result_hash,
        )
        self.assertEqual(self.paths.result.read_bytes(), result_bytes)

    def test_rejected_cases_never_own_partial_rows(self) -> None:
        proposal_hash = ensure_proposal(self.paths, _proposal_arrays())
        ensure_shard(self.paths, _shard_arrays(proposal_hash))

        wrong = list(_case_records())
        wrong[1] = CaseCommitRecord(
            case_id=71,
            accepted=True,
            required_bits=63,
            evaluated_bits=63,
            failed_bits=0,
            first_row=2,
            row_count=1,
            metrics={},
        )
        with self.assertRaisesRegex(ValueError, "row ownership"):
            commit_batch(self.paths, cases=wrong, metadata={})

        shard = _shard_arrays(proposal_hash)
        shard["case_local_index"] = np.asarray(
            [0, 2, 0, 2, 2],
            dtype=np.int32,
        )
        different_paths = BatchPaths.under(
            self.root,
            family="tanaka",
            split="train",
            batch_id=8,
        )
        proposal = _proposal_arrays()
        proposal["batch_id"] = np.asarray(8, dtype=np.int64)
        next_hash = ensure_proposal(different_paths, proposal)
        shard["proposal_sha256"] = np.asarray(next_hash)
        with self.assertRaisesRegex(ValueError, "ordered block"):
            ensure_shard(different_paths, shard)

    def test_all_rejected_batch_commits_without_a_shard(self) -> None:
        ensure_proposal(self.paths, _proposal_arrays())
        cases = tuple(
            CaseCommitRecord(
                case_id=case_id,
                accepted=False,
                required_bits=32,
                evaluated_bits=32,
                failed_bits=32,
                first_row=-1,
                row_count=0,
                metrics={"maximum_error": None},
            )
            for case_id in (70, 71, 72)
        )
        commit_batch(self.paths, cases=cases, metadata={"accepted": 0})
        self.assertFalse(self.paths.shard.exists())
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.COMMITTED)

    def test_existing_proposal_or_shard_must_replay_exactly(self) -> None:
        proposal = _proposal_arrays()
        proposal_hash = ensure_proposal(self.paths, proposal)
        changed = _proposal_arrays()
        changed["cell_id"] = np.asarray([1, 1, 0], dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "proposal differs"):
            ensure_proposal(self.paths, changed)

        shard = _shard_arrays(proposal_hash)
        ensure_shard(self.paths, shard)
        changed_shard = _shard_arrays(proposal_hash)
        changed_shard["eta"] = changed_shard["eta"].copy()
        changed_shard["eta"][0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(RuntimeError, "shard differs"):
            ensure_shard(self.paths, changed_shard)

    def test_hash_and_orphan_corruption_are_detected(self) -> None:
        self.paths.result.parent.mkdir(parents=True, exist_ok=True)
        self.paths.result.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "without its proposal"):
            inspect_batch(self.paths)

        self.paths.result.unlink()
        proposal_hash = ensure_proposal(self.paths, _proposal_arrays())
        ensure_shard(self.paths, _shard_arrays(proposal_hash))
        commit_batch(self.paths, cases=_case_records(), metadata={})
        payload = json.loads(self.paths.result.read_text(encoding="utf-8"))
        payload["shard_sha256"] = "b" * 64
        self.paths.result.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "different shard"):
            inspect_batch(self.paths)

    def test_fatal_failure_is_terminal_and_bound_to_proposal(self) -> None:
        ensure_proposal(self.paths, _proposal_arrays())
        failure_hash = record_fatal_failure(
            self.paths,
            phase="integrate",
            exception_type="RuntimeError",
            message="injected",
            telemetry={"completed_steps": 3},
        )
        self.assertEqual(failure_hash, file_sha256(self.paths.failure))
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.FAILED)
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            ensure_proposal(self.paths, _proposal_arrays())
        with self.assertRaisesRegex(RuntimeError, "failed batch"):
            commit_batch(self.paths, cases=_case_records(), metadata={})

    def test_nonfinite_json_and_invalid_selected_times_are_rejected(self) -> None:
        proposal_hash = ensure_proposal(self.paths, _proposal_arrays())
        shard = _shard_arrays(proposal_hash)
        shard["selected_dense_index"] = np.asarray(
            [0, 0, 0, 5, 10],
            dtype=np.int32,
        )
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            ensure_shard(self.paths, shard)

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
