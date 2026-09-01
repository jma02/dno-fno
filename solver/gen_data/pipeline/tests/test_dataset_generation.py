"""CPU tests for whole-batch dataset generation and restart behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import batch_path
from solver.gen_data.pipeline.dataset_generation import (
    MAX_RETRIES_PER_PARAMETER_GROUP,
    DatasetChunkConfig,
    dataset_generation_lock,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    DatasetSplit,
    ParameterGroupTarget,
    PhysicalFamilyId,
    build_next_attempt_batch,
)
from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheck,
    SimulationCheckResult,
)
from solver.gen_data.pipeline.types import BatchPlanArrays
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)


class InjectedInterruption(RuntimeError):
    """Controlled interruption before a completed batch is written."""


def _chunk_config(
    root: Path,
    *,
    simulation_targets: tuple[ParameterGroupTarget, ...] | None = None,
    batch_size: int = 2,
    configuration: Mapping[str, object] | None = None,
) -> DatasetChunkConfig:
    return DatasetChunkConfig(
        root=root,
        family_name="tanaka",
        family_id=PhysicalFamilyId.TANAKA,
        dataset_split=DatasetSplit.TRAIN,
        worker_stream_id=3,
        simulation_targets=simulation_targets
        or (
            ParameterGroupTarget("low", 2),
            ParameterGroupTarget("moderate", 2),
        ),
        batch_size=batch_size,
        first_attempt_index=0,
        configuration=dict(configuration or {"nx": 8}),
    )


def _batch_plan(
    chunk_config: DatasetChunkConfig,
    assignments: Sequence[AttemptAssignment],
) -> BatchPlanArrays:
    return build_batch_plan(
        assignments,
        tuple(
            {
                "parameter_group": assignment.parameter_group_id,
                "attempt_index": assignment.simulation_key.attempt_index,
            }
            for assignment in assignments
        ),
        metadata={"generation": chunk_config.to_json_record()},
    )


def _outcome(
    assignment: AttemptAssignment,
    *,
    accepted: bool,
) -> SimulationOutcome:
    required = SimulationCheck.OUTSIDE_SUPPORT
    rows = None
    if accepted:
        value = float(assignment.simulation_key.attempt_index + 1)
        field = np.full((1, 8), value, dtype=np.float64)
        rows = AcceptedSimulationRows(
            eta=field,
            xi=field / 2.0,
            gxi=-field,
            depth=1.0,
            time=np.asarray([0.0], dtype=np.float64),
        )
    return SimulationOutcome(
        decision=SimulationCheckResult(
            required=required,
            evaluated=required,
            failed=SimulationCheck.NONE if accepted else required,
        ),
        rows=rows,
        metrics={"attempt_index": assignment.simulation_key.attempt_index},
    )


class FakeExecutor:
    """Save deterministic batches while rejecting selected attempts."""

    def __init__(
        self,
        chunk_config: DatasetChunkConfig,
        *,
        rejected_attempts: frozenset[int] = frozenset(),
    ) -> None:
        self.chunk_config = chunk_config
        self.rejected_attempts = rejected_attempts
        self.calls: list[tuple[int, tuple[AttemptAssignment, ...]]] = []

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> Path:
        self.calls.append((batch_id, assignments))
        path = batch_path(
            self.chunk_config.root,
            family=self.chunk_config.family_name,
            split=self.chunk_config.dataset_split.value,
            batch_id=batch_id,
        )
        outcomes = tuple(
            _outcome(
                assignment,
                accepted=(
                    assignment.simulation_key.attempt_index
                    not in self.rejected_attempts
                ),
            )
            for assignment in assignments
        )
        commit_simulation_outcomes(
            path,
            _batch_plan(self.chunk_config, assignments),
            outcomes,
            metadata={
                "attempted": len(outcomes),
                "accepted": sum(outcome.decision.accepted for outcome in outcomes),
            },
        )
        return path


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_rejections_schedule_replacements_and_resume_completed_batches(
        self,
    ) -> None:
        chunk_config = _chunk_config(self.root)
        executor = FakeExecutor(
            chunk_config,
            rejected_attempts=frozenset({1}),
        )

        state = generate_simulations(chunk_config, executor)

        self.assertTrue(state.complete)
        self.assertEqual(
            dict(state.accepted_simulation_counts),
            {"low": 2, "moderate": 2},
        )
        self.assertEqual(
            [
                assignment.parameter_group_id
                for _, assignments in executor.calls
                for assignment in assignments
            ],
            ["low", "moderate", "moderate", "low", "moderate"],
        )
        self.assertEqual([len(batch) for _, batch in executor.calls], [2, 2, 1])
        self.assertEqual(state, scan_dataset_generation(chunk_config))

        class MustNotRun:
            def __call__(
                self,
                assignments: tuple[AttemptAssignment, ...],
                *,
                batch_id: int,
            ) -> Path:
                raise AssertionError((assignments, batch_id))

        self.assertTrue(generate_simulations(chunk_config, MustNotRun()).complete)

    def test_interrupted_batch_is_absent_and_rerun_from_the_start(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
        )
        interrupted_assignments: tuple[AttemptAssignment, ...] | None = None

        def interrupt(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> Path:
            nonlocal interrupted_assignments
            interrupted_assignments = assignments
            raise InjectedInterruption(f"batch {batch_id} stopped")

        with self.assertRaisesRegex(InjectedInterruption, "batch 0 stopped"):
            generate_simulations(chunk_config, interrupt)

        path = batch_path(
            self.root,
            family="tanaka",
            split="train",
            batch_id=0,
        )
        self.assertFalse(path.exists())
        interrupted_state = scan_dataset_generation(chunk_config)
        self.assertEqual(interrupted_state.completed_batches, ())
        self.assertEqual(sum(interrupted_state.simulation_attempt_counts.values()), 0)

        executor = FakeExecutor(chunk_config)
        completed = generate_simulations(chunk_config, executor)
        self.assertTrue(completed.complete)
        self.assertEqual(executor.calls[0][1], interrupted_assignments)

    def test_generation_stops_after_retry_limit(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 32),),
            batch_size=32,
        )
        executor = FakeExecutor(
            chunk_config,
            rejected_attempts=frozenset(range(64)),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "attempt limits reached.*low: accepted=0/32, attempts=64/64",
        ):
            generate_simulations(chunk_config, executor)

        state = scan_dataset_generation(chunk_config)
        self.assertFalse(state.complete)
        self.assertEqual(dict(state.simulation_attempt_counts), {"low": 64})
        self.assertEqual(len(state.completed_batches), 2)

    def test_scanner_rejects_batch_gaps(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 1),),
            batch_size=1,
        )
        executor = FakeExecutor(chunk_config)
        assignments = build_next_attempt_batch(
            chunk_config.simulation_targets,
            {},
            {},
            chunk_config.maximum_attempts_by_parameter_group,
            family_id=int(chunk_config.family_id),
            dataset_split=chunk_config.dataset_split,
            worker_stream_id=chunk_config.worker_stream_id,
            first_attempt_index=0,
            batch_size=1,
        )
        executor(assignments, batch_id=1)

        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            scan_dataset_generation(chunk_config)

    def test_generation_config_records_retry_limit(self) -> None:
        chunk_config = _chunk_config(self.root, configuration={"nx": 64})
        self.assertEqual(
            chunk_config.to_json_record()["maximum_retries_per_parameter_group"],
            MAX_RETRIES_PER_PARAMETER_GROUP,
        )

    def test_single_writer_lock_fails_fast(self) -> None:
        chunk_config = _chunk_config(self.root)
        with dataset_generation_lock(chunk_config):
            with self.assertRaisesRegex(RuntimeError, "another"):
                generate_simulations(chunk_config, FakeExecutor(chunk_config))


if __name__ == "__main__":
    unittest.main()
