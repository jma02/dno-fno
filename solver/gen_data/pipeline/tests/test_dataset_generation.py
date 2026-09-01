"""CPU tests for restartable dataset generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    BatchStatus,
    save_batch_plan,
    inspect_batch,
    record_fatal_failure,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    SimulationKey,
    ParameterGroupTarget,
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.types import BatchPlanArrays
from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheckResult,
    SimulationCheck,
)
from solver.gen_data.pipeline.dataset_generation import (
    MAX_RETRIES_PER_PARAMETER_GROUP,
    DatasetChunkConfig,
    dataset_generation_lock,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)


class InjectedInterruption(RuntimeError):
    """Controlled nonterminal interruption used by restart tests."""


def _chunk_config(
    root: Path,
    *,
    simulation_targets: tuple[ParameterGroupTarget, ...] | None = None,
    batch_size: int = 2,
    configuration: Mapping[str, object] | None = None,
) -> DatasetChunkConfig:
    target_values = simulation_targets or (
        ParameterGroupTarget("low", 2),
        ParameterGroupTarget("moderate", 2),
    )
    return DatasetChunkConfig(
        root=root,
        family_name="tanaka",
        family_id=PhysicalFamilyId.TANAKA,
        dataset_split=DatasetSplit.TRAIN,
        worker_stream_id=3,
        simulation_targets=target_values,
        parameter_group_codes={
            target.parameter_group_id: index
            for index, target in enumerate(target_values)
        },
        batch_size=batch_size,
        first_attempt_index=0,
        configuration=dict(
            configuration
            or {
                "contract": {
                    "role": "reduced_wiring_evidence_only",
                    "nx": 8,
                }
            }
        ),
    )


def _batch_plan(
    spec: DatasetChunkConfig,
    assignments: Sequence[AttemptAssignment],
) -> BatchPlanArrays:
    return build_batch_plan(
        assignments,
        tuple(
            {
                "schema": "fake_simulation_spec_v1",
                "simulation_id": assignment.simulation_key.simulation_id,
                "parameter_group_id": assignment.parameter_group_id,
                "attempt_index": assignment.simulation_key.attempt_index,
            }
            for assignment in assignments
        ),
        parameter_group_codes=spec.parameter_group_codes,
        metadata={"run": spec.to_json_record()},
    )


def _decision(*, accepted: bool) -> SimulationCheckResult:
    reason = SimulationCheck.OUTSIDE_SUPPORT
    return SimulationCheckResult(
        required=reason,
        evaluated=reason,
        failed=SimulationCheck.NONE if accepted else reason,
    )


def _outcome(
    assignment: AttemptAssignment,
    *,
    accepted: bool,
) -> SimulationOutcome:
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
        decision=_decision(accepted=accepted),
        rows=rows,
        metrics={"attempt_index": assignment.simulation_key.attempt_index},
    )


class FakeExecutor:
    """Deterministic executor with selected simulation-level rejections."""

    def __init__(
        self,
        spec: DatasetChunkConfig,
        *,
        rejected_attempts: frozenset[int] = frozenset(),
    ) -> None:
        self.spec = spec
        self.rejected_attempts = rejected_attempts
        self.calls: list[tuple[int, tuple[AttemptAssignment, ...]]] = []

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths:
        self.calls.append((batch_id, assignments))
        proposal = _batch_plan(
            self.spec,
            assignments,
        )
        paths = BatchPaths.for_batch(
            self.spec.root,
            family=self.spec.family_name,
            split=self.spec.dataset_split.value,
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
            paths,
            proposal,
            outcomes,
            metadata={
                "attempted": len(outcomes),
                "accepted": sum(outcome.decision.accepted for outcome in outcomes),
            },
        )
        return paths


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_rejection_schedules_same_cell_replacement_and_short_final_batch(
        self,
    ) -> None:
        spec = _chunk_config(self.root)
        executor = FakeExecutor(spec, rejected_attempts=frozenset({1}))

        with mock.patch(
            "solver.gen_data.pipeline.dataset_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_simulations(spec, executor)
            self.assertEqual(scanner.call_count, 1)

        self.assertTrue(state.complete)
        self.assertIsNone(state.pending_batch)
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
        self.assertEqual(
            [len(assignments) for _, assignments in executor.calls],
            [2, 2, 1],
        )
        self.assertEqual(state, scan_dataset_generation(spec))

        class MustNotRun:
            def __call__(
                self,
                assignments: tuple[AttemptAssignment, ...],
                *,
                batch_id: int,
            ) -> BatchPaths:
                raise AssertionError((assignments, batch_id))

        replay = generate_simulations(spec, MustNotRun())
        self.assertTrue(replay.complete)

    def test_generation_stops_after_32_retries(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 32),),
            batch_size=32,
        )
        executor = FakeExecutor(
            spec,
            rejected_attempts=frozenset(range(64)),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "attempt limits reached.*low: accepted=0/32, attempts=64/64",
        ):
            generate_simulations(spec, executor)

        state = scan_dataset_generation(spec)
        self.assertFalse(state.complete)
        self.assertEqual(dict(state.accepted_simulation_counts), {"low": 0})
        self.assertEqual(dict(state.simulation_attempt_counts), {"low": 64})
        self.assertEqual(
            [len(assignments) for _, assignments in executor.calls],
            [32, 32],
        )
        self.assertFalse(
            BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=2,
            ).batch_plan.exists()
        )

    def test_pending_last_allowed_batch_is_replayed(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 32),),
            batch_size=32,
        )
        reject_first_batch = FakeExecutor(
            spec,
            rejected_attempts=frozenset(range(32)),
        )

        def interrupt_after_proposal(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            if batch_id == 0:
                return reject_first_batch(assignments, batch_id=batch_id)
            proposal = _batch_plan(spec, assignments)
            paths = BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=batch_id,
            )
            save_batch_plan(paths, proposal)
            raise InjectedInterruption("after final allowed batch plan")

        with self.assertRaisesRegex(
            InjectedInterruption,
            "after final allowed batch plan",
        ):
            generate_simulations(spec, interrupt_after_proposal)

        pending = scan_dataset_generation(spec)
        self.assertIsNotNone(pending.pending_batch)
        self.assertEqual(dict(pending.simulation_attempt_counts), {"low": 64})

        executor = FakeExecutor(spec)
        completed = generate_simulations(spec, executor)
        self.assertTrue(completed.complete)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(dict(completed.simulation_attempt_counts), {"low": 64})

    def test_proposal_only_interruption_replays_exact_assignments(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
        )
        interrupted_calls: list[tuple[AttemptAssignment, ...]] = []

        def interrupt_after_proposal(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            interrupted_calls.append(assignments)
            proposal = _batch_plan(spec, assignments)
            paths = BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=batch_id,
            )
            save_batch_plan(paths, proposal)
            raise InjectedInterruption("after batch plan")

        with self.assertRaisesRegex(InjectedInterruption, "after batch plan"):
            generate_simulations(spec, interrupt_after_proposal)

        pending = scan_dataset_generation(spec).pending_batch
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.assignments, interrupted_calls[0])

        resumed_executor = FakeExecutor(spec)
        with mock.patch(
            "solver.gen_data.pipeline.dataset_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_simulations(spec, resumed_executor)
            self.assertEqual(scanner.call_count, 1)
        self.assertTrue(state.complete)
        self.assertEqual(resumed_executor.calls[0][1], interrupted_calls[0])
        self.assertEqual(state, scan_dataset_generation(spec))

    def test_shard_interruption_replays_without_replacing_the_shard(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
        )
        interrupted_executor = FakeExecutor(spec)
        with (
            mock.patch(
                "solver.gen_data.pipeline.writer.commit_batch",
                side_effect=InjectedInterruption("after shard"),
            ),
            self.assertRaisesRegex(InjectedInterruption, "after shard"),
        ):
            generate_simulations(spec, interrupted_executor)

        pending = scan_dataset_generation(spec).pending_batch
        self.assertIsNotNone(pending)
        assert pending is not None
        shard_bytes = pending.paths.shard.read_bytes()

        with mock.patch(
            "solver.gen_data.pipeline.dataset_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_simulations(spec, FakeExecutor(spec))
            self.assertEqual(scanner.call_count, 1)
        self.assertTrue(state.complete)
        self.assertEqual(pending.paths.shard.read_bytes(), shard_bytes)
        self.assertEqual(state, scan_dataset_generation(spec))

    def test_failed_batch_stops_without_scheduling_a_later_batch(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
        )
        calls = 0

        def fail_batch(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            nonlocal calls
            calls += 1
            proposal = _batch_plan(spec, assignments)
            paths = BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=batch_id,
            )
            save_batch_plan(paths, proposal)
            record_fatal_failure(
                paths,
                phase="construct",
                exception_type="RuntimeError",
                message="injected fatal error",
                telemetry={"completed_simulations": 0},
            )
            return paths

        with self.assertRaisesRegex(RuntimeError, "batch 0 failed"):
            generate_simulations(spec, fail_batch)
        self.assertEqual(calls, 1)

        with self.assertRaisesRegex(RuntimeError, "batch 0 failed"):
            generate_simulations(spec, fail_batch)
        self.assertEqual(calls, 1)
        self.assertFalse(
            BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=1,
            ).batch_plan.exists()
        )

    def test_incremental_state_validates_the_persisted_assignments(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
            batch_size=1,
        )

        def persist_a_different_cell(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            persisted = (
                AttemptAssignment(
                    simulation_key=assignments[0].simulation_key,
                    parameter_group_id="moderate",
                ),
            )
            paths = BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=batch_id,
            )
            commit_simulation_outcomes(
                paths,
                _batch_plan(spec, persisted),
                (_outcome(persisted[0], accepted=True),),
                metadata={"injected_wrong_cell": True},
            )
            return paths

        with self.assertRaisesRegex(
            RuntimeError,
            "persisted batch-plan assignments differ",
        ):
            generate_simulations(spec, persist_a_different_cell)
        with self.assertRaisesRegex(
            RuntimeError,
            "deterministic simulation schedule",
        ):
            scan_dataset_generation(spec)

    def test_scanner_rejects_batch_gaps_and_wrong_worker_stream(self) -> None:
        gap_root = self.root / "gap"
        gap_spec = _chunk_config(
            gap_root,
            simulation_targets=(
                ParameterGroupTarget("low", 1),
                ParameterGroupTarget("moderate", 1),
            ),
        )
        assignments = (
            AttemptAssignment(
                simulation_key=SimulationKey(
                    family_id=int(gap_spec.family_id),
                    dataset_split=gap_spec.dataset_split,
                    worker_stream_id=gap_spec.worker_stream_id,
                    attempt_index=0,
                ),
                parameter_group_id="low",
            ),
        )
        gap_paths = BatchPaths.for_batch(
            gap_spec.root,
            family=gap_spec.family_name,
            split=gap_spec.dataset_split.value,
            batch_id=1,
        )
        save_batch_plan(
            gap_paths,
            _batch_plan(gap_spec, assignments),
        )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            scan_dataset_generation(gap_spec)

        worker_stream_root = self.root / "worker_stream"
        worker_stream_spec = _chunk_config(
            worker_stream_root,
            simulation_targets=(ParameterGroupTarget("low", 1),),
            batch_size=1,
        )
        wrong_assignment = AttemptAssignment(
            simulation_key=SimulationKey(
                family_id=int(worker_stream_spec.family_id),
                dataset_split=worker_stream_spec.dataset_split,
                worker_stream_id=worker_stream_spec.worker_stream_id + 1,
                attempt_index=0,
            ),
            parameter_group_id="low",
        )
        wrong_proposal = build_batch_plan(
            (wrong_assignment,),
            ({"schema": "wrong_worker_stream_v1"},),
            parameter_group_codes=worker_stream_spec.parameter_group_codes,
            metadata={},
        )
        wrong_paths = BatchPaths.for_batch(
            worker_stream_spec.root,
            family=worker_stream_spec.family_name,
            split=worker_stream_spec.dataset_split.value,
            batch_id=0,
        )
        save_batch_plan(wrong_paths, wrong_proposal)
        with self.assertRaisesRegex(RuntimeError, "worker_stream_id"):
            scan_dataset_generation(worker_stream_spec)

    def test_scanner_does_not_trust_tampered_result_acceptance(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 1),),
            batch_size=1,
        )
        state = generate_simulations(spec, FakeExecutor(spec))
        result_path = state.committed[0].result
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["simulations"][0]["accepted"] = False
        result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "quality masks"):
            scan_dataset_generation(spec)

    def test_generation_config_records_retry_limit(self) -> None:
        first = _chunk_config(self.root / "first", configuration={"nx": 64})
        self.assertEqual(
            first.to_json_record()["maximum_retries_per_parameter_group"],
            MAX_RETRIES_PER_PARAMETER_GROUP,
        )

    def test_single_writer_lock_fails_fast(self) -> None:
        spec = _chunk_config(self.root)
        with dataset_generation_lock(spec):
            with self.assertRaisesRegex(RuntimeError, "another"):
                generate_simulations(spec, FakeExecutor(spec))

        state = scan_dataset_generation(spec)
        self.assertFalse(state.complete)

    def test_executor_must_return_standard_terminal_paths(self) -> None:
        spec = _chunk_config(
            self.root,
            simulation_targets=(ParameterGroupTarget("low", 1),),
            batch_size=1,
        )

        def proposal_only(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            proposal = _batch_plan(spec, assignments)
            paths = BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.dataset_split.value,
                batch_id=batch_id,
            )
            save_batch_plan(paths, proposal)
            return paths

        with self.assertRaisesRegex(RuntimeError, "terminal batch record"):
            generate_simulations(spec, proposal_only)
        self.assertEqual(
            inspect_batch(
                BatchPaths.for_batch(
                    spec.root,
                    family=spec.family_name,
                    split=spec.dataset_split.value,
                    batch_id=0,
                )
            ).status,
            BatchStatus.PLAN_SAVED,
        )


if __name__ == "__main__":
    unittest.main()
