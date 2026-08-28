"""CPU tests for restartable valid-case generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    BatchStatus,
    ensure_proposal,
    inspect_batch,
    record_fatal_failure,
)
from solver.gen_data.pipeline.case_allocation import (
    AttemptAssignment,
    CaseKey,
    SampleCellTarget,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.case_checks import (
    CaseCheckResult,
    CaseCheck,
)
from solver.gen_data.pipeline.valid_case_generation import (
    DatasetGenerationSpec,
    dataset_generation_lock,
    generate_valid_cases,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.writer import (
    AcceptedCaseRows,
    CaseOutcome,
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
)


class InjectedInterruption(RuntimeError):
    """Controlled nonterminal interruption used by restart tests."""


def _run_spec(
    root: Path,
    *,
    case_targets: tuple[SampleCellTarget, ...] | None = None,
    batch_size: int = 2,
    maximum_attempts_per_accepted_case: int = 4,
    configuration: Mapping[str, object] | None = None,
) -> DatasetGenerationSpec:
    target_values = case_targets or (
        SampleCellTarget("low", 2),
        SampleCellTarget("moderate", 2),
    )
    return DatasetGenerationSpec(
        root=root,
        family_name="tanaka",
        family_id=PhysicalFamilyId.TANAKA,
        revision_id=1,
        split_id=SplitId.TRAIN,
        stream_id=3,
        case_targets=target_values,
        cell_codes={
            target.cell_id: index for index, target in enumerate(target_values)
        },
        batch_size=batch_size,
        first_attempt_index=0,
        maximum_attempts_per_accepted_case=(maximum_attempts_per_accepted_case),
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


def _proposal_arrays(
    spec: DatasetGenerationSpec,
    assignments: Sequence[AttemptAssignment],
    *,
    batch_id: int,
) -> dict[str, np.ndarray]:
    return build_proposal_arrays(
        assignments,
        tuple(
            {
                "schema": "fake_case_spec_v1",
                "case_id": assignment.case_key.case_id,
                "cell_id": assignment.cell_id,
                "attempt_index": assignment.case_key.attempt_index,
            }
            for assignment in assignments
        ),
        cell_codes=spec.cell_codes,
        batch_id=batch_id,
        metadata={"run": spec.to_json_record()},
    )


def _decision(*, accepted: bool) -> CaseCheckResult:
    reason = CaseCheck.OUTSIDE_SUPPORT
    return CaseCheckResult(
        required=reason,
        evaluated=reason,
        failed=CaseCheck.NONE if accepted else reason,
    )


def _outcome(
    assignment: AttemptAssignment,
    *,
    accepted: bool,
) -> CaseOutcome:
    rows = None
    if accepted:
        value = float(assignment.case_key.attempt_index + 1)
        field = np.full((1, 8), value, dtype=np.float64)
        rows = AcceptedCaseRows(
            eta=field,
            xi=field / 2.0,
            gxi=-field,
            depth=1.0,
            time=np.asarray([0.0], dtype=np.float64),
            selected_dense_index=np.asarray([0], dtype=np.int32),
        )
    return CaseOutcome(
        decision=_decision(accepted=accepted),
        rows=rows,
        metrics={"attempt_index": assignment.case_key.attempt_index},
    )


class FakeExecutor:
    """Deterministic executor with selected case-level rejections."""

    def __init__(
        self,
        spec: DatasetGenerationSpec,
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
        proposal = _proposal_arrays(
            self.spec,
            assignments,
            batch_id=batch_id,
        )
        paths = batch_paths_for_assignments(
            self.spec.root,
            assignments,
            family_name=self.spec.family_name,
            batch_id=batch_id,
        )
        outcomes = tuple(
            _outcome(
                assignment,
                accepted=(
                    assignment.case_key.attempt_index not in self.rejected_attempts
                ),
            )
            for assignment in assignments
        )
        commit_case_outcomes(
            paths,
            proposal,
            outcomes,
            metadata={
                "attempted": len(outcomes),
                "accepted": sum(outcome.decision.accepted for outcome in outcomes),
            },
        )
        return paths


class ValidCaseGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_rejection_schedules_same_cell_replacement_and_short_final_batch(
        self,
    ) -> None:
        spec = _run_spec(self.root)
        executor = FakeExecutor(spec, rejected_attempts=frozenset({1}))

        with mock.patch(
            "solver.gen_data.pipeline.valid_case_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_valid_cases(spec, executor)
            self.assertEqual(scanner.call_count, 1)

        self.assertTrue(state.complete)
        self.assertIsNone(state.pending)
        self.assertIsNone(state.terminal_failure)
        self.assertEqual(
            dict(state.accepted_by_cell),
            {"low": 2, "moderate": 2},
        )
        self.assertEqual(state.next_attempt_index, 5)
        self.assertEqual(state.next_batch_id, 3)
        self.assertEqual(
            [
                assignment.cell_id
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

        replay = generate_valid_cases(spec, MustNotRun())
        self.assertTrue(replay.complete)
        self.assertEqual(replay.next_attempt_index, 5)

    def test_rejection_loop_stops_at_cap_and_restart_does_not_execute(
        self,
    ) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 4),),
            batch_size=3,
            maximum_attempts_per_accepted_case=2,
        )
        executor = FakeExecutor(
            spec,
            rejected_attempts=frozenset({0, 1, 2, 4, 5, 6, 7}),
        )

        state = generate_valid_cases(spec, executor)

        self.assertFalse(state.complete)
        self.assertIsNone(state.pending)
        self.assertIsNone(state.terminal_failure)
        self.assertIsNotNone(state.attempt_limit_failure)
        assert state.attempt_limit_failure is not None
        self.assertEqual(
            state.attempt_limit_failure.exhausted_cells,
            ("low",),
        )
        self.assertIn("attempted=8", state.attempt_limit_failure.message)
        self.assertIn("ceiling=8", state.attempt_limit_failure.message)
        self.assertIn("accepted=1", state.attempt_limit_failure.message)
        self.assertIn("target=4", state.attempt_limit_failure.message)
        self.assertEqual(dict(state.attempted_by_cell), {"low": 8})
        self.assertEqual(dict(state.accepted_by_cell), {"low": 1})
        self.assertEqual(
            [len(assignments) for _, assignments in executor.calls],
            [3, 3, 2],
        )
        self.assertFalse(
            BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.split_id.value,
                batch_id=3,
            ).proposal.exists()
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

        restarted = generate_valid_cases(spec, MustNotRun())
        self.assertEqual(restarted, state)

    def test_pending_final_attempt_replays_before_cap_exhaustion(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1),),
            batch_size=1,
            maximum_attempts_per_accepted_case=1,
        )

        def interrupt_after_proposal(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            proposal = _proposal_arrays(spec, assignments, batch_id=batch_id)
            paths = batch_paths_for_assignments(
                spec.root,
                assignments,
                family_name=spec.family_name,
                batch_id=batch_id,
            )
            ensure_proposal(paths, proposal)
            raise InjectedInterruption("after final allowed proposal")

        with self.assertRaisesRegex(
            InjectedInterruption,
            "after final allowed proposal",
        ):
            generate_valid_cases(spec, interrupt_after_proposal)

        pending = scan_dataset_generation(spec)
        self.assertIsNotNone(pending.pending)
        self.assertIsNone(pending.attempt_limit_failure)
        self.assertEqual(dict(pending.attempted_by_cell), {"low": 1})

        executor = FakeExecutor(spec)
        completed = generate_valid_cases(spec, executor)
        self.assertTrue(completed.complete)
        self.assertIsNone(completed.attempt_limit_failure)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(dict(completed.attempted_by_cell), {"low": 1})

    def test_proposal_only_interruption_replays_exact_assignments(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1), SampleCellTarget("moderate", 1)),
        )
        interrupted_calls: list[tuple[AttemptAssignment, ...]] = []

        def interrupt_after_proposal(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            interrupted_calls.append(assignments)
            proposal = _proposal_arrays(spec, assignments, batch_id=batch_id)
            paths = batch_paths_for_assignments(
                spec.root,
                assignments,
                family_name=spec.family_name,
                batch_id=batch_id,
            )
            ensure_proposal(paths, proposal)
            raise InjectedInterruption("after proposal")

        with self.assertRaisesRegex(InjectedInterruption, "after proposal"):
            generate_valid_cases(spec, interrupt_after_proposal)

        pending = scan_dataset_generation(spec).pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.status, BatchStatus.PROPOSED)
        self.assertEqual(pending.assignments, interrupted_calls[0])

        resumed_executor = FakeExecutor(spec)
        with mock.patch(
            "solver.gen_data.pipeline.valid_case_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_valid_cases(spec, resumed_executor)
            self.assertEqual(scanner.call_count, 1)
        self.assertTrue(state.complete)
        self.assertEqual(resumed_executor.calls[0][1], interrupted_calls[0])
        self.assertEqual(state, scan_dataset_generation(spec))

    def test_shard_interruption_replays_without_replacing_the_shard(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1), SampleCellTarget("moderate", 1)),
        )
        interrupted_executor = FakeExecutor(spec)
        with (
            mock.patch(
                "solver.gen_data.pipeline.writer.commit_batch",
                side_effect=InjectedInterruption("after shard"),
            ),
            self.assertRaisesRegex(InjectedInterruption, "after shard"),
        ):
            generate_valid_cases(spec, interrupted_executor)

        pending = scan_dataset_generation(spec).pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.status, BatchStatus.SHARD_WRITTEN)
        shard_bytes = pending.paths.shard.read_bytes()

        with mock.patch(
            "solver.gen_data.pipeline.valid_case_generation.scan_dataset_generation",
            wraps=scan_dataset_generation,
        ) as scanner:
            state = generate_valid_cases(spec, FakeExecutor(spec))
            self.assertEqual(scanner.call_count, 1)
        self.assertTrue(state.complete)
        self.assertEqual(pending.paths.shard.read_bytes(), shard_bytes)
        self.assertEqual(state, scan_dataset_generation(spec))

    def test_terminal_failure_stops_without_scheduling_a_later_batch(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1), SampleCellTarget("moderate", 1)),
        )
        calls = 0

        def fail_batch(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            nonlocal calls
            calls += 1
            proposal = _proposal_arrays(spec, assignments, batch_id=batch_id)
            paths = batch_paths_for_assignments(
                spec.root,
                assignments,
                family_name=spec.family_name,
                batch_id=batch_id,
            )
            ensure_proposal(paths, proposal)
            record_fatal_failure(
                paths,
                phase="construct",
                exception_type="RuntimeError",
                message="injected fatal error",
                telemetry={"completed_cases": 0},
            )
            return paths

        state = generate_valid_cases(spec, fail_batch)
        self.assertFalse(state.complete)
        self.assertIsNotNone(state.terminal_failure)
        self.assertEqual(calls, 1)
        self.assertEqual(state, scan_dataset_generation(spec))

        replay = generate_valid_cases(spec, fail_batch)
        self.assertIsNotNone(replay.terminal_failure)
        self.assertEqual(calls, 1)
        self.assertFalse(
            BatchPaths.for_batch(
                spec.root,
                family=spec.family_name,
                split=spec.split_id.value,
                batch_id=1,
            ).proposal.exists()
        )

    def test_incremental_state_validates_the_persisted_assignments(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1), SampleCellTarget("moderate", 1)),
            batch_size=1,
        )

        def persist_a_different_cell(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            persisted = (
                AttemptAssignment(
                    case_key=assignments[0].case_key,
                    cell_id="moderate",
                ),
            )
            paths = batch_paths_for_assignments(
                spec.root,
                persisted,
                family_name=spec.family_name,
                batch_id=batch_id,
            )
            commit_case_outcomes(
                paths,
                _proposal_arrays(spec, persisted, batch_id=batch_id),
                (_outcome(persisted[0], accepted=True),),
                metadata={"injected_wrong_cell": True},
            )
            return paths

        with self.assertRaisesRegex(
            RuntimeError,
            "persisted proposal assignments differ",
        ):
            generate_valid_cases(spec, persist_a_different_cell)
        with self.assertRaisesRegex(
            RuntimeError,
            "deterministic case schedule",
        ):
            scan_dataset_generation(spec)

    def test_scanner_rejects_batch_gaps_and_wrong_stream(self) -> None:
        gap_root = self.root / "gap"
        gap_spec = _run_spec(
            gap_root,
            case_targets=(SampleCellTarget("low", 1), SampleCellTarget("moderate", 1)),
        )
        assignments = (
            AttemptAssignment(
                case_key=CaseKey(
                    family_id=int(gap_spec.family_id),
                    revision_id=gap_spec.revision_id,
                    split_id=gap_spec.split_id,
                    stream_id=gap_spec.stream_id,
                    attempt_index=0,
                ),
                cell_id="low",
            ),
        )
        gap_paths = BatchPaths.for_batch(
            gap_spec.root,
            family=gap_spec.family_name,
            split=gap_spec.split_id.value,
            batch_id=1,
        )
        ensure_proposal(
            gap_paths,
            _proposal_arrays(gap_spec, assignments, batch_id=1),
        )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            scan_dataset_generation(gap_spec)

        stream_root = self.root / "stream"
        stream_spec = _run_spec(
            stream_root,
            case_targets=(SampleCellTarget("low", 1),),
            batch_size=1,
        )
        wrong_assignment = AttemptAssignment(
            case_key=CaseKey(
                family_id=int(stream_spec.family_id),
                revision_id=stream_spec.revision_id,
                split_id=stream_spec.split_id,
                stream_id=stream_spec.stream_id + 1,
                attempt_index=0,
            ),
            cell_id="low",
        )
        wrong_proposal = build_proposal_arrays(
            (wrong_assignment,),
            ({"schema": "wrong_stream_v1"},),
            cell_codes=stream_spec.cell_codes,
            batch_id=0,
            metadata={},
        )
        wrong_paths = BatchPaths.for_batch(
            stream_spec.root,
            family=stream_spec.family_name,
            split=stream_spec.split_id.value,
            batch_id=0,
        )
        ensure_proposal(wrong_paths, wrong_proposal)
        with self.assertRaisesRegex(RuntimeError, "stream_id"):
            scan_dataset_generation(stream_spec)

    def test_scanner_does_not_trust_tampered_result_acceptance(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1),),
            batch_size=1,
        )
        state = generate_valid_cases(spec, FakeExecutor(spec))
        result_path = state.committed[0].result
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["cases"][0]["accepted"] = False
        result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "quality masks"):
            scan_dataset_generation(spec)

    def test_generation_spec_is_strict_and_immutable(self) -> None:
        first = _run_spec(self.root / "first", configuration={"nx": 64})
        self.assertEqual(
            first.to_json_record()["maximum_attempts_per_accepted_case"],
            4,
        )
        self.assertEqual(
            first.to_json_record()["schema"],
            "paper_dataset_accepted_quota_run_v2",
        )
        for invalid_limit in (0, -1, True, 1.5):
            with self.subTest(invalid_limit=invalid_limit):
                with self.assertRaises(ValueError):
                    replace(
                        first,
                        root=self.root / f"invalid_limit_{invalid_limit}",
                        maximum_attempts_per_accepted_case=invalid_limit,
                    )
        for field_name in (
            "revision_id",
            "stream_id",
            "first_attempt_index",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(TypeError, "integer"):
                    replace(first, **{field_name: 1.5})

        source_configuration = {"contract": {"nx": 64}}
        immutable = _run_spec(
            self.root / "immutable",
            configuration=source_configuration,
        )
        source_configuration["contract"]["nx"] = 128
        frozen_contract = immutable.configuration["contract"]
        self.assertIsInstance(frozen_contract, Mapping)
        assert isinstance(frozen_contract, Mapping)
        self.assertEqual(frozen_contract["nx"], 64)
        with self.assertRaises(TypeError):
            frozen_contract["nx"] = 128  # type: ignore[index]

    def test_single_writer_lock_fails_fast(self) -> None:
        spec = _run_spec(self.root)
        with dataset_generation_lock(spec):
            with self.assertRaisesRegex(RuntimeError, "another"):
                generate_valid_cases(spec, FakeExecutor(spec))

        state = scan_dataset_generation(spec)
        self.assertFalse(state.complete)
        self.assertEqual(state.next_batch_id, 0)
        self.assertEqual(state.next_attempt_index, 0)

    def test_executor_must_return_standard_terminal_paths(self) -> None:
        spec = _run_spec(
            self.root,
            case_targets=(SampleCellTarget("low", 1),),
            batch_size=1,
        )

        def proposal_only(
            assignments: tuple[AttemptAssignment, ...],
            *,
            batch_id: int,
        ) -> BatchPaths:
            proposal = _proposal_arrays(spec, assignments, batch_id=batch_id)
            paths = batch_paths_for_assignments(
                spec.root,
                assignments,
                family_name=spec.family_name,
                batch_id=batch_id,
            )
            ensure_proposal(paths, proposal)
            return paths

        with self.assertRaisesRegex(RuntimeError, "terminal batch record"):
            generate_valid_cases(spec, proposal_only)
        self.assertEqual(
            inspect_batch(
                BatchPaths.for_batch(
                    spec.root,
                    family=spec.family_name,
                    split=spec.split_id.value,
                    batch_id=0,
                )
            ).status,
            BatchStatus.PROPOSED,
        )


if __name__ == "__main__":
    unittest.main()
