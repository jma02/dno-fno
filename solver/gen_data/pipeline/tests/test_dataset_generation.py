"""CPU tests for whole-batch dataset generation and restart behavior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import batch_path
from solver.gen_data.pipeline.dataset_generation import (
    DatasetChunkConfig,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.simulation_allocation import (
    DatasetSplit,
    PhysicalFamilyId,
    select_next_parameter_groups,
)
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
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
    simulation_targets: Mapping[str, int] | None = None,
    batch_size: int = 2,
) -> DatasetChunkConfig:
    return DatasetChunkConfig(
        root=root,
        family_name="tanaka",
        family_id=PhysicalFamilyId.TANAKA,
        dataset_split=DatasetSplit.TRAIN,
        simulation_targets=simulation_targets or {"low": 2, "moderate": 2},
        batch_size=batch_size,
    )


def _batch_plan(
    parameter_group_ids: tuple[str, ...],
    *,
    first_attempt_number: int,
    chunk_config: DatasetChunkConfig,
) -> BatchPlanArrays:
    return build_batch_plan(
        parameter_group_ids,
        tuple(
            {
                "parameter_group": parameter_group_id,
                "attempt_number": first_attempt_number + offset,
            }
            for offset, parameter_group_id in enumerate(parameter_group_ids)
        ),
        family_id=int(chunk_config.family_id),
        dataset_split=chunk_config.dataset_split,
    )


def _outcome(
    attempt_number: int,
    *,
    accepted: bool,
) -> SimulationOutcome:
    rows = None
    if accepted:
        value = float(attempt_number + 1)
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
            accepted=accepted,
            outside_support=not accepted,
        ),
        rows=rows,
        metrics={"attempt_number": attempt_number},
    )


class FakeGenerator:
    """Save deterministic batches while rejecting selected attempts."""

    def __init__(
        self,
        chunk_config: DatasetChunkConfig,
        *,
        rejected_attempts: frozenset[int] = frozenset(),
    ) -> None:
        self.chunk_config = chunk_config
        self.rejected_attempts = rejected_attempts
        self.calls: list[tuple[int, int, tuple[str, ...]]] = []

    def __call__(
        self,
        parameter_group_ids: tuple[str, ...],
        *,
        first_attempt_number: int,
        batch_id: int,
    ) -> Path:
        self.calls.append((batch_id, first_attempt_number, parameter_group_ids))
        path = batch_path(
            self.chunk_config.root,
            family=self.chunk_config.family_name,
            split=self.chunk_config.dataset_split.value,
            batch_id=batch_id,
        )
        outcomes = tuple(
            _outcome(
                first_attempt_number + offset,
                accepted=(first_attempt_number + offset not in self.rejected_attempts),
            )
            for offset in range(len(parameter_group_ids))
        )
        commit_simulation_outcomes(
            path,
            _batch_plan(
                parameter_group_ids,
                first_attempt_number=first_attempt_number,
                chunk_config=self.chunk_config,
            ),
            outcomes,
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
        generator = FakeGenerator(
            chunk_config,
            rejected_attempts=frozenset({1}),
        )

        state = generate_simulations(chunk_config, generator)

        self.assertTrue(state.complete)
        self.assertEqual(
            dict(state.accepted_simulation_counts),
            {"low": 2, "moderate": 2},
        )
        self.assertEqual(
            [
                parameter_group_id
                for _, _, parameter_group_ids in generator.calls
                for parameter_group_id in parameter_group_ids
            ],
            ["low", "moderate", "moderate", "low", "moderate"],
        )
        self.assertEqual(
            [len(parameter_group_ids) for _, _, parameter_group_ids in generator.calls],
            [2, 2, 1],
        )
        self.assertEqual(
            [first_attempt_number for _, first_attempt_number, _ in generator.calls],
            [0, 2, 4],
        )
        self.assertEqual(state, scan_dataset_generation(chunk_config))

        class MustNotRun:
            def __call__(
                self,
                parameter_group_ids: tuple[str, ...],
                *,
                first_attempt_number: int,
                batch_id: int,
            ) -> Path:
                raise AssertionError(
                    (parameter_group_ids, first_attempt_number, batch_id)
                )

        self.assertTrue(generate_simulations(chunk_config, MustNotRun()).complete)

    def test_interrupted_batch_is_absent_and_rerun_from_the_start(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets={"low": 1, "moderate": 1},
        )
        interrupted_parameter_groups: tuple[str, ...] | None = None
        interrupted_first_attempt: int | None = None

        def interrupt(
            parameter_group_ids: tuple[str, ...],
            *,
            first_attempt_number: int,
            batch_id: int,
        ) -> Path:
            nonlocal interrupted_first_attempt, interrupted_parameter_groups
            interrupted_parameter_groups = parameter_group_ids
            interrupted_first_attempt = first_attempt_number
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

        generator = FakeGenerator(chunk_config)
        completed = generate_simulations(chunk_config, generator)
        self.assertTrue(completed.complete)
        self.assertEqual(generator.calls[0][1], interrupted_first_attempt)
        self.assertEqual(generator.calls[0][2], interrupted_parameter_groups)

    def test_generation_stops_after_retry_limit(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets={"low": 32},
            batch_size=32,
        )
        generator = FakeGenerator(
            chunk_config,
            rejected_attempts=frozenset(range(64)),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "attempt limits reached.*low: accepted=0/32, attempts=64/64",
        ):
            generate_simulations(chunk_config, generator)

        state = scan_dataset_generation(chunk_config)
        self.assertFalse(state.complete)
        self.assertEqual(dict(state.simulation_attempt_counts), {"low": 64})
        self.assertEqual(len(state.completed_batches), 2)

    def test_scanner_rejects_batch_gaps(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets={"low": 1},
            batch_size=1,
        )
        generator = FakeGenerator(chunk_config)
        parameter_group_ids = select_next_parameter_groups(
            chunk_config.simulation_targets,
            {},
            {},
            chunk_config.maximum_attempts_by_parameter_group,
            batch_size=1,
        )
        generator(parameter_group_ids, first_attempt_number=0, batch_id=1)

        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            scan_dataset_generation(chunk_config)

    def test_scanner_orders_double_digit_batch_ids_numerically(self) -> None:
        chunk_config = _chunk_config(
            self.root,
            simulation_targets={"low": 12},
            batch_size=1,
        )
        completed = generate_simulations(chunk_config, FakeGenerator(chunk_config))

        self.assertEqual(
            tuple(path.stem for path in completed.completed_batches),
            tuple(f"batch_{batch_id:06d}" for batch_id in range(12)),
        )
        self.assertEqual(completed, scan_dataset_generation(chunk_config))


if __name__ == "__main__":
    unittest.main()
