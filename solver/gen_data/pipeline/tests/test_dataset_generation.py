"""CPU tests for whole-batch dataset generation and restart behavior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import batch_path, save_completed_batch
from solver.gen_data.pipeline.dataset_generation import (
    BatchGenerator,
    GenerationResult,
    generate_simulations,
)
from solver.gen_data.pipeline.types import (
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)


FAMILY_NAME = "tanaka"
FAMILY_ID = PhysicalFamilyId.TANAKA
DATASET_SPLIT = DatasetSplit.TRAIN


class InjectedInterruption(RuntimeError):
    """Controlled interruption before a completed batch is written."""


def _generate(
    root: Path,
    accepted_targets: Mapping[str, int],
    batch_size: int,
    generate_batch: BatchGenerator,
) -> GenerationResult:
    return generate_simulations(
        root,
        family_id=FAMILY_ID,
        dataset_split=DATASET_SPLIT,
        accepted_targets=accepted_targets,
        batch_size=batch_size,
        generate_batch=generate_batch,
    )


def _fake_generator(
    *,
    rejected_attempts: frozenset[int] = frozenset(),
) -> tuple[BatchGenerator, list[tuple[int, tuple[str, ...]]]]:
    calls: list[tuple[int, tuple[str, ...]]] = []

    def generate_batch(
        parameter_group_ids: tuple[str, ...],
        first_attempt_number: int,
        output_path: Path,
    ) -> None:
        calls.append((first_attempt_number, parameter_group_ids))
        rows_by_simulation: list[SimulationRows | None] = []
        for offset in range(len(parameter_group_ids)):
            attempt_number = first_attempt_number + offset
            rows: SimulationRows | None = None
            if attempt_number not in rejected_attempts:
                value = float(attempt_number + 1)
                field = np.full((1, 8), value, dtype=np.float64)
                rows = SimulationRows(
                    eta=field,
                    xi=field / 2.0,
                    gxi=-field,
                    depth=1.0,
                    time=np.asarray([0.0], dtype=np.float64),
                )
            rows_by_simulation.append(rows)

        save_completed_batch(
            output_path,
            parameter_group_ids,
            rows_by_simulation,
            family_id=FAMILY_ID,
            dataset_split=DATASET_SPLIT,
        )

    return generate_batch, calls


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_rejections_schedule_replacements_and_resume_completed_batches(
        self,
    ) -> None:
        targets = {"low": 2, "moderate": 2}
        generator, calls = _fake_generator(rejected_attempts=frozenset({1}))

        attempts, paths = _generate(self.root, targets, 2, generator)

        self.assertEqual(attempts, {"low": 2, "moderate": 3})
        self.assertEqual(
            [group for _, groups in calls for group in groups],
            ["low", "moderate", "moderate", "low", "moderate"],
        )
        self.assertEqual([len(groups) for _, groups in calls], [2, 2, 1])
        self.assertEqual([first_attempt for first_attempt, _ in calls], [0, 2, 4])

        def must_not_run(
            parameter_group_ids: tuple[str, ...],
            first_attempt_number: int,
            output_path: Path,
        ) -> None:
            raise AssertionError(
                (parameter_group_ids, first_attempt_number, output_path)
            )

        self.assertEqual(
            _generate(self.root, targets, 2, must_not_run),
            (attempts, paths),
        )

    def test_interrupted_batch_is_absent_and_rerun_from_the_start(self) -> None:
        targets = {"low": 1, "moderate": 1}
        interrupted_groups: tuple[str, ...] | None = None
        interrupted_first_attempt: int | None = None

        def interrupt(
            parameter_group_ids: tuple[str, ...],
            first_attempt_number: int,
            output_path: Path,
        ) -> None:
            del output_path
            nonlocal interrupted_first_attempt, interrupted_groups
            interrupted_groups = parameter_group_ids
            interrupted_first_attempt = first_attempt_number
            raise InjectedInterruption("batch stopped")

        with self.assertRaisesRegex(InjectedInterruption, "batch stopped"):
            _generate(self.root, targets, 2, interrupt)

        path = batch_path(
            self.root,
            family=FAMILY_NAME,
            split=DATASET_SPLIT.value,
            batch_id=0,
        )
        self.assertFalse(path.exists())

        generator, calls = _fake_generator()
        _, paths = _generate(self.root, targets, 2, generator)
        self.assertEqual(calls[0], (interrupted_first_attempt, interrupted_groups))
        self.assertEqual(paths, (path,))

    def test_generation_stops_after_retry_limit(self) -> None:
        targets = {"low": 32}
        generator, _ = _fake_generator(rejected_attempts=frozenset(range(96)))

        with self.assertRaisesRegex(
            RuntimeError,
            "attempt limits reached.*low: accepted=0/32, attempts=96/96",
        ):
            _generate(self.root, targets, 32, generator)

        self.assertEqual(
            len(tuple((self.root / "batches" / FAMILY_NAME / "train").glob("*.npz"))),
            3,
        )

    def test_scanner_rejects_batch_gaps(self) -> None:
        targets = {"low": 1}
        generator, _ = _fake_generator()
        generator(
            ("low",),
            0,
            batch_path(
                self.root,
                family=FAMILY_NAME,
                split=DATASET_SPLIT.value,
                batch_id=1,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            _generate(self.root, targets, 1, generator)

    def test_scanner_orders_double_digit_batch_ids_numerically(self) -> None:
        targets = {"low": 12}
        generator, _ = _fake_generator()
        completed = _generate(self.root, targets, 1, generator)

        self.assertEqual(
            tuple(path.stem for path in completed[1]),
            tuple(f"batch_{batch_id:06d}" for batch_id in range(12)),
        )


if __name__ == "__main__":
    unittest.main()
