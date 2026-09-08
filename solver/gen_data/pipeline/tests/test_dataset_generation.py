"""CPU tests for whole-batch dataset generation and restart behavior."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.dataset_generation import (
    BatchGenerator,
    GenerationResult,
    generate_simulations,
)
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
    RequestedSimulationsPerGroup,
)


FAMILY_NAME = "tanaka"
FAMILY_ID = PhysicalFamilyId.TANAKA
SEED = 2026072210


class InjectedInterruption(RuntimeError):
    """Controlled interruption before a completed batch is written."""


def _generate(
    root: Path,
    requested_simulations_per_group: RequestedSimulationsPerGroup,
    batch_size: int,
    generate_batch: BatchGenerator,
) -> GenerationResult:
    return generate_simulations(
        root,
        family_id=FAMILY_ID,
        seed=SEED,
        requested_simulations_per_group=requested_simulations_per_group,
        batch_size=batch_size,
        generate_batch=generate_batch,
    )


def _fake_generator(
    *,
    rejected_attempts: frozenset[int] = frozenset(),
    interrupt_after_batches: int | None = None,
) -> tuple[BatchGenerator, list[tuple[int, tuple[str, ...]]]]:
    calls: list[tuple[int, tuple[str, ...]]] = []

    def generate_batch(
        parameter_group_ids: tuple[str, ...],
        first_attempt_number: int,
        output_path: Path,
    ) -> None:
        if len(calls) == interrupt_after_batches:
            raise InjectedInterruption("batch stopped")
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
            seed=SEED,
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
        targets = {"main_m1_q0": 2, "main_m1_q1": 2}
        generator, calls = _fake_generator(rejected_attempts=frozenset({1}))

        attempts, paths = _generate(self.root, targets, 2, generator)

        self.assertEqual(attempts, {"main_m1_q0": 3, "main_m1_q1": 2})
        self.assertEqual(
            calls,
            [
                (0, ("main_m1_q0", "main_m1_q0")),
                (2, ("main_m1_q0",)),
                (3, ("main_m1_q1", "main_m1_q1")),
            ],
        )
        must_not_run, _ = _fake_generator(interrupt_after_batches=0)
        self.assertEqual(
            _generate(self.root, targets, 2, must_not_run),
            (attempts, paths),
        )

    def test_every_interrupted_batch_resumes_identically(self) -> None:
        targets = {
            "main_m1_q0": 3,
            "main_m1_q1": 2,
            "main_m2_q0": 0,
        }
        rejected = frozenset({1, 4})
        generator, reference_calls = _fake_generator(rejected_attempts=rejected)
        reference_attempts, reference_paths = _generate(
            self.root / "reference", targets, 2, generator
        )
        for completed_count in range(len(reference_calls)):
            with self.subTest(completed_count=completed_count):
                root = self.root / str(completed_count)
                interrupt, completed_calls = _fake_generator(
                    rejected_attempts=rejected,
                    interrupt_after_batches=completed_count,
                )
                with self.assertRaises(InjectedInterruption):
                    _generate(root, targets, 2, interrupt)
                directory = root / "batches" / FAMILY_NAME
                self.assertEqual(len(tuple(directory.glob("*.npz"))), completed_count)
                generator, resumed_calls = _fake_generator(rejected_attempts=rejected)
                attempts, paths = _generate(root, targets, 2, generator)
                self.assertEqual(attempts, reference_attempts)
                self.assertEqual(completed_calls + resumed_calls, reference_calls)
                for actual, expected in zip(paths, reference_paths, strict=True):
                    with np.load(actual) as a, np.load(expected) as b:
                        self.assertEqual(a.files, b.files)
                        for name in a.files:
                            np.testing.assert_array_equal(a[name], b[name])

    def test_resume_rejects_a_different_sampling_seed(self) -> None:
        generator, _ = _fake_generator()
        targets = {"main_m1_q0": 1}
        _generate(self.root, targets, 1, generator)
        must_not_run, calls = _fake_generator(interrupt_after_batches=0)
        with self.assertRaisesRegex(RuntimeError, "different seed"):
            generate_simulations(
                self.root,
                family_id=FAMILY_ID,
                seed=SEED + 1,
                requested_simulations_per_group=targets,
                batch_size=1,
                generate_batch=must_not_run,
            )
        self.assertEqual(calls, [])

    def test_generation_respects_attempt_limit(self) -> None:
        for requested, batch_size in ((32, 32), (5, 3)):
            with self.subTest(requested=requested, batch_size=batch_size):
                targets = {"main_m1_q0": requested}
                limit = 2 * requested
                generator, calls = _fake_generator(
                    rejected_attempts=frozenset(range(limit))
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"attempt limit reached.*main_m1_q0: successful=0/{requested}, attempts={limit}/{limit}",
                ):
                    _generate(
                        self.root / str(requested), targets, batch_size, generator
                    )
                self.assertEqual(sum(len(groups) for _, groups in calls), limit)
                must_not_run, _ = _fake_generator(interrupt_after_batches=0)
                with self.assertRaisesRegex(RuntimeError, "attempt limit reached"):
                    _generate(
                        self.root / str(requested), targets, batch_size, must_not_run
                    )

                generator, _ = _fake_generator(
                    rejected_attempts=frozenset(range(requested))
                )
                root = self.root / f"last_attempt_{requested}"
                attempts, paths = _generate(root, targets, batch_size, generator)
                self.assertEqual(attempts, {"main_m1_q0": limit})
                self.assertEqual(
                    _generate(root, targets, batch_size, must_not_run),
                    (attempts, paths),
                )

    def test_resume_rejects_saved_quota_and_retry_overflows(self) -> None:
        for requested, saved_count, rejected, error in (
            (1, 2, frozenset(), "accepted target"),
            (1, 3, frozenset(range(3)), "attempt limit"),
            (0, 1, frozenset({0}), "attempt limit"),
        ):
            with self.subTest(requested=requested, saved_count=saved_count):
                root = self.root / str(saved_count)
                generator, _ = _fake_generator(rejected_attempts=rejected)
                generator(
                    ("main_m1_q1",) * saved_count,
                    0,
                    root / "batches" / FAMILY_NAME / "batch_000000.npz",
                )
                must_not_run, _ = _fake_generator(interrupt_after_batches=0)
                with self.assertRaisesRegex(RuntimeError, error):
                    _generate(
                        root,
                        {"main_m1_q0": 1, "main_m1_q1": requested},
                        2,
                        must_not_run,
                    )

    def test_resume_fails_when_a_saved_batch_is_missing(self) -> None:
        targets = {"main_m1_q0": 1}
        generator, _ = _fake_generator()
        generator(
            ("main_m1_q0",),
            0,
            self.root / "batches" / FAMILY_NAME / "batch_000001.npz",
        )

        with self.assertRaises(FileNotFoundError):
            _generate(self.root, targets, 1, generator)

    def test_scanner_orders_double_digit_batch_ids_numerically(self) -> None:
        targets = {"main_m1_q0": 12}
        generator, _ = _fake_generator()
        completed = _generate(self.root, targets, 1, generator)

        self.assertEqual(
            tuple(path.stem for path in completed[1]),
            tuple(f"batch_{batch_id:06d}" for batch_id in range(12)),
        )
        must_not_run, _ = _fake_generator(interrupt_after_batches=0)
        self.assertEqual(_generate(self.root, targets, 1, must_not_run), completed)


if __name__ == "__main__":
    unittest.main()
