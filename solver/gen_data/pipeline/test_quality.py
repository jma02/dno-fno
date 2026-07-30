"""Focused tests for generated-data quality and identity contracts."""
from __future__ import annotations

import unittest

from solver.gen_data.pipeline import (
    GeneratorRevisionRecord,
    QualityDecision,
    QualityReason,
    QualityScope,
    SampleRecord,
    TrajectoryRecord,
    reasons_from_bits,
)


class QualityDecisionTest(unittest.TestCase):
    def test_acceptance_requires_every_required_check(self) -> None:
        required = QualityReason.NONFINITE_STATE | QualityReason.HAMILTONIAN_DRIFT
        incomplete = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=required,
            evaluated=QualityReason.NONFINITE_STATE,
            failed=QualityReason.NONE,
        )
        complete = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=required,
            evaluated=required,
            failed=QualityReason.NONE,
        )

        self.assertFalse(incomplete.accepted)
        self.assertEqual(incomplete.missing, QualityReason.HAMILTONIAN_DRIFT)
        self.assertTrue(complete.accepted)

    def test_optional_diagnostic_failure_does_not_change_acceptance(self) -> None:
        decision = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=QualityReason.NONFINITE_STATE,
            evaluated=(
                QualityReason.NONFINITE_STATE | QualityReason.OUTSIDE_SUPPORT
            ),
            failed=QualityReason.OUTSIDE_SUPPORT,
        )

        self.assertTrue(decision.accepted)

    def test_required_failure_rejects(self) -> None:
        decision = QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=(
                QualityReason.NONFINITE_STATE | QualityReason.OUTSIDE_SUPPORT
            ),
            evaluated=(
                QualityReason.NONFINITE_STATE | QualityReason.OUTSIDE_SUPPORT
            ),
            failed=QualityReason.OUTSIDE_SUPPORT,
        )

        self.assertFalse(decision.accepted)

    def test_failed_check_must_have_been_evaluated(self) -> None:
        with self.assertRaisesRegex(ValueError, "marked as evaluated"):
            QualityDecision(
                scope=QualityScope.SAMPLE,
                required=QualityReason.NONFINITE_TARGET,
                evaluated=QualityReason.NONE,
                failed=QualityReason.NONFINITE_TARGET,
            )

    def test_uint32_round_trip_uses_stable_bits(self) -> None:
        self.assertEqual(int(QualityReason.NONFINITE_STATE), 1 << 0)
        self.assertEqual(int(QualityReason.INCOMPLETE_TRAJECTORY), 1 << 9)
        persisted = int(
            QualityReason.GL2_STAGE_RESIDUAL | QualityReason.TEMPORAL_DEFECT
        )
        decision = QualityDecision.from_bits(
            scope=QualityScope.GENERATOR_REVISION,
            required_bits=persisted,
            evaluated_bits=persisted,
            failed_bits=0,
        )

        self.assertEqual(decision.evaluated_bits, persisted)
        self.assertEqual(
            reasons_from_bits(decision.evaluated_bits),
            (
                QualityReason.GL2_STAGE_RESIDUAL,
                QualityReason.TEMPORAL_DEFECT,
            ),
        )
        self.assertTrue(decision.accepted)

    def test_from_bits_accepts_numpy_style_integer_values(self) -> None:
        class ArrayInteger:
            def __index__(self) -> int:
                return 1 << 0

        decision = QualityDecision.from_bits(
            scope=QualityScope.SAMPLE,
            required_bits=ArrayInteger(),
            evaluated_bits=ArrayInteger(),
            failed_bits=0,
        )

        self.assertTrue(decision.accepted)


class GenerationRecordTest(unittest.TestCase):
    def test_sample_retains_trajectory_and_generator_identity(self) -> None:
        generator = GeneratorRevisionRecord("tanaka", "git:abc123")
        trajectory = TrajectoryRecord(generator, "run-7:case-24")
        sample = SampleRecord(trajectory, frame_index=19)

        self.assertEqual(sample.trajectory.trajectory_id, "run-7:case-24")
        self.assertEqual(sample.trajectory.generator_revision.family_id, "tanaka")

    def test_negative_frame_index_is_invalid(self) -> None:
        generator = GeneratorRevisionRecord("tanaka", "git:abc123")
        trajectory = TrajectoryRecord(generator, "run-7:case-24")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SampleRecord(trajectory, frame_index=-1)


if __name__ == "__main__":
    unittest.main()
