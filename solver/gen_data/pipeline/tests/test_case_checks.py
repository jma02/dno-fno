"""Focused tests for generated-case checks."""

from __future__ import annotations

import unittest

from solver.gen_data.pipeline.case_checks import (
    CaseCheckResult,
    CaseCheck,
    checks_from_bits,
)


class CaseCheckResultTest(unittest.TestCase):
    def test_acceptance_requires_every_required_check(self) -> None:
        required = CaseCheck.NONFINITE_STATE | CaseCheck.HAMILTONIAN_DRIFT
        incomplete = CaseCheckResult(
            required=required,
            evaluated=CaseCheck.NONFINITE_STATE,
            failed=CaseCheck.NONE,
        )
        complete = CaseCheckResult(
            required=required,
            evaluated=required,
            failed=CaseCheck.NONE,
        )

        self.assertFalse(incomplete.accepted)
        self.assertEqual(incomplete.missing, CaseCheck.HAMILTONIAN_DRIFT)
        self.assertTrue(complete.accepted)

    def test_optional_diagnostic_failure_does_not_change_acceptance(self) -> None:
        decision = CaseCheckResult(
            required=CaseCheck.NONFINITE_STATE,
            evaluated=(CaseCheck.NONFINITE_STATE | CaseCheck.OUTSIDE_SUPPORT),
            failed=CaseCheck.OUTSIDE_SUPPORT,
        )

        self.assertTrue(decision.accepted)

    def test_required_failure_rejects(self) -> None:
        decision = CaseCheckResult(
            required=(CaseCheck.NONFINITE_STATE | CaseCheck.OUTSIDE_SUPPORT),
            evaluated=(CaseCheck.NONFINITE_STATE | CaseCheck.OUTSIDE_SUPPORT),
            failed=CaseCheck.OUTSIDE_SUPPORT,
        )

        self.assertFalse(decision.accepted)

    def test_failed_check_must_have_been_evaluated(self) -> None:
        with self.assertRaisesRegex(ValueError, "marked as evaluated"):
            CaseCheckResult(
                required=CaseCheck.NONFINITE_TARGET,
                evaluated=CaseCheck.NONE,
                failed=CaseCheck.NONFINITE_TARGET,
            )

    def test_uint32_round_trip_uses_stable_bits(self) -> None:
        self.assertEqual(int(CaseCheck.NONFINITE_STATE), 1 << 0)
        self.assertEqual(int(CaseCheck.INCOMPLETE_TRAJECTORY), 1 << 9)
        persisted = int(CaseCheck.GL2_STAGE_RESIDUAL | CaseCheck.TEMPORAL_DEFECT)
        decision = CaseCheckResult.from_bits(
            required_bits=persisted,
            evaluated_bits=persisted,
            failed_bits=0,
        )

        self.assertEqual(int(decision.evaluated), persisted)
        self.assertEqual(
            checks_from_bits(decision.evaluated),
            (
                CaseCheck.GL2_STAGE_RESIDUAL,
                CaseCheck.TEMPORAL_DEFECT,
            ),
        )
        self.assertTrue(decision.accepted)

    def test_from_bits_accepts_numpy_style_integer_values(self) -> None:
        class ArrayInteger:
            def __index__(self) -> int:
                return 1 << 0

        decision = CaseCheckResult.from_bits(
            required_bits=ArrayInteger(),
            evaluated_bits=ArrayInteger(),
            failed_bits=0,
        )

        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
