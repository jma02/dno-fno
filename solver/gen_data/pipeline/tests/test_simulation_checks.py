"""Focused tests for generated-simulation checks."""

from __future__ import annotations

import unittest

from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheckResult,
    SimulationCheck,
    checks_from_bits,
)


class SimulationCheckResultTest(unittest.TestCase):
    def test_acceptance_requires_every_required_check(self) -> None:
        required = SimulationCheck.NONFINITE_STATE | SimulationCheck.HAMILTONIAN_DRIFT
        incomplete = SimulationCheckResult(
            required=required,
            evaluated=SimulationCheck.NONFINITE_STATE,
            failed=SimulationCheck.NONE,
        )
        complete = SimulationCheckResult(
            required=required,
            evaluated=required,
            failed=SimulationCheck.NONE,
        )

        self.assertFalse(incomplete.accepted)
        self.assertEqual(incomplete.missing, SimulationCheck.HAMILTONIAN_DRIFT)
        self.assertTrue(complete.accepted)

    def test_optional_diagnostic_failure_does_not_change_acceptance(self) -> None:
        decision = SimulationCheckResult(
            required=SimulationCheck.NONFINITE_STATE,
            evaluated=(
                SimulationCheck.NONFINITE_STATE | SimulationCheck.OUTSIDE_SUPPORT
            ),
            failed=SimulationCheck.OUTSIDE_SUPPORT,
        )

        self.assertTrue(decision.accepted)

    def test_required_failure_rejects(self) -> None:
        decision = SimulationCheckResult(
            required=(
                SimulationCheck.NONFINITE_STATE | SimulationCheck.OUTSIDE_SUPPORT
            ),
            evaluated=(
                SimulationCheck.NONFINITE_STATE | SimulationCheck.OUTSIDE_SUPPORT
            ),
            failed=SimulationCheck.OUTSIDE_SUPPORT,
        )

        self.assertFalse(decision.accepted)

    def test_failed_check_must_have_been_evaluated(self) -> None:
        with self.assertRaisesRegex(ValueError, "marked as evaluated"):
            SimulationCheckResult(
                required=SimulationCheck.NONFINITE_TARGET,
                evaluated=SimulationCheck.NONE,
                failed=SimulationCheck.NONFINITE_TARGET,
            )

    def test_uint32_round_trip_uses_stable_bits(self) -> None:
        self.assertEqual(int(SimulationCheck.NONFINITE_STATE), 1)
        self.assertEqual(int(SimulationCheck.INCOMPLETE_TRAJECTORY), 512)
        persisted = int(
            SimulationCheck.GL2_STAGE_RESIDUAL | SimulationCheck.TEMPORAL_DEFECT
        )
        decision = SimulationCheckResult.from_bits(
            required_bits=persisted,
            evaluated_bits=persisted,
            failed_bits=0,
        )

        self.assertEqual(int(decision.evaluated), persisted)
        self.assertEqual(
            checks_from_bits(decision.evaluated),
            (
                SimulationCheck.GL2_STAGE_RESIDUAL,
                SimulationCheck.TEMPORAL_DEFECT,
            ),
        )
        self.assertTrue(decision.accepted)

    def test_from_bits_accepts_numpy_style_integer_values(self) -> None:
        class ArrayInteger:
            def __index__(self) -> int:
                return 1

        decision = SimulationCheckResult.from_bits(
            required_bits=ArrayInteger(),
            evaluated_bits=ArrayInteger(),
            failed_bits=0,
        )

        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
