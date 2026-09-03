"""Focused tests for generated-simulation checks."""

from __future__ import annotations

import unittest

from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult


class SimulationCheckResultTest(unittest.TestCase):
    def test_failure_conditions_default_to_false(self) -> None:
        decision = SimulationCheckResult(accepted=True)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.failed_checks, ())

    def test_acceptance_is_independent_of_diagnostic_conditions(self) -> None:
        decision = SimulationCheckResult(
            accepted=True,
            outside_support=True,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.failed_checks, ("outside_support",))

    def test_failed_checks_uses_stable_field_order(self) -> None:
        decision = SimulationCheckResult(
            accepted=False,
            nonfinite_state=True,
            nonpositive_water_height=True,
            integration_failure=True,
            incomplete_trajectory=True,
        )

        self.assertEqual(
            decision.failed_checks,
            (
                "nonfinite_state",
                "nonpositive_water_height",
                "integration_failure",
                "incomplete_trajectory",
            ),
        )


if __name__ == "__main__":
    unittest.main()
