"""Unittest-discoverable wrappers for the focused IC-panel checks."""
from __future__ import annotations

import unittest

from solver.evals.tests_ic_panel import (
    test_panel_load_and_registry_timestep_precedence,
    test_panel_validation_rejects_wrong_count_and_identity,
    test_truth_cache_keys_on_panel_digest,
)


class IcPanelTest(unittest.TestCase):
    def test_load_and_registry_timestep_precedence(self) -> None:
        test_panel_load_and_registry_timestep_precedence()

    def test_validation_rejects_wrong_count_and_identity(self) -> None:
        test_panel_validation_rejects_wrong_count_and_identity()

    def test_truth_cache_keys_on_panel_digest(self) -> None:
        test_truth_cache_keys_on_panel_digest()


if __name__ == "__main__":
    unittest.main()
