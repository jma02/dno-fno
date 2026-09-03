"""Record why a generated simulation was accepted or rejected."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationCheckResult:
    """Acceptance decision and independent failure conditions."""

    accepted: bool
    nonfinite_state: bool = False
    nonfinite_target: bool = False
    nonpositive_water_height: bool = False
    hamiltonian_drift: bool = False
    integration_failure: bool = False
    outside_support: bool = False
    incomplete_trajectory: bool = False

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Return stable names for the failure conditions that occurred."""

        checks = (
            ("nonfinite_state", self.nonfinite_state),
            ("nonfinite_target", self.nonfinite_target),
            ("nonpositive_water_height", self.nonpositive_water_height),
            ("hamiltonian_drift", self.hamiltonian_drift),
            ("integration_failure", self.integration_failure),
            ("outside_support", self.outside_support),
            ("incomplete_trajectory", self.incomplete_trajectory),
        )
        return tuple(name for name, failed in checks if failed)
