"""CPU end-to-end tests for the common static Stokes writer."""

from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    AttemptAssignment,
    SimulationKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.simulation_checks import SimulationCheck  # noqa: E402
from solver.gen_data.stokes_sampling import (  # noqa: E402
    STOKES_SAMPLE_CELL_IDS,
    StokesSample,
    sample_stokes_simulation,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
    evaluate_static_stokes_sample,
)

jax.config.update("jax_enable_x64", True)


def assignment(
    cell_index: int,
    *,
    attempt_index: int,
    split_id: SplitId = SplitId.VALIDATION,
) -> AttemptAssignment:
    """Return a fixed Stokes assignment for integration tests."""

    return AttemptAssignment(
        simulation_key=SimulationKey(
            family_id=PhysicalFamilyId.STOKES,
            revision_id=1,
            split_id=split_id,
            stream_id=9,
            attempt_index=attempt_index,
        ),
        cell_id=STOKES_SAMPLE_CELL_IDS[cell_index],
    )


def reduced_contract() -> StaticStokesContract:
    """Return the explicitly labeled CPU wiring contract."""

    return StaticStokesContract(
        nx=64,
        length=2.0 * math.pi,
        dno_order=2,
        pad_factor=2,
        maximum_wavenumber=24.0,
        role="reduced_wiring_evidence_only",
    )


class StokesStaticPipelineTest(unittest.TestCase):
    def test_contract_requires_an_explicit_reduced_evidence_label(self) -> None:
        self.assertEqual(
            (
                PAPER_STATIC_STOKES_CONTRACT.nx,
                PAPER_STATIC_STOKES_CONTRACT.dno_order,
                PAPER_STATIC_STOKES_CONTRACT.pad_factor,
                PAPER_STATIC_STOKES_CONTRACT.maximum_wavenumber,
                PAPER_STATIC_STOKES_CONTRACT.role,
            ),
            (1024, 6, 8, 128.0, "paper_dataset"),
        )
        with self.assertRaisesRegex(ValueError, "must be labeled"):
            StaticStokesContract(
                nx=64,
                dno_order=2,
                pad_factor=2,
                maximum_wavenumber=24.0,
            )
        self.assertEqual(
            reduced_contract().role,
            "reduced_wiring_evidence_only",
        )

    def test_nonfinite_target_is_a_zero_row_rejection(self) -> None:
        sample = sample_stokes_simulation(assignment(2, attempt_index=30))
        contract = reduced_contract()

        def nonfinite_target(
            eta: jax.Array | np.ndarray,
            xi: jax.Array | np.ndarray,
            depth: float | jax.Array | np.ndarray,
            *,
            nx: int,
            length: float,
            dno_order: int,
            pad_factor: int,
            maximum_wavenumber: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del depth, nx, length, dno_order, pad_factor, maximum_wavenumber
            eta_array = jnp.asarray(eta)
            xi_array = jnp.asarray(xi)
            return eta_array, xi_array, jnp.full_like(eta_array, jnp.nan)

        outcome = evaluate_static_stokes_sample(
            sample,
            contract=contract,
            target_evaluator=nonfinite_target,
        )
        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertTrue(outcome.decision.failed & SimulationCheck.NONFINITE_TARGET)
        self.assertEqual(outcome.metrics["target_finite"], False)

    def test_constructor_and_target_exceptions_propagate(self) -> None:
        sample = sample_stokes_simulation(assignment(2, attempt_index=31))
        contract = reduced_contract()

        def failing_constructor(
            sample: StokesSample,
            contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            del sample, contract
            raise ValueError("injected constructor bug")

        with self.assertRaisesRegex(ValueError, "injected constructor bug"):
            evaluate_static_stokes_sample(
                sample,
                contract=contract,
                state_constructor=failing_constructor,
            )

        def failing_target(
            eta: jax.Array | np.ndarray,
            xi: jax.Array | np.ndarray,
            depth: float | jax.Array | np.ndarray,
            *,
            nx: int,
            length: float,
            dno_order: int,
            pad_factor: int,
            maximum_wavenumber: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del eta, xi, depth, nx, length, dno_order, pad_factor, maximum_wavenumber
            raise RuntimeError("injected target bug")

        with self.assertRaisesRegex(RuntimeError, "injected target bug"):
            evaluate_static_stokes_sample(
                sample,
                contract=contract,
                target_evaluator=failing_target,
            )


if __name__ == "__main__":
    unittest.main()
