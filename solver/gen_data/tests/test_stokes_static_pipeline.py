"""CPU tests for constructing and labeling static Stokes states."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.stokes_sampling import StokesSample  # noqa: E402
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_DNO_ORDER,
    PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
    PAPER_STATIC_STOKES_NX,
    PAPER_STATIC_STOKES_PAD_FACTOR,
    evaluate_static_stokes_sample,
)

jax.config.update("jax_enable_x64", True)


class StokesStaticPipelineTest(unittest.TestCase):
    def test_invalid_constructed_states_are_rejected_before_dno_evaluation(
        self,
    ) -> None:
        sample = StokesSample("finite", 14, 0.2, 0.0, 0.001)
        zeros = jnp.zeros(PAPER_STATIC_STOKES_NX, dtype=jnp.float64)
        for eta, failed_check in (
            (jnp.full_like(zeros, jnp.nan), "nonfinite_state"),
            (jnp.full_like(zeros, -sample.depth), "nonpositive_water_height"),
        ):
            with (
                self.subTest(failed_check=failed_check),
                patch(
                    "solver.gen_data.stokes_static_pipeline.stokes_eta_xi_at_phase",
                    return_value=(eta, zeros),
                ),
                patch(
                    "solver.gen_data.stokes_static_pipeline.compute_dno_target"
                ) as target,
            ):
                outcome = evaluate_static_stokes_sample(sample)

            self.assertFalse(outcome.decision.accepted)
            self.assertIsNone(outcome.rows)
            self.assertTrue(getattr(outcome.decision, failed_check))
            self.assertEqual(outcome.metrics, {})
            target.assert_not_called()

    def test_target_finiteness_controls_whether_the_row_is_retained(self) -> None:
        sample = StokesSample("deep", 3, 4.0, 0.0, 0.005)
        zeros = jnp.zeros(PAPER_STATIC_STOKES_NX, dtype=jnp.float64)

        for target_q, accepted in (
            (zeros, True),
            (jnp.full_like(zeros, jnp.nan), False),
        ):
            with (
                self.subTest(accepted=accepted),
                patch(
                    "solver.gen_data.stokes_static_pipeline.stokes_eta_xi_at_phase",
                    return_value=(zeros, jnp.ones_like(zeros)),
                ) as constructor,
                patch(
                    "solver.gen_data.stokes_static_pipeline.compute_dno_target",
                    return_value=(zeros, zeros, target_q),
                ) as target,
            ):
                outcome = evaluate_static_stokes_sample(sample)

            self.assertEqual(outcome.decision.accepted, accepted)
            self.assertEqual(outcome.decision.nonfinite_target, not accepted)
            self.assertEqual(outcome.metrics, {})
            self.assertEqual(outcome.rows is not None, accepted)
            constructor.assert_called_once()
            target.assert_called_once()
            target_eta, target_xi, _ = target.call_args.args
            self.assertTrue(np.isfinite(np.asarray(target_eta)).all())
            self.assertEqual(float(jnp.mean(target_xi)), 0.0)
            self.assertEqual(
                target.call_args.kwargs,
                {
                    "nx": PAPER_STATIC_STOKES_NX,
                    "length": 2.0 * np.pi,
                    "dno_order": PAPER_STATIC_STOKES_DNO_ORDER,
                    "pad_factor": PAPER_STATIC_STOKES_PAD_FACTOR,
                    "maximum_wavenumber": PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
                },
            )
            if outcome.rows is not None:
                self.assertEqual(outcome.rows.eta.shape, (1, PAPER_STATIC_STOKES_NX))
                np.testing.assert_array_equal(outcome.rows.time, np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
