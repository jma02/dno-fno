"""Tests for the frozen paper-dataset DNO target."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.reference import (  # noqa: E402
    PAPER_DNO_TARGET,
    DiscreteDnoTarget,
    evaluate_discrete_dno_target,
)

jax.config.update("jax_enable_x64", True)


class DiscreteDnoTargetTest(unittest.TestCase):
    def test_paper_definition_is_frozen(self) -> None:
        self.assertEqual(
            (
                PAPER_DNO_TARGET.nx,
                PAPER_DNO_TARGET.dno_order,
                PAPER_DNO_TARGET.pad_factor,
                PAPER_DNO_TARGET.maximum_wavenumber,
            ),
            (1024, 6, 8, 128.0),
        )
        self.assertAlmostEqual(PAPER_DNO_TARGET.length, 2.0 * np.pi)

    def test_evaluator_promotes_stored_float32_fields_to_float64(self) -> None:
        definition = DiscreteDnoTarget(
            nx=64,
            dno_order=2,
            pad_factor=2,
            maximum_wavenumber=8.0,
        )
        x = 2.0 * np.pi * jnp.arange(64, dtype=jnp.float32) / 64.0
        eta_input, xi_input, target = evaluate_discrete_dno_target(
            0.02 * jnp.cos(3.0 * x),
            0.03 * jnp.sin(4.0 * x),
            jnp.asarray(1.0, dtype=jnp.float32),
            definition=definition,
        )

        self.assertEqual(eta_input.dtype, jnp.float64)
        self.assertEqual(xi_input.dtype, jnp.float64)
        self.assertEqual(target.dtype, jnp.float64)

    def test_unresolved_input_modes_do_not_change_the_target(self) -> None:
        definition = DiscreteDnoTarget(
            nx=64,
            dno_order=6,
            pad_factor=8,
            maximum_wavenumber=8.0,
        )
        x = 2.0 * np.pi * jnp.arange(64, dtype=jnp.float64) / 64.0
        eta = 0.02 * jnp.cos(3.0 * x)
        xi = 0.03 * jnp.sin(4.0 * x)
        high_mode = 0.1 * jnp.cos(12.0 * x)

        base = evaluate_discrete_dno_target(
            eta,
            xi,
            1.0,
            definition=definition,
        )
        perturbed = evaluate_discrete_dno_target(
            eta + high_mode,
            xi - high_mode,
            1.0,
            definition=definition,
        )

        for base_field, perturbed_field in zip(base, perturbed):
            np.testing.assert_allclose(
                np.asarray(base_field),
                np.asarray(perturbed_field),
                rtol=1e-12,
                atol=1e-13,
            )

    def test_target_is_mean_zero_and_translation_covariant(self) -> None:
        definition = DiscreteDnoTarget(
            nx=64,
            dno_order=6,
            pad_factor=8,
            maximum_wavenumber=8.0,
        )
        x = 2.0 * np.pi * jnp.arange(64, dtype=jnp.float64) / 64.0
        eta = 0.02 * jnp.cos(3.0 * x) + 0.01 * jnp.sin(5.0 * x)
        xi = 0.03 * jnp.sin(4.0 * x) + 2.0
        shift = 7

        _, _, target = evaluate_discrete_dno_target(
            eta,
            xi,
            0.8,
            definition=definition,
        )
        _, _, shifted_target = evaluate_discrete_dno_target(
            jnp.roll(eta, shift),
            jnp.roll(xi, shift),
            0.8,
            definition=definition,
        )

        self.assertLess(abs(float(jnp.mean(target))), 1e-14)
        np.testing.assert_allclose(
            np.asarray(shifted_target),
            np.asarray(jnp.roll(target, shift)),
            rtol=1e-11,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
