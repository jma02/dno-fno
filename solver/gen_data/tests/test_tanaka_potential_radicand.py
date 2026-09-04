"""Focused CPU tests for the Tanaka surface-potential real-root gate."""

from __future__ import annotations

import math
import os
from typing import NamedTuple
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    TanakaPotentialRadicandError,
    build_tanaka_initial_conditions,
)
from solver.gen_data.tanaka_sampling import TanakaCrest  # noqa: E402
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    make_default_tanaka_template,
)

jax.config.update("jax_enable_x64", True)

LENGTH = 2.0 * math.pi

FakeTanakaProfiles = NamedTuple(
    "FakeTanakaProfiles",
    [
        ("x_profile", jax.Array),
        ("eta_profile", jax.Array),
        ("theta_profile", jax.Array),
        ("froude", jax.Array),
    ],
)


def _profiles(
    amplitudes: tuple[float, ...],
    *,
    froude: tuple[float, ...] | None = None,
) -> FakeTanakaProfiles:
    count = len(amplitudes)
    return FakeTanakaProfiles(
        jnp.tile(jnp.asarray((-1.0, 0.0, 1.0)), (count, 1)),
        jnp.asarray([(0.0, amplitude, 0.0) for amplitude in amplitudes]),
        jnp.zeros((count, 3), dtype=jnp.float64),
        jnp.asarray((1.0,) * count if froude is None else froude),
    )


def _build_with_profiles(profiles: FakeTanakaProfiles) -> tuple[jax.Array, jax.Array]:
    count = profiles.eta_profile.shape[0]
    with patch(
        "solver.gen_data.tanaka_initial_conditions.solve_modified_tanaka_batched",
        return_value=profiles,
    ):
        return build_tanaka_initial_conditions(
            make_default_tanaka_template(nx=32, dno_order=0, pad_factor=1),
            np.ones(count, dtype=np.float64),
            tuple((TanakaCrest(0.1, 0.0, 1),) for _ in range(count)),
            length=LENGTH,
            nx=32,
            gravity=1.0,
        )


class TanakaPotentialRadicandTest(unittest.TestCase):
    def test_finite_negative_radicand_reports_invalid_simulations(self) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            _build_with_profiles(_profiles((0.1, 0.75)))

        self.assertEqual(caught.exception.invalid_simulation_indices, (1,))

    def test_nonfinite_speed_is_a_fatal_numerical_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "speeds") as caught:
            _build_with_profiles(_profiles((0.1, 0.1), froude=(1.0, np.nan)))

        self.assertNotIsInstance(caught.exception, TanakaPotentialRadicandError)

    def test_nonfinite_radicand_is_a_fatal_numerical_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "radicand") as caught:
            _build_with_profiles(_profiles((0.1, np.nan)))

        self.assertNotIsInstance(caught.exception, TanakaPotentialRadicandError)

    def test_real_profiles_preserve_translation_and_direction_symmetries(self) -> None:
        nx = 64
        template = make_default_tanaka_template(
            depth=1.0,
            gravity=1.0,
            direction=1,
            nx=nx,
            length=LENGTH,
            center=0.0,
            dno_order=6,
            pad_factor=8,
        )
        eta, xi = build_tanaka_initial_conditions(
            template,
            np.asarray((0.35, 0.35, 0.35), dtype=np.float64),
            (
                (TanakaCrest(0.45, 0.0, 1),),
                (TanakaCrest(0.45, LENGTH / 2.0, 1),),
                (TanakaCrest(0.45, 0.0, -1),),
            ),
            length=LENGTH,
            nx=nx,
            gravity=1.0,
        )
        eta_host = np.asarray(eta)
        xi_host = np.asarray(xi)
        self.assertTrue(np.isfinite(eta_host).all())
        self.assertTrue(np.isfinite(xi_host).all())
        np.testing.assert_allclose(
            eta_host[1], np.roll(eta_host[0], nx // 2), rtol=2e-11, atol=2e-13
        )
        np.testing.assert_allclose(
            xi_host[1], np.roll(xi_host[0], nx // 2), rtol=2e-11, atol=2e-13
        )
        np.testing.assert_allclose(eta_host[2], eta_host[0], rtol=2e-11, atol=2e-13)
        np.testing.assert_allclose(xi_host[2], -xi_host[0], rtol=2e-11, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
