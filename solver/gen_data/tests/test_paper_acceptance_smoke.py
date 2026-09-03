"""CPU smoke tests for finite-depth Stokes dataset support.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests.test_paper_acceptance_smoke
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.reference_solutions.stokes_wave import (  # noqa: E402
    finite_depth_eta_harmonics,
    finite_depth_stokes_in_ursell_support,
    finite_depth_stokes_wave_height,
    stokes_eta_xi,
)

jax.config.update("jax_enable_x64", True)


LENGTH = 2.0 * np.pi
DEPTH = 1.0
GRAVITY = 1.0


class PaperDatasetAcceptanceSmokeTest(unittest.TestCase):
    def test_finite_stokes_constructor_support(self) -> None:
        length = 164.0
        k0 = 2.0 * np.pi * 14 / length
        self.assertFalse(
            bool(
                finite_depth_stokes_in_ursell_support(
                    k0, depth=1.0, gravity=1.0, a0=0.22543466384990768
                )
            )
        )
        self.assertTrue(
            bool(
                finite_depth_stokes_in_ursell_support(
                    k0, depth=1.0, gravity=1.0, a0=0.085
                )
            )
        )

    def test_finite_stokes_harmonics_vectorize_over_parameter_draws(self) -> None:
        k0 = 2.0 * np.pi * np.asarray([14.0, 14.0]) / 164.0
        amplitudes = np.asarray([0.22543466384990768, 0.09589740544385852])
        harmonics = finite_depth_eta_harmonics(
            k0=k0,
            depth=np.ones(2),
            gravity=1.0,
            a0=amplitudes,
        )

        self.assertEqual(harmonics.shape, (2, 5))
        np.testing.assert_allclose(
            np.asarray(harmonics[0]),
            np.asarray(
                finite_depth_eta_harmonics(
                    k0=k0[0],
                    depth=1.0,
                    gravity=1.0,
                    a0=amplitudes[0],
                )
            ),
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(harmonics[1]),
            np.asarray(
                finite_depth_eta_harmonics(
                    k0=k0[1],
                    depth=1.0,
                    gravity=1.0,
                    a0=amplitudes[1],
                )
            ),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_wave_height_bound_contains_a_dense_phase_evaluation(self) -> None:
        k0 = 2.0 * np.pi * 14.0 / 164.0
        harmonics = np.asarray(
            finite_depth_eta_harmonics(
                k0=k0,
                depth=1.0,
                gravity=1.0,
                a0=0.09589740544385852,
            )
        )
        phases = np.linspace(0.0, 2.0 * np.pi, 200_000, endpoint=False)
        modes = np.arange(1.0, 6.0)
        elevation = np.sum(
            harmonics[:, None] * np.cos(modes[:, None] * phases),
            axis=0,
        )
        dense_height = float(np.max(elevation) - np.min(elevation))
        upper_bound = float(
            finite_depth_stokes_wave_height(
                k0,
                depth=1.0,
                gravity=1.0,
                a0=0.09589740544385852,
                phase_points=64,
            )
        )

        self.assertGreaterEqual(upper_bound, dense_height)

    def test_finite_stokes_support_depends_only_on_kh_and_ka(self) -> None:
        wavenumbers = np.asarray([0.6, 1.2], dtype=np.float64)
        depths = np.asarray([1.0, 0.5], dtype=np.float64)
        amplitudes = 0.05 / wavenumbers
        harmonics = np.asarray(
            finite_depth_eta_harmonics(
                k0=wavenumbers,
                depth=depths,
                gravity=1.0,
                a0=amplitudes,
            )
        )

        np.testing.assert_allclose(
            harmonics[0] / amplitudes[0],
            harmonics[1] / amplitudes[1],
            rtol=1e-13,
            atol=1e-14,
        )
        in_support = finite_depth_stokes_in_ursell_support(
            k0=wavenumbers,
            depth=depths,
            gravity=1.0,
            a0=amplitudes,
        )
        np.testing.assert_array_equal(
            np.asarray(in_support),
            np.asarray([True, True]),
        )

        near_shallow_boundary = finite_depth_stokes_in_ursell_support(
            k0=np.asarray([0.5, 0.5]),
            depth=np.ones(2),
            gravity=1.0,
            a0=np.asarray([0.075, 0.08]),
        )
        np.testing.assert_array_equal(
            np.asarray(near_shallow_boundary),
            np.asarray([True, False]),
        )

    def test_finite_stokes_translation_and_phase_zero_reflection(self) -> None:
        nx = 256
        x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
        parameters = {
            "time": jnp.asarray(0.0, dtype=jnp.float64),
            "n0": 14,
            "a0": 0.005,
            "length": LENGTH,
            "depth": 0.1,
            "gravity": GRAVITY,
            "ichoi": 1,
        }
        eta, xi = stokes_eta_xi(x=x, **parameters)
        shifted_eta, shifted_xi = stokes_eta_xi(
            x=x + LENGTH / nx,
            **parameters,
        )
        reflected_eta, reflected_xi = stokes_eta_xi(x=-x, **parameters)

        np.testing.assert_allclose(
            np.asarray(shifted_eta),
            np.asarray(jnp.roll(eta, -1)),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(shifted_xi),
            np.asarray(jnp.roll(xi, -1)),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(reflected_eta),
            np.asarray(eta),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(reflected_xi),
            -np.asarray(xi),
            rtol=1e-12,
            atol=1e-13,
        )
        self.assertGreater(float(jnp.min(parameters["depth"] + eta)), 0.0)


if __name__ == "__main__":
    unittest.main()
