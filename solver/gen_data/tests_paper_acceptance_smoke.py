"""CPU smoke test crossing Stokes construction, GL2, and acceptance.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests_paper_acceptance_smoke
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
from solver.gen_data.pipeline.acceptance import (  # noqa: E402
    RefinementTrajectory,
    evaluate_finite_stokes_support,
    evaluate_temporal_refinement,
)
from solver.solvers.time_integrator import (  # noqa: E402
    State,
    make_solver_params,
    rollout,
)

jax.config.update("jax_enable_x64", True)


LENGTH = 2.0 * np.pi
DEPTH = 1.0
GRAVITY = 1.0


def _as_refinement_trajectory(
    payload: dict[str, jax.Array],
) -> RefinementTrajectory:
    return RefinementTrajectory(
        times=np.asarray(payload["times"], dtype=np.float64),
        eta=np.asarray(payload["eta"], dtype=np.float64),
        xi=np.asarray(payload["xi"], dtype=np.float64),
        gxi=np.asarray(payload["gxi"], dtype=np.float64),
    )


def _rollout_pair(saved_spacing: float) -> tuple[
    RefinementTrajectory,
    RefinementTrajectory,
]:
    nx = 32
    x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
    eta, xi = stokes_eta_xi(
        x=x,
        time=jnp.asarray(0.0, dtype=jnp.float64),
        n0=1,
        a0=0.02,
        length=LENGTH,
        depth=DEPTH,
        gravity=GRAVITY,
        ichoi=1,
    )
    initial_state = State(eta=eta, xi=xi)
    params = make_solver_params(
        nx=nx,
        length=LENGTH,
        depth=DEPTH,
        gravity=GRAVITY,
        dno_order=1,
        pad_factor=2,
        filter_fraction=0.5,
    )
    times = jnp.asarray([0.0, saved_spacing], dtype=jnp.float64)
    common = {
        "initial_state": initial_state,
        "times": times,
        "params": params,
        "save_gxi": True,
        "method": "gl2_if",
        "implicit_iterations": 8,
    }
    coarse = rollout(**common, substeps_per_interval=1)
    fine = rollout(**common, substeps_per_interval=2)
    jax.block_until_ready(fine["gxi"])
    return _as_refinement_trajectory(coarse), _as_refinement_trajectory(fine)


class PaperDatasetAcceptanceSmokeTest(unittest.TestCase):
    def test_finite_stokes_constructor_support(self) -> None:
        length = 164.0
        k0 = 2.0 * np.pi * 14 / length
        wavelength = 2.0 * np.pi / k0
        rejected_height = finite_depth_stokes_wave_height(
            k0, depth=1.0, gravity=1.0, a0=0.22543466384990768
        )
        accepted_height = finite_depth_stokes_wave_height(
            k0, depth=1.0, gravity=1.0, a0=0.085
        )
        rejected_metrics, rejected_decision = evaluate_finite_stokes_support(
            wave_height_upper_bound=float(rejected_height),
            wavelength=float(wavelength),
            depth=1.0,
        )
        accepted_metrics, accepted_decision = evaluate_finite_stokes_support(
            wave_height_upper_bound=float(accepted_height),
            wavelength=float(wavelength),
            depth=1.0,
        )

        self.assertFalse(rejected_decision.accepted)
        self.assertFalse(
            bool(
                finite_depth_stokes_in_ursell_support(
                    k0, depth=1.0, gravity=1.0, a0=0.22543466384990768
                )
            )
        )
        self.assertGreater(rejected_metrics.ursell_upper_bound, 96.8)
        self.assertTrue(accepted_decision.accepted)
        self.assertTrue(
            bool(
                finite_depth_stokes_in_ursell_support(
                    k0, depth=1.0, gravity=1.0, a0=0.085
                )
            )
        )
        self.assertLess(accepted_metrics.ursell_upper_bound, 25.0)

    def test_finite_stokes_harmonics_vectorize_over_parameter_draws(self) -> None:
        k0 = 2.0 * np.pi * np.asarray([14.0, 14.0]) / 164.0
        amplitudes = np.asarray(
            [0.22543466384990768, 0.09589740544385852]
        )
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

    def test_small_step_pair_passes(self) -> None:
        coarse, fine = _rollout_pair(saved_spacing=0.1)
        metrics, decision = evaluate_temporal_refinement(
            coarse,
            fine,
            depth=DEPTH,
            gravity=GRAVITY,
            length=LENGTH,
            maximum_wavenumber=8.0,
        )

        self.assertTrue(decision.accepted)
        self.assertLess(metrics.maximum_error, 1e-6)

    def test_large_step_pair_fails(self) -> None:
        coarse, fine = _rollout_pair(saved_spacing=2.0)
        metrics, decision = evaluate_temporal_refinement(
            coarse,
            fine,
            depth=DEPTH,
            gravity=GRAVITY,
            length=LENGTH,
            maximum_wavenumber=8.0,
        )

        self.assertFalse(decision.accepted)
        self.assertGreater(metrics.maximum_error, 2e-3)


if __name__ == "__main__":
    unittest.main()
