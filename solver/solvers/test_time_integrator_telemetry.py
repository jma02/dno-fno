"""CPU tests for convergence-controlled GL2 stage telemetry."""

from __future__ import annotations

import math
import unittest

import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import rollout_surrogate
from solver.solvers.dno_series_jax import dno_series_eval
from solver.solvers.time_integrator import (
    SpectralState,
    State,
    gauss_legendre_2_if_step,
    gauss_legendre_2_if_step_with_telemetry,
    make_solver_params,
    nonlinear_ramp_factor,
    relative_implicit_stage_residual,
    rollout,
)


class NonlinearRampTest(unittest.TestCase):
    def test_default_is_one_and_dommermuth_formula_is_literal(self) -> None:
        default = make_solver_params(16, 2.0 * math.pi, 1.0)
        ramped = make_solver_params(
            16,
            2.0 * math.pi,
            1.0,
            nonlinear_ramp_time=10.0,
            nonlinear_ramp_order=4,
        )

        self.assertEqual(float(nonlinear_ramp_factor(0.0, default)), 1.0)
        self.assertEqual(float(nonlinear_ramp_factor(0.0, ramped)), 0.0)
        np.testing.assert_allclose(
            nonlinear_ramp_factor(10.0, ramped),
            1.0 - math.exp(-1.0),
            rtol=1e-7,
            atol=0.0,
        )

    def test_simulationwise_times_and_ramp_times_pair_elementwise(self) -> None:
        params = make_solver_params(
            16,
            2.0 * math.pi,
            1.0,
            nonlinear_ramp_time=jnp.asarray([10.0, 20.0]),
            nonlinear_ramp_order=4,
        )
        factors = nonlinear_ramp_factor(jnp.asarray([10.0, 20.0]), params)

        self.assertEqual(factors.shape, (2, 1))
        np.testing.assert_allclose(
            factors[:, 0],
            np.full(2, 1.0 - math.exp(-1.0)),
            rtol=1e-7,
            atol=0.0,
        )


class RelativeImplicitStageResidualTest(unittest.TestCase):
    def test_is_componentwise_scale_invariant_and_zero_safe(self) -> None:
        stage1 = SpectralState(
            eta_hat=jnp.asarray([1.0, 2.0j]),
            xi_hat=jnp.asarray([3.0, -1.0j]),
        )
        stage2 = SpectralState(
            eta_hat=jnp.asarray([0.5, -1.0j]),
            xi_hat=jnp.asarray([2.0, 4.0j]),
        )
        mapped1 = SpectralState(
            eta_hat=stage1.eta_hat + jnp.asarray([0.01, -0.02j]),
            xi_hat=stage1.xi_hat + jnp.asarray([0.03, 0.01j]),
        )
        mapped2 = SpectralState(
            eta_hat=stage2.eta_hat + jnp.asarray([-0.02, 0.01j]),
            xi_hat=stage2.xi_hat + jnp.asarray([0.01, -0.04j]),
        )

        residual = relative_implicit_stage_residual(
            stage1,
            stage2,
            mapped1,
            mapped2,
        )

        def rescale(state: SpectralState) -> SpectralState:
            return SpectralState(
                eta_hat=7.0 * state.eta_hat,
                xi_hat=0.03 * state.xi_hat,
            )

        rescaled_residual = relative_implicit_stage_residual(
            rescale(stage1),
            rescale(stage2),
            rescale(mapped1),
            rescale(mapped2),
        )
        zero = SpectralState(
            eta_hat=jnp.zeros(4, dtype=stage1.eta_hat.dtype),
            xi_hat=jnp.zeros(4, dtype=stage1.xi_hat.dtype),
        )

        np.testing.assert_allclose(
            residual,
            rescaled_residual,
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertEqual(
            float(relative_implicit_stage_residual(zero, zero, zero, zero)),
            0.0,
        )


class GaussLegendreTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nx = 16
        self.x = 2.0 * jnp.pi * jnp.arange(self.nx) / self.nx
        self.params = make_solver_params(
            self.nx,
            2.0 * math.pi,
            1.0,
            dno_order=1,
            pad_factor=2,
        )
        self.zero = State(
            eta=jnp.zeros(self.nx, dtype=self.x.dtype),
            xi=jnp.zeros(self.nx, dtype=self.x.dtype),
        )
        self.mild = State(
            eta=0.03 * jnp.cos(self.x),
            xi=0.02 * jnp.sin(self.x),
        )

    def test_surrogate_true_dno_matches_fixed_truth_without_truth_ramping(self) -> None:
        initial = State(
            eta=jnp.stack((self.zero.eta, self.mild.eta)),
            xi=jnp.stack((self.zero.xi, self.mild.xi + 0.01)),
        )
        times = jnp.asarray([0.0, 0.1, 0.2])
        for fraction in (1.0, 2.0 / 3.0):
            with self.subTest(filter_fraction=fraction):
                params = self.params._replace(filter_fraction=fraction)

                def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
                    return dno_series_eval(
                        eta,
                        xi,
                        params.k,
                        params.depth,
                        params.dno_order,
                        pad_factor=params.pad_factor,
                    )

                truth = rollout(
                    initial,
                    times,
                    params,
                    save_gxi=False,
                    substeps_per_interval=2,
                    implicit_iterations=4,
                    zero_mean_xi=True,
                )
                surrogate = rollout_surrogate(
                    initial, times, params, predict, substeps=2
                )
                ramped = rollout_surrogate(
                    initial,
                    times,
                    params._replace(nonlinear_ramp_time=10.0),
                    predict,
                    substeps=2,
                )
                for field in ("eta", "xi"):
                    np.testing.assert_allclose(
                        surrogate[field], truth[field], rtol=1e-12, atol=1e-14
                    )
                for field in surrogate:
                    np.testing.assert_array_equal(ramped[field], surrogate[field])
                np.testing.assert_allclose(
                    surrogate["xi"].mean(axis=-1), 0.0, rtol=0.0, atol=1e-14
                )

    def test_zero_state_converges_without_picard_update(self) -> None:
        result = gauss_legendre_2_if_step_with_telemetry(
            self.zero,
            0.0,
            0.05,
            self.params,
            max_iterations=4,
            residual_tolerance=1e-12,
        )

        self.assertEqual(float(result.telemetry.residual), 0.0)
        self.assertEqual(int(result.telemetry.iterations), 0)
        self.assertTrue(bool(result.telemetry.converged))
        self.assertTrue(bool(result.telemetry.stage_finite))
        self.assertTrue(bool(result.telemetry.state_finite))
        self.assertFalse(bool(result.telemetry.hit_iteration_cap))
        np.testing.assert_array_equal(result.state.eta, self.zero.eta)
        np.testing.assert_array_equal(result.state.xi, self.zero.xi)

    def test_iteration_cap_marks_finite_but_unsolved_stage(self) -> None:
        result = gauss_legendre_2_if_step_with_telemetry(
            self.mild,
            0.0,
            0.05,
            self.params,
            max_iterations=0,
            residual_tolerance=1e-12,
        )

        self.assertGreater(float(result.telemetry.residual), 1e-12)
        self.assertEqual(int(result.telemetry.iterations), 0)
        self.assertFalse(bool(result.telemetry.converged))
        self.assertTrue(bool(result.telemetry.stage_finite))
        self.assertTrue(bool(result.telemetry.state_finite))
        self.assertTrue(bool(result.telemetry.hit_iteration_cap))

    def test_converged_step_agrees_with_fixed_eight_update_step(self) -> None:
        controlled = gauss_legendre_2_if_step_with_telemetry(
            self.mild,
            0.0,
            0.05,
            self.params,
            max_iterations=8,
            residual_tolerance=1e-12,
        )
        fixed = gauss_legendre_2_if_step(
            self.mild,
            0.0,
            0.05,
            self.params,
            iterations=8,
        )

        self.assertTrue(bool(controlled.telemetry.converged))
        self.assertLess(int(controlled.telemetry.iterations), 8)
        np.testing.assert_allclose(
            controlled.state.eta,
            fixed.eta,
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            controlled.state.xi,
            fixed.xi,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_nonfinite_stage_is_distinct_from_iteration_cap(self) -> None:
        bad_eta = self.mild.eta.at[0].set(jnp.nan)
        result = gauss_legendre_2_if_step_with_telemetry(
            State(eta=bad_eta, xi=self.mild.xi),
            0.0,
            0.05,
            self.params,
            max_iterations=4,
            residual_tolerance=1e-8,
        )

        self.assertFalse(bool(result.telemetry.converged))
        self.assertFalse(bool(result.telemetry.stage_finite))
        self.assertFalse(bool(result.telemetry.state_finite))
        self.assertFalse(bool(result.telemetry.hit_iteration_cap))

    def test_batched_rollout_records_every_actual_substep(self) -> None:
        initial = State(
            eta=jnp.stack((self.zero.eta, self.mild.eta)),
            xi=jnp.stack((self.zero.xi, self.mild.xi)),
        )
        result = rollout(
            initial,
            jnp.asarray([0.0, 0.1, 0.2]),
            self.params,
            save_gxi=False,
            substeps_per_interval=2,
            implicit_iterations=0,
            implicit_residual_tolerance=1e-12,
        )

        self.assertEqual(result["gl2_stage_residual"].shape, (4, 2))
        self.assertEqual(result["gl2_iterations"].shape, (4, 2))
        np.testing.assert_allclose(
            result["gl2_step_times"],
            np.asarray([0.0, 0.05, 0.1, 0.15]),
        )
        np.testing.assert_allclose(
            result["gl2_step_dts"],
            np.full(4, 0.05),
        )
        np.testing.assert_array_equal(
            result["gl2_all_stages_converged"],
            np.asarray([True, False]),
        )
        np.testing.assert_array_equal(
            result["gl2_first_failed_step"],
            np.asarray([-1, 0], dtype=np.int32),
        )
        self.assertTrue(bool(jnp.all(result["gl2_hit_iteration_cap"][:, 1])))

    def test_fixed_iteration_output_is_unchanged(self) -> None:
        params = make_solver_params(
            self.nx,
            2.0 * math.pi,
            1.3,
            dno_order=2,
            pad_factor=2,
        )
        state = State(
            eta=0.05 * jnp.cos(2.0 * self.x) + 0.01 * jnp.sin(3.0 * self.x),
            xi=0.04 * jnp.sin(2.0 * self.x) - 0.007 * jnp.cos(self.x),
        )

        result = gauss_legendre_2_if_step(
            state,
            0.17,
            0.031,
            params,
            iterations=4,
        )

        np.testing.assert_allclose(
            (
                float(jnp.linalg.norm(result.eta)),
                float(jnp.linalg.norm(result.xi)),
            ),
            (0.14425030059710908, 0.11483650773805007),
            rtol=2e-6,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
