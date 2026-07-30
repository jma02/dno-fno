"""CPU tests for the paper-corpus Benjamin--Feir construction.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests_benjamin_feir_jcp09
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

from solver.data.stokes_truth_jax import stokes_eta_xi  # noqa: E402
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    build_initial_conditions,
    deep_water_proxy_depth,
    feasible_mode_pairs,
    instability_band_fraction,
    is_supported,
    sample_parameters,
)
from solver.gen_data.pipeline.acceptance import (  # noqa: E402
    RefinementTrajectory,
    evaluate_temporal_refinement,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    rollout,
)

jax.config.update("jax_enable_x64", True)

LENGTH = 2.0 * np.pi
GRAVITY = 1.0


def _canonical_parameters() -> dict[str, np.ndarray]:
    return {
        "n_carr": np.asarray([9], dtype=np.int32),
        "side_offset": np.asarray([2], dtype=np.int32),
        "n_l": np.asarray([7], dtype=np.int32),
        "n_r": np.asarray([11], dtype=np.int32),
        "eps_carrier": np.asarray([0.13], dtype=np.float64),
        "eps_pert": np.asarray([0.10], dtype=np.float64),
        "phase": np.asarray([-np.pi / 4.0], dtype=np.float64),
        "depth": np.asarray([5.0], dtype=np.float64),
    }


def _canonical_carrier(x: jax.Array) -> tuple[jax.Array, jax.Array, float]:
    carrier_mode = 9
    carrier_wavenumber = float(carrier_mode)
    carrier_amplitude = 0.13 / carrier_wavenumber
    bare_amplitude = carrier_amplitude
    for _ in range(8):
        bare_steepness = carrier_wavenumber * bare_amplitude
        factor = (
            1.0
            + bare_steepness**2 / 8.0
            + 121.0 * bare_steepness**4 / 192.0
        )
        bare_amplitude = carrier_amplitude / factor
    eta, xi = stokes_eta_xi(
        x=x,
        time=jnp.asarray(0.0, dtype=x.dtype),
        n0=carrier_mode,
        a0=bare_amplitude,
        length=LENGTH,
        depth=deep_water_proxy_depth(LENGTH),
        gravity=GRAVITY,
        ichoi=0,
    )
    return eta, xi, carrier_amplitude


def _as_refinement_trajectory(
    payload: dict[str, jax.Array],
) -> RefinementTrajectory:
    return RefinementTrajectory(
        times=np.asarray(payload["times"], dtype=np.float64),
        eta=np.asarray(payload["eta"], dtype=np.float64),
        xi=np.asarray(payload["xi"], dtype=np.float64),
        gxi=np.asarray(payload["gxi"], dtype=np.float64),
    )


def _fixed_band_relative_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    maximum_mode: int,
) -> float:
    coarse_coefficients = np.fft.rfft(coarse) / coarse.shape[-1]
    fine_coefficients = np.fft.rfft(fine) / fine.shape[-1]
    difference = coarse_coefficients[: maximum_mode + 1] - fine_coefficients[
        : maximum_mode + 1
    ]
    reference = fine_coefficients[: maximum_mode + 1]
    numerator = np.sqrt(
        np.abs(difference[0]) ** 2
        + 2.0 * np.sum(np.abs(difference[1:]) ** 2)
    )
    denominator = np.sqrt(
        np.abs(reference[0]) ** 2
        + 2.0 * np.sum(np.abs(reference[1:]) ** 2)
    )
    return float(numerator / denominator)


class BenjaminFeirJCP09Test(unittest.TestCase):
    def test_support_is_exactly_the_declared_instability_band(self) -> None:
        self.assertTrue(bool(is_supported(9, 2, 0.13, 0.10)))
        self.assertTrue(bool(is_supported(10, 2, 0.11, 0.10)))
        self.assertFalse(bool(is_supported(9, 4, 0.13, 0.10)))
        self.assertFalse(bool(is_supported(4, 4, 0.13, 0.10)))
        self.assertAlmostEqual(
            float(instability_band_fraction(9, 2, 0.13)),
            0.6043647702,
            places=9,
        )
        pairs = {tuple(pair) for pair in feasible_mode_pairs().tolist()}
        self.assertIn((10, 2), pairs)

    def test_sampler_never_rewrites_or_leaves_support(self) -> None:
        parameters = sample_parameters(
            np.random.default_rng(20260725),
            batch_size=8192,
            length=LENGTH,
        )
        supported = is_supported(
            parameters["n_carr"],
            parameters["side_offset"],
            parameters["eps_carrier"],
            parameters["eps_pert"],
        )
        self.assertTrue(np.all(supported))
        self.assertTrue(np.all(parameters["n_l"] >= 1))
        self.assertTrue(
            np.all(parameters["n_r"] == parameters["n_carr"] + parameters["side_offset"])
        )
        self.assertTrue(
            np.allclose(
                parameters["depth"],
                deep_water_proxy_depth(LENGTH),
                rtol=0.0,
                atol=0.0,
            )
        )
        self.assertIn(
            (10, 2),
            set(
                zip(
                    parameters["n_carr"].tolist(),
                    parameters["side_offset"].tolist(),
                    strict=True,
                )
            ),
        )

    def test_canonical_state_is_literal_equation_33(self) -> None:
        nx = 256
        x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
        eta, xi = build_initial_conditions(
            x=x,
            parameters=_canonical_parameters(),
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        eta_carrier, xi_carrier, amplitude = _canonical_carrier(x)
        phase_left = 7.0 * x - np.pi / 4.0
        phase_right = 11.0 * x - np.pi / 4.0
        sideband_amplitude = 0.10 * amplitude
        expected_eta = (
            eta_carrier
            + sideband_amplitude * jnp.cos(phase_left)
            + sideband_amplitude * jnp.cos(phase_right)
        )
        expected_xi = (
            xi_carrier
            + sideband_amplitude
            / jnp.sqrt(7.0)
            * jnp.exp(7.0 * expected_eta)
            * jnp.sin(phase_left)
            + sideband_amplitude
            / jnp.sqrt(11.0)
            * jnp.exp(11.0 * expected_eta)
            * jnp.sin(phase_right)
        )
        expected_xi -= jnp.mean(expected_xi)
        np.testing.assert_allclose(eta[0], expected_eta, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(xi[0], expected_xi, rtol=2e-14, atol=2e-14)
        self.assertLess(float(jnp.abs(jnp.mean(xi[0]))), 1e-15)

    def test_constructor_is_batched_periodic_and_has_no_empirical_cross_modes(
        self,
    ) -> None:
        nx = 256
        shift = 13
        x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
        parameters = sample_parameters(
            np.random.default_rng(11),
            batch_size=4,
            length=LENGTH,
        )
        eta, xi = build_initial_conditions(
            x=x,
            parameters=parameters,
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        translated_eta, translated_xi = build_initial_conditions(
            x=x + shift * LENGTH / nx,
            parameters=parameters,
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        np.testing.assert_allclose(
            translated_eta,
            np.roll(np.asarray(eta), -shift, axis=-1),
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            translated_xi,
            np.roll(np.asarray(xi), -shift, axis=-1),
            rtol=2e-13,
            atol=2e-13,
        )

        canonical_eta, _ = build_initial_conditions(
            x=x,
            parameters=_canonical_parameters(),
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        eta_coefficients = np.fft.rfft(np.asarray(canonical_eta[0])) / nx
        self.assertLess(abs(eta_coefficients[16]), 1e-15)
        self.assertLess(abs(eta_coefficients[20]), 1e-15)

    def test_canonical_fixed_band_spatial_refinement(self) -> None:
        states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for nx in (128, 256):
            x, wavenumbers = build_grid(nx, LENGTH)
            eta, xi = build_initial_conditions(
                x=jnp.asarray(x, dtype=jnp.float64),
                parameters=_canonical_parameters(),
                length=LENGTH,
                gravity=GRAVITY,
                dtype=jnp.float64,
            )
            gxi = dno_series_eval(
                eta[0],
                xi[0],
                jnp.asarray(wavenumbers, dtype=jnp.float64),
                deep_water_proxy_depth(LENGTH),
                6,
                pad_factor=8,
            )
            states.append(
                (
                    np.asarray(eta[0]),
                    np.asarray(xi[0]),
                    np.asarray(gxi),
                )
            )
        errors = tuple(
            _fixed_band_relative_error(states[0][index], states[1][index], 32)
            for index in range(3)
        )
        self.assertLess(max(errors), 1e-7, msg=f"fixed-band errors: {errors}")

    def test_canonical_short_time_refinement(self) -> None:
        nx = 128
        x, wavenumbers = build_grid(nx, LENGTH)
        eta, xi = build_initial_conditions(
            x=jnp.asarray(x, dtype=jnp.float64),
            parameters=_canonical_parameters(),
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        depth = deep_water_proxy_depth(LENGTH)
        wavenumbers = jnp.asarray(wavenumbers, dtype=jnp.float64)
        solver_parameters = SolverParams(
            nx=nx,
            length=LENGTH,
            depth=depth,
            gravity=GRAVITY,
            dno_order=6,
            pad_factor=8,
            filter_fraction=0.25,
            k=wavenumbers,
            g0=make_linear_dno_symbol(wavenumbers, depth),
        )
        common = {
            "initial_state": State(eta=eta[0], xi=xi[0]),
            "times": jnp.asarray([0.0, 0.08], dtype=jnp.float64),
            "params": solver_parameters,
            "save_gxi": True,
            "method": "gl2_if",
            "implicit_iterations": 8,
        }
        coarse = rollout(**common, substeps_per_interval=8)
        fine = rollout(**common, substeps_per_interval=16)
        jax.block_until_ready(fine["gxi"])
        metrics, decision = evaluate_temporal_refinement(
            _as_refinement_trajectory(coarse),
            _as_refinement_trajectory(fine),
            depth=depth,
            gravity=GRAVITY,
            length=LENGTH,
            maximum_wavenumber=32.0,
        )
        self.assertTrue(decision.accepted)
        self.assertLess(metrics.maximum_error, 1.5e-4)


if __name__ == "__main__":
    unittest.main()
