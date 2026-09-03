"""CPU tests for the paper-dataset Benjamin--Feir construction.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests.test_benjamin_feir_jcp09
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

from solver.reference_solutions.stokes_wave import stokes_eta_xi  # noqa: E402
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    BENJAMIN_FEIR_MODE_PAIRS,
    build_initial_conditions,
    deep_water_proxy_depth,
    focused_steepness_carrier_upper_bound,
    focused_steepness_proxy,
    instability_band_fraction,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
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
        "translation": np.asarray([0.0], dtype=np.float64),
        "depth": np.asarray([5.0], dtype=np.float64),
    }


def _canonical_carrier(x: jax.Array) -> tuple[jax.Array, jax.Array, float]:
    carrier_mode = 9
    carrier_wavenumber = float(carrier_mode)
    carrier_amplitude = 0.13 / carrier_wavenumber
    bare_amplitude = carrier_amplitude
    for _ in range(8):
        bare_steepness = carrier_wavenumber * bare_amplitude
        factor = 1.0 + bare_steepness**2 / 8.0 + 121.0 * bare_steepness**4 / 192.0
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


def _fixed_band_relative_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    maximum_mode: int,
) -> float:
    coarse_coefficients = np.fft.rfft(coarse) / coarse.shape[-1]
    fine_coefficients = np.fft.rfft(fine) / fine.shape[-1]
    difference = (
        coarse_coefficients[: maximum_mode + 1] - fine_coefficients[: maximum_mode + 1]
    )
    reference = fine_coefficients[: maximum_mode + 1]
    numerator = np.sqrt(
        np.abs(difference[0]) ** 2 + 2.0 * np.sum(np.abs(difference[1:]) ** 2)
    )
    denominator = np.sqrt(
        np.abs(reference[0]) ** 2 + 2.0 * np.sum(np.abs(reference[1:]) ** 2)
    )
    return float(numerator / denominator)


class BenjaminFeirJCP09Test(unittest.TestCase):
    def test_mode_pairs_intersect_the_instability_band(self) -> None:
        self.assertAlmostEqual(
            float(instability_band_fraction(9, 2, 0.13)),
            0.6043647702,
            places=9,
        )
        pairs = set(BENJAMIN_FEIR_MODE_PAIRS)
        self.assertEqual(len(pairs), 66)
        self.assertIn((10, 2), pairs)

    def test_focused_proxy_has_the_closed_form_sample_support_endpoint(self) -> None:
        focused_limit = (1.0 + np.sqrt(2.0)) / 10.0
        self.assertAlmostEqual(
            float(focused_steepness_proxy(10, 2, 0.10)),
            focused_limit,
            places=15,
        )
        self.assertAlmostEqual(
            float(
                focused_steepness_carrier_upper_bound(
                    10,
                    2,
                    focused_steepness_limit=focused_limit,
                )
            ),
            0.10,
            places=15,
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

    def test_constructor_is_batched_translation_covariant_and_has_no_cross_modes(
        self,
    ) -> None:
        nx = 256
        shift = 13
        x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
        parameters = {
            name: np.repeat(values, 4)
            for name, values in _canonical_parameters().items()
        }
        parameters["translation"] = np.asarray(
            [0.0, 0.4, 1.2, 2.7],
            dtype=np.float64,
        )
        eta, xi = build_initial_conditions(
            x=x,
            parameters=parameters,
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        translated_parameters = {
            **parameters,
            "translation": np.mod(
                parameters["translation"] + shift * LENGTH / nx,
                LENGTH,
            ),
        }
        translated_eta, translated_xi = build_initial_conditions(
            x=x,
            parameters=translated_parameters,
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        np.testing.assert_allclose(
            translated_eta,
            np.roll(np.asarray(eta), shift, axis=-1),
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            translated_xi,
            np.roll(np.asarray(xi), shift, axis=-1),
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

    def test_relative_sideband_and_resonant_quartet_phases_are_fixed(
        self,
    ) -> None:
        nx = 256
        translation = 0.371
        x = jnp.asarray(LENGTH * np.arange(nx) / nx, dtype=jnp.float64)
        parameters = _canonical_parameters()
        parameters["translation"] = np.asarray(
            [translation],
            dtype=np.float64,
        )
        eta, _ = build_initial_conditions(
            x=x,
            parameters=parameters,
            length=LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        coefficients = np.fft.rfft(np.asarray(eta[0])) / nx

        carrier_phase = coefficients[9] * np.exp(1j * 9.0 * translation)
        left_phase = coefficients[7] * np.exp(1j * 7.0 * translation)
        right_phase = coefficients[11] * np.exp(1j * 11.0 * translation)
        np.testing.assert_allclose(
            carrier_phase / abs(carrier_phase),
            1.0 + 0.0j,
            rtol=0.0,
            atol=2e-13,
        )
        fixed_sideband_phase = np.exp(-1j * np.pi / 4.0)
        np.testing.assert_allclose(
            left_phase / abs(left_phase),
            fixed_sideband_phase,
            rtol=0.0,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            right_phase / abs(right_phase),
            fixed_sideband_phase,
            rtol=0.0,
            atol=2e-13,
        )
        quartet = coefficients[7] * coefficients[11] / coefficients[9] ** 2
        np.testing.assert_allclose(
            quartet / abs(quartet),
            -1j,
            rtol=0.0,
            atol=2e-13,
        )

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


if __name__ == "__main__":
    unittest.main()
