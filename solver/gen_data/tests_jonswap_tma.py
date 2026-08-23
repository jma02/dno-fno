"""Focused CPU tests for the public JONSWAP/TMA constructor."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.jonswap_tma import (
    PAPER_PEAK_STEEPNESS_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
    PAPER_RESOLVED_BAND_TRANSITION_FRACTION,
    PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
    PAPER_RESOLVED_BAND_WINDOW,
    PAPER_SHALLOW_PEAK_MODES,
    JonswapTmaParameters,
    JonswapTmaState,
    ResolvedBand,
    build_jonswap_tma_initial_condition,
    finite_depth_angular_frequency,
    finite_depth_group_velocity,
    is_in_paper_support,
    jonswap_tma_spectrum,
    paper_support_violations,
    positive_mode_wavenumbers,
    relative_frequency_interval_fits,
    resolved_band_window,
    sample_jonswap_tma_phases,
    tma_depth_factor,
)


FloatArray = NDArray[np.float64]
LENGTH = 2.0 * np.pi
BAND = ResolvedBand(
    length=LENGTH,
    maximum_wavenumber=128.0,
    transition_wavenumber=96.0,
)


def periodic_grid(nx: int) -> FloatArray:
    """Return the endpoint-excluded grid on the fixed periodic domain."""

    return np.arange(nx, dtype=np.float64) * LENGTH / nx


def linear_hamiltonian(
    eta: FloatArray,
    xi: FloatArray,
    *,
    depth: float,
    gravity: float,
) -> float:
    """Evaluate the discrete flat-surface Hamiltonian."""

    nx = eta.size
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=LENGTH / nx)
    symbol = wavenumbers * np.tanh(wavenumbers * depth)
    flat_dno_xi = np.fft.irfft(symbol * np.fft.rfft(xi), n=nx)
    return float(0.5 * LENGTH * np.mean(xi * flat_dno_xi + gravity * eta**2))


class JonswapTmaFormulaTest(unittest.TestCase):
    def test_paper_band_is_the_frozen_support_aware_window(self) -> None:
        band = ResolvedBand()
        self.assertEqual(
            band.maximum_wavenumber,
            PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
        )
        self.assertEqual(
            band.transition_wavenumber,
            PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
        )
        self.assertEqual(band.quadrature_order, PAPER_RESOLVED_BAND_QUADRATURE_ORDER)
        self.assertEqual(PAPER_RESOLVED_BAND_WINDOW, "cosine_squared_density_v1")
        self.assertEqual(PAPER_RESOLVED_BAND_TRANSITION_FRACTION, 0.75)
        self.assertEqual(
            PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
            4.0 * max(PAPER_SHALLOW_PEAK_MODES),
        )
        self.assertEqual(
            PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER
            - PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
            32.0,
        )

        depths = np.geomspace(1.0e-3, 1.0e3, 25)
        frequency_ratio = np.asarray(
            [
                finite_depth_angular_frequency(
                    np.asarray([PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER]),
                    depth=float(depth),
                    gravity=1.0,
                )[0]
                / finite_depth_angular_frequency(
                    np.asarray([max(PAPER_SHALLOW_PEAK_MODES)]),
                    depth=float(depth),
                    gravity=1.0,
                )[0]
                for depth in depths
            ],
            dtype=np.float64,
        )
        self.assertTrue(np.all(frequency_ratio >= 2.0))

    def test_tma_shallow_and_deep_limits(self) -> None:
        shallow = np.asarray([1.0e-5, 2.0e-5], dtype=np.float64)
        shallow_ratio = tma_depth_factor(shallow) / (0.5 * shallow**2)
        np.testing.assert_allclose(
            shallow_ratio, np.ones_like(shallow), rtol=1.0e-9, atol=0.0
        )
        self.assertEqual(float(tma_depth_factor(np.asarray([0.0]))[0]), 0.0)
        self.assertAlmostEqual(
            float(tma_depth_factor(np.asarray([30.0]))[0]), 1.0, places=14
        )

    def test_group_velocity_is_dispersion_derivative(self) -> None:
        wavenumbers = np.asarray([0.8, 3.0, 12.0], dtype=np.float64)
        depth = 0.7
        gravity = 1.0
        step = 1.0e-6
        finite_difference = (
            finite_depth_angular_frequency(
                wavenumbers + step, depth=depth, gravity=gravity
            )
            - finite_depth_angular_frequency(
                wavenumbers - step, depth=depth, gravity=gravity
            )
        ) / (2.0 * step)
        analytic = finite_depth_group_velocity(
            wavenumbers, depth=depth, gravity=gravity
        )
        np.testing.assert_allclose(
            analytic, finite_difference, rtol=3.0e-10, atol=2.0e-10
        )

    def test_spectrum_is_normalized_nonnegative_and_windowed(self) -> None:
        parameters = JonswapTmaParameters(
            depth=0.2,
            significant_height=0.02,
            peak_wavenumber=9.0,
            peak_enhancement=3.3,
            right_moving_fraction=0.5,
        )
        spectrum = jonswap_tma_spectrum(parameters, band=BAND)
        self.assertEqual(spectrum.wavenumbers.shape, spectrum.energy_fractions.shape)
        self.assertGreaterEqual(float(np.min(spectrum.energy_fractions)), 0.0)
        self.assertAlmostEqual(float(np.sum(spectrum.energy_fractions)), 1.0)
        self.assertLessEqual(
            float(np.max(spectrum.wavenumbers)), BAND.maximum_wavenumber
        )

        probe = np.asarray([0.0, 96.0, 112.0, 128.0, 140.0])
        expected = np.asarray([1.0, 1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(
            resolved_band_window(probe, band=BAND), expected, atol=1.0e-15
        )

    def test_relative_frequency_spectrum_is_normalized_and_truncated(self) -> None:
        parameters = JonswapTmaParameters(0.2, 0.016, 9.0, 3.3, 0.5)
        interval = (
            PAPER_RELATIVE_FREQUENCY_MINIMUM,
            PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )
        spectrum = jonswap_tma_spectrum(
            parameters,
            band=BAND,
            relative_frequency_interval=interval,
        )
        frequencies = finite_depth_angular_frequency(
            spectrum.wavenumbers,
            depth=parameters.depth,
            gravity=1.0,
        )
        peak_frequency = finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber]),
            depth=parameters.depth,
            gravity=1.0,
        )[0]
        self.assertAlmostEqual(float(np.sum(spectrum.energy_fractions)), 1.0)
        self.assertTrue(
            np.all(spectrum.energy_fractions[frequencies > 2.6 * peak_frequency] == 0.0)
        )
        self.assertTrue(
            np.all(spectrum.energy_fractions[frequencies < 0.4 * peak_frequency] == 0.0)
        )

    def test_relative_frequency_spectrum_fails_if_upper_endpoint_is_unresolved(
        self,
    ) -> None:
        parameters = JonswapTmaParameters(1.5 / 24.0, 0.004, 24.0, 3.3, 0.5)
        self.assertFalse(
            relative_frequency_interval_fits(
                parameters,
                band=BAND,
                relative_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
            )
        )
        with self.assertRaisesRegex(ValueError, "exceeds the resolved band"):
            jonswap_tma_spectrum(
                parameters,
                band=BAND,
                relative_frequency_interval=(
                    PAPER_RELATIVE_FREQUENCY_MINIMUM,
                    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
                ),
            )

class JonswapTmaSupportTest(unittest.TestCase):
    def test_three_declared_strata(self) -> None:
        examples = {
            "shallow": JonswapTmaParameters(0.04, 0.008, 20.0, 3.3, 0.5),
            "finite": JonswapTmaParameters(0.2, 0.016, 9.0, 1.0, 1.0),
            "deep": JonswapTmaParameters(10.0, 0.016, 9.0, 5.0, 0.0),
        }
        for stratum, parameters in examples.items():
            with self.subTest(stratum=stratum):
                self.assertTrue(is_in_paper_support(parameters, stratum=stratum))
                self.assertEqual(
                    paper_support_violations(parameters, stratum=stratum), ()
                )

    def test_global_peak_steepness_boundary_is_closed(self) -> None:
        for stratum, template in {
            "shallow": JonswapTmaParameters(0.05, 0.008, 20.0, 3.3, 0.5),
            "finite": JonswapTmaParameters(0.5, 0.02, 8.0, 3.3, 0.5),
            "deep": JonswapTmaParameters(10.0, 0.02, 8.0, 3.3, 0.5),
        }.items():
            boundary_height = (
                2.0 * PAPER_PEAK_STEEPNESS_MAXIMUM / template.peak_wavenumber
            )
            inside = JonswapTmaParameters(
                template.depth,
                boundary_height,
                template.peak_wavenumber,
                template.peak_enhancement,
                template.right_moving_fraction,
            )
            with self.subTest(stratum=stratum, side="inside"):
                self.assertEqual(
                    paper_support_violations(inside, stratum=stratum), ()
                )
            outside = JonswapTmaParameters(
                template.depth,
                np.nextafter(boundary_height, np.inf),
                template.peak_wavenumber,
                template.peak_enhancement,
                template.right_moving_fraction,
            )
            with self.subTest(stratum=stratum, side="outside"):
                self.assertIn(
                    "JONSWAP/TMA k_p H_s/2 must not exceed 0.08",
                    paper_support_violations(outside, stratum=stratum),
                )

    def test_support_rejects_unsupported_discrete_parameters(self) -> None:

        unsupported_gamma = JonswapTmaParameters(0.2, 0.02, 9.0, 2.0, 0.5)

        self.assertIn(
            "peak_enhancement must be one of {1, 3.3, 5}",
            paper_support_violations(unsupported_gamma, stratum="finite"),
        )

    def test_relative_frequency_resolution_is_an_input_only_support_condition(
        self,
    ) -> None:
        unresolved = JonswapTmaParameters(1.5 / 24.0, 0.004, 24.0, 3.3, 0.5)
        self.assertIn(
            "JONSWAP/TMA relative upper frequency must fit in the resolved band",
            paper_support_violations(
                unresolved,
                stratum="shallow",
                band=BAND,
                relative_frequency_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
            ),
        )


class JonswapTmaStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = JonswapTmaParameters(
            depth=0.2,
            significant_height=0.02,
            peak_wavenumber=9.0,
            peak_enhancement=3.3,
            right_moving_fraction=0.5,
        )
        self.phase_right, self.phase_left = sample_jonswap_tma_phases(
            np.random.default_rng(20260725), band=BAND
        )

    def build(self, nx: int = 1024) -> JonswapTmaState:
        return build_jonswap_tma_initial_condition(
            periodic_grid(nx),
            parameters=self.parameters,
            phase_right=self.phase_right,
            phase_left=self.phase_left,
            band=BAND,
        )

    def test_linear_energy_is_phase_independent(self) -> None:
        state = self.build()
        expected = LENGTH * (self.parameters.significant_height / 4.0) ** 2
        actual = linear_hamiltonian(
            state.eta,
            state.xi,
            depth=self.parameters.depth,
            gravity=1.0,
        )
        self.assertAlmostEqual(actual, expected, places=14)

        second_right, second_left = sample_jonswap_tma_phases(
            np.random.default_rng(9), band=BAND
        )
        second = build_jonswap_tma_initial_condition(
            periodic_grid(1024),
            parameters=self.parameters,
            phase_right=second_right,
            phase_left=second_left,
            band=BAND,
        )
        second_energy = linear_hamiltonian(
            second.eta,
            second.xi,
            depth=self.parameters.depth,
            gravity=1.0,
        )
        self.assertAlmostEqual(second_energy, expected, places=14)

    def test_counterpropagating_cancellation_is_not_rescaled(self) -> None:
        phases = self.phase_right
        state = build_jonswap_tma_initial_condition(
            periodic_grid(1024),
            parameters=self.parameters,
            phase_right=phases,
            phase_left=phases + np.pi,
            band=BAND,
        )
        self.assertLess(float(np.linalg.norm(state.eta)), 1.0e-13)
        self.assertGreater(float(np.linalg.norm(state.xi)), 1.0e-3)
        expected = LENGTH * (self.parameters.significant_height / 4.0) ** 2
        self.assertAlmostEqual(
            linear_hamiltonian(
                state.eta,
                state.xi,
                depth=self.parameters.depth,
                gravity=1.0,
            ),
            expected,
            places=14,
        )

    def test_each_direction_satisfies_the_linear_kinematic_equation(self) -> None:
        x = periodic_grid(1024)
        wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=LENGTH / x.size)
        flat_symbol = wavenumbers * np.tanh(wavenumbers * self.parameters.depth)
        for right_fraction in (0.0, 1.0):
            parameters = JonswapTmaParameters(
                depth=self.parameters.depth,
                significant_height=self.parameters.significant_height,
                peak_wavenumber=self.parameters.peak_wavenumber,
                peak_enhancement=self.parameters.peak_enhancement,
                right_moving_fraction=right_fraction,
            )
            state = build_jonswap_tma_initial_condition(
                x,
                parameters=parameters,
                phase_right=self.phase_right,
                phase_left=self.phase_left,
                band=BAND,
            )
            g0_xi = np.fft.irfft(flat_symbol * np.fft.rfft(state.xi), n=x.size)
            mode_frequency = np.sqrt(
                state.spectrum.wavenumbers
                * np.tanh(state.spectrum.wavenumbers * parameters.depth)
            )
            if right_fraction == 1.0:
                eta_time_derivative = np.sum(
                    mode_frequency[:, None]
                    * state.amplitude_right[:, None]
                    * np.sin(
                        state.spectrum.wavenumbers[:, None] * x[None, :]
                        + state.phase_right[:, None]
                    ),
                    axis=0,
                )
            else:
                eta_time_derivative = -np.sum(
                    mode_frequency[:, None]
                    * state.amplitude_left[:, None]
                    * np.sin(
                        state.spectrum.wavenumbers[:, None] * x[None, :]
                        + state.phase_left[:, None]
                    ),
                    axis=0,
                )
            with self.subTest(right_fraction=right_fraction):
                np.testing.assert_allclose(
                    g0_xi,
                    eta_time_derivative,
                    rtol=0.0,
                    atol=2.0e-14,
                )

    def test_translation_covariance_and_fixed_band_spatial_agreement(self) -> None:
        nx = 1024
        shift_cells = 13
        shift = shift_cells * LENGTH / nx
        original = self.build(nx)
        translated = build_jonswap_tma_initial_condition(
            periodic_grid(nx) + shift,
            parameters=self.parameters,
            phase_right=self.phase_right,
            phase_left=self.phase_left,
            band=BAND,
        )
        np.testing.assert_allclose(
            translated.eta,
            np.roll(original.eta, -shift_cells),
            rtol=0.0,
            atol=4.0e-15,
        )
        np.testing.assert_allclose(
            translated.xi,
            np.roll(original.xi, -shift_cells),
            rtol=0.0,
            atol=4.0e-15,
        )

        coarse = self.build(512)
        for coarse_field, fine_field in (
            (coarse.eta, original.eta),
            (coarse.xi, original.xi),
        ):
            fine_at_coarse_points = fine_field[::2]
            relative_error = np.linalg.norm(
                coarse_field - fine_at_coarse_points
            ) / np.linalg.norm(fine_at_coarse_points)
            self.assertLess(float(relative_error), 2.0e-13)

    def test_all_constructed_modes_are_explicit(self) -> None:
        state = self.build()
        expected = positive_mode_wavenumbers(band=BAND)
        np.testing.assert_array_equal(state.spectrum.wavenumbers, expected)
        self.assertEqual(state.phase_right.shape, expected.shape)
        self.assertEqual(state.phase_left.shape, expected.shape)


if __name__ == "__main__":
    unittest.main()
