"""Focused CPU tests for the JONSWAP/TMA constructor."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.jonswap_tma import (
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
    JonswapTmaParameters,
    JonswapTmaState,
    ResolvedBand,
    build_jonswap_tma_initial_condition,
    finite_depth_angular_frequency,
    finite_depth_group_velocity,
    jonswap_tma_spectrum,
    positive_mode_wavenumbers,
    tma_depth_factor,
)


FloatArray = NDArray[np.float64]
LENGTH = 2.0 * np.pi
BAND = ResolvedBand(LENGTH, PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER)


def periodic_grid(nx: int) -> FloatArray:
    return np.arange(nx, dtype=np.float64) * LENGTH / nx


def random_phases(seed: int) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    number_of_modes = positive_mode_wavenumbers(band=BAND).size
    return (
        rng.uniform(0.0, 2.0 * np.pi, size=number_of_modes),
        rng.uniform(0.0, 2.0 * np.pi, size=number_of_modes),
    )


def linear_hamiltonian(
    eta: FloatArray,
    xi: FloatArray,
    *,
    depth: float,
    gravity: float,
) -> float:
    nx = eta.size
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=LENGTH / nx)
    symbol = wavenumbers * np.tanh(wavenumbers * depth)
    flat_dno_xi = np.fft.irfft(symbol * np.fft.rfft(xi), n=nx)
    return float(0.5 * LENGTH * np.mean(xi * flat_dno_xi + gravity * eta**2))


class JonswapTmaFormulaTest(unittest.TestCase):
    def test_paper_spectrum_uses_the_fixed_sharp_relative_frequency_band(self) -> None:
        self.assertEqual(BAND.maximum_wavenumber, 128.0)
        self.assertEqual(PAPER_RESOLVED_BAND_QUADRATURE_ORDER, 16)
        self.assertEqual(
            (PAPER_RELATIVE_FREQUENCY_MINIMUM, PAPER_RELATIVE_FREQUENCY_MAXIMUM),
            (0.5, 2.5),
        )
        parameters = JonswapTmaParameters(0.2, 0.016, 9.0, 3.3, 0.5)
        spectrum = jonswap_tma_spectrum(parameters, band=BAND)
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

        self.assertEqual(spectrum.wavenumbers.shape, spectrum.energy_fractions.shape)
        self.assertGreaterEqual(float(np.min(spectrum.energy_fractions)), 0.0)
        self.assertAlmostEqual(float(np.sum(spectrum.energy_fractions)), 1.0)
        self.assertTrue(
            np.all(spectrum.energy_fractions[frequencies > 2.6 * peak_frequency] == 0.0)
        )
        self.assertTrue(
            np.all(spectrum.energy_fractions[frequencies < 0.4 * peak_frequency] == 0.0)
        )

    def test_tma_shallow_and_deep_limits(self) -> None:
        shallow = np.asarray([1.0e-5, 2.0e-5])
        np.testing.assert_allclose(
            tma_depth_factor(shallow) / (0.5 * shallow**2),
            np.ones_like(shallow),
            rtol=1.0e-9,
            atol=0.0,
        )
        self.assertEqual(float(tma_depth_factor(np.asarray([0.0]))[0]), 0.0)
        self.assertAlmostEqual(
            float(tma_depth_factor(np.asarray([30.0]))[0]), 1.0, places=14
        )

    def test_group_velocity_is_the_dispersion_derivative(self) -> None:
        wavenumbers = np.asarray([0.8, 3.0, 12.0])
        depth = 0.7
        step = 1.0e-6
        finite_difference = (
            finite_depth_angular_frequency(wavenumbers + step, depth=depth, gravity=1.0)
            - finite_depth_angular_frequency(
                wavenumbers - step, depth=depth, gravity=1.0
            )
        ) / (2.0 * step)
        np.testing.assert_allclose(
            finite_depth_group_velocity(wavenumbers, depth=depth, gravity=1.0),
            finite_difference,
            rtol=3.0e-10,
            atol=2.0e-10,
        )


class JonswapTmaStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = JonswapTmaParameters(0.2, 0.02, 9.0, 3.3, 0.5)
        self.phase_right, self.phase_left = random_phases(20260725)

    def build(self, nx: int = 1024) -> JonswapTmaState:
        return build_jonswap_tma_initial_condition(
            periodic_grid(nx),
            parameters=self.parameters,
            phase_right=self.phase_right,
            phase_left=self.phase_left,
            band=BAND,
        )

    def test_linear_energy_is_phase_independent(self) -> None:
        expected = LENGTH * (self.parameters.significant_height / 4.0) ** 2
        state = self.build()
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
        phase_right, phase_left = random_phases(9)
        second = build_jonswap_tma_initial_condition(
            periodic_grid(1024),
            parameters=self.parameters,
            phase_right=phase_right,
            phase_left=phase_left,
            band=BAND,
        )
        self.assertAlmostEqual(
            linear_hamiltonian(
                second.eta,
                second.xi,
                depth=self.parameters.depth,
                gravity=1.0,
            ),
            expected,
            places=14,
        )

    def test_counterpropagating_cancellation_is_not_rescaled(self) -> None:
        state = build_jonswap_tma_initial_condition(
            periodic_grid(1024),
            parameters=self.parameters,
            phase_right=self.phase_right,
            phase_left=self.phase_right + np.pi,
            band=BAND,
        )
        self.assertLess(float(np.linalg.norm(state.eta)), 1.0e-13)
        self.assertGreater(float(np.linalg.norm(state.xi)), 1.0e-3)
        self.assertAlmostEqual(
            linear_hamiltonian(
                state.eta,
                state.xi,
                depth=self.parameters.depth,
                gravity=1.0,
            ),
            LENGTH * (self.parameters.significant_height / 4.0) ** 2,
            places=14,
        )

    def test_each_direction_satisfies_the_linear_kinematic_equation(self) -> None:
        x = periodic_grid(1024)
        wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=LENGTH / x.size)
        flat_symbol = wavenumbers * np.tanh(wavenumbers * self.parameters.depth)
        spectrum = jonswap_tma_spectrum(self.parameters, band=BAND)
        mode_frequency = np.sqrt(
            spectrum.wavenumbers * np.tanh(spectrum.wavenumbers * self.parameters.depth)
        )
        variance = (self.parameters.significant_height / 4.0) ** 2
        amplitude = np.sqrt(2.0 * variance * spectrum.energy_fractions)
        for right_fraction in (0.0, 1.0):
            parameters = self.parameters._replace(right_moving_fraction=right_fraction)
            state = build_jonswap_tma_initial_condition(
                x,
                parameters=parameters,
                phase_right=self.phase_right,
                phase_left=self.phase_left,
                band=BAND,
            )
            flat_dno_xi = np.fft.irfft(flat_symbol * np.fft.rfft(state.xi), n=x.size)
            phases = self.phase_right if right_fraction == 1.0 else self.phase_left
            direction = 1.0 if right_fraction == 1.0 else -1.0
            eta_time_derivative = direction * np.sum(
                mode_frequency[:, None]
                * amplitude[:, None]
                * np.sin(spectrum.wavenumbers[:, None] * x[None, :] + phases[:, None]),
                axis=0,
            )
            with self.subTest(right_fraction=right_fraction):
                np.testing.assert_allclose(
                    flat_dno_xi,
                    eta_time_derivative,
                    rtol=0.0,
                    atol=2.0e-14,
                )

    def test_translation_covariance_and_grid_independence(self) -> None:
        nx = 1024
        shift_cells = 13
        original = self.build(nx)
        translated = build_jonswap_tma_initial_condition(
            periodic_grid(nx) + shift_cells * LENGTH / nx,
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


if __name__ == "__main__":
    unittest.main()
