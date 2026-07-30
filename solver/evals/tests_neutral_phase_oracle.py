"""Focused invariants for the phase-only oracle projection."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.neutral_phase_oracle import (
    localized_translation_component,
    modal_phase_component,
)

jax.config.update("jax_enable_x64", True)


def _grid(n: int = 256) -> tuple[jax.Array, jax.Array]:
    x = 2.0 * jnp.pi * jnp.arange(n, dtype=jnp.float64) / n
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    return x, k


def test_rigid_translation_defect_is_recovered_exactly() -> None:
    x, k = _grid()
    eta = (jnp.cos(3.0 * x) + 0.3 * jnp.sin(7.0 * x))[None, :]
    eta_x = jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(eta)))
    defect = -0.0125 * eta_x
    component = modal_phase_component(
        eta, defect, k, k_max=128.0, relative_energy_floor=1e-14
    )
    np.testing.assert_allclose(component, defect, rtol=1e-11, atol=1e-12)


def test_amplitude_growth_defect_is_left_untouched() -> None:
    x, k = _grid()
    eta = (jnp.cos(4.0 * x) + 0.2 * jnp.cos(9.0 * x))[None, :]
    component = modal_phase_component(
        eta, 0.03 * eta, k, k_max=128.0, relative_energy_floor=1e-14
    )
    np.testing.assert_allclose(component, 0.0, atol=1e-12)


def test_constant_defect_is_invisible() -> None:
    x, k = _grid()
    eta = jnp.cos(5.0 * x)[None, :]
    base = 0.02 * jnp.sin(5.0 * x)[None, :]
    baseline = modal_phase_component(
        eta, base, k, k_max=128.0, relative_energy_floor=1e-14
    )
    shifted = modal_phase_component(
        eta, base + 19.0, k, k_max=128.0, relative_energy_floor=1e-14
    )
    np.testing.assert_allclose(shifted, baseline, rtol=1e-12, atol=1e-12)


def test_projection_is_translation_equivariant() -> None:
    x, k = _grid()
    eta = (jnp.cos(2.0 * x) + 0.4 * jnp.sin(11.0 * x))[None, :]
    defect = (0.1 * jnp.sin(2.0 * x) - 0.02 * jnp.cos(11.0 * x))[None, :]
    component = modal_phase_component(
        eta, defect, k, k_max=128.0, relative_energy_floor=1e-14
    )
    shift = 31
    shifted = modal_phase_component(
        jnp.roll(eta, shift, axis=-1),
        jnp.roll(defect, shift, axis=-1),
        k,
        k_max=128.0,
        relative_energy_floor=1e-14,
    )
    np.testing.assert_allclose(
        shifted, jnp.roll(component, shift, axis=-1), rtol=1e-11, atol=1e-12
    )


def test_localized_projection_recovers_rigid_defect_with_correct_sign() -> None:
    x, k = _grid()
    eta = (jnp.cos(3.0 * x) + 0.3 * jnp.sin(7.0 * x))[None, :]
    eta_x = jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(eta)))
    defect = 0.0125 * eta_x
    for window_depths in (1.0, 4.0):
        component = localized_translation_component(
            eta,
            defect,
            k,
            jnp.asarray([0.05]),
            window_depths=window_depths,
            energy_floor_relative=0.0,
        )
        np.testing.assert_allclose(component, defect, rtol=1e-10, atol=1e-11)


def test_localized_projection_preserves_opposite_packet_defects() -> None:
    x, k = _grid(512)

    def periodic_distance(center: float) -> jax.Array:
        return jnp.angle(jnp.exp(1j * (x - center)))

    left = jnp.exp(-0.5 * (periodic_distance(1.5) / 0.16) ** 2)
    right = 0.7 * jnp.exp(-0.5 * (periodic_distance(4.5) / 0.16) ** 2)
    eta = (left + right)[None, :]
    eta_x = jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(eta)))
    left_mask = jnp.abs(periodic_distance(1.5)) < 0.5
    right_mask = jnp.abs(periodic_distance(4.5)) < 0.5
    coefficient = jnp.where(left_mask, 0.02, jnp.where(right_mask, -0.01, 0.0))
    defect = coefficient[None, :] * eta_x
    component = localized_translation_component(
        eta,
        defect,
        k,
        jnp.asarray([0.05]),
        window_depths=1.0,
        energy_floor_relative=1e-6,
    )

    def fitted_coefficient(mask: jax.Array) -> jax.Array:
        weighted_eta_x = jnp.where(mask[None, :], eta_x, 0.0)
        return jnp.sum(weighted_eta_x * component) / jnp.sum(weighted_eta_x**2)

    np.testing.assert_allclose(fitted_coefficient(left_mask), 0.02, rtol=2e-3)
    np.testing.assert_allclose(fitted_coefficient(right_mask), -0.01, rtol=2e-3)


def test_localized_projection_is_translation_equivariant() -> None:
    x, k = _grid()
    eta = (jnp.cos(2.0 * x) + 0.4 * jnp.sin(11.0 * x))[None, :]
    defect = (0.1 * jnp.sin(2.0 * x) - 0.02 * jnp.cos(11.0 * x))[None, :]
    depth = jnp.asarray([0.08])
    component = localized_translation_component(
        eta,
        defect,
        k,
        depth,
        window_depths=4.0,
        energy_floor_relative=1e-3,
    )
    shift = 31
    shifted = localized_translation_component(
        jnp.roll(eta, shift, axis=-1),
        jnp.roll(defect, shift, axis=-1),
        k,
        depth,
        window_depths=4.0,
        energy_floor_relative=1e-3,
    )
    np.testing.assert_allclose(
        shifted, jnp.roll(component, shift, axis=-1), rtol=1e-10, atol=1e-11
    )


if __name__ == "__main__":
    tests = (
        test_rigid_translation_defect_is_recovered_exactly,
        test_amplitude_growth_defect_is_left_untouched,
        test_constant_defect_is_invisible,
        test_projection_is_translation_equivariant,
        test_localized_projection_recovers_rigid_defect_with_correct_sign,
        test_localized_projection_preserves_opposite_packet_defects,
        test_localized_projection_is_translation_equivariant,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
