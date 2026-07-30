"""CPU invariants for the soliton-selective spectral growth guard."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import (
    _apply_eta_growth_guard,
    _positive_elevation_soliton_mask,
)
from solver.solvers.time_integrator import State


jax.config.update("jax_enable_x64", True)


def _grid(n: int = 256) -> tuple[jax.Array, jax.Array]:
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    return x, k


def test_positive_elevation_selector_separates_soliton_and_wave_train() -> None:
    x, _ = _grid()
    distance = jnp.minimum(jnp.abs(x - jnp.pi), 2.0 * jnp.pi - jnp.abs(x - jnp.pi))
    soliton = jnp.exp(-(distance / 0.2) ** 2)
    wave_train = jnp.cos(4.0 * x)
    selected = _positive_elevation_soliton_mask(
        jnp.stack((soliton, wave_train)),
        negative_energy_threshold=1e-3,
    )
    np.testing.assert_array_equal(selected, np.asarray([True, False]))


def test_inactive_guard_is_identity() -> None:
    x, k = _grid()
    eta = jnp.sin(7.0 * x)
    xi = jnp.cos(9.0 * x)
    state = State(eta=eta, xi=xi)
    guarded = _apply_eta_growth_guard(
        state,
        k,
        eta.size,
        eta_amp0=jnp.asarray(1e-12),
        depth=jnp.asarray(0.2),
        activation_mask=jnp.asarray(False),
        k_lo=64.0,
        k_hi=128.0,
        abs_floor=0.0,
        growth_factor=100.0,
        sharpness=10.0,
        houli_a=0.69,
        houli_m=4.0,
        k_eff=128.0,
        kh_eff=12.0,
        filter_xi=True,
    )
    np.testing.assert_allclose(guarded.eta, eta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(guarded.xi, xi, rtol=1e-13, atol=1e-13)


def test_depth_adaptation_preserves_shallow_mode_and_damps_deep_mode() -> None:
    x, k = _grid()
    mode = jnp.cos(80.0 * x)
    eta = jnp.stack((mode, mode))
    state = State(eta=eta, xi=jnp.zeros_like(eta))
    guarded = _apply_eta_growth_guard(
        state,
        k,
        eta.shape[-1],
        eta_amp0=jnp.zeros((2,), dtype=eta.dtype),
        depth=jnp.asarray([[0.2], [0.05]], dtype=eta.dtype),
        activation_mask=jnp.asarray([True, True]),
        k_lo=64.0,
        k_hi=128.0,
        abs_floor=1e-12,
        growth_factor=100.0,
        sharpness=100.0,
        houli_a=0.69,
        houli_m=4.0,
        k_eff=128.0,
        kh_eff=12.0,
        filter_xi=False,
    )
    amplitude = jnp.max(jnp.abs(guarded.eta), axis=-1)
    assert float(amplitude[0]) < 0.01
    assert float(amplitude[1]) > 0.95


if __name__ == "__main__":
    tests = (
        test_positive_elevation_selector_separates_soliton_and_wave_train,
        test_inactive_guard_is_identity,
        test_depth_adaptation_preserves_shallow_mode_and_damps_deep_mode,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
