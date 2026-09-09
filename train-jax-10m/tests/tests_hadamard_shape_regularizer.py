"""CPU tests for :mod:`hadamard_shape_regularizer`.

Run directly; pytest is not required:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True \
      .venv/bin/python train-jax-10m/tests/tests_hadamard_shape_regularizer.py
"""

from __future__ import annotations

import os
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hadamard_shape_regularizer import (  # noqa: E402
    compute_hadamard_reg,
    projected_sobolev_energy,
)

jax.config.update("jax_enable_x64", True)


Array = jax.Array
TestFn = Callable[[], None]
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def _wavenumbers(nx: int) -> Array:
    dx = 2.0 * np.pi / nx
    return (2.0 * np.pi * jnp.fft.fftfreq(nx, d=dx)).astype(jnp.float64)


def _spectral_dx(field: Array, k: Array) -> Array:
    return jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(field, axis=-1), axis=-1))


def _g0(field: Array, k: Array) -> Array:
    return jnp.real(jnp.fft.ifft(jnp.abs(k) * jnp.fft.fft(field, axis=-1), axis=-1))


def _g01(eta: Array, xi: Array) -> Array:
    """Deep-water G0+G1, whose Hadamard identity is exact at eta=0."""
    k = _wavenumbers(eta.shape[-1])
    g0_xi = _g0(xi, k)
    xi_x = _spectral_dx(xi, k)
    g1 = -_g0(eta * g0_xi, k) - _spectral_dx(eta * xi_x, k)
    return g0_xi + g1


def _identity_norm_inputs(eta: Array, xi: Array) -> Array:
    return jnp.stack((eta, xi), axis=-1)


def _identity_targets(values: Array) -> Array:
    return values


def _g01_apply(variables: dict[str, Any], inputs: Array, depth: Array) -> Array:
    del variables, depth
    return _g01(inputs[..., 0], inputs[..., 1])[..., None]


def _g0_apply(variables: dict[str, Any], inputs: Array, depth: Array) -> Array:
    del variables, depth
    k = _wavenumbers(inputs.shape[-2])
    return _g0(inputs[..., 1], k)[..., None]


def _base_inputs(
    nx: int = 64, batch_size: int = 3
) -> tuple[Array, Array, Array, Array]:
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
    eta = jnp.zeros((batch_size, nx), dtype=jnp.float64)
    amplitudes = jnp.linspace(0.7, 1.1, batch_size)[:, None]
    xi = amplitudes * (
        0.08 * jnp.cos(3.0 * x)[None, :] + 0.03 * jnp.sin(7.0 * x)[None, :]
    )
    depth = jnp.zeros((batch_size, 1), dtype=jnp.float64)
    return eta, xi, depth, _wavenumbers(nx)


_hadamard_loss = partial(
    compute_hadamard_reg,
    k_max=20.0,
    fd_step_max=1e-3,
    eta_scale_floor=2e-2,
    denominator_floor=1e-24,
)


def test_projected_sobolev_energy_matches_known_mode() -> None:
    nx = 64
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
    k = _wavenumbers(nx)
    field = jnp.cos(3.0 * x)[None, :]
    l2_energy = projected_sobolev_energy(field, k, 20.0, 0)
    np.testing.assert_allclose(np.asarray(l2_energy), [0.5], rtol=1e-13)
    energy = projected_sobolev_energy(field, k, 20.0, 1)
    np.testing.assert_allclose(np.asarray(energy), np.asarray([5.0]), rtol=1e-13)

    projected_out = projected_sobolev_energy(field, k, 2.0, 1)
    np.testing.assert_allclose(np.asarray(projected_out), 0.0, atol=1e-25)


def test_relative_probe_is_scaled_and_bandlimited() -> None:
    nx = 64
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
    eta = jnp.stack(
        (
            0.2 * jnp.cos(2.0 * x),
            0.02 * jnp.cos(2.0 * x),
            jnp.zeros_like(x),
        )
    )
    _, xi, depth, k = _base_inputs(nx)
    surfaces: list[Array] = []

    def recording_apply(
        variables: dict[str, Any], inputs: Array, depth: Array
    ) -> Array:
        surfaces.append(inputs[..., 0])
        return _g01_apply(variables, inputs, depth)

    _hadamard_loss(
        jax.random.PRNGKey(4),
        recording_apply,
        {},
        eta,
        xi,
        depth,
        _identity_norm_inputs,
        _identity_targets,
        k,
        jnp.float64,
        k_max=12.0,
        fd_step_min=1e-3,
        fd_step_max=1e-3,
        eta_scale_floor=5e-3,
    )

    base, perturbed, repeated = surfaces
    np.testing.assert_array_equal(base, eta)
    np.testing.assert_array_equal(repeated, eta)
    perturbation = perturbed - base
    expected_scale = jnp.maximum(jnp.sqrt(jnp.mean(eta * eta, axis=-1)), 5e-3)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.mean(perturbation * perturbation, axis=-1)),
        1e-3 * expected_scale,
        rtol=1e-12,
    )
    np.testing.assert_allclose(jnp.mean(perturbation, axis=-1), 0.0, atol=1e-15)
    perturbation_hat = jnp.fft.fft(perturbation, axis=-1)
    assert float(jnp.max(jnp.abs(perturbation_hat[:, jnp.abs(k) > 12.0]))) < 1e-13


def test_exact_g01_has_zero_flat_surface_defect() -> None:
    eta, xi, depth, k = _base_inputs()
    loss = _hadamard_loss(
        rng=jax.random.PRNGKey(8),
        apply_fn=_g01_apply,
        model_params={},
        eta_phys=eta,
        xi_phys=xi,
        batch_depth_local=depth,
        norm_inputs_fn=_identity_norm_inputs,
        denorm_targets_fn=_identity_targets,
        k=k,
        dtype=jnp.float64,
    )
    assert float(loss) < 1e-18


def test_g0_only_detects_missing_shape_derivative() -> None:
    eta, xi, depth, k = _base_inputs()
    loss = _hadamard_loss(
        rng=jax.random.PRNGKey(8),
        apply_fn=_g0_apply,
        model_params={},
        eta_phys=eta,
        xi_phys=xi,
        batch_depth_local=depth,
        norm_inputs_fn=_identity_norm_inputs,
        denorm_targets_fn=_identity_targets,
        k=k,
        dtype=jnp.float64,
    )
    assert float(loss) > 0.99


def test_normalizers_and_output_mean_match_production() -> None:
    eta, xi, depth, k = _base_inputs()
    eta_scale = jnp.asarray(2.5, dtype=jnp.float64)
    xi_scale = jnp.asarray(3.5, dtype=jnp.float64)
    target_scale = jnp.asarray(4.5, dtype=jnp.float64)

    def norm_inputs(eta_phys: Array, xi_phys: Array) -> Array:
        return jnp.stack((eta_phys / eta_scale, xi_phys / xi_scale), axis=-1)

    def denorm_targets(values: Array) -> Array:
        return values * target_scale

    def scaled_apply(
        variables: dict[str, Any], inputs: Array, batch_depth: Array
    ) -> Array:
        del variables, batch_depth
        eta_phys = inputs[..., 0] * eta_scale
        xi_phys = inputs[..., 1] * xi_scale
        # The removable DC offset checks rollout-compatible mean subtraction.
        return (_g01(eta_phys, xi_phys) / target_scale + 7.0)[..., None]

    loss = _hadamard_loss(
        jax.random.PRNGKey(3),
        scaled_apply,
        {},
        eta,
        xi,
        depth,
        norm_inputs,
        denorm_targets,
        k,
        jnp.float64,
    )
    assert float(loss) < 1e-17


def test_loss_is_jittable_and_differentiable() -> None:
    eta, xi, depth, k = _base_inputs(batch_size=2)
    x = jnp.arange(eta.shape[-1], dtype=jnp.float64) * (2.0 * jnp.pi / eta.shape[-1])
    eta = 0.015 * jnp.cos(2.0 * x)[None, :] * jnp.ones((2, 1))

    def perturbed_apply(
        variables: dict[str, Any], inputs: Array, batch_depth: Array
    ) -> Array:
        del batch_depth
        alpha = variables["params"]["alpha"]
        eta_local = inputs[..., 0]
        xi_local = inputs[..., 1]
        output = _g01(eta_local, xi_local) + alpha * eta_local**2 * xi_local
        return output[..., None]

    def loss_of_params(params: dict[str, Array]) -> Array:
        return _hadamard_loss(
            jax.random.PRNGKey(15),
            perturbed_apply,
            params,
            eta,
            xi,
            depth,
            _identity_norm_inputs,
            _identity_targets,
            k,
            jnp.float64,
            fd_step_min=2e-3,
            fd_step_max=2e-3,
        )

    params = {"alpha": jnp.asarray(0.4, dtype=jnp.float64)}
    value, gradient = jax.jit(jax.value_and_grad(loss_of_params))(params)
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient["alpha"]))
    assert abs(float(gradient["alpha"])) > 1e-10


TESTS: list[tuple[str, TestFn]] = [
    ("math/projected_sobolev_energy", test_projected_sobolev_energy_matches_known_mode),
    (
        "probe/relative_scaled_bandlimited",
        test_relative_probe_is_scaled_and_bandlimited,
    ),
    ("identity/exact_g01_flat", test_exact_g01_has_zero_flat_surface_defect),
    ("identity/g0_detects_missing_g1", test_g0_only_detects_missing_shape_derivative),
    (
        "integration/normalizers_and_mean",
        test_normalizers_and_output_mean_match_production,
    ),
    ("integration/jit_and_grad", test_loss_is_jittable_and_differentiable),
]


def _run_test(name: str, test_fn: TestFn) -> None:
    try:
        test_fn()
    except Exception as error:  # noqa: BLE001 - standalone test harness
        detail = f"{type(error).__name__}: {error}"
        FAILED.append((name, detail))
        print(f"[FAIL] {name} ({detail})", flush=True)
        traceback.print_exc()
    else:
        PASSED.append(name)
        print(f"[PASS] {name}", flush=True)


def main() -> int:
    for name, test_fn in TESTS:
        _run_test(name, test_fn)
    print(f"summary: {len(PASSED)} pass, {len(FAILED)} fail")
    return int(bool(FAILED))


if __name__ == "__main__":
    sys.exit(main())
