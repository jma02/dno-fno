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
from typing import Any, Callable, Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR.parent))
sys.path.insert(0, str(TRAIN_DIR))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hadamard_shape_regularizer import (  # noqa: E402
    compute_hadamard_loss,
    compute_local_response_loss,
    projected_sobolev_energy,
)
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.solvers.dno_series_jax import build_grid  # noqa: E402

jax.config.update("jax_enable_x64", True)


Array = jax.Array
TestFn = Callable[[], None]
ResponseMode = Literal["secant", "teacher", "targeted"]
RESPONSE_MODES: tuple[ResponseMode, ...] = ("secant", "teacher", "targeted")
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
    compute_hadamard_loss,
    k_max=20.0,
    fd_step_max=1e-3,
    min_surface_rms=2e-2,
    denominator_eps=1e-24,
)

_local_response_loss = partial(
    compute_local_response_loss,
    length=2.0 * np.pi,
    k_max=8.0,
    fd_step_min=1e-3,
    fd_step_max=1e-3,
    min_surface_rms=1e-3,
    denominator_eps=1e-24,
)


def _local_inputs(nx: int = 64) -> tuple[Array, Array, Array, Array]:
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
    amplitudes = jnp.asarray([0.8, 1.0, 1.2])[:, None]
    eta = amplitudes * (0.005 * jnp.cos(x) + 0.001 * jnp.sin(3.0 * x))
    xi = amplitudes * (0.03 * jnp.sin(x) + 0.01 * jnp.cos(3.0 * x))
    depth = jnp.log(jnp.asarray([0.3, 1.4, 8.0]))[:, None]
    return eta, xi, depth, _wavenumbers(nx)


def _m6(eta: Array, xi: Array, log_depth: Array) -> Array:
    return compute_dno_target(
        eta,
        xi,
        jnp.exp(log_depth),
        nx=eta.shape[-1],
        length=2.0 * np.pi,
        dno_order=6,
        pad_factor=8,
        maximum_wavenumber=128.0,
    )[2]


def _m6_apply(variables: dict[str, Any], inputs: Array, depth: Array) -> Array:
    gain = variables["params"].get("gain", 1.0)
    return (gain * _m6(inputs[..., 0], inputs[..., 1], depth))[..., None]


def _wrong_response_apply(
    variables: dict[str, Any], inputs: Array, depth: Array
) -> Array:
    del depth
    gain = variables["params"]["gain"]
    eta, xi = inputs[..., 0], inputs[..., 1]
    k = _wavenumbers(eta.shape[-1])
    response = (
        gain * _g01(eta, xi)
        + 0.6 * _spectral_dx(eta, k) * _spectral_dx(xi, k)
        + 0.1 * xi**2
    )
    return (response + 7.0)[..., None]


def _continuum_response(
    eta: Array,
    xi: Array,
    zeta: Array,
    chi: Array,
    depth: Array,
    k: Array,
) -> Array:
    eta_x, xi_x = _spectral_dx(eta, k), _spectral_dx(xi, k)
    b_velocity = (_m6(eta, xi, depth) + eta_x * xi_x) / (1.0 + eta_x**2)
    v_velocity = xi_x - b_velocity * eta_x
    return _m6(eta, chi - zeta * b_velocity, depth) - _spectral_dx(zeta * v_velocity, k)


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
        min_surface_rms=5e-3,
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


def test_m6_continuum_anchor_and_truncation_discrepancy() -> None:
    nx = 64
    x, k = build_grid(nx, 2.0 * np.pi)
    depth = jnp.log(jnp.asarray([0.35, 1.7]))[:, None]
    xi = jnp.broadcast_to(0.03 * jnp.sin(x) + 0.01 * jnp.cos(3.0 * x), (2, nx))
    zeta = jnp.broadcast_to(0.03 * jnp.cos(2.0 * x) - 0.015 * jnp.sin(3.0 * x), (2, nx))
    chi = jnp.broadcast_to(0.025 * jnp.sin(x) + 0.01 * jnp.cos(4.0 * x), (2, nx))
    errors: list[list[float]] = []
    for amplitude in (0.005, 0.12):
        eta = jnp.broadcast_to(
            amplitude * (jnp.cos(x) + 0.2 * jnp.sin(3.0 * x)), (2, nx)
        )
        continuum = _continuum_response(eta, xi, zeta, chi, depth, k)
        local_errors = []
        for step in (1e-3, 1e-4):
            plus = _m6(eta + step * zeta, xi + step * chi, depth)
            minus = _m6(eta - step * zeta, xi - step * chi, depth)
            central = (plus - minus) / (2.0 * step)
            local_errors.append(
                float(jnp.linalg.norm(central - continuum) / jnp.linalg.norm(continuum))
            )
        errors.append(local_errors)
    assert errors[0][0] < 3e-9
    assert errors[0][1] < errors[0][0] / 20.0
    # M6 is a truncated operator: its derivative need not satisfy the continuum
    # identity exactly. The larger-amplitude mismatch survives a smaller step.
    assert errors[1][1] > 1e-6
    np.testing.assert_allclose(errors[1][0], errors[1][1], rtol=2e-3)


def test_exact_m6_has_zero_joint_secant_defect() -> None:
    eta, xi, depth, k = _local_inputs()
    for mode in ("secant", "targeted"):
        loss = _local_response_loss(
            jax.random.PRNGKey(21),
            _m6_apply,
            {},
            eta,
            xi,
            depth,
            _identity_norm_inputs,
            _identity_targets,
            k,
            jnp.float64,
            mode=mode,
        )
        assert float(loss) < 1e-20, (mode, float(loss))


def test_local_response_gain_value_gradient_and_jit_agree() -> None:
    eta, xi, depth, k = _local_inputs()
    mode: ResponseMode
    for mode in RESPONSE_MODES:

        def loss_of_gain(gain: Array) -> Array:
            return _local_response_loss(
                jax.random.PRNGKey(12),
                _m6_apply,
                {"gain": gain},
                eta,
                xi,
                depth,
                _identity_norm_inputs,
                _identity_targets,
                k,
                jnp.float64,
                mode=mode,
            )

        gain = jnp.asarray(0.65, dtype=jnp.float64)
        eager = jax.value_and_grad(loss_of_gain)(gain)
        compiled = jax.jit(jax.value_and_grad(loss_of_gain))(gain)
        np.testing.assert_allclose(
            np.asarray(eager), np.asarray(compiled), rtol=2e-8, atol=1e-10
        )
        assert float(eager[0]) > 1e-3
        assert abs(float(eager[1])) > 0.1
        if mode != "teacher":
            np.testing.assert_allclose(eager[0], (gain - 1.0) ** 2, rtol=1e-8)
            np.testing.assert_allclose(eager[1], 2.0 * (gain - 1.0), rtol=1e-8)


def test_local_response_frozen_reference_and_probe_constraints() -> None:
    eta, xi, depth, k = _local_inputs()
    gain = jnp.asarray(0.7, dtype=jnp.float64)
    step = 1e-3
    mode: ResponseMode
    for mode in RESPONSE_MODES:
        evaluations: list[tuple[Array, Array]] = []

        def recording_apply(
            variables: dict[str, Any], inputs: Array, batch_depth: Array
        ) -> Array:
            evaluations.append((inputs, batch_depth))
            return _wrong_response_apply(variables, inputs, batch_depth)

        actual = _local_response_loss(
            jax.random.PRNGKey(31),
            recording_apply,
            {"gain": gain},
            eta,
            xi,
            depth,
            _identity_norm_inputs,
            _identity_targets,
            k,
            jnp.float64,
            mode=mode,
        )
        assert len(evaluations) == (3 if mode == "targeted" else 2)
        for _, model_depth in evaluations:
            np.testing.assert_array_equal(model_depth, depth)
        plus, minus = (evaluation[0] for evaluation in evaluations[-2:])
        np.testing.assert_allclose(
            (plus + minus) / 2.0, jnp.stack((eta, xi), axis=-1), atol=1e-15
        )
        zeta, chi = jnp.moveaxis((plus - minus) / (2.0 * step), -1, 0)
        for direction in (zeta, chi):
            np.testing.assert_allclose(jnp.mean(direction, axis=-1), 0.0, atol=1e-13)
            spectrum = jnp.fft.fft(direction, axis=-1)
            assert float(jnp.max(jnp.abs(spectrum[:, jnp.abs(k) > 8.0]))) < 1e-11
        assert float(jnp.linalg.norm(zeta)) > 0.0
        assert float(jnp.linalg.norm(chi)) > 0.0

        reference = (
            _continuum_response(eta, xi, zeta, chi, depth, k)
            if mode == "teacher"
            else (
                _m6(plus[..., 0], plus[..., 1], depth)
                - _m6(minus[..., 0], minus[..., 1], depth)
            )
            / (2.0 * step)
        )
        denominator = projected_sobolev_energy(reference, k, 8.0, 1) + 1e-24

        def frozen_loss(parameter: Array) -> Array:
            q_plus = _wrong_response_apply(
                {"params": {"gain": parameter}}, plus, depth
            )[..., 0]
            q_minus = _wrong_response_apply(
                {"params": {"gain": parameter}}, minus, depth
            )[..., 0]
            central = (q_plus - q_minus) / (2.0 * step)
            central -= jnp.mean(central, axis=-1, keepdims=True)
            return jnp.mean(
                projected_sobolev_energy(central - reference, k, 8.0, 1) / denominator
            )

        def actual_loss(parameter: Array) -> Array:
            return _local_response_loss(
                jax.random.PRNGKey(31),
                _wrong_response_apply,
                {"gain": parameter},
                eta,
                xi,
                depth,
                _identity_norm_inputs,
                _identity_targets,
                k,
                jnp.float64,
                mode=mode,
            )

        expected, expected_grad = jax.value_and_grad(frozen_loss)(gain)
        actual_grad = jax.grad(actual_loss)(gain)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
        # The recorded directions and independently computed teacher are fixed
        # here; equality excludes differentiating through the flow-defect probe.
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-8, atol=1e-10)


def test_local_response_stops_state_and_teacher_gradients() -> None:
    eta, xi, depth, k = _local_inputs()
    mode: ResponseMode
    for mode in RESPONSE_MODES:

        def loss_of_state(surface: Array, potential: Array, log_depth: Array) -> Array:
            return _local_response_loss(
                jax.random.PRNGKey(7),
                _wrong_response_apply,
                {"gain": jnp.asarray(0.7)},
                surface,
                potential,
                log_depth,
                _identity_norm_inputs,
                _identity_targets,
                k,
                jnp.float64,
                mode=mode,
            )

        gradients = jax.grad(loss_of_state, argnums=(0, 1, 2))(eta, xi, depth)
        for gradient in gradients:
            np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


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
    (
        "local_response/m6_continuum_anchor",
        test_m6_continuum_anchor_and_truncation_discrepancy,
    ),
    ("local_response/exact_m6_secants", test_exact_m6_has_zero_joint_secant_defect),
    (
        "local_response/value_grad_jit",
        test_local_response_gain_value_gradient_and_jit_agree,
    ),
    (
        "local_response/frozen_reference_and_probes",
        test_local_response_frozen_reference_and_probe_constraints,
    ),
    (
        "local_response/stopped_state_and_teacher",
        test_local_response_stops_state_and_teacher_gradients,
    ),
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
