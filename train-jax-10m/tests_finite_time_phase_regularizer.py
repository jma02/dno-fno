"""CPU invariants for the finite-time full-state rollout objective."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finite_time_phase_regularizer import (
    FiniteTimePhaseConfig,
    full_state_path_loss,
    gl2_if_path,
    sample_source_microbatch,
)
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol
from solver.solvers.time_integrator import (
    State,
    gauss_legendre_2_if_step,
    make_solver_params,
)
from stage_tangent_regularizer import ReferenceIFParams, make_reference_F
from source_conditioning import parse_source_ids, source_conditioning_for_dataset


jax.config.update("jax_enable_x64", True)


def _path_problem(
    *,
    substeps: int = 2,
    nx: int = 64,
) -> tuple[State, jax.Array, jax.Array, FiniteTimePhaseConfig]:
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
    time = jnp.arange(1, substeps + 1, dtype=x.dtype)[:, None, None]
    eta = jnp.cos(5.0 * x)[None, None, :] * (1.0 + 0.01 * time)
    xi = 0.4 * jnp.sin(5.0 * x)[None, None, :] * (1.0 - 0.02 * time)
    path = State(eta=eta, xi=xi)
    depth = jnp.asarray([0.4], dtype=x.dtype)
    k_rfft = jnp.fft.rfftfreq(nx, d=1.0 / nx)
    config = FiniteTimePhaseConfig(
        substeps=substeps,
        k_max=nx // 4,
        filter_fraction=0.25,
    )
    return path, depth, k_rfft, config


def test_exact_path_has_zero_loss() -> None:
    path, depth, k_rfft, config = _path_problem()
    loss, diagnostics = full_state_path_loss(
        path,
        path,
        depth,
        k_rfft,
        jnp.ones((1,), dtype=depth.dtype),
        config,
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        diagnostics["raw_growth_rate_rms"], 0.0, atol=1e-15
    )
    assert all(value.shape == () for value in diagnostics.values())


def test_xi_only_error_is_detected() -> None:
    reference, depth, k_rfft, config = _path_problem()
    model = State(
        eta=reference.eta,
        xi=reference.xi + 0.02 * jnp.roll(reference.xi, 1, axis=-1),
    )
    loss, diagnostics = full_state_path_loss(
        model,
        reference,
        depth,
        k_rfft,
        jnp.ones((1,), dtype=depth.dtype),
        config,
    )

    assert float(loss) > 0.0
    assert float(diagnostics["raw_growth_rate_rms"]) > 0.0


def test_translation_is_classified_as_coherent_phase() -> None:
    reference, depth, k_rfft, config = _path_problem()
    model = State(
        eta=jnp.roll(reference.eta, 1, axis=-1),
        xi=jnp.roll(reference.xi, 1, axis=-1),
    )
    loss, diagnostics = full_state_path_loss(
        model,
        reference,
        depth,
        k_rfft,
        jnp.ones((1,), dtype=depth.dtype),
        config,
    )

    assert float(loss) > 0.0
    assert float(diagnostics["phase_growth_rate_rms"]) > 1.0
    assert float(diagnostics["action_growth_rate_rms"]) < 1e-10
    assert float(diagnostics["polarization_growth_rate_rms"]) < 1e-6


def test_path_loss_gradient_is_finite_and_nonzero() -> None:
    reference, depth, k_rfft, config = _path_problem()

    def objective(amplitude: jax.Array) -> jax.Array:
        model = State(
            eta=amplitude * reference.eta,
            xi=amplitude * reference.xi,
        )
        return full_state_path_loss(
            model,
            reference,
            depth,
            k_rfft,
            jnp.ones((1,), dtype=depth.dtype),
            config,
        )[0]

    gradient = jax.grad(objective)(jnp.asarray(1.1, dtype=depth.dtype))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 1e-6


def test_source_sampler_selects_only_requested_rows() -> None:
    rows, nx = 12, 8
    eta = jnp.arange(rows * nx, dtype=jnp.float64).reshape(rows, nx)
    xi = -eta
    depth = jnp.arange(rows, dtype=eta.dtype) + 1.0
    source = jnp.asarray([5, 7, 5, 8, 5, 7, 5, 9, 5, 6, 5, 14])
    eta_sub, xi_sub, depth_sub, valid = sample_source_microbatch(
        jax.random.PRNGKey(4),
        eta,
        xi,
        depth,
        source,
        jnp.asarray(5, dtype=source.dtype),
        4,
    )

    assert bool(jnp.all(valid == 1.0))
    assert bool(jnp.all(jnp.isin(depth_sub - 1.0, jnp.where(source == 5)[0])))
    np.testing.assert_allclose(xi_sub, -eta_sub)


def test_source_sampler_uses_schema_v2_tanaka_without_changing_legacy_ids() -> None:
    rows, nx = 8, 4
    eta = jnp.arange(rows * nx, dtype=jnp.float64).reshape(rows, nx)
    xi = -eta
    depth = jnp.arange(rows, dtype=eta.dtype) + 1.0
    source = jnp.asarray([1, 2, 3, 4, 2, 3, 1, 4], dtype=jnp.int32)
    schema_v2 = source_conditioning_for_dataset({"split_id": object()})
    assert schema_v2.finite_time_source_ids == (2, 3)
    assert (
        parse_source_ids(None, automatic=schema_v2.finite_time_source_ids)
        == (2, 3)
    )
    assert (
        parse_source_ids("3,2", automatic=schema_v2.finite_time_source_ids)
        == (3, 2)
    )
    eta_sub, xi_sub, depth_sub, valid = sample_source_microbatch(
        jax.random.PRNGKey(9),
        eta,
        xi,
        depth,
        source,
        jnp.asarray(schema_v2.tanaka_source_ids[0], dtype=source.dtype),
        2,
    )
    assert bool(jnp.all(valid == 1.0))
    assert bool(jnp.all(jnp.isin(depth_sub - 1.0, jnp.asarray([1, 4]))))
    np.testing.assert_allclose(xi_sub, -eta_sub)

    legacy = source_conditioning_for_dataset({})
    assert legacy.tanaka_source_ids == (5, 6, 14)
    assert legacy.finite_time_source_ids == (5, 6, 14, 7, 8, 9)
    assert 2 not in legacy.tanaka_source_ids
    assert 2 not in legacy.finite_time_source_ids


def test_reference_path_matches_production_gl2_step() -> None:
    nx = 32
    length = 2.0 * np.pi
    depth_value = 0.4
    dt = 0.01
    x, k = build_grid(nx, length)
    x = x.astype(jnp.float64)
    k = k.astype(jnp.float64)
    initial = State(
        eta=(0.01 * jnp.cos(2.0 * x))[None, :],
        xi=(0.02 * jnp.sin(3.0 * x))[None, :],
    )
    depth = jnp.asarray([depth_value], dtype=x.dtype)
    g0 = make_linear_dno_symbol(k, depth_value)
    reference_params = ReferenceIFParams(
        k=k,
        g0=g0[None, :],
        depth=depth,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        filter_fraction=0.25,
        nx=nx,
    )
    path = gl2_if_path(
        make_reference_F(reference_params, zero_mean_gxi=True),
        initial,
        reference_params,
        dt=dt,
        substeps=1,
        picard_iterations=4,
        rematerialize=False,
    )

    production_params = make_solver_params(
        nx=nx,
        length=length,
        depth=depth_value,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        filter_fraction=0.25,
    )
    production = gauss_legendre_2_if_step(
        State(eta=initial.eta[0], xi=initial.xi[0]),
        0.0,
        dt,
        production_params,
        iterations=4,
    )
    production = State(
        eta=production.eta,
        xi=production.xi - jnp.mean(production.xi),
    )

    np.testing.assert_allclose(path.eta[0, 0], production.eta, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(path.xi[0, 0], production.xi, rtol=1e-11, atol=1e-13)


if __name__ == "__main__":
    tests = (
        test_exact_path_has_zero_loss,
        test_xi_only_error_is_detected,
        test_translation_is_classified_as_coherent_phase,
        test_path_loss_gradient_is_finite_and_nonzero,
        test_source_sampler_selects_only_requested_rows,
        test_source_sampler_uses_schema_v2_tanaka_without_changing_legacy_ids,
        test_reference_path_matches_production_gl2_step,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
