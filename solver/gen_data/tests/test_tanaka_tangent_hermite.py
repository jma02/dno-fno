"""Focused CPU checks for tangent-aware Tanaka profile placement.

Run directly; pytest is not required:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=True \
      uv run python solver/gen_data/tests/test_tanaka_tangent_hermite.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    TANAKA_FINE_FACTOR,
    _validate_tanaka_profile_batch,
    build_per_simulation_initial_conditions,
    cubic_hermite_zero_exterior,
    place_tanaka_profile_periodic,
    tanaka_periodic_image_radius,
)
from solver.gen_data.tanaka_sampling import TanakaCrest  # noqa: E402
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_QC_UPPER,
    ModifiedTanakaBatchSolution,
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
    validate_solved_amplitudes,
)

jax.config.update("jax_enable_x64", True)

LENGTH = 2.0 * math.pi
NX = 1024
FILTER_FRACTION = 0.25
SIMULATION31_DEPTH = 0.26861433760407505
SIMULATION31_SPECS = [
    TanakaCrest(
        alpha=0.05548354495289499,
        center=3.5548496920089176,
        direction=-1,
    ),
    TanakaCrest(
        alpha=0.053531413616445936,
        center=5.032703928715591,
        direction=1,
    ),
]


@dataclass(frozen=True)
class TanakaFixture:
    profile_solution: ModifiedTanakaBatchSolution
    eta: jax.Array
    xi: jax.Array
    gxi: jax.Array
    k: jax.Array


def relative_l2(left: jax.Array, right: jax.Array) -> float:
    denominator = jnp.maximum(jnp.linalg.norm(right), jnp.finfo(right.dtype).tiny)
    return float(jnp.linalg.norm(left - right) / denominator)


@cache
def build_fixture() -> TanakaFixture:
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=1.0,
        direction=1,
        nx=NX,
        length=LENGTH,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    profile_amplitudes = jnp.asarray((0.05, 0.25, 0.45), dtype=jnp.float64)
    profile_solution = solve_modified_tanaka_batched(
        template,
        profile_amplitudes,
        centers=jnp.zeros_like(profile_amplitudes),
        directions=jnp.ones_like(profile_amplitudes),
    )

    direction_center = 0.731
    simulation_specs = [
        SIMULATION31_SPECS,
        [TanakaCrest(0.10, direction_center, 1)],
        [TanakaCrest(0.10, direction_center, -1)],
    ]
    eta, xi = build_per_simulation_initial_conditions(
        template_params=template,
        simulation_h_ref=np.asarray((SIMULATION31_DEPTH, 0.10, 0.10)),
        simulation_specs=simulation_specs,
        length=LENGTH,
        nx=NX,
        gravity=1.0,
    )
    _, k = build_grid(NX, LENGTH)
    eta = apply_lowpass(eta, k, FILTER_FRACTION)
    xi = apply_lowpass(xi, k, FILTER_FRACTION)
    xi = xi - jnp.mean(xi, axis=-1, keepdims=True)
    depths = jnp.asarray((SIMULATION31_DEPTH, 0.10, 0.10))[:, None]
    gxi = apply_lowpass(
        dno_series_eval(
            eta,
            xi,
            k,
            depths,
            6,
            pad_factor=8,
        ),
        k,
        FILTER_FRACTION,
    )
    jax.block_until_ready(gxi)
    return TanakaFixture(
        profile_solution=profile_solution,
        eta=eta,
        xi=xi,
        gxi=gxi,
        k=k,
    )


def test_cubic_hermite_exact_and_zero_exterior() -> None:
    x_nodes = jnp.asarray((-1.0, -0.35, 0.2, 0.61, 1.0))

    def polynomial(x: jax.Array) -> jax.Array:
        return 0.5 * x**3 - 0.25 * x**2 - x + 0.75

    def derivative(x: jax.Array) -> jax.Array:
        return 1.5 * x**2 - 0.5 * x - 1.0

    y_nodes = polynomial(x_nodes)
    slopes = derivative(x_nodes)
    x_eval = jnp.linspace(-1.0, 1.0, 101)
    actual = cubic_hermite_zero_exterior(
        x_nodes,
        y_nodes,
        slopes,
        x_eval,
    )
    np.testing.assert_allclose(actual, polynomial(x_eval), rtol=5e-13, atol=5e-13)

    tangent = jax.vmap(
        jax.grad(
            lambda x: cubic_hermite_zero_exterior(
                x_nodes,
                y_nodes,
                slopes,
                x,
            )
        )
    )(x_nodes)
    np.testing.assert_allclose(tangent, slopes, rtol=5e-13, atol=5e-13)

    outside = jnp.asarray((-2.0, -1.01, 1.01, 2.0))
    outside_values = cubic_hermite_zero_exterior(
        x_nodes,
        y_nodes,
        slopes,
        outside,
    )
    outside_tangents = jax.vmap(
        jax.grad(
            lambda x: cubic_hermite_zero_exterior(
                x_nodes,
                y_nodes,
                slopes,
                x,
            )
        )
    )(outside)
    np.testing.assert_array_equal(outside_values, jnp.zeros_like(outside))
    np.testing.assert_array_equal(outside_tangents, jnp.zeros_like(outside))


def test_profile_validation_rejects_nonincreasing_knots() -> None:
    x_profile = jnp.asarray(((-1.0, 0.0, 0.0, 0.5, 1.0),))
    eta_profile = jnp.zeros_like(x_profile)
    theta_profile = jnp.zeros_like(x_profile)
    try:
        _validate_tanaka_profile_batch(
            x_profile,
            eta_profile,
            theta_profile,
        )
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("Duplicate Tanaka knots were not rejected.")


def test_small_profiles_realize_their_requested_amplitudes() -> None:
    requested = jnp.asarray(
        (1.0e-8, 2.4102831187118947e-5, 1.0e-3),
        dtype=jnp.float64,
    )
    template = make_default_tanaka_template(
        nx=32,
        dno_order=0,
        pad_factor=1,
    )
    assert template.qc_upper == DEFAULT_QC_UPPER
    assert template.outer_iterations == DEFAULT_OUTER_ITERATIONS
    solution = solve_modified_tanaka_batched(template, requested)
    achieved = jnp.max(solution.eta_profile, axis=-1)
    validate_solved_amplitudes(solution.eta_profile, requested)
    np.testing.assert_allclose(
        achieved,
        requested,
        rtol=1.0e-6,
        atol=1.0e-14,
    )
    jax.clear_caches()


def test_profile_amplitude_validation_rejects_the_old_floor() -> None:
    eta_profile = jnp.asarray(
        ((0.0, 4.0e-4, 9.947786769e-4, 4.0e-4, 0.0),),
        dtype=jnp.float64,
    )
    requested = jnp.asarray((1.0e-5,), dtype=jnp.float64)
    try:
        validate_solved_amplitudes(eta_profile, requested)
    except ValueError as error:
        assert "requested=1.0000000000000001e-05" in str(error)
        assert "achieved=0.0009947786769" in str(error)
    else:
        raise AssertionError("The old Tanaka amplitude floor was not rejected.")


def test_profiles_remain_nonnegative_and_monotone() -> None:
    solution = build_fixture().profile_solution
    for x_profile, eta_profile, theta_profile in zip(
        solution.x_profile,
        solution.eta_profile,
        solution.theta_profile,
    ):
        slopes = jnp.tan(theta_profile)
        midpoints = 0.5 * (x_profile[:-1] + x_profile[1:])
        samples = jnp.sort(jnp.concatenate((x_profile, midpoints)))
        values = cubic_hermite_zero_exterior(
            x_profile,
            eta_profile,
            slopes,
            samples,
        )
        crest = int(jnp.argmax(values))
        scale = float(jnp.max(jnp.abs(values)))
        tolerance = 1e-12 * max(scale, 1.0)
        assert float(jnp.min(values)) >= -tolerance
        assert float(jnp.min(jnp.diff(values[: crest + 1]))) >= -tolerance
        assert float(jnp.max(jnp.diff(values[crest:]))) <= tolerance


def test_periodic_placement_translation_and_image_convergence() -> None:
    solution = build_fixture().profile_solution
    x_profile = solution.x_profile[0]
    eta_profile = solution.eta_profile[0]
    theta_profile = solution.theta_profile[0]
    depth = jnp.asarray(0.35)
    x_fine = (LENGTH / (NX * TANAKA_FINE_FACTOR)) * jnp.arange(
        NX * TANAKA_FINE_FACTOR,
        dtype=jnp.float64,
    )
    image_radius = tanaka_periodic_image_radius(
        x_profile,
        depth,
        LENGTH,
    )
    center = jnp.asarray(0.123)

    def place(target_center: jax.Array, radius: int) -> jax.Array:
        return place_tanaka_profile_periodic(
            x_profile,
            eta_profile,
            theta_profile,
            depth,
            target_center,
            x_fine,
            length=LENGTH,
            nx=NX,
            image_radius=radius,
        )

    production = apply_lowpass(
        place(center, image_radius),
        build_fixture().k,
        FILTER_FRACTION,
    )
    reference = apply_lowpass(
        place(center, image_radius + 2),
        build_fixture().k,
        FILTER_FRACTION,
    )
    assert relative_l2(production, reference) < 1e-11

    period_shifted = place(center + LENGTH, image_radius)
    assert relative_l2(period_shifted, place(center, image_radius)) < 1e-13

    dx = LENGTH / NX
    grid_shifted = place(center + dx, image_radius)
    assert relative_l2(grid_shifted, jnp.roll(place(center, image_radius), 1)) < 1e-11


def test_simulation31_tail_and_direction_regression() -> None:
    fixture = build_fixture()
    for field in (fixture.eta, fixture.xi, fixture.gxi):
        assert bool(jnp.all(jnp.isfinite(field)))

    expected_tail_norms = {
        "eta": 3.828750871099434e-09,
        "xi": 6.50918614445819e-11,
        "gxi": 4.315824664769325e-09,
    }
    for name, field in (
        ("eta", fixture.eta[0]),
        ("xi", fixture.xi[0]),
        ("gxi", fixture.gxi[0]),
    ):
        coefficients = np.fft.rfft(np.asarray(field)) / NX
        tail_norm = float(np.linalg.norm(coefficients[80:129]))
        np.testing.assert_allclose(
            tail_norm,
            expected_tail_norms[name],
            rtol=5e-5,
            atol=1e-14,
        )

    np.testing.assert_allclose(fixture.eta[1], fixture.eta[2], rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(fixture.xi[1], -fixture.xi[2], rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(fixture.gxi[1], -fixture.gxi[2], rtol=2e-11, atol=2e-13)


def test_simulation31_one_interval_production_rollout() -> None:
    fixture = build_fixture()
    defaults = make_normalized_rollout_settings()._replace(
        filter_fraction=FILTER_FRACTION
    )
    depth = jnp.asarray(((SIMULATION31_DEPTH,),), dtype=jnp.float64)
    params = SolverParams(
        nx=NX,
        length=LENGTH,
        depth=depth,
        gravity=1.0,
        dno_order=defaults.dno_order,
        pad_factor=defaults.pad_factor,
        filter_fraction=defaults.filter_fraction,
        k=fixture.k,
        g0=make_linear_dno_symbol(fixture.k, depth),
    )
    params = cast_solver_params_dtype(params, jnp.float64)
    rollout = batched_rollout(
        cast_state_dtype(
            State(
                eta=fixture.eta[:1],
                xi=fixture.xi[:1],
            ),
            jnp.float64,
        ),
        jnp.asarray((0.0, 0.08), dtype=jnp.float64),
        params,
        save_gxi=True,
        substeps_per_interval=defaults.substeps_per_interval,
        method=defaults.method,
        implicit_iterations=defaults.implicit_iterations,
        implicit_relaxation=defaults.implicit_relaxation,
        zero_mean_xi=defaults.zero_mean_xi,
    )
    jax.block_until_ready(rollout["gxi"])
    eta = np.asarray(rollout["eta"])
    xi = np.asarray(rollout["xi"])
    gxi = np.asarray(rollout["gxi"])
    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(xi))
    assert np.all(np.isfinite(gxi))
    assert float(np.min(SIMULATION31_DEPTH + eta)) > 0.1 * SIMULATION31_DEPTH
    assert float(np.max(np.abs(np.mean(xi, axis=-1)))) < 1e-12

    for field in (eta, xi, gxi):
        spectrum = np.fft.rfft(field, axis=-1)
        assert float(np.max(np.abs(spectrum[..., 129:]))) < 1e-10

    dx = LENGTH / NX
    hamiltonian = 0.5 * dx * np.sum(xi * gxi + eta**2, axis=-1)
    drift = np.max(
        np.abs(hamiltonian - hamiltonian[:1])
        / np.maximum(np.abs(hamiltonian[:1]), np.finfo(np.float64).tiny)
    )
    assert float(drift) < 1e-8


def main() -> None:
    tests = (
        test_cubic_hermite_exact_and_zero_exterior,
        test_profile_validation_rejects_nonincreasing_knots,
        test_small_profiles_realize_their_requested_amplitudes,
        test_profile_amplitude_validation_rejects_the_old_floor,
        test_profiles_remain_nonnegative_and_monotone,
        test_periodic_placement_translation_and_image_convergence,
        test_simulation31_tail_and_direction_regression,
        test_simulation31_one_interval_production_rollout,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)


if __name__ == "__main__":
    main()
