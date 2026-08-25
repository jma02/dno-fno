"""Cross-family CPU smoke for the shared refinement decision.

The Benjamin--Feir arm exercises the revision-4 JCP09 constructor used by the
current paper dataset.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests.test_paper_acceptance_cross_family
"""
from __future__ import annotations

import os
import unittest
from functools import cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    build_initial_conditions as build_benjamin_feir_initial_conditions,
    deep_water_proxy_depth,
)
from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    build_per_case_initial_conditions,
)
from solver.gen_data.tanaka_sampling import TanakaCrest  # noqa: E402
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
    apply_lowpass,
    rollout,
)
from solver.reference_solutions.stokes_wave import (  # noqa: E402
    stokes_eta_xi_at_phase,
)
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    make_default_tanaka_template,
)

jax.config.update("jax_enable_x64", True)

NX = 256
LENGTH = 2.0 * np.pi
GRAVITY = 1.0
FILTER_FRACTION = 0.25
DELIVERED_WAVENUMBER = 32.0
FAMILY_NAMES = (
    "finite_stokes",
    "tanaka",
    "current_benjamin_feir",
)


@cache
def _build_cross_family_batch() -> tuple[State, jax.Array, jax.Array]:
    x, k = build_grid(NX, LENGTH)
    x = jnp.asarray(x, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)

    stokes_eta, stokes_xi = stokes_eta_xi_at_phase(
        x=x,
        phase=0.0,
        n0=2,
        a0=0.025,
        length=LENGTH,
        depth=0.5,
        gravity=GRAVITY,
        ichoi=1,
    )
    stokes_eta = stokes_eta[None, :]
    stokes_xi = stokes_xi[None, :]

    tanaka_depth = 0.08
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=GRAVITY,
        direction=1,
        nx=NX,
        length=LENGTH,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    tanaka_eta, tanaka_xi = build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=np.asarray([tanaka_depth], dtype=np.float64),
        case_specs=[
            [
                TanakaCrest(
                    alpha=0.10,
                    center=np.pi,
                    direction=1,
                )
            ]
        ],
        length=LENGTH,
        nx=NX,
        gravity=GRAVITY,
    )

    bf_params = {
        "n_carr": np.asarray([5], dtype=np.int32),
        "side_offset": np.asarray([1], dtype=np.int32),
        "eps_carrier": np.asarray([0.08], dtype=np.float64),
        "eps_pert": np.asarray([0.10], dtype=np.float64),
        "translation": np.asarray([0.0], dtype=np.float64),
    }
    bf_eta, bf_xi = build_benjamin_feir_initial_conditions(
        x=x,
        parameters=bf_params,
        length=LENGTH,
        gravity=GRAVITY,
        dtype=jnp.float64,
    )

    eta = jnp.concatenate(
        (stokes_eta, tanaka_eta, bf_eta),
        axis=0,
    )
    xi = jnp.concatenate(
        (stokes_xi, tanaka_xi, bf_xi),
        axis=0,
    )
    eta = apply_lowpass(eta, k, FILTER_FRACTION)
    xi = apply_lowpass(xi, k, FILTER_FRACTION)
    xi = xi - jnp.mean(xi, axis=-1, keepdims=True)
    depths = jnp.asarray(
        [[0.5], [tanaka_depth], [deep_water_proxy_depth(LENGTH)]],
        dtype=jnp.float64,
    )
    jax.block_until_ready(xi)
    return State(eta=eta, xi=xi), depths, k


@cache
def _cross_family_rollout_pair() -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
]:
    state, depths, k = _build_cross_family_batch()
    params = SolverParams(
        nx=NX,
        length=LENGTH,
        depth=depths,
        gravity=GRAVITY,
        dno_order=6,
        pad_factor=8,
        filter_fraction=FILTER_FRACTION,
        k=k,
        g0=make_linear_dno_symbol(k, depths),
    )
    times = jnp.asarray([0.0, 0.08], dtype=jnp.float64)
    common = {
        "initial_state": state,
        "times": times,
        "params": params,
        "save_gxi": True,
        "method": "gl2_if",
        "implicit_iterations": 8,
        "zero_mean_xi": True,
    }
    coarse = rollout(**common, substeps_per_interval=8)
    fine = rollout(**common, substeps_per_interval=16)
    jax.block_until_ready(fine["gxi"])
    return coarse, fine, depths


def _trajectory(
    payload: dict[str, jax.Array],
    family_index: int,
) -> RefinementTrajectory:
    return RefinementTrajectory(
        times=np.asarray(payload["times"], dtype=np.float64),
        eta=np.asarray(payload["eta"][:, family_index], dtype=np.float64),
        xi=np.asarray(payload["xi"][:, family_index], dtype=np.float64),
        gxi=np.asarray(payload["gxi"][:, family_index], dtype=np.float64),
    )


class CrossFamilyAcceptanceSmokeTest(unittest.TestCase):
    def test_constructors_produce_finite_graph_valued_bandlimited_states(
        self,
    ) -> None:
        state, depths, _ = _build_cross_family_batch()
        eta = np.asarray(state.eta)
        xi = np.asarray(state.xi)
        depth_values = np.asarray(depths)

        self.assertTrue(np.isfinite(eta).all())
        self.assertTrue(np.isfinite(xi).all())
        self.assertTrue(np.all(depth_values + eta > 0.0))
        for field in (eta, xi):
            spectrum = np.fft.rfft(field, axis=-1)
            self.assertLess(float(np.max(np.abs(spectrum[:, 33:]))), 1e-10)

    def test_dno_targets_translate_with_the_constructed_states(self) -> None:
        state, depths, k = _build_cross_family_batch()
        shift = 7
        gxi = apply_lowpass(
            dno_series_eval(
                state.eta,
                state.xi,
                k,
                depths,
                6,
                pad_factor=8,
            ),
            k,
            FILTER_FRACTION,
        )
        shifted_gxi = apply_lowpass(
            dno_series_eval(
                jnp.roll(state.eta, shift, axis=-1),
                jnp.roll(state.xi, shift, axis=-1),
                k,
                depths,
                6,
                pad_factor=8,
            ),
            k,
            FILTER_FRACTION,
        )
        jax.block_until_ready(shifted_gxi)

        np.testing.assert_allclose(
            np.asarray(shifted_gxi),
            np.asarray(jnp.roll(gxi, shift, axis=-1)),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_supported_cross_family_pair_passes(self) -> None:
        coarse, fine, depths = _cross_family_rollout_pair()
        for family_index, family_name in enumerate(FAMILY_NAMES):
            metrics, decision = evaluate_temporal_refinement(
                _trajectory(coarse, family_index),
                _trajectory(fine, family_index),
                depth=float(depths[family_index, 0]),
                gravity=GRAVITY,
                length=LENGTH,
                maximum_wavenumber=DELIVERED_WAVENUMBER,
            )

            with self.subTest(family=family_name):
                self.assertTrue(decision.accepted)
                self.assertLess(metrics.maximum_error, 1e-7)


if __name__ == "__main__":
    unittest.main()
