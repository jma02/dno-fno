"""Fixed-band spatial-refinement smoke for the Stokes generator revision.

This is method-level validation, not a per-sample acceptance rule.

Run with:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
      uv run python -m unittest solver.gen_data.tests_stokes_spatial_smoke
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

from solver.data.stokes_truth_jax import (  # noqa: E402
    finite_depth_stokes_in_ursell_support,
    stokes_eta_xi,
    stokes_eta_xi_at_phase,
)
from solver.gen_data.generate_stokes_dataset import (  # noqa: E402
    build_paper_stokes_batch,
    build_stokes_states,
)
from solver.gen_data.pipeline.reference import (  # noqa: E402
    PAPER_DNO_TARGET,
    evaluate_discrete_dno_target,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
)

jax.config.update("jax_enable_x64", True)


def _fixed_band_relative_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    *,
    maximum_mode: int,
    denominator_floor: float = 1e-30,
) -> float:
    coarse_coefficients = np.fft.rfft(coarse)[: maximum_mode + 1] / coarse.size
    fine_coefficients = np.fft.rfft(fine)[: maximum_mode + 1] / fine.size
    difference = coarse_coefficients - fine_coefficients

    def real_field_norm(coefficients: np.ndarray) -> float:
        return float(
            np.sqrt(
                abs(coefficients[0]) ** 2
                + 2.0 * np.sum(np.abs(coefficients[1:]) ** 2)
            )
        )

    return real_field_norm(difference) / (
        real_field_norm(fine_coefficients) + denominator_floor
    )


def _build_stokes_fields(nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = 2.0 * np.pi
    x, k = build_grid(nx, length)
    eta, xi = stokes_eta_xi(
        x=jnp.asarray(x, dtype=jnp.float64),
        time=jnp.asarray(0.0, dtype=jnp.float64),
        n0=2,
        a0=0.025,
        length=length,
        depth=0.5,
        gravity=1.0,
        ichoi=1,
    )
    gxi = dno_series_eval(
        eta,
        xi,
        jnp.asarray(k, dtype=jnp.float64),
        0.5,
        6,
        pad_factor=8,
    )
    jax.block_until_ready(gxi)
    return tuple(
        np.asarray(field, dtype=np.float64)
        for field in (eta, xi, gxi)
    )


class StokesSpatialRefinementSmokeTest(unittest.TestCase):
    def test_paper_batch_uses_the_frozen_projected_target(self) -> None:
        x, k = build_grid(
            PAPER_DNO_TARGET.nx,
            PAPER_DNO_TARGET.length,
        )
        n0 = jnp.asarray([26], dtype=jnp.int32)
        a0 = jnp.asarray([0.005], dtype=jnp.float64)
        depth = jnp.asarray([0.2], dtype=jnp.float64)
        phase = jnp.asarray([0.4], dtype=jnp.float64)
        raw_eta, raw_xi = build_stokes_states(
            x=jnp.asarray(x, dtype=jnp.float64),
            length=PAPER_DNO_TARGET.length,
            gravity=1.0,
            n0=n0,
            a0=a0,
            depth=depth,
            phase=phase,
            ichoi=1,
        )
        expected = evaluate_discrete_dno_target(
            raw_eta,
            raw_xi,
            depth[:, None],
        )
        actual = build_paper_stokes_batch(
            gravity=1.0,
            n0=n0,
            a0=a0,
            depth=depth,
            phase=phase,
            ichoi=1,
        )
        raw_eta_coefficients = np.fft.fft(np.asarray(raw_eta), axis=-1)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        raw_eta_coefficients[
                            ..., np.abs(np.asarray(k)) > 128.0
                        ]
                    )
                )
            ),
            1e-6,
        )

        for actual_field, expected_field in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(actual_field),
                np.asarray(expected_field),
                rtol=1e-13,
                atol=1e-14,
            )
            coefficients = np.fft.fft(np.asarray(actual_field), axis=-1)
            self.assertLess(
                float(
                    np.max(
                        np.abs(
                            coefficients[
                                ..., np.abs(np.asarray(k)) > 128.0
                            ]
                        )
                    )
                ),
                1e-10,
            )
        self.assertLess(abs(float(jnp.mean(actual[2]))), 1e-14)

    def test_paper_batch_rejects_noncanonical_gravity(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires gravity"):
            build_paper_stokes_batch(
                gravity=9.81,
                n0=jnp.asarray([2]),
                a0=jnp.asarray([0.01]),
                depth=jnp.asarray([1.0]),
                phase=jnp.asarray([0.0]),
                ichoi=1,
            )

    def test_phase_is_a_translation_and_xi_has_zero_mean(self) -> None:
        nx = 128
        length = 2.0 * np.pi
        x, _ = build_grid(nx, length)
        shift_points = 7
        n0 = 3
        phase = n0 * length * shift_points / nx

        for ichoi, depth in ((0, 10.0), (1, 0.8)):
            eta_zero, xi_zero = stokes_eta_xi_at_phase(
                x=jnp.asarray(x, dtype=jnp.float64),
                phase=0.0,
                n0=n0,
                a0=0.02,
                length=length,
                depth=depth,
                gravity=1.0,
                ichoi=ichoi,
            )
            eta_shifted, xi_shifted = stokes_eta_xi_at_phase(
                x=jnp.asarray(x, dtype=jnp.float64),
                phase=phase,
                n0=n0,
                a0=0.02,
                length=length,
                depth=depth,
                gravity=1.0,
                ichoi=ichoi,
            )

            np.testing.assert_allclose(
                np.asarray(eta_shifted),
                np.roll(np.asarray(eta_zero), -shift_points),
                rtol=1e-12,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                np.asarray(xi_shifted),
                np.roll(np.asarray(xi_zero), -shift_points),
                rtol=1e-12,
                atol=1e-13,
            )
            self.assertLess(abs(float(jnp.mean(xi_shifted))), 1e-14)

    def test_independent_grids_agree_on_fixed_delivered_band(self) -> None:
        self.assertTrue(
            bool(
                finite_depth_stokes_in_ursell_support(
                    k0=2.0,
                    depth=0.5,
                    gravity=1.0,
                    a0=0.025,
                )
            )
        )
        coarse_fields = _build_stokes_fields(32)
        fine_fields = _build_stokes_fields(64)
        errors = tuple(
            _fixed_band_relative_error(
                coarse,
                fine,
                maximum_mode=12,
            )
            for coarse, fine in zip(coarse_fields, fine_fields)
        )

        self.assertLess(max(errors), 1e-6)


if __name__ == "__main__":
    unittest.main()
