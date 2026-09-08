"""Scientific checks for trajectory-family initial-state construction."""

from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    sample_benjamin_feir_simulation,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    JonswapTmaState,
    ResolvedBand,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    sample_tanaka_simulation,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
)

jax.config.update("jax_enable_x64", True)


def _wiring_config() -> RolloutNumerics:
    return RolloutNumerics(
        nx=64,
        target_nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        integration_dno_order=0,
        label_dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        target_maximum_wavenumber=16.0,
        saved_dt=0.02,
        substeps_per_saved_frame=1,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        internal_hamiltonian_drift_threshold=None,
    )


def _resolved_band(config: RolloutNumerics) -> ResolvedBand:
    return ResolvedBand(
        config.length,
        config.target_maximum_wavenumber,
    )


def _assert_initial_batch(
    test: unittest.TestCase,
    batch: TrajectoryInitialBatch,
    config: RolloutNumerics,
) -> None:
    test.assertEqual(batch.eta0.shape, (1, config.nx))
    test.assertEqual(batch.xi0.shape, batch.eta0.shape)
    test.assertEqual(batch.depths.shape, (1,))
    test.assertEqual(batch.eta0.dtype, np.float64)
    test.assertEqual(batch.xi0.dtype, np.float64)
    test.assertEqual(batch.depths.dtype, np.float64)
    test.assertTrue(np.isfinite(batch.eta0).all())
    test.assertTrue(np.isfinite(batch.xi0).all())
    test.assertTrue(np.all(batch.depths[:, None] + batch.eta0 > 0.0))
    test.assertLess(abs(float(np.mean(batch.xi0))), 1.0e-14)

    wavenumbers = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            config.nx,
            d=config.length / config.nx,
        )
    )
    outside = np.abs(wavenumbers) > config.target_maximum_wavenumber
    for field in (batch.eta0, batch.xi0):
        test.assertLess(
            float(np.max(np.abs(np.fft.fft(field, axis=-1)[:, outside]))),
            1.0e-10,
        )


class TrajectoryFamilyAdapterTest(unittest.TestCase):
    def test_all_families_construct_finite_states_with_exact_replay(self) -> None:
        config = _wiring_config()
        band = _resolved_band(config)

        tanaka_samples = tuple(
            sample_tanaka_simulation(
                "main_m1_q0",
                seed=2026072205,
                attempt_number=19,
            )
            for _ in range(2)
        )
        bf_samples = tuple(
            sample_benjamin_feir_simulation(
                "n_c_04__delta_n_01",
                seed=2026072205,
                attempt_number=23,
            )
            for _ in range(2)
        )
        jonswap_samples = tuple(
            sample_jonswap_tma_simulation(
                "finite__gamma_1__right_0",
                seed=2026072205,
                attempt_number=29,
                band=band,
            )
            for _ in range(2)
        )

        pairs = (
            (
                "tanaka",
                construct_tanaka_trajectory_batch((tanaka_samples[0],), config),
                construct_tanaka_trajectory_batch((tanaka_samples[1],), config),
            ),
            (
                "benjamin_feir",
                construct_benjamin_feir_trajectory_batch((bf_samples[0],), config),
                construct_benjamin_feir_trajectory_batch((bf_samples[1],), config),
            ),
            (
                "jonswap_tma",
                construct_jonswap_tma_trajectory_batch(
                    (jonswap_samples[0],), config, band=band
                )[0],
                construct_jonswap_tma_trajectory_batch(
                    (jonswap_samples[1],), config, band=band
                )[0],
            ),
        )
        for family, first, replay in pairs:
            with self.subTest(family=family):
                assert first is not None and replay is not None
                _assert_initial_batch(self, first, config)
                np.testing.assert_array_equal(first.eta0, replay.eta0)
                np.testing.assert_array_equal(first.xi0, replay.xi0)
                np.testing.assert_array_equal(first.depths, replay.depths)

    def test_paper_jonswap_preserves_spectral_band_and_filters_invalid_states(
        self,
    ) -> None:
        config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        band = _resolved_band(config)
        sample = sample_jonswap_tma_simulation(
            "finite__gamma_1__right_0",
            seed=2026072205,
            attempt_number=31,
            band=band,
        )
        initial, indices = construct_jonswap_tma_trajectory_batch(
            (sample,), config, band=band
        )
        assert initial is not None
        self.assertEqual(indices, (0,))

        self.assertEqual(band.maximum_wavenumber, 128.0)
        _assert_initial_batch(self, initial, config)

        wavenumbers = np.asarray(
            2.0 * np.pi * np.fft.rfftfreq(config.nx, d=config.length / config.nx),
            dtype=np.float64,
        )
        frequencies = finite_depth_angular_frequency(
            wavenumbers,
            depth=sample.parameters.depth,
            gravity=config.gravity,
        )
        peak_frequency = finite_depth_angular_frequency(
            np.asarray([sample.parameters.peak_wavenumber]),
            depth=sample.parameters.depth,
            gravity=config.gravity,
        )[0]
        outside_window = (frequencies < 0.4 * peak_frequency) | (
            frequencies > 2.6 * peak_frequency
        )
        self.assertLess(
            float(np.max(np.abs(np.fft.rfft(initial.eta0[0])[outside_window]))),
            1.0e-10,
        )

        samples = tuple(
            sample._replace(parameters=sample.parameters._replace(depth=float(i + 1)))
            for i in range(4)
        )
        eta = np.repeat(
            np.asarray([np.nan, 0.1, -3.0, 0.2])[:, None], config.nx, axis=1
        )
        xi = np.arange(4)[:, None] * np.sin(
            2.0 * np.pi * np.arange(config.nx) / config.nx
        )
        for expected_indices in ((1, 3), ()):
            eta_fields = eta if expected_indices else np.full_like(eta, np.nan)
            with (
                self.subTest(expected_indices=expected_indices),
                patch(
                    "solver.gen_data.trajectory_family_adapters.build_jonswap_tma_initial_condition",
                    side_effect=map(JonswapTmaState, eta_fields, xi),
                ) as build,
            ):
                initial, indices = construct_jonswap_tma_trajectory_batch(
                    samples, config, band=band
                )
            self.assertEqual(build.call_count, len(samples))
            self.assertEqual(indices, expected_indices)
            if not indices:
                self.assertIsNone(initial)
                continue
            assert initial is not None
            np.testing.assert_allclose(initial.eta0, eta[list(indices)], atol=1e-15)
            np.testing.assert_allclose(initial.xi0, xi[list(indices)], atol=1e-15)
            np.testing.assert_array_equal(initial.depths, [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
