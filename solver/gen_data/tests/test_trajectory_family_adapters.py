"""Scientific checks for trajectory-family initial-state construction."""

from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    sample_benjamin_feir_simulation,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    ResolvedBand,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
)
from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
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
                TANAKA_PARAMETER_GROUP_IDS[0],
                dataset_split=DatasetSplit.TEST,
                attempt_number=19,
            )
            for _ in range(2)
        )
        bf_samples = tuple(
            sample_benjamin_feir_simulation(
                BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0],
                dataset_split=DatasetSplit.TEST,
                attempt_number=23,
            )
            for _ in range(2)
        )
        jonswap_samples = tuple(
            sample_jonswap_tma_simulation(
                JONSWAP_TMA_PARAMETER_GROUP_IDS[9],
                dataset_split=DatasetSplit.TEST,
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
                ),
                construct_jonswap_tma_trajectory_batch(
                    (jonswap_samples[1],), config, band=band
                ),
            ),
        )
        for family, first, replay in pairs:
            with self.subTest(family=family):
                _assert_initial_batch(self, first, config)
                np.testing.assert_array_equal(first.eta0, replay.eta0)
                np.testing.assert_array_equal(first.xi0, replay.xi0)
                np.testing.assert_array_equal(first.depths, replay.depths)

    def test_paper_jonswap_uses_the_sharp_relative_frequency_band(self) -> None:
        config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        band = _resolved_band(config)
        sample = sample_jonswap_tma_simulation(
            JONSWAP_TMA_PARAMETER_GROUP_IDS[9],
            dataset_split=DatasetSplit.TEST,
            attempt_number=31,
            band=band,
        )
        initial = construct_jonswap_tma_trajectory_batch((sample,), config, band=band)

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


if __name__ == "__main__":
    unittest.main()
