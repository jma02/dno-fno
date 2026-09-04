"""CPU wiring tests for the three trajectory-family adapters.

The reduced ``N=64``, ``M=0`` GL2 runs below test software wiring only.  They
are not spatial-, order-, or full-horizon validation of the paper config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import TypeAlias
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RELATIVE_FREQUENCY_WINDOW,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.types import (  # noqa: E402
    BatchPlanArrays,
    DatasetSplit,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.trajectory_subsampling import (  # noqa: E402
    subsample_trajectories,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    build_batch_plan,
    commit_simulation_outcomes,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
    resolved_band_for_config,
    sample_benjamin_feir_simulations,
    sample_jonswap_tma_simulations,
    sample_tanaka_simulations,
)

jax.config.update("jax_enable_x64", True)


SpecificationRecords: TypeAlias = tuple[Mapping[str, object], ...]


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a JSON object")
    return value


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array")
    return value


def _require_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a JSON number")
    return float(value)


def wiring_config() -> RolloutNumerics:
    """Return a deliberately reduced real-GL2 software-wiring config."""

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


def assert_valid_initial_batch(
    test: unittest.TestCase,
    batch: TrajectoryInitialBatch,
    *,
    config: RolloutNumerics,
) -> None:
    """Check dtype, graph, band, mean, and JSON invariants."""

    test.assertEqual(batch.eta0.dtype, np.float64)
    test.assertEqual(batch.xi0.dtype, np.float64)
    test.assertEqual(batch.depths.dtype, np.float64)
    test.assertEqual(batch.eta0.shape, (1, config.nx))
    test.assertEqual(batch.xi0.shape, (1, config.nx))
    test.assertTrue(np.isfinite(batch.eta0).all())
    test.assertTrue(np.isfinite(batch.xi0).all())
    test.assertTrue(np.all(batch.depths[:, None] + batch.eta0 > 0.0))
    test.assertLess(float(np.max(np.abs(np.mean(batch.xi0, axis=-1)))), 1.0e-14)

    modes = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            config.nx,
            d=config.length / config.nx,
        )
    )
    for field in (batch.eta0, batch.xi0):
        coefficients = np.fft.fft(field, axis=-1)
        outside = coefficients[:, np.abs(modes) > config.maximum_wavenumber]
        test.assertLess(float(np.max(np.abs(outside))), 1.0e-11)
    json.dumps(
        batch.specification_records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sample_and_construct_family(
    family: TrajectoryFamily,
    *,
    parameter_group_id: str,
    attempt_number: int,
    config: RolloutNumerics,
    root: Path,
    batch_id: int,
) -> tuple[TrajectoryInitialBatch, SpecificationRecords, Path, BatchPlanArrays]:
    """Sample and construct one simulation without writing its batch."""

    output_path = root / f"{family}_{batch_id:08d}.npz"
    if family == "tanaka":
        sampled = sample_tanaka_simulations(
            (parameter_group_id,),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=attempt_number,
            config=config,
        )
        initial = construct_tanaka_trajectory_batch(sampled)
    elif family == "benjamin_feir":
        sampled = sample_benjamin_feir_simulations(
            (parameter_group_id,),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=attempt_number,
            config=config,
        )
        initial = construct_benjamin_feir_trajectory_batch(sampled)
    else:
        sampled = sample_jonswap_tma_simulations(
            (parameter_group_id,),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=attempt_number,
            config=config,
        )
        initial = construct_jonswap_tma_trajectory_batch(sampled)

    return (
        initial,
        sampled.specification_records,
        output_path,
        build_batch_plan(
            sampled.parameter_group_ids,
            sampled.specification_records,
            family_id={
                "tanaka": PhysicalFamilyId.TANAKA,
                "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
                "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
            }[family],
            dataset_split=DatasetSplit.TEST,
        ),
    )


class TrajectoryFamilyAdapterTest(unittest.TestCase):
    def test_paper_jonswap_constructs_on_target_band_before_wide_evolution(
        self,
    ) -> None:
        config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[9]
        sampled = sample_jonswap_tma_simulations(
            (parameter_group_id,),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=29,
            config=config,
        )
        record = sampled.specification_records[0]
        band = resolved_band_for_config(config)

        self.assertEqual(config.nx, 2048)
        self.assertEqual(config.maximum_wavenumber, 704.0)
        self.assertEqual(config.target_nx, 1024)
        self.assertEqual(config.target_maximum_wavenumber, 128.0)
        self.assertEqual(band.maximum_wavenumber, 128.0)
        self.assertEqual(band.transition_wavenumber, 96.0)
        self.assertEqual(len(_require_list(record["phase_right"])), 128)
        self.assertEqual(len(_require_list(record["phase_left"])), 128)
        initial_projection = _require_mapping(record["initial_projection"])
        self.assertEqual(
            initial_projection["maximum_wavenumber"],
            128.0,
        )
        constructor_settings = _require_mapping(record["constructor_settings"])
        self.assertEqual(
            constructor_settings["density_window"],
            PAPER_RELATIVE_FREQUENCY_WINDOW,
        )
        initial = construct_jonswap_tma_trajectory_batch(sampled)

        metrics = initial.construction_metrics[0]
        self.assertEqual(
            metrics["initial_discrete_peak_wavenumber"],
            round(sampled.samples[0].parameters.peak_wavenumber),
        )
        self.assertGreaterEqual(
            _require_number(metrics["initial_half_maximum_spectral_cell_count"]),
            1,
        )
        self.assertAlmostEqual(
            _require_number(metrics["initial_realized_height_ratio"]),
            1.0,
            places=13,
        )
        self.assertLess(
            _require_number(metrics["initial_linear_hamiltonian_relative_error"]),
            1.0e-12,
        )
        self.assertGreater(
            _require_number(metrics["initial_minimum_water_column"]),
            0.0,
        )

        wavenumbers = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                config.nx,
                d=config.length / config.nx,
            )
        )
        outside_target = np.abs(wavenumbers) > 128.0
        for field in (initial.eta0, initial.xi0):
            coefficients = np.fft.fft(field, axis=-1)
            self.assertLess(
                float(np.max(np.abs(coefficients[:, outside_target]))),
                1.0e-10,
            )

    def test_jonswap_uses_the_published_relative_frequency_band(
        self,
    ) -> None:
        config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[9]
        sampled = sample_jonswap_tma_simulations(
            (parameter_group_id,),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=31,
            config=config,
        )
        record = sampled.specification_records[0]
        self.assertEqual(
            record["initial_condition_constructor"],
            "relative_frequency_jonswap_tma_linear_state_v2",
        )
        constructor_settings = _require_mapping(record["constructor_settings"])
        self.assertEqual(
            constructor_settings["density_window"],
            PAPER_RELATIVE_FREQUENCY_WINDOW,
        )
        self.assertEqual(
            constructor_settings["relative_frequency_minimum"],
            PAPER_RELATIVE_FREQUENCY_MINIMUM,
        )
        self.assertEqual(
            constructor_settings["relative_frequency_maximum"],
            PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )

        initial = construct_jonswap_tma_trajectory_batch(sampled)

        parameters = sampled.samples[0].parameters
        wavenumbers = np.asarray(
            2.0
            * np.pi
            * np.fft.rfftfreq(
                config.nx,
                d=config.length / config.nx,
            ),
            dtype=np.float64,
        )
        frequencies = finite_depth_angular_frequency(
            wavenumbers,
            depth=parameters.depth,
            gravity=config.gravity,
        )
        peak_frequency = finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber]),
            depth=parameters.depth,
            gravity=config.gravity,
        )[0]
        coefficients = np.fft.rfft(initial.eta0[0])
        far_outside = (frequencies > 2.6 * peak_frequency) | (
            frequencies < 0.4 * peak_frequency
        )
        self.assertLess(float(np.max(np.abs(coefficients[far_outside]))), 1.0e-10)

    def test_all_families_construct_finite_graph_states_with_exact_replay(
        self,
    ) -> None:
        config = wiring_config()
        families: tuple[tuple[TrajectoryFamily, str, int], ...] = (
            ("tanaka", TANAKA_PARAMETER_GROUP_IDS[0], 19),
            ("benjamin_feir", BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0], 23),
            ("jonswap_tma", JONSWAP_TMA_PARAMETER_GROUP_IDS[9], 29),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constructed: dict[TrajectoryFamily, TrajectoryInitialBatch] = {}
            for batch_id, (family, parameter_group_id, attempt_number) in enumerate(
                families
            ):
                with self.subTest(family=family):
                    first, first_records, proposed_path, _ = (
                        _sample_and_construct_family(
                            family,
                            parameter_group_id=parameter_group_id,
                            attempt_number=attempt_number,
                            config=config,
                            root=root,
                            batch_id=batch_id,
                        )
                    )
                    second, replayed_records, _, _ = _sample_and_construct_family(
                        family,
                        parameter_group_id=parameter_group_id,
                        attempt_number=attempt_number,
                        config=config,
                        root=root,
                        batch_id=batch_id,
                    )
                    self.assertEqual(first_records, replayed_records)
                    if family == "benjamin_feir":
                        bf_record = first_records[0]
                        self.assertNotIn("schema", bf_record)
                        self.assertEqual(
                            bf_record["initial_condition_constructor"],
                            ("jcp09_equation_33_with_project_fifth_order_carrier_v2"),
                        )
                    self.assertFalse(proposed_path.exists())
                    assert_valid_initial_batch(self, first, config=config)
                    np.testing.assert_array_equal(first.eta0, second.eta0)
                    np.testing.assert_array_equal(first.xi0, second.xi0)
                    np.testing.assert_array_equal(first.depths, second.depths)
                    self.assertEqual(
                        first.specification_records,
                        second.specification_records,
                    )
                    constructed[family] = first

            random_sea = constructed["jonswap_tma"]
            record = random_sea.specification_records[0]
            band = resolved_band_for_config(config)
            phase_right = _require_list(record["phase_right"])
            phase_left = _require_list(record["phase_left"])
            self.assertEqual(
                len(phase_right),
                int(np.floor(band.maximum_wavenumber)),
            )
            self.assertEqual(len(phase_left), len(phase_right))
            constructor_settings = _require_mapping(record["constructor_settings"])
            self.assertEqual(
                constructor_settings["density_window"],
                PAPER_RELATIVE_FREQUENCY_WINDOW,
            )

    def test_jonswap_construction_refuses_a_changed_density_window(self) -> None:
        config = wiring_config()
        sampled = sample_jonswap_tma_simulations(
            (JONSWAP_TMA_PARAMETER_GROUP_IDS[9],),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=35,
            config=config,
        )
        changed_settings = {
            **sampled.construction_settings,
            "density_window": "changed_window",
        }
        changed = replace(sampled, construction_settings=changed_settings)
        with self.assertRaisesRegex(ValueError, "density window"):
            construct_jonswap_tma_trajectory_batch(changed)

    def test_bf_and_jonswap_real_gl2_to_committed_manifest_wiring(
        self,
    ) -> None:
        """Exercise the real CPU generator; numerical settings are smoke-only."""

        config = wiring_config()
        simulations: tuple[tuple[TrajectoryFamily, str, int], ...] = (
            ("benjamin_feir", BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0], 41),
            ("jonswap_tma", JONSWAP_TMA_PARAMETER_GROUP_IDS[9], 43),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            for batch_id, (family, parameter_group_id, attempt_number) in enumerate(
                simulations
            ):
                initial, _, batch_path, batch_plan = _sample_and_construct_family(
                    family,
                    parameter_group_id=parameter_group_id,
                    attempt_number=attempt_number,
                    config=config,
                    root=root,
                    batch_id=batch_id,
                )
                self.assertFalse(batch_path.exists())
                saved_time_count = 200 if family == "benjamin_feir" else 16
                saved_times = config.saved_dt * np.arange(
                    saved_time_count, dtype=np.float64
                )
                execution = execute_trajectory_batch(
                    initial.eta0,
                    initial.xi0,
                    initial.depths,
                    (saved_times,) * initial.eta0.shape[0],
                    config=config,
                )
                self.assertTrue(execution[0].decision.accepted)
                outcomes = subsample_trajectories(
                    execution,
                    initial.depths,
                    family=family,
                    length=config.length,
                )
                commit_simulation_outcomes(
                    batch_path,
                    batch_plan,
                    outcomes,
                )
                paths.append(batch_path)

            view = build_dataset_view(
                root,
                tuple(paths),
                name="trajectory_family_adapter_smoke",
                length=config.length,
            )
            manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_trajectories"], 2)
            self.assertEqual(manifest["n_rows"], 216)
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_accepted"],
                    np.asarray([True, True]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([200, 16]),
                )


if __name__ == "__main__":
    unittest.main()
