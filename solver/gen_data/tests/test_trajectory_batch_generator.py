"""Behavioral tests for trajectory-family batch generation."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
import os
from pathlib import Path
import tempfile
from typing import TypeAlias
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    sample_benjamin_feir_simulation,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    ResolvedBand,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.batch_storage import (  # noqa: E402
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    GenerationResult,
    generate_simulations,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedTrajectoryBatch,
)
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.trajectory_subsampling import (  # noqa: E402
    subsample_trajectories,
)
from solver.gen_data.pipeline.types import (  # noqa: E402
    DatasetSplit,
    PhysicalFamilyId,
    SimulationRows,
)
from solver.gen_data.trajectory_batch_generator import (  # noqa: E402
    generate_trajectory_batch,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    JonswapInitialStateDomainError,
    TrajectoryInitialBatch,
)

jax.config.update("jax_enable_x64", True)

FloatArray: TypeAlias = np.ndarray
Constructor: TypeAlias = Callable[
    [tuple[object, ...], RolloutNumerics], TrajectoryInitialBatch
]


def _config() -> RolloutNumerics:
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
        saved_dt=0.08,
        substeps_per_saved_frame=2,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        internal_hamiltonian_drift_threshold=None,
    )


def _generate(
    root: Path,
    family: TrajectoryFamily,
    numerical: RolloutNumerics,
    parameter_group_ids: tuple[str, ...],
    targets: tuple[int, ...],
    *,
    batch_size: int = 2,
) -> GenerationResult:
    return generate_simulations(
        root,
        family_id=PhysicalFamilyId[family.upper()],
        dataset_split=DatasetSplit.TEST,
        requested_per_group=dict(zip(parameter_group_ids, targets, strict=True)),
        batch_size=batch_size,
        generate_batch=partial(
            generate_trajectory_batch,
            dataset_split=DatasetSplit.TEST,
            family=family,
            numerical=numerical,
            solver_batch_size=batch_size if family == "jonswap_tma" else None,
        ),
    )


def _marker_constructor() -> tuple[Constructor, list[int]]:
    calls: list[int] = []
    next_marker = 1

    def construct(
        samples: tuple[object, ...], numerical: RolloutNumerics
    ) -> TrajectoryInitialBatch:
        nonlocal next_marker
        calls.append(len(samples))
        markers = np.arange(
            next_marker,
            next_marker + len(samples),
            dtype=np.float64,
        )
        next_marker += len(samples)
        eta0 = np.repeat(markers[:, None], numerical.nx, axis=1)
        return TrajectoryInitialBatch(
            eta0,
            np.zeros_like(eta0),
            np.full(len(samples), 10.0, dtype=np.float64),
        )

    return construct, calls


def _fast_integrator(
    failed_markers: frozenset[int] = frozenset(),
) -> tuple[Callable[..., IntegratedTrajectoryBatch], list[tuple[int, float]]]:
    calls: list[tuple[int, float]] = []

    def integrate(
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del depths
        batch_size, nx = eta0.shape
        calls.append((batch_size, float(saved_times[-1])))
        eta = np.repeat(eta0[None], saved_times.size, axis=0)
        xi = np.repeat(xi0[None], saved_times.size, axis=0)
        step_shape = (
            (saved_times.size - 1) * config.substeps_per_saved_frame,
            batch_size,
        )
        converged = np.ones(step_shape, dtype=np.bool_)
        markers = np.rint(eta0[:, 0]).astype(int)
        converged[:, np.isin(markers, tuple(failed_markers))] = False
        return IntegratedTrajectoryBatch(
            eta,
            xi,
            np.zeros((saved_times.size, batch_size, nx), dtype=np.float64),
            converged,
            None,
        )

    return integrate, calls


def _integrate_jonswap_without_adjustment(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[FloatArray, ...],
    peak_periods: FloatArray,
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationRows | None, ...]:
    del peak_periods, solver_batch_size
    return subsample_trajectories(
        execute_trajectory_batch(
            initial.eta0,
            initial.xi0,
            initial.depths,
            time_grids,
            config=numerical,
        ),
        initial.depths,
        family="jonswap_tma",
        length=numerical.length,
    )


def _last_saved_times(path: Path) -> list[float]:
    batch = load_completed_batch(path)
    assert batch.shard is not None
    simulation_indices = batch.shard["simulation_local_index"]
    return [
        float(np.max(batch.shard["time"][simulation_indices == index]))
        for index in range(len(batch.parameter_group_ids))
    ]


class TrajectoryBatchGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_failed_rollout_retains_its_sibling_and_is_replaced(self) -> None:
        numerical = _config()
        group = "n_c_04__delta_n_01"
        constructor, constructor_calls = _marker_constructor()
        integrator, integration_calls = _fast_integrator(frozenset({1}))
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_benjamin_feir_trajectory_batch",
                new=constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=integrator,
            ),
        ):
            attempts, completed = _generate(
                self.root,
                "benjamin_feir",
                numerical,
                (group,),
                (2,),
            )

        self.assertEqual(attempts, {group: 3})
        self.assertEqual(constructor_calls, [2, 1])
        self.assertEqual([call[0] for call in integration_calls], [2, 1])
        first = load_completed_batch(completed[0])
        np.testing.assert_array_equal(first.accepted_simulations, (False, True))
        assert first.shard is not None
        np.testing.assert_array_equal(
            np.unique(first.shard["simulation_local_index"]),
            np.asarray([1]),
        )

    def test_tanaka_construction_rejects_only_the_named_simulation(self) -> None:
        numerical = _config()
        group = "main_m1_q0"
        base_constructor, calls = _marker_constructor()
        first_call = True

        class TanakaFailure(ValueError):
            invalid_simulation_indices = (0,)

        def reject_first(
            samples: tuple[object, ...], config: RolloutNumerics
        ) -> TrajectoryInitialBatch:
            nonlocal first_call
            if first_call:
                first_call = False
                calls.append(len(samples))
                raise TanakaFailure
            return base_constructor(samples, config)

        integrator, _ = _fast_integrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "TanakaPotentialRadicandError",
                new=TanakaFailure,
            ),
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_tanaka_trajectory_batch",
                new=reject_first,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=integrator,
            ),
        ):
            _, completed = _generate(
                self.root,
                "tanaka",
                numerical,
                (group,),
                (2,),
            )

        self.assertEqual(calls[:2], [2, 1])
        first = load_completed_batch(completed[0])
        np.testing.assert_array_equal(first.accepted_simulations, (False, True))
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))

    def test_jonswap_construction_rejects_only_the_named_simulation(self) -> None:
        numerical = _config()
        group = "finite__gamma_1__right_0"
        base_constructor, calls = _marker_constructor()
        first_call = True

        def reject_first(
            samples: tuple[object, ...],
            config: RolloutNumerics,
            *,
            band: object,
        ) -> TrajectoryInitialBatch:
            nonlocal first_call
            del band
            if first_call:
                first_call = False
                calls.append(len(samples))
                raise JonswapInitialStateDomainError((0,), (False,), (True,))
            return base_constructor(samples, config)

        integrator, _ = _fast_integrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_jonswap_tma_trajectory_batch",
                new=reject_first,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=integrator,
            ),
            mock.patch(
                "solver.gen_data.jonswap_horizon_generator."
                "integrate_and_subsample_jonswap",
                new=_integrate_jonswap_without_adjustment,
            ),
        ):
            _, completed = _generate(
                self.root,
                "jonswap_tma",
                numerical,
                (group,),
                (2,),
            )

        self.assertEqual(calls[:2], [2, 1])
        first = load_completed_batch(completed[0])
        np.testing.assert_array_equal(first.accepted_simulations, (False, True))
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))

    def test_unexpected_constructor_error_is_not_recorded_as_rejection(self) -> None:
        numerical = _config()
        output = batch_path(
            self.root,
            family="tanaka",
            split="test",
            batch_id=0,
        )

        def fail(
            samples: tuple[object, ...], config: RolloutNumerics
        ) -> TrajectoryInitialBatch:
            del samples, config
            raise ValueError("constructor bug")

        with mock.patch(
            "solver.gen_data.trajectory_batch_generator.construct_tanaka_trajectory_batch",
            new=fail,
        ):
            with self.assertRaisesRegex(ValueError, "constructor bug"):
                _generate(
                    self.root,
                    "tanaka",
                    numerical,
                    ("main_m1_q0",),
                    (1,),
                    batch_size=1,
                )
        self.assertFalse(output.exists())

    def test_jonswap_simulations_use_their_own_peak_period_horizons(self) -> None:
        numerical = _config()
        groups = (
            "finite__gamma_1__right_0",
            "deep__gamma_1__right_0",
        )
        constructor, _ = _marker_constructor()
        integrator, calls = _fast_integrator()

        def construct(
            samples: tuple[object, ...],
            config: RolloutNumerics,
            *,
            band: object,
        ) -> TrajectoryInitialBatch:
            del band
            return constructor(samples, config)

        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_jonswap_tma_trajectory_batch",
                new=construct,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=integrator,
            ),
            mock.patch(
                "solver.gen_data.jonswap_horizon_generator."
                "integrate_and_subsample_jonswap",
                new=_integrate_jonswap_without_adjustment,
            ),
        ):
            _, completed = _generate(
                self.root,
                "jonswap_tma",
                numerical,
                groups,
                (1, 1),
            )

        band = ResolvedBand(numerical.length, numerical.target_maximum_wavenumber)
        expected = []
        for offset, group in enumerate(groups):
            sample = sample_jonswap_tma_simulation(
                group,
                dataset_split=DatasetSplit.TEST,
                attempt_number=offset,
                band=band,
            )
            frequency = finite_depth_angular_frequency(
                np.asarray([sample.parameters.peak_wavenumber], dtype=np.float64),
                depth=sample.parameters.depth,
                gravity=numerical.gravity,
            )[0]
            intended = 16.0 * 2.0 * math.pi / float(frequency)
            expected.append(
                math.floor(intended / numerical.saved_dt) * numerical.saved_dt
            )

        np.testing.assert_allclose(
            _last_saved_times(completed[0]), expected, rtol=0.0, atol=1.0e-13
        )
        self.assertEqual(calls, [(2, max(expected))])
        np.testing.assert_array_equal(
            load_completed_batch(completed[0]).accepted_simulations,
            (True, True),
        )

    def test_benjamin_feir_simulations_use_their_own_carrier_horizons(self) -> None:
        numerical = _config()
        groups = (
            "n_c_04__delta_n_01",
            "n_c_20__delta_n_07",
        )
        constructor, _ = _marker_constructor()
        integrator, calls = _fast_integrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_benjamin_feir_trajectory_batch",
                new=constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=integrator,
            ),
        ):
            _, completed = _generate(
                self.root,
                "benjamin_feir",
                numerical,
                groups,
                (1, 1),
            )

        expected = []
        for offset, group in enumerate(groups):
            sample = sample_benjamin_feir_simulation(
                group,
                dataset_split=DatasetSplit.TEST,
                attempt_number=offset,
            )
            carrier_wavenumber = 2.0 * math.pi * sample.carrier_mode / numerical.length
            intended = (
                100.0
                * 2.0
                * math.pi
                / math.sqrt(numerical.gravity * carrier_wavenumber)
            )
            expected.append(
                math.floor(intended / numerical.saved_dt) * numerical.saved_dt
            )

        self.assertNotEqual(*expected)
        np.testing.assert_allclose(
            _last_saved_times(completed[0]), expected, rtol=0.0, atol=1.0e-13
        )
        self.assertEqual(calls, [(2, max(expected))])

    def test_jonswap_requires_solver_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "solver_batch_size"):
            generate_trajectory_batch(
                ("shallow__gamma_1__right_0",),
                0,
                self.root / "batch.npz",
                dataset_split=DatasetSplit.TEST,
                family="jonswap_tma",
                numerical=_config(),
            )


if __name__ == "__main__":
    unittest.main()
