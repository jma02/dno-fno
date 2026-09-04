"""CPU tests for rollout-family batch execution."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
import math
import os
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
)
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    deep_water_proxy_depth,
)
from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    TanakaPotentialRadicandError,
    TanakaRadicandFailure,
    _validate_tanaka_surface_potential_radicand,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.pipeline.batch_storage import (  # noqa: E402
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.types import (  # noqa: E402
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    GenerationResult,
    generate_simulations,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
    TrajectoryFamily,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedTrajectoryBatch,
    GL2BatchTelemetry,
)
from solver.gen_data.pipeline.writer import SimulationOutcome  # noqa: E402
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
    TanakaCrest,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    JonswapInitialStateDomainError,
    JonswapInitialStateFailure,
    SampledSimulations,
    TrajectoryInitialBatch,
    construct_tanaka_trajectory_batch,
    sample_tanaka_simulations,
)
from solver.gen_data.trajectory_batch_generator import (  # noqa: E402
    SimulationTimeGrid,
    generate_trajectory_batch,
    integrate_and_subsample_trajectories,
)

jax.config.update("jax_enable_x64", True)


class InjectedInterruption(OSError):
    """Controlled nonterminal interruption."""


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
    *,
    parameter_group_ids: tuple[str, ...],
    targets: tuple[int, ...],
    batch_size: int = 2,
) -> GenerationResult:
    return generate_simulations(
        root,
        family_id=PhysicalFamilyId[family.upper()],
        dataset_split=DatasetSplit.TEST,
        accepted_targets=dict(zip(parameter_group_ids, targets, strict=True)),
        batch_size=batch_size,
        generate_batch=partial(
            generate_trajectory_batch,
            dataset_split=DatasetSplit.TEST,
            family=family,
            numerical=numerical,
            solver_batch_size=batch_size if family == "jonswap_tma" else None,
        ),
    )


def _integrate_jonswap_without_adjustment(
    initial: TrajectoryInitialBatch,
    time_grids: tuple[SimulationTimeGrid, ...],
    *,
    numerical: RolloutNumerics,
    solver_batch_size: int,
) -> tuple[SimulationOutcome, ...]:
    del solver_batch_size
    return integrate_and_subsample_trajectories(
        initial,
        time_grids,
        family="jonswap_tma",
        numerical=numerical,
    )


def _tanaka_failure_component(
    local_simulation_index: int,
    global_component_index: int,
    *,
    component_within_simulation: int = 0,
) -> TanakaRadicandFailure:
    return TanakaRadicandFailure(
        local_simulation_index=local_simulation_index,
        component_within_simulation=component_within_simulation,
        global_component_index=global_component_index,
        alpha=0.2,
        center=0.3,
        direction=1,
        depth=0.25,
        unsigned_speed=1.0,
        speed_squared=1.0,
        minimum_radicand=-1.0e-6,
        minimum_radicand_over_speed_squared=-1.0e-6,
        minimum_radicand_grid_index=3,
        minimum_radicand_x=0.4,
        negative_count=1,
        nonfinite_count=0,
        first_nonfinite_grid_index=None,
        first_nonfinite_x=None,
    )


class MarkerConstructor:
    """Construct constant marker fields for fast generator tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.markers_by_record: dict[str, int] = {}

    def __call__(
        self,
        sampled: SampledSimulations[object],
        *,
        selected_local_indices: tuple[int, ...] | None = None,
    ) -> TrajectoryInitialBatch:
        all_indices = tuple(range(len(sampled.parameter_group_ids)))
        selected = selected_local_indices or all_indices
        self.calls.append(selected)
        records = tuple(sampled.specification_records[index] for index in selected)
        marker_values = []
        for record in records:
            key = json.dumps(record, sort_keys=True)
            if key not in self.markers_by_record:
                self.markers_by_record[key] = len(self.markers_by_record) + 1
            marker_values.append(self.markers_by_record[key])
        markers = np.asarray(marker_values, dtype=np.float64)
        eta0 = np.repeat(
            markers[:, None],
            sampled.config.nx,
            axis=1,
        )
        return TrajectoryInitialBatch(
            eta0=eta0,
            xi0=np.zeros_like(eta0),
            depths=np.asarray(
                [
                    (
                        float(cast(float, record["depth"]))
                        if "depth" in record
                        else deep_water_proxy_depth(sampled.config.length)
                    )
                    for record in records
                ],
                dtype=np.float64,
            ),
            specification_records=records,
        )


class JonswapRejectingConstructor:
    """Declare attempt zero invalid while retaining its proposed siblings."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.base = MarkerConstructor()
        self.rejected_record: str | None = None

    def __call__(
        self,
        sampled: SampledSimulations[object],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch:
        selected = selected_local_indices or tuple(
            range(len(sampled.parameter_group_ids))
        )
        self.calls.append(selected)
        if self.rejected_record is None:
            self.rejected_record = json.dumps(
                sampled.specification_records[0], sort_keys=True
            )
        invalid_positions = tuple(
            position
            for position, original_index in enumerate(selected)
            if json.dumps(sampled.specification_records[original_index], sort_keys=True)
            == self.rejected_record
        )
        if invalid_positions:
            failures = tuple(
                JonswapInitialStateFailure(
                    local_simulation_index=position,
                    state_finite=True,
                    minimum_water_column=-0.01,
                )
                for position in invalid_positions
            )
            raise JonswapInitialStateDomainError(failures)
        return self.base(
            sampled,
            selected_local_indices=selected,
        )


class FastRolloutIntegrator:
    """Return valid synthetic rollouts, optionally rejecting selected markers."""

    def __init__(
        self,
        *,
        failed_production_markers: frozenset[int] = frozenset(),
    ) -> None:
        self.failed_production_markers = failed_production_markers
        self.calls: list[tuple[int, int, float]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del depths
        batch_size, nx = eta0.shape
        self.calls.append(
            (config.substeps_per_saved_frame, batch_size, float(saved_times[-1]))
        )
        eta = np.repeat(eta0[None, :, :], saved_times.size, axis=0)
        xi = np.repeat(xi0[None, :, :], saved_times.size, axis=0)
        substeps = config.substeps_per_saved_frame
        telemetry_shape = ((saved_times.size - 1) * substeps, batch_size)
        converged = np.ones(telemetry_shape, dtype=np.bool_)
        markers = np.rint(eta0[:, 0]).astype(int)
        converged[:, np.isin(markers, tuple(self.failed_production_markers))] = False
        return IntegratedTrajectoryBatch(
            eta=np.asarray(eta, dtype=np.float64),
            xi=np.asarray(xi, dtype=np.float64),
            gxi=np.zeros((saved_times.size, batch_size, nx), dtype=np.float64),
            gl2=GL2BatchTelemetry(
                stage_residual=np.zeros(telemetry_shape, dtype=np.float64),
                converged=converged,
                stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
                state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            ),
        )


class TrajectoryBatchGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_failed_trajectory_retains_sibling_and_replaces_same_parameter_group(
        self,
    ) -> None:
        numerical = _config()
        parameter_group_id = BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0]
        constructor = MarkerConstructor()
        rollouts = FastRolloutIntegrator(failed_production_markers=frozenset({1}))
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_benjamin_feir_trajectory_batch",
                new=constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
            ),
        ):
            attempts, completed = _generate(
                self.root,
                "benjamin_feir",
                numerical,
                parameter_group_ids=(parameter_group_id,),
                targets=(2,),
            )

        self.assertEqual(attempts, {parameter_group_id: 3})
        self.assertEqual(constructor.calls, [(0, 1), (0,)])
        first = load_completed_batch(completed[0])
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True],
        )
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))
        self.assertEqual(first.shard["eta"].shape[0], 200)
        second = load_completed_batch(completed[1])
        self.assertEqual(
            str(second.plan["parameter_group_id"][0]),
            parameter_group_id,
        )

    def test_declared_tanaka_failure_rejects_only_named_simulation(self) -> None:
        numerical = _config()
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        base_constructor = MarkerConstructor()
        rejected_first_simulation = False

        def declared_constructor(
            sampled: SampledSimulations[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            nonlocal rejected_first_simulation
            selected = selected_local_indices or tuple(
                range(len(sampled.parameter_group_ids))
            )
            invalid_positions = (0,) if not rejected_first_simulation else ()
            rejected_first_simulation = True
            if invalid_positions:
                raise TanakaPotentialRadicandError(
                    "negative_or_nonfinite_surface_potential_radicand",
                    tuple(
                        _tanaka_failure_component(position, position)
                        for position in invalid_positions
                    ),
                )
            return base_constructor(
                sampled,
                selected_local_indices=selected,
            )

        rollouts = FastRolloutIntegrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_tanaka_trajectory_batch",
                new=declared_constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
            ),
        ):
            _, completed = _generate(
                self.root / "declared",
                "tanaka",
                numerical,
                parameter_group_ids=(parameter_group_id,),
                targets=(2,),
            )

        first = load_completed_batch(completed[0])
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True],
        )
        self.assertEqual(
            first.simulations[0].metrics["construction_status"],
            "declared_outside_support",
        )
        self.assertEqual(
            first.simulations[0].failed_checks,
            ("outside_support",),
        )
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))

        def unexpected_constructor(
            sampled: SampledSimulations[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del sampled, selected_local_indices
            raise ValueError("unclassified constructor bug")

        unexpected_root = self.root / "unexpected"
        with mock.patch(
            "solver.gen_data.trajectory_batch_generator."
            "construct_tanaka_trajectory_batch",
            new=unexpected_constructor,
        ):
            with self.assertRaisesRegex(ValueError, "unclassified"):
                _generate(
                    unexpected_root,
                    "tanaka",
                    numerical,
                    parameter_group_ids=(parameter_group_id,),
                    targets=(1,),
                    batch_size=1,
                )
        self.assertFalse(
            batch_path(
                unexpected_root,
                family="tanaka",
                split="test",
                batch_id=0,
            ).exists()
        )

    def test_declared_jonswap_graph_failure_retains_sibling_and_replays(
        self,
    ) -> None:
        numerical = _config()
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[9]

        def run(
            root: Path,
        ) -> tuple[
            tuple[dict[str, int], tuple[Path, ...]],
            JonswapRejectingConstructor,
        ]:
            constructor = JonswapRejectingConstructor()
            rollouts = FastRolloutIntegrator()
            with (
                mock.patch(
                    "solver.gen_data.trajectory_batch_generator."
                    "construct_jonswap_tma_trajectory_batch",
                    new=constructor,
                ),
                mock.patch(
                    "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                    new=rollouts,
                ),
                mock.patch(
                    "solver.gen_data.jonswap_horizon_generator."
                    "integrate_and_subsample_jonswap",
                    new=_integrate_jonswap_without_adjustment,
                ),
            ):
                state = _generate(
                    root,
                    "jonswap_tma",
                    numerical,
                    parameter_group_ids=(parameter_group_id,),
                    targets=(2,),
                )
            return state, constructor

        state, constructor = run(self.root / "first")
        _, completed = state
        self.assertEqual(constructor.calls, [(0, 1), (1,), (0,)])
        first = load_completed_batch(completed[0])
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True],
        )
        rejected = first.simulations[0]
        self.assertEqual(
            rejected.failed_checks,
            ("nonpositive_water_height", "outside_support"),
        )
        failure = json.loads(str(rejected.metrics["construction_failure_json"]))
        self.assertEqual(failure["index_space"], "original_batch_plan_local_index")
        self.assertEqual(failure["invalid_simulation_indices"], [0])
        self.assertEqual(
            failure["constructor_subbatch_invalid_simulation_indices"],
            [0],
        )
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))

        replay, replay_constructor = run(self.root / "replay")
        _, replay_completed = replay
        self.assertEqual(replay_constructor.calls, constructor.calls)
        self.assertEqual(
            [load_completed_batch(path).simulations for path in replay_completed],
            [load_completed_batch(path).simulations for path in completed],
        )

    def test_constructor_generated_finite_radicand_failure_is_recoverable(
        self,
    ) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            _validate_tanaka_surface_potential_radicand(
                np.asarray(((1.0, -1.0e-8),), dtype=np.float64),
                x_grid=np.asarray((0.0, 0.5), dtype=np.float64),
                speed_per_crest=np.asarray((1.0,), dtype=np.float64),
                simulation_h_ref=np.asarray((0.2,), dtype=np.float64),
                flat_specs=[TanakaCrest(0.2, 0.3, 1)],
                crest_simulation_ids=np.asarray((0,), dtype=np.int32),
                components_within_simulation=(0,),
            )
        self.assertTrue(caught.exception.is_recoverable)
        self.assertEqual(caught.exception.invalid_simulation_indices, (0,))

    def test_speed_failures_are_not_recoverable(
        self,
    ) -> None:
        for speed in (0.0, math.nan):
            with (
                self.subTest(speed=speed),
                self.assertRaises(TanakaPotentialRadicandError) as caught,
            ):
                _validate_tanaka_surface_potential_radicand(
                    np.asarray(((1.0, 2.0),), dtype=np.float64),
                    x_grid=np.asarray((0.0, 0.5), dtype=np.float64),
                    speed_per_crest=np.asarray((speed,), dtype=np.float64),
                    simulation_h_ref=np.asarray((0.2,), dtype=np.float64),
                    flat_specs=[TanakaCrest(0.2, 0.3, 1)],
                    crest_simulation_ids=np.asarray((0,), dtype=np.int32),
                    components_within_simulation=(0,),
                )
            self.assertFalse(caught.exception.is_recoverable)

    def test_nonfinite_radicands_are_not_recoverable(
        self,
    ) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            _validate_tanaka_surface_potential_radicand(
                np.asarray(((1.0, math.nan),), dtype=np.float64),
                x_grid=np.asarray((0.0, 0.5), dtype=np.float64),
                speed_per_crest=np.asarray((1.0,), dtype=np.float64),
                simulation_h_ref=np.asarray((0.2,), dtype=np.float64),
                flat_specs=[TanakaCrest(0.2, 0.3, 1)],
                crest_simulation_ids=np.asarray((0,), dtype=np.int32),
                components_within_simulation=(0,),
            )
        self.assertFalse(caught.exception.is_recoverable)

    def test_two_round_tanaka_rejections_archive_original_indices(self) -> None:
        numerical = _config()
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        base_constructor = MarkerConstructor()
        rejection_round = 0

        def rejecting_constructor(
            sampled: SampledSimulations[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            nonlocal rejection_round
            selected = selected_local_indices or tuple(
                range(len(sampled.parameter_group_ids))
            )
            if rejection_round < 2:
                position = rejection_round
                rejection_round += 1
                raise TanakaPotentialRadicandError(
                    "negative_or_nonfinite_surface_potential_radicand",
                    (_tanaka_failure_component(position, position),),
                )
            return base_constructor(
                sampled,
                selected_local_indices=selected,
            )

        rollouts = FastRolloutIntegrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_tanaka_trajectory_batch",
                new=rejecting_constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
            ),
        ):
            _, completed = _generate(
                self.root,
                "tanaka",
                numerical,
                parameter_group_ids=(parameter_group_id,),
                targets=(3,),
                batch_size=3,
            )

        first = load_completed_batch(completed[0])
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True, False],
        )
        second_failure = json.loads(
            str(first.simulations[2].metrics["construction_failure_json"])
        )
        self.assertEqual(second_failure["invalid_simulation_indices"], [2])
        self.assertEqual(
            second_failure["constructor_subbatch_invalid_simulation_indices"],
            [1],
        )
        component = second_failure["components"][0]
        self.assertEqual(component["local_simulation_index"], 2)
        self.assertEqual(component["global_component_index"], 2)
        self.assertEqual(component["constructor_subbatch_local_simulation_index"], 1)
        self.assertEqual(
            component["constructor_subbatch_global_component_index"],
            1,
        )

    def test_selected_tanaka_adapter_verifies_full_proposal_then_subsets(
        self,
    ) -> None:
        numerical = _config()
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        sampled = sample_tanaka_simulations(
            (parameter_group_id, parameter_group_id),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=19,
            config=numerical,
        )
        observed_depths: list[np.ndarray] = []

        def fake_builder(**arguments):
            depths = np.asarray(arguments["simulation_h_ref"], dtype=np.float64)
            simulation_specs = arguments["simulation_specs"]
            self.assertEqual(len(simulation_specs), 1)
            observed_depths.append(depths.copy())
            zeros = np.zeros(
                (depths.size, numerical.nx),
                dtype=np.float64,
            )
            return zeros, zeros

        with mock.patch(
            "solver.gen_data.trajectory_family_adapters."
            "build_per_simulation_initial_conditions",
            side_effect=fake_builder,
        ):
            selected = construct_tanaka_trajectory_batch(
                sampled,
                selected_local_indices=(1,),
            )
        self.assertEqual(selected.eta0.shape, (1, numerical.nx))
        self.assertEqual(
            selected.specification_records,
            (sampled.specification_records[1],),
        )
        np.testing.assert_array_equal(
            observed_depths[0],
            np.asarray([sampled.samples[1].depth], dtype=np.float64),
        )
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            construct_tanaka_trajectory_batch(
                sampled,
                selected_local_indices=(1, 0),
            )

    def test_jonswap_simulations_use_distinct_per_simulation_peak_period_grids(
        self,
    ) -> None:
        numerical = _config()
        parameter_group_ids = (
            JONSWAP_TMA_PARAMETER_GROUP_IDS[9],
            JONSWAP_TMA_PARAMETER_GROUP_IDS[18],
        )
        rollouts = FastRolloutIntegrator()
        with (
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
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
                parameter_group_ids=parameter_group_ids,
                targets=(1, 1),
            )

        batch = load_completed_batch(completed[0])
        records = [
            json.loads(str(value)) for value in batch.plan["simulation_spec_json"]
        ]
        expected_terminal_times = []
        for record in records:
            frequency = float(
                finite_depth_angular_frequency(
                    np.asarray([record["peak_wavenumber"]], dtype=np.float64),
                    depth=float(record["depth"]),
                    gravity=numerical.gravity,
                )[0]
            )
            intended = 16.0 * 2.0 * math.pi / frequency
            expected_terminal_times.append(
                math.floor(
                    (intended + 1.0e-12 * numerical.saved_dt) / numerical.saved_dt
                )
                * numerical.saved_dt
            )
        np.testing.assert_allclose(
            [
                float(cast(float, simulation.metrics["realized_terminal_time"]))
                for simulation in batch.simulations
            ],
            expected_terminal_times,
            rtol=0.0,
            atol=1.0e-13,
        )
        self.assertTrue(all(batch_size == 2 for _, batch_size, _ in rollouts.calls))
        self.assertEqual(len(rollouts.calls), 1)
        self.assertEqual(
            [terminal for _, _, terminal in rollouts.calls],
            [max(expected_terminal_times)],
        )
        assert batch.shard is not None
        self.assertEqual(
            np.bincount(batch.shard["simulation_local_index"]).tolist(),
            [16, 16],
        )
        for simulation in batch.simulations:
            self.assertIn("initial_discrete_peak_wavenumber", simulation.metrics)
            self.assertIn(
                "initial_linear_hamiltonian_relative_error",
                simulation.metrics,
            )

    def test_benjamin_feir_simulations_use_distinct_carrier_period_grids(self) -> None:
        numerical = _config()
        parameter_group_ids = (
            BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0],
            BENJAMIN_FEIR_PARAMETER_GROUP_IDS[-1],
        )
        rollouts = FastRolloutIntegrator()
        constructor = MarkerConstructor()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_benjamin_feir_trajectory_batch",
                new=constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
            ),
        ):
            _, completed = _generate(
                self.root,
                "benjamin_feir",
                numerical,
                parameter_group_ids=parameter_group_ids,
                targets=(1, 1),
            )

        batch = load_completed_batch(completed[0])
        records = [
            json.loads(str(value)) for value in batch.plan["simulation_spec_json"]
        ]
        expected_terminal_times = []
        for record in records:
            carrier_wavenumber = (
                2.0 * math.pi * float(record["carrier_mode"]) / numerical.length
            )
            intended = 2.0 * math.pi / math.sqrt(numerical.gravity * carrier_wavenumber)
            intended *= 100.0
            expected_terminal_times.append(
                math.floor(intended / numerical.saved_dt) * numerical.saved_dt
            )
        np.testing.assert_allclose(
            [
                float(cast(float, simulation.metrics["realized_terminal_time"]))
                for simulation in batch.simulations
            ],
            expected_terminal_times,
            rtol=0.0,
            atol=1.0e-13,
        )
        self.assertNotEqual(*expected_terminal_times)
        self.assertEqual(len(rollouts.calls), 1)
        self.assertEqual(rollouts.calls[0][1], 2)
        self.assertEqual(rollouts.calls[0][2], max(expected_terminal_times))
        assert batch.shard is not None
        self.assertEqual(
            np.bincount(batch.shard["simulation_local_index"]).tolist(),
            [200, 200],
        )

    def test_interrupted_batch_restarts_without_partial_artifacts(self) -> None:
        numerical = _config()
        parameter_group_id = BENJAMIN_FEIR_PARAMETER_GROUP_IDS[1]
        interrupted_specs: list[tuple[Mapping[str, object], ...]] = []

        def interrupt_constructor(
            sampled: SampledSimulations[object],
            *,
            selected_local_indices: tuple[int, ...] | None = None,
        ) -> TrajectoryInitialBatch:
            del selected_local_indices
            interrupted_specs.append(sampled.specification_records)
            raise InjectedInterruption("during batch construction")

        expected_path = batch_path(
            self.root,
            family="benjamin_feir",
            split="test",
            batch_id=0,
        )
        with mock.patch(
            "solver.gen_data.trajectory_batch_generator."
            "construct_benjamin_feir_trajectory_batch",
            new=interrupt_constructor,
        ):
            with self.assertRaisesRegex(
                InjectedInterruption, "during batch construction"
            ):
                _generate(
                    self.root,
                    "benjamin_feir",
                    numerical,
                    parameter_group_ids=(parameter_group_id,),
                    targets=(1,),
                    batch_size=1,
                )
        self.assertFalse(expected_path.exists())

        constructor = MarkerConstructor()
        rollouts = FastRolloutIntegrator()
        with (
            mock.patch(
                "solver.gen_data.trajectory_batch_generator."
                "construct_benjamin_feir_trajectory_batch",
                new=constructor,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=rollouts,
            ),
        ):
            _generate(
                self.root,
                "benjamin_feir",
                numerical,
                parameter_group_ids=(parameter_group_id,),
                targets=(1,),
                batch_size=1,
            )
        completed = load_completed_batch(expected_path)
        self.assertEqual(
            tuple(
                json.loads(str(value))
                for value in completed.plan["simulation_spec_json"]
            ),
            interrupted_specs[0],
        )

    def test_paper_jonswap_requires_solver_batch_size(self) -> None:
        numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[0]

        with self.assertRaisesRegex(ValueError, "solver_batch_size"):
            generate_trajectory_batch(
                (parameter_group_id,),
                0,
                self.root / "batch.npz",
                dataset_split=DatasetSplit.TEST,
                family="jonswap_tma",
                numerical=numerical,
            )


if __name__ == "__main__":
    unittest.main()
