"""CPU tests for rollout-family batch execution."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    sample_benjamin_feir_simulation,
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
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
    DatasetChunkState,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG,
    PAPER_JONSWAP_ROLLOUT_CONFIG,
    PAPER_TANAKA_ROLLOUT_CONFIG,
    RolloutConfig,
    TrajectoryExecutionConfig,
    TrajectoryFrameSelectionConfig,
    paper_trajectory_execution,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedTrajectoryBatch,
    GL2BatchTelemetry,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
    TanakaCrest,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    JonswapInitialStateDomainError,
    JonswapInitialStateFailure,
    PreparedTrajectoryBatch,
    TrajectoryInitialBatch,
    construct_tanaka_trajectory_batch,
    prepare_trajectory_batch,
    sample_tanaka_simulations,
)
from solver.gen_data.trajectory_batch_generator import (  # noqa: E402
    TrajectoryBatchGenerator,
    _benjamin_feir_time_grid,
    _jonswap_time_grid,
)

jax.config.update("jax_enable_x64", True)


class InjectedInterruption(OSError):
    """Controlled nonterminal interruption."""


def _contract() -> RolloutConfig:
    return RolloutConfig(
        nx=64,
        target_nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        target_dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        target_maximum_wavenumber=16.0,
        dt=0.04,
        saved_dt=0.08,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        target_time_chunk_size=2,
    )


def _execution(family: str) -> TrajectoryExecutionConfig:
    return TrajectoryExecutionConfig(
        family=family,  # type: ignore[arg-type]
        numerical=_contract(),
        frame_selection=TrajectoryFrameSelectionConfig(
            tanaka_count=3,
            tanaka_alpha=0.5,
            tanaka_sigma_steps=1.0,
            benjamin_feir_count=3,
            jonswap_tma_count=3,
        ),
        fixed_terminal_time=0.16 if family == "tanaka" else None,
        period_count=(
            16 if family == "jonswap_tma" else 1 if family == "benjamin_feir" else None
        ),
        jonswap_quadrature_order=4 if family == "jonswap_tma" else None,
    )


def _run_spec(
    root: Path,
    execution: TrajectoryExecutionConfig,
    *,
    parameter_group_ids: tuple[str, ...],
    targets: tuple[int, ...],
    batch_size: int = 2,
) -> DatasetChunkConfig:
    family_id = {
        "tanaka": PhysicalFamilyId.TANAKA,
        "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
        "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
    }[execution.family]
    return DatasetChunkConfig(
        root=root,
        family_name=execution.family,
        family_id=family_id,
        dataset_split=DatasetSplit.TEST,
        simulation_targets=dict(zip(parameter_group_ids, targets, strict=True)),
        batch_size=batch_size,
    )


def _sample_depth(sample: object) -> float:
    depth = getattr(sample, "depth", None)
    if depth is not None:
        return float(depth)
    parameters = getattr(sample, "parameters")
    return float(parameters.depth)


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
        proposed: PreparedTrajectoryBatch[object],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch:
        all_indices = tuple(range(len(proposed.sampled.parameter_group_ids)))
        selected = selected_local_indices or all_indices
        self.calls.append(selected)
        samples = tuple(proposed.sampled.samples[index] for index in selected)
        records = tuple(
            proposed.sampled.specification_records[index] for index in selected
        )
        marker_values = []
        for record in records:
            key = json.dumps(record, sort_keys=True)
            if key not in self.markers_by_record:
                self.markers_by_record[key] = len(self.markers_by_record) + 1
            marker_values.append(self.markers_by_record[key])
        markers = np.asarray(marker_values, dtype=np.float64)
        eta0 = np.repeat(
            markers[:, None],
            proposed.sampled.contract.nx,
            axis=1,
        )
        return TrajectoryInitialBatch(
            eta0=eta0,
            xi0=np.zeros_like(eta0),
            depths=np.asarray(
                [_sample_depth(sample) for sample in samples],
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
        proposed: PreparedTrajectoryBatch[object],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch:
        selected = selected_local_indices or tuple(
            range(len(proposed.sampled.parameter_group_ids))
        )
        self.calls.append(selected)
        if self.rejected_record is None:
            self.rejected_record = json.dumps(
                proposed.sampled.specification_records[0], sort_keys=True
            )
        invalid_positions = tuple(
            position
            for position, original_index in enumerate(selected)
            if json.dumps(
                proposed.sampled.specification_records[original_index], sort_keys=True
            )
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
            proposed,
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
        self.calls: list[tuple[float, int, float]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutConfig,
    ) -> IntegratedTrajectoryBatch:
        del depths
        batch_size, nx = eta0.shape
        self.calls.append((config.dt, batch_size, float(saved_times[-1])))
        eta = np.repeat(eta0[None, :, :], saved_times.size, axis=0)
        xi = np.repeat(xi0[None, :, :], saved_times.size, axis=0)
        substeps = int(round(config.saved_dt / config.dt))
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


class TrajectoryBatchGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_failed_trajectory_retains_sibling_and_replaces_same_parameter_group(
        self,
    ) -> None:
        execution = _execution("benjamin_feir")
        parameter_group_id = BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0]
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(2,),
        )
        constructor = MarkerConstructor()
        rollouts = FastRolloutIntegrator(failed_production_markers=frozenset({1}))
        generator = TrajectoryBatchGenerator(
            chunk_config=spec,
            execution=execution,
            constructor=constructor,
            rollout_integrator=rollouts,
        )

        state = generate_simulations(spec, generator)

        self.assertTrue(state.complete)
        self.assertEqual(
            dict(state.accepted_simulation_counts), {parameter_group_id: 2}
        )
        self.assertEqual(constructor.calls, [(0, 1), (0,)])
        first = load_completed_batch(state.completed_batches[0])
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True],
        )
        assert first.shard is not None
        self.assertTrue(np.all(first.shard["simulation_local_index"] == 1))
        self.assertEqual(first.shard["eta"].shape[0], 3)
        second = load_completed_batch(state.completed_batches[1])
        self.assertEqual(
            str(second.plan["parameter_group_id"][0]),
            parameter_group_id,
        )

    def test_declared_tanaka_failure_rejects_only_named_simulation(self) -> None:
        execution = _execution("tanaka")
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        spec = _run_spec(
            self.root / "declared",
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(2,),
        )
        base_constructor = MarkerConstructor()
        rejected_first_simulation = False

        def declared_constructor(
            proposed: PreparedTrajectoryBatch[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            nonlocal rejected_first_simulation
            selected = selected_local_indices or tuple(
                range(len(proposed.sampled.parameter_group_ids))
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
                proposed,
                selected_local_indices=selected,
            )

        generator = TrajectoryBatchGenerator(
            chunk_config=spec,
            execution=execution,
            constructor=declared_constructor,
            rollout_integrator=FastRolloutIntegrator(),
        )
        state = generate_simulations(spec, generator)

        self.assertTrue(state.complete)
        first = load_completed_batch(state.completed_batches[0])
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

        unexpected_spec = _run_spec(
            self.root / "unexpected",
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(1,),
            batch_size=1,
        )

        def unexpected_constructor(
            proposed: PreparedTrajectoryBatch[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del proposed, selected_local_indices
            raise ValueError("unclassified constructor bug")

        unexpected = TrajectoryBatchGenerator(
            chunk_config=unexpected_spec,
            execution=execution,
            constructor=unexpected_constructor,
            rollout_integrator=FastRolloutIntegrator(),
        )
        with self.assertRaisesRegex(ValueError, "unclassified"):
            generate_simulations(unexpected_spec, unexpected)
        self.assertEqual(
            scan_dataset_generation(unexpected_spec).completed_batches,
            (),
        )

    def test_declared_jonswap_graph_failure_retains_sibling_and_replays(
        self,
    ) -> None:
        execution = _execution("jonswap_tma")
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[9]

        def run(root: Path) -> tuple[DatasetChunkState, JonswapRejectingConstructor]:
            spec = _run_spec(
                root,
                execution,
                parameter_group_ids=(parameter_group_id,),
                targets=(2,),
            )
            constructor = JonswapRejectingConstructor()
            state = generate_simulations(
                spec,
                TrajectoryBatchGenerator(
                    chunk_config=spec,
                    execution=execution,
                    constructor=constructor,
                    rollout_integrator=FastRolloutIntegrator(),
                ),
            )
            return state, constructor

        state, constructor = run(self.root / "first")
        self.assertTrue(state.complete)
        self.assertEqual(constructor.calls, [(0, 1), (1,), (0,)])
        first = load_completed_batch(state.completed_batches[0])
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
        self.assertTrue(replay.complete)
        self.assertEqual(replay_constructor.calls, constructor.calls)
        self.assertEqual(
            [
                load_completed_batch(path).simulations
                for path in replay.completed_batches
            ],
            [
                load_completed_batch(path).simulations
                for path in state.completed_batches
            ],
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
        execution = _execution("tanaka")
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(3,),
            batch_size=3,
        )
        base_constructor = MarkerConstructor()
        rejection_round = 0

        def rejecting_constructor(
            proposed: PreparedTrajectoryBatch[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            nonlocal rejection_round
            selected = selected_local_indices or tuple(
                range(len(proposed.sampled.parameter_group_ids))
            )
            if rejection_round < 2:
                position = rejection_round
                rejection_round += 1
                raise TanakaPotentialRadicandError(
                    "negative_or_nonfinite_surface_potential_radicand",
                    (_tanaka_failure_component(position, position),),
                )
            return base_constructor(
                proposed,
                selected_local_indices=selected,
            )

        state = generate_simulations(
            spec,
            TrajectoryBatchGenerator(
                chunk_config=spec,
                execution=execution,
                constructor=rejecting_constructor,
                rollout_integrator=FastRolloutIntegrator(),
            ),
        )

        self.assertTrue(state.complete)
        first = load_completed_batch(state.completed_batches[0])
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
        execution = _execution("tanaka")
        parameter_group_id = TANAKA_PARAMETER_GROUP_IDS[0]
        sampled = sample_tanaka_simulations(
            (parameter_group_id, parameter_group_id),
            dataset_split=DatasetSplit.TEST,
            first_attempt_number=19,
            contract=execution.numerical,
        )
        proposed = prepare_trajectory_batch(
            sampled,
            root=self.root,
            family_name="tanaka",
            family_id=PhysicalFamilyId.TANAKA,
            dataset_split=DatasetSplit.TEST,
            batch_id=0,
        )
        observed_depths: list[np.ndarray] = []

        def fake_builder(**arguments):
            depths = np.asarray(arguments["simulation_h_ref"], dtype=np.float64)
            simulation_specs = arguments["simulation_specs"]
            self.assertEqual(len(simulation_specs), 1)
            observed_depths.append(depths.copy())
            zeros = np.zeros(
                (depths.size, execution.numerical.nx),
                dtype=np.float64,
            )
            return zeros, zeros

        with mock.patch(
            "solver.gen_data.trajectory_family_adapters."
            "build_per_simulation_initial_conditions",
            side_effect=fake_builder,
        ):
            selected = construct_tanaka_trajectory_batch(
                proposed,
                selected_local_indices=(1,),
            )
        self.assertEqual(selected.eta0.shape, (1, execution.numerical.nx))
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
                proposed,
                selected_local_indices=(1, 0),
            )

    def test_jonswap_simulations_use_distinct_per_simulation_peak_period_grids(
        self,
    ) -> None:
        execution = _execution("jonswap_tma")
        parameter_group_ids = (
            JONSWAP_TMA_PARAMETER_GROUP_IDS[9],
            JONSWAP_TMA_PARAMETER_GROUP_IDS[18],
        )
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=parameter_group_ids,
            targets=(1, 1),
        )
        rollouts = FastRolloutIntegrator()
        generator = TrajectoryBatchGenerator(
            chunk_config=spec,
            execution=execution,
            rollout_integrator=rollouts,
        )
        state = generate_simulations(spec, generator)
        self.assertTrue(state.complete)

        batch = load_completed_batch(state.completed_batches[0])
        records = [
            json.loads(str(value)) for value in batch.plan["simulation_spec_json"]
        ]
        expected_terminal_times = []
        for record in records:
            frequency = float(
                finite_depth_angular_frequency(
                    np.asarray([record["peak_wavenumber"]], dtype=np.float64),
                    depth=float(record["depth"]),
                    gravity=execution.numerical.gravity,
                )[0]
            )
            intended = 16.0 * 2.0 * math.pi / frequency
            expected_terminal_times.append(
                math.floor(
                    (intended + 1.0e-12 * execution.numerical.saved_dt)
                    / execution.numerical.saved_dt
                )
                * execution.numerical.saved_dt
            )
        np.testing.assert_allclose(
            [
                float(simulation.metrics["realized_terminal_time"])
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
            [3, 3],
        )
        for simulation in batch.simulations:
            self.assertIn("initial_discrete_peak_wavenumber", simulation.metrics)
            self.assertIn(
                "initial_linear_hamiltonian_relative_error",
                simulation.metrics,
            )

    def test_benjamin_feir_simulations_use_distinct_carrier_period_grids(self) -> None:
        execution = _execution("benjamin_feir")
        parameter_group_ids = (
            BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0],
            BENJAMIN_FEIR_PARAMETER_GROUP_IDS[-1],
        )
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=parameter_group_ids,
            targets=(1, 1),
        )
        rollouts = FastRolloutIntegrator()
        generator = TrajectoryBatchGenerator(
            chunk_config=spec,
            execution=execution,
            constructor=MarkerConstructor(),
            rollout_integrator=rollouts,
        )
        state = generate_simulations(spec, generator)
        self.assertTrue(state.complete)

        batch = load_completed_batch(state.completed_batches[0])
        records = [
            json.loads(str(value)) for value in batch.plan["simulation_spec_json"]
        ]
        expected_terminal_times = []
        for record in records:
            carrier_wavenumber = float(record["carrier_wavenumber"])
            intended = (
                2.0
                * math.pi
                / math.sqrt(execution.numerical.gravity * carrier_wavenumber)
            )
            expected_terminal_times.append(
                math.floor(intended / execution.numerical.saved_dt)
                * execution.numerical.saved_dt
            )
        np.testing.assert_allclose(
            [
                float(simulation.metrics["realized_terminal_time"])
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
            [3, 3],
        )

    def test_jonswap_horizon_is_strictly_floored_at_rounding_boundary(
        self,
    ) -> None:
        execution = _execution("jonswap_tma")
        intended = 7.99999999999996
        angular_frequency = 16.0 * 2.0 * math.pi / intended
        sample = SimpleNamespace(
            parameters=SimpleNamespace(
                peak_wavenumber=1.0,
                depth=1.0,
            )
        )
        with mock.patch(
            "solver.gen_data.trajectory_batch_generator.finite_depth_angular_frequency",
            return_value=np.asarray([angular_frequency], dtype=np.float64),
        ):
            grid = _jonswap_time_grid(sample, execution)  # type: ignore[arg-type]
        self.assertLessEqual(grid.realized_terminal_time, intended)
        self.assertLess(
            intended - grid.realized_terminal_time,
            execution.numerical.saved_dt,
        )
        self.assertEqual(grid.realized_terminal_time, 7.92)

    def test_benjamin_feir_horizon_is_one_hundred_carrier_periods(
        self,
    ) -> None:
        execution = paper_trajectory_execution("benjamin_feir")
        samples = tuple(
            sample_benjamin_feir_simulation(
                parameter_group_id,
                dataset_split=DatasetSplit.TEST,
                attempt_number=index,
            )
            for index, parameter_group_id in enumerate(
                (
                    BENJAMIN_FEIR_PARAMETER_GROUP_IDS[0],
                    BENJAMIN_FEIR_PARAMETER_GROUP_IDS[-1],
                )
            )
        )
        for sample in samples:
            grid = _benjamin_feir_time_grid(sample, execution)
            carrier_period = (
                2.0
                * math.pi
                / math.sqrt(execution.numerical.gravity * sample.carrier_wavenumber)
            )
            intended = 100.0 * carrier_period
            self.assertEqual(grid.intended_terminal_time, intended)
            self.assertLessEqual(grid.realized_terminal_time, intended)
            self.assertLess(
                intended - grid.realized_terminal_time,
                execution.numerical.saved_dt,
            )

        self.assertEqual(execution.period_count, 100)
        self.assertNotEqual(
            samples[0].carrier_wavenumber,
            samples[1].carrier_wavenumber,
        )
        self.assertNotEqual(
            _benjamin_feir_time_grid(samples[0], execution).realized_terminal_time,
            _benjamin_feir_time_grid(samples[1], execution).realized_terminal_time,
        )

    def test_interrupted_batch_restarts_without_partial_artifacts(self) -> None:
        execution = _execution("benjamin_feir")
        parameter_group_id = BENJAMIN_FEIR_PARAMETER_GROUP_IDS[1]
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(1,),
            batch_size=1,
        )
        interrupted_specs: list[tuple[Mapping[str, object], ...]] = []

        def interrupt_constructor(
            proposed: PreparedTrajectoryBatch[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del selected_local_indices
            interrupted_specs.append(proposed.sampled.specification_records)
            self.assertFalse(proposed.path.exists())
            raise InjectedInterruption("during batch construction")

        interrupted = TrajectoryBatchGenerator(
            chunk_config=spec,
            execution=execution,
            constructor=interrupt_constructor,
            rollout_integrator=FastRolloutIntegrator(),
        )
        with self.assertRaisesRegex(InjectedInterruption, "during batch construction"):
            generate_simulations(spec, interrupted)

        expected_path = batch_path(
            self.root,
            family="benjamin_feir",
            split="test",
            batch_id=0,
        )
        self.assertFalse(expected_path.exists())
        self.assertEqual(scan_dataset_generation(spec).completed_batches, ())

        state = generate_simulations(
            spec,
            TrajectoryBatchGenerator(
                chunk_config=spec,
                execution=execution,
                constructor=MarkerConstructor(),
                rollout_integrator=FastRolloutIntegrator(),
            ),
        )
        self.assertTrue(state.complete)
        completed = load_completed_batch(expected_path)
        self.assertEqual(
            tuple(
                json.loads(str(value))
                for value in completed.plan["simulation_spec_json"]
            ),
            interrupted_specs[0],
        )

    def test_paper_numerical_contract_is_family_specific(self) -> None:
        tanaka_execution = paper_trajectory_execution("tanaka")
        tanaka = tanaka_execution.numerical
        self.assertEqual(tanaka, PAPER_TANAKA_ROLLOUT_CONFIG)
        self.assertEqual(tanaka.maximum_wavenumber, 256.0)
        self.assertEqual(tanaka.target_maximum_wavenumber, 128.0)
        self.assertIsNone(tanaka.internal_hamiltonian_drift_threshold)
        self.assertEqual(tanaka_execution.fixed_terminal_time, 200.0)

        benjamin_feir_execution = paper_trajectory_execution("benjamin_feir")
        benjamin_feir = benjamin_feir_execution.numerical
        self.assertEqual(benjamin_feir, PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG)
        self.assertEqual(benjamin_feir.nx, 1024)
        self.assertEqual(benjamin_feir.maximum_wavenumber, 256.0)
        self.assertEqual(benjamin_feir.target_nx, 1024)
        self.assertEqual(benjamin_feir_execution.period_count, 100)

        jonswap = paper_trajectory_execution("jonswap_tma")
        jonswap_numerical = jonswap.numerical
        self.assertEqual(jonswap_numerical, PAPER_JONSWAP_ROLLOUT_CONFIG)
        self.assertEqual(jonswap_numerical.nx, 2048)
        self.assertEqual(jonswap_numerical.target_nx, 1024)
        self.assertEqual(jonswap.period_count, 16)
        self.assertEqual(jonswap_numerical.maximum_wavenumber, 704.0)
        self.assertEqual(jonswap_numerical.gl2_iteration_cap, 5)
        self.assertEqual(jonswap_numerical.target_nx, 1024)

        for numerical in (benjamin_feir, jonswap_numerical):
            self.assertEqual(numerical.dno_order, 4)
            self.assertEqual(numerical.target_dno_order, 6)
            self.assertEqual(
                numerical.target_maximum_wavenumber,
                128.0,
            )
            self.assertEqual(
                numerical.internal_hamiltonian_drift_threshold,
                1.0e-3,
            )

        self.assertIsNotNone(jonswap.jonswap_adjustment)
        self.assertIsNone(
            paper_trajectory_execution("benjamin_feir").jonswap_adjustment
        )

    def test_paper_jonswap_requires_adjustment_generator(self) -> None:
        execution = paper_trajectory_execution("jonswap_tma")
        parameter_group_id = JONSWAP_TMA_PARAMETER_GROUP_IDS[0]
        spec = _run_spec(
            self.root,
            execution,
            parameter_group_ids=(parameter_group_id,),
            targets=(1,),
            batch_size=1,
        )

        with self.assertRaisesRegex(ValueError, "dedicated generator"):
            TrajectoryBatchGenerator(
                chunk_config=spec,
                execution=execution,
            )


if __name__ == "__main__":
    unittest.main()
