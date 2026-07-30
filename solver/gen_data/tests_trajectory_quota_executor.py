"""CPU tests for accepted-quota rollout-family execution."""

from __future__ import annotations

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

from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BENJAMIN_FEIR_POPULATION_CELLS,
)
from solver.gen_data.generate_tanaka_dataset_v2 import (  # noqa: E402
    TanakaPotentialRadicandError,
    _validate_tanaka_surface_potential_radicand,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    JONSWAP_TMA_POPULATION_CELLS,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    BatchStatus,
    file_sha256,
    inspect_batch,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    AcceptedQuotaRunSpec,
    run_accepted_quotas,
    scan_quota_run,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    ResidualControlledArm,
    ResidualControlledGL2Contract,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
)
from solver.gen_data.multi_crest import CrestSpec  # noqa: E402
from solver.gen_data.tanaka_population import (  # noqa: E402
    TANAKA_POPULATION_CELLS,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    PersistedTrajectoryProposal,
    TrajectoryInitialBatch,
    construct_tanaka_trajectory_batch,
    persist_sampled_trajectory_proposal,
    sample_tanaka_trajectory_cases,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TRAJECTORY_REQUIRED_CHECKS,
    DeclaredConstructionFailure,
    DeclaredTrajectoryFatalError,
    TrajectoryExecutionConfig,
    TrajectoryHorizonPolicy,
    TrajectoryQuotaExecutor,
    _jonswap_time_grid,
    classify_tanaka_construction_failure,
)

jax.config.update("jax_enable_x64", True)


class InjectedInterruption(OSError):
    """Controlled nonterminal interruption."""


class FakeDeclaredTanakaFailure(ValueError):
    def __init__(self, record: dict[str, object]) -> None:
        super().__init__("declared Tanaka construction failure")
        self.failure_record = record


def _contract() -> ResidualControlledGL2Contract:
    return ResidualControlledGL2Contract(
        nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        production_dt=0.04,
        saved_dt=0.08,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        refinement_tolerance=1.0e-3,
        relative_floor=1.0e-12,
        target_time_chunk_size=2,
    )


def _execution(family: str) -> TrajectoryExecutionConfig:
    return TrajectoryExecutionConfig(
        family=family,  # type: ignore[arg-type]
        role="reduced_wiring_evidence_only",
        numerical=_contract(),
        horizon=(
            TrajectoryHorizonPolicy.jonswap_peak_periods(16)
            if family == "jonswap_tma"
            else TrajectoryHorizonPolicy.fixed(0.16)
        ),
        stored_time_policy=StoredTimePolicy(
            tanaka_count=3,
            tanaka_alpha=0.5,
            tanaka_sigma_steps=1.0,
            benjamin_feir_count=3,
            benjamin_feir_alpha=0.5,
            benjamin_feir_sigma_steps=1.0,
            random_sea_count=3,
        ),
        jonswap_quadrature_order=4 if family == "jonswap_tma" else None,
    )


def _run_spec(
    root: Path,
    execution: TrajectoryExecutionConfig,
    *,
    cell_ids: tuple[str, ...],
    targets: tuple[int, ...],
    batch_size: int = 2,
    execution_record: dict[str, object] | None = None,
) -> AcceptedQuotaRunSpec:
    return AcceptedQuotaRunSpec(
        root=root,
        family_name=execution.family,
        family_id={
            "tanaka": PhysicalFamilyId.TANAKA,
            "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
            "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
        }[execution.family],
        revision_id=1,
        split_id=SplitId.TEST,
        stream_id=13,
        quotas=tuple(
            CellQuota(cell_id, target) for cell_id, target in zip(cell_ids, targets)
        ),
        cell_codes={cell_id: index for index, cell_id in enumerate(cell_ids)},
        batch_size=batch_size,
        first_attempt_index=0,
        configuration={
            "trajectory_execution": (execution_record or execution.to_json_record())
        },
    )


def _sample_depth(sample: object) -> float:
    depth = getattr(sample, "depth", None)
    if depth is not None:
        return float(depth)
    parameters = getattr(sample, "parameters")
    return float(parameters.depth)


def _tanaka_failure_component(
    local_case_index: int,
    global_component_index: int,
    *,
    component_within_case: int = 0,
) -> dict[str, object]:
    return {
        "local_case_index": local_case_index,
        "component_within_case": component_within_case,
        "global_component_index": global_component_index,
        "alpha": 0.2,
        "center": 0.3,
        "direction": 1,
        "depth": 0.25,
        "unsigned_speed": 1.0,
        "speed_squared": 1.0,
        "minimum_radicand": -1.0e-6,
        "minimum_radicand_over_speed_squared": -1.0e-6,
        "minimum_radicand_grid_index": 3,
        "minimum_radicand_x": 0.4,
        "negative_count": 1,
        "nonfinite_count": 0,
        "first_nonfinite_grid_index": None,
        "first_nonfinite_x": None,
    }


class MarkerConstructor:
    """Construct constant markers after checking the durable proposal."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(
        self,
        proposed: PersistedTrajectoryProposal[object],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch:
        self.assert_proposed(proposed)
        all_indices = tuple(range(len(proposed.sampled.assignments)))
        selected = selected_local_indices or all_indices
        self.calls.append(selected)
        assignments = tuple(proposed.sampled.assignments[index] for index in selected)
        samples = tuple(proposed.sampled.samples[index] for index in selected)
        records = tuple(
            proposed.sampled.specification_records[index] for index in selected
        )
        markers = np.asarray(
            [assignment.case_key.attempt_index + 1 for assignment in assignments],
            dtype=np.float64,
        )
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

    @staticmethod
    def assert_proposed(
        proposed: PersistedTrajectoryProposal[object],
    ) -> None:
        status = inspect_batch(
            proposed.paths,
            expected_fingerprint=proposed.config_fingerprint,
        ).status
        if status not in (BatchStatus.PROPOSED, BatchStatus.SHARD_WRITTEN):
            raise AssertionError(f"constructor observed status {status}")


class FastArmExecutor:
    """Return valid synthetic arms, optionally rejecting selected markers."""

    def __init__(
        self,
        *,
        incomplete_production_markers: frozenset[int] = frozenset(),
    ) -> None:
        self.incomplete_production_markers = incomplete_production_markers
        self.calls: list[tuple[float, int, float]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
    ) -> ResidualControlledArm:
        del depths
        batch_size, nx = eta0.shape
        self.calls.append((dt, batch_size, float(saved_times[-1])))
        eta = np.repeat(eta0[None, :, :], saved_times.size, axis=0)
        xi = np.repeat(xi0[None, :, :], saved_times.size, axis=0)
        complete = np.ones(batch_size, dtype=np.bool_)
        if math.isclose(dt, contract.production_dt, abs_tol=1.0e-15):
            markers = np.rint(eta0[:, 0]).astype(int)
            complete = np.asarray(
                [
                    marker not in self.incomplete_production_markers
                    for marker in markers
                ],
                dtype=np.bool_,
            )
        substeps = int(round(contract.saved_dt / dt))
        telemetry_shape = ((saved_times.size - 1) * substeps, batch_size)
        return ResidualControlledArm(
            dt=dt,
            times=np.asarray(saved_times, dtype=np.float64),
            eta=np.asarray(eta, dtype=np.float64),
            xi=np.asarray(xi, dtype=np.float64),
            q_ref=np.zeros((saved_times.size, batch_size, nx), dtype=np.float64),
            complete=complete,
            gl2_stage_residual=np.zeros(telemetry_shape, dtype=np.float64),
            gl2_iterations=np.ones(telemetry_shape, dtype=np.int32),
            gl2_converged=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_stage_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_state_finite=np.ones(telemetry_shape, dtype=np.bool_),
            gl2_hit_iteration_cap=np.zeros(telemetry_shape, dtype=np.bool_),
        )


def _declared_classifier(
    error: Exception,
) -> DeclaredConstructionFailure | None:
    if not isinstance(error, FakeDeclaredTanakaFailure):
        return None
    indices = error.failure_record["invalid_case_indices"]
    assert isinstance(indices, list)
    return DeclaredConstructionFailure(
        local_indices=tuple(indices),
        record=error.failure_record,
    )


class TrajectoryQuotaExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_incomplete_trajectory_retains_sibling_and_replaces_same_cell(
        self,
    ) -> None:
        execution = _execution("benjamin_feir")
        cell_id = BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(2,),
        )
        constructor = MarkerConstructor()
        arms = FastArmExecutor(
            incomplete_production_markers=frozenset({1})
        )
        executor = TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
            constructor=constructor,
            arm_executor=arms,
        )

        state = run_accepted_quotas(spec, executor)

        self.assertTrue(state.complete)
        self.assertEqual(dict(state.accepted_by_cell), {cell_id: 2})
        self.assertEqual(state.next_attempt_index, 3)
        self.assertEqual(constructor.calls, [(0, 1), (0,)])
        first_result = json.loads(state.committed[0].result.read_text(encoding="utf-8"))
        self.assertEqual(
            [case["accepted"] for case in first_result["cases"]],
            [False, True],
        )
        self.assertTrue(
            all(
                case["required_bits"] == int(TRAJECTORY_REQUIRED_CHECKS)
                for case in first_result["cases"]
            )
        )
        self.assertEqual(
            first_result["cases"][0]["evaluated_bits"]
            & int(TRAJECTORY_REQUIRED_CHECKS),
            int(TRAJECTORY_REQUIRED_CHECKS),
        )
        with np.load(state.committed[0].shard, allow_pickle=False) as shard:
            self.assertTrue(np.all(shard["case_local_index"] == 1))
            self.assertEqual(shard["eta"].shape[0], 3)
        with np.load(
            state.committed[1].proposal,
            allow_pickle=False,
        ) as proposal:
            replacement = json.loads(str(proposal["case_spec_json"][0]))
        self.assertEqual(replacement["attempt_index"], 2)
        self.assertEqual(replacement["cell_id"], cell_id)

    def test_declared_tanaka_failure_rejects_only_named_case(self) -> None:
        execution = _execution("tanaka")
        cell_id = TANAKA_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root / "declared",
            execution,
            cell_ids=(cell_id,),
            targets=(2,),
        )
        base_constructor = MarkerConstructor()

        def declared_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            selected = selected_local_indices or tuple(
                range(len(proposed.sampled.assignments))
            )
            invalid_positions = [
                position
                for position, original_index in enumerate(selected)
                if (
                    proposed.sampled.assignments[original_index].case_key.attempt_index
                    == 0
                )
            ]
            if invalid_positions:
                raise FakeDeclaredTanakaFailure(
                    {
                        "schema": "tanaka_potential_radicand_failure_v1",
                        "reason": "negative_surface_potential_radicand",
                        "invalid_case_indices": invalid_positions,
                        "components": [
                            _tanaka_failure_component(position, position)
                            for position in invalid_positions
                        ],
                    }
                )
            return base_constructor(
                proposed,
                selected_local_indices=selected,
            )

        executor = TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
            constructor=declared_constructor,
            construction_failure_classifier=_declared_classifier,
            arm_executor=FastArmExecutor(),
        )
        state = run_accepted_quotas(spec, executor)

        self.assertTrue(state.complete)
        self.assertEqual(state.next_attempt_index, 3)
        first_result = json.loads(state.committed[0].result.read_text(encoding="utf-8"))
        self.assertEqual(
            [case["accepted"] for case in first_result["cases"]],
            [False, True],
        )
        self.assertEqual(
            first_result["cases"][0]["metrics"]["construction_status"],
            "declared_outside_support",
        )
        self.assertEqual(
            first_result["cases"][0]["required_bits"],
            int(TRAJECTORY_REQUIRED_CHECKS),
        )
        self.assertEqual(
            first_result["cases"][0]["evaluated_bits"],
            int(QualityReason.OUTSIDE_SUPPORT),
        )
        with np.load(state.committed[0].shard, allow_pickle=False) as shard:
            self.assertTrue(np.all(shard["case_local_index"] == 1))

        unexpected_spec = _run_spec(
            self.root / "unexpected",
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )

        def unexpected_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del proposed, selected_local_indices
            raise ValueError("unclassified constructor bug")

        unexpected = TrajectoryQuotaExecutor(
            run_spec=unexpected_spec,
            execution=execution,
            constructor=unexpected_constructor,
            construction_failure_classifier=_declared_classifier,
            arm_executor=FastArmExecutor(),
        )
        with self.assertRaisesRegex(ValueError, "unclassified"):
            run_accepted_quotas(unexpected_spec, unexpected)
        pending = scan_quota_run(unexpected_spec).pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.status, BatchStatus.PROPOSED)
        self.assertFalse(pending.paths.result.exists())

    def test_default_classifier_recognizes_only_the_declared_tanaka_error(
        self,
    ) -> None:
        record = {
            "schema": "tanaka_potential_radicand_failure_v1",
            "reason": "negative_or_nonfinite_surface_potential_radicand",
            "invalid_case_indices": [0, 2],
            "components": [
                _tanaka_failure_component(0, 0),
                _tanaka_failure_component(2, 2),
            ],
        }
        failure = classify_tanaka_construction_failure(
            TanakaPotentialRadicandError(record)
        )
        self.assertEqual(
            failure,
            DeclaredConstructionFailure(
                local_indices=(0, 2),
                record=record,
            ),
        )
        self.assertIsNone(
            classify_tanaka_construction_failure(
                ValueError("unclassified constructor bug")
            )
        )
        malformed = {
            **record,
            "schema": "not_the_declared_schema",
        }
        with self.assertRaisesRegex(ValueError, "schema"):
            classify_tanaka_construction_failure(
                TanakaPotentialRadicandError(malformed)
            )
        with self.assertRaisesRegex(TypeError, "components"):
            classify_tanaka_construction_failure(
                TanakaPotentialRadicandError(
                    {
                        **record,
                        "components": [],
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            classify_tanaka_construction_failure(
                TanakaPotentialRadicandError(
                    {
                        **record,
                        "invalid_case_indices": [0],
                        "components": [
                            {
                                "local_case_index": 0,
                                "component_within_case": 0,
                                "global_component_index": 0,
                            }
                        ],
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "repeats a component"):
            classify_tanaka_construction_failure(
                TanakaPotentialRadicandError(
                    {
                        **record,
                        "invalid_case_indices": [0],
                        "components": [
                            _tanaka_failure_component(0, 0),
                            _tanaka_failure_component(0, 0),
                        ],
                    }
                )
            )
        inconsistent_components = (
            (
                {"unsigned_speed": None},
                "speed fields",
            ),
            (
                {"speed_squared": 2.0},
                "speed_squared disagrees",
            ),
            (
                {"minimum_radicand_over_speed_squared": -2.0e-6},
                "scaled minimum disagrees",
            ),
            (
                {"negative_count": 0},
                "negative-radicand count disagrees",
            ),
        )
        for changed_fields, message in inconsistent_components:
            with self.subTest(changed_fields=changed_fields):
                component = {
                    **_tanaka_failure_component(0, 0),
                    **changed_fields,
                }
                with self.assertRaisesRegex(ValueError, message):
                    classify_tanaka_construction_failure(
                        TanakaPotentialRadicandError(
                            {
                                **record,
                                "invalid_case_indices": [0],
                                "components": [component],
                            }
                        )
                    )

    def test_malformed_tanaka_diagnostic_remains_pending(self) -> None:
        execution = _execution("tanaka")
        cell_id = TANAKA_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )

        def malformed_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del proposed, selected_local_indices
            component = _tanaka_failure_component(0, 0)
            component["unsigned_speed"] = None
            raise TanakaPotentialRadicandError(
                {
                    "schema": "tanaka_potential_radicand_failure_v1",
                    "reason": "negative_or_nonfinite_surface_potential_radicand",
                    "invalid_case_indices": [0],
                    "components": [component],
                }
            )

        executor = TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
            constructor=malformed_constructor,
            arm_executor=FastArmExecutor(),
        )
        with self.assertRaisesRegex(ValueError, "speed fields"):
            run_accepted_quotas(spec, executor)
        pending = scan_quota_run(spec).pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.status, BatchStatus.PROPOSED)
        self.assertFalse(pending.paths.result.exists())

    def test_default_classifier_accepts_the_constructor_generated_record(
        self,
    ) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            _validate_tanaka_surface_potential_radicand(
                np.asarray(((1.0, -1.0e-8),), dtype=np.float64),
                x_grid=np.asarray((0.0, 0.5), dtype=np.float64),
                speed_per_crest=np.asarray((1.0,), dtype=np.float64),
                case_h_ref=np.asarray((0.2,), dtype=np.float64),
                flat_specs=[CrestSpec(0.2, 0.3, 1)],
                crest_case_ids=np.asarray((0,), dtype=np.int32),
                components_within_case=(0,),
            )
        failure = classify_tanaka_construction_failure(caught.exception)
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.local_indices, (0,))

    def test_default_classifier_does_not_resample_speed_failures(
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
                    case_h_ref=np.asarray((0.2,), dtype=np.float64),
                    flat_specs=[CrestSpec(0.2, 0.3, 1)],
                    crest_case_ids=np.asarray((0,), dtype=np.int32),
                    components_within_case=(0,),
                )
            failure = classify_tanaka_construction_failure(caught.exception)
            self.assertIsNone(failure)

    def test_default_classifier_does_not_resample_nonfinite_radicands(
        self,
    ) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            _validate_tanaka_surface_potential_radicand(
                np.asarray(((1.0, math.nan),), dtype=np.float64),
                x_grid=np.asarray((0.0, 0.5), dtype=np.float64),
                speed_per_crest=np.asarray((1.0,), dtype=np.float64),
                case_h_ref=np.asarray((0.2,), dtype=np.float64),
                flat_specs=[CrestSpec(0.2, 0.3, 1)],
                crest_case_ids=np.asarray((0,), dtype=np.int32),
                components_within_case=(0,),
            )
        self.assertIsNone(
            classify_tanaka_construction_failure(caught.exception)
        )

    def test_two_round_tanaka_rejections_archive_original_indices(self) -> None:
        execution = _execution("tanaka")
        cell_id = TANAKA_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(3,),
            batch_size=3,
        )
        base_constructor = MarkerConstructor()

        def rejecting_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            selected = selected_local_indices or tuple(
                range(len(proposed.sampled.assignments))
            )
            attempts = tuple(
                proposed.sampled.assignments[index].case_key.attempt_index
                for index in selected
            )
            failed_attempt = 0 if 0 in attempts else 2 if 2 in attempts else None
            if failed_attempt is not None:
                position = attempts.index(failed_attempt)
                raise TanakaPotentialRadicandError(
                    {
                        "schema": "tanaka_potential_radicand_failure_v1",
                        "reason": ("negative_or_nonfinite_surface_potential_radicand"),
                        "invalid_case_indices": [position],
                        "components": [_tanaka_failure_component(position, position)],
                    }
                )
            return base_constructor(
                proposed,
                selected_local_indices=selected,
            )

        state = run_accepted_quotas(
            spec,
            TrajectoryQuotaExecutor(
                run_spec=spec,
                execution=execution,
                constructor=rejecting_constructor,
                arm_executor=FastArmExecutor(),
            ),
        )

        self.assertTrue(state.complete)
        self.assertEqual(state.next_attempt_index, 5)
        first_result = json.loads(state.committed[0].result.read_text())
        self.assertEqual(
            [case["accepted"] for case in first_result["cases"]],
            [False, True, False],
        )
        second_failure_json = first_result["cases"][2]["metrics"][
            "construction_failure_json"
        ]
        second_failure = json.loads(second_failure_json)
        self.assertEqual(second_failure["invalid_case_indices"], [2])
        self.assertEqual(
            second_failure["constructor_subbatch_invalid_case_indices"],
            [1],
        )
        component = second_failure["components"][0]
        self.assertEqual(component["local_case_index"], 2)
        self.assertEqual(component["global_component_index"], 2)
        self.assertEqual(component["constructor_subbatch_local_case_index"], 1)
        self.assertEqual(
            component["constructor_subbatch_global_component_index"],
            1,
        )

    def test_explicit_fatal_error_writes_terminal_sidecar(self) -> None:
        execution = _execution("benjamin_feir")
        cell_id = BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )

        def fatal_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del proposed, selected_local_indices
            raise DeclaredTrajectoryFatalError(
                "deterministic test failure",
                phase="construction",
                telemetry={"test_code": 17},
            )

        state = run_accepted_quotas(
            spec,
            TrajectoryQuotaExecutor(
                run_spec=spec,
                execution=execution,
                constructor=fatal_constructor,
                arm_executor=FastArmExecutor(),
            ),
        )
        self.assertFalse(state.complete)
        self.assertIsNotNone(state.terminal_failure)
        assert state.terminal_failure is not None
        failure = json.loads(state.terminal_failure.failure.read_text())
        self.assertEqual(failure["phase"], "construction")
        self.assertEqual(
            failure["exception_type"],
            "DeclaredTrajectoryFatalError",
        )
        self.assertEqual(failure["telemetry"], {"test_code": 17})

    def test_selected_tanaka_adapter_verifies_full_proposal_then_subsets(
        self,
    ) -> None:
        execution = _execution("tanaka")
        cell_id = TANAKA_POPULATION_CELLS[0].cell_id
        assignments = tuple(
            AttemptAssignment(
                case_key=CaseKey(
                    family_id=int(PhysicalFamilyId.TANAKA),
                    revision_id=1,
                    split_id=SplitId.TEST,
                    stream_id=21,
                    attempt_index=index,
                ),
                cell_id=cell_id,
            )
            for index in (19, 20)
        )
        sampled = sample_tanaka_trajectory_cases(
            assignments,
            contract=execution.numerical,
        )
        proposed = persist_sampled_trajectory_proposal(
            sampled,
            root=self.root,
            family_name="tanaka",
            batch_id=0,
            cell_codes={cell_id: 0},
            config_fingerprint="a" * 64,
            metadata={"test_scope": "selected_tanaka_adapter"},
        )
        observed_depths: list[np.ndarray] = []

        def fake_builder(**arguments):
            depths = np.asarray(arguments["case_h_ref"], dtype=np.float64)
            case_specs = arguments["case_specs"]
            self.assertEqual(len(case_specs), 1)
            observed_depths.append(depths.copy())
            zeros = np.zeros(
                (depths.size, execution.numerical.nx),
                dtype=np.float64,
            )
            return zeros, zeros

        with mock.patch(
            "solver.gen_data.trajectory_family_adapters."
            "build_per_case_initial_conditions",
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

    def test_jonswap_cases_use_distinct_per_case_peak_period_grids(self) -> None:
        execution = _execution("jonswap_tma")
        cell_ids = (
            JONSWAP_TMA_POPULATION_CELLS[9].cell_id,
            JONSWAP_TMA_POPULATION_CELLS[18].cell_id,
        )
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=cell_ids,
            targets=(1, 1),
        )
        arms = FastArmExecutor()
        executor = TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
            constructor=MarkerConstructor(),
            arm_executor=arms,
        )
        state = run_accepted_quotas(spec, executor)
        self.assertTrue(state.complete)

        with np.load(state.committed[0].proposal, allow_pickle=False) as proposal:
            records = [json.loads(str(value)) for value in proposal["case_spec_json"]]
            metadata = json.loads(str(proposal["metadata_json"]))
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
        observed_grids = metadata["case_time_grids"]
        np.testing.assert_allclose(
            [grid["realized_terminal_time"] for grid in observed_grids],
            expected_terminal_times,
            rtol=0.0,
            atol=1.0e-13,
        )
        self.assertTrue(all(batch_size == 2 for _, batch_size, _ in arms.calls))
        self.assertEqual(len(arms.calls), 1)
        self.assertEqual(
            [terminal for _, _, terminal in arms.calls],
            [max(expected_terminal_times)],
        )
        with np.load(state.committed[0].shard, allow_pickle=False) as shard:
            self.assertEqual(
                np.bincount(shard["case_local_index"]).tolist(),
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
            "solver.gen_data.trajectory_quota_executor.finite_depth_angular_frequency",
            return_value=np.asarray([angular_frequency], dtype=np.float64),
        ):
            grid = _jonswap_time_grid(sample, execution)  # type: ignore[arg-type]
        self.assertLessEqual(grid.realized_terminal_time, intended)
        self.assertLess(
            intended - grid.realized_terminal_time,
            execution.numerical.saved_dt,
        )
        self.assertEqual(grid.realized_terminal_time, 7.92)

    def test_proposed_and_shard_written_batches_resume_exactly(self) -> None:
        execution = _execution("benjamin_feir")
        cell_id = BENJAMIN_FEIR_POPULATION_CELLS[1].cell_id

        proposal_spec = _run_spec(
            self.root / "proposal",
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )

        def interrupt_constructor(
            proposed: PersistedTrajectoryProposal[object],
            *,
            selected_local_indices: tuple[int, ...] | None,
        ) -> TrajectoryInitialBatch:
            del selected_local_indices
            MarkerConstructor.assert_proposed(proposed)
            raise InjectedInterruption("after proposal")

        interrupted = TrajectoryQuotaExecutor(
            run_spec=proposal_spec,
            execution=execution,
            constructor=interrupt_constructor,
            arm_executor=FastArmExecutor(),
        )
        with self.assertRaisesRegex(InjectedInterruption, "after proposal"):
            run_accepted_quotas(proposal_spec, interrupted)
        proposal_pending = scan_quota_run(proposal_spec).pending
        self.assertIsNotNone(proposal_pending)
        assert proposal_pending is not None
        proposal_hash = file_sha256(proposal_pending.paths.proposal)
        proposal_state = run_accepted_quotas(
            proposal_spec,
            TrajectoryQuotaExecutor(
                run_spec=proposal_spec,
                execution=execution,
                constructor=MarkerConstructor(),
                arm_executor=FastArmExecutor(),
            ),
        )
        self.assertTrue(proposal_state.complete)
        self.assertEqual(
            file_sha256(proposal_pending.paths.proposal),
            proposal_hash,
        )

        shard_spec = _run_spec(
            self.root / "shard",
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )
        shard_executor = TrajectoryQuotaExecutor(
            run_spec=shard_spec,
            execution=execution,
            constructor=MarkerConstructor(),
            arm_executor=FastArmExecutor(),
        )
        with (
            mock.patch(
                "solver.gen_data.pipeline.writer.commit_batch",
                side_effect=InjectedInterruption("after shard"),
            ),
            self.assertRaisesRegex(InjectedInterruption, "after shard"),
        ):
            run_accepted_quotas(shard_spec, shard_executor)
        shard_pending = scan_quota_run(shard_spec).pending
        self.assertIsNotNone(shard_pending)
        assert shard_pending is not None
        self.assertEqual(shard_pending.status, BatchStatus.SHARD_WRITTEN)
        shard_hash = file_sha256(shard_pending.paths.shard)
        shard_state = run_accepted_quotas(shard_spec, shard_executor)
        self.assertTrue(shard_state.complete)
        self.assertEqual(file_sha256(shard_pending.paths.shard), shard_hash)

    def test_execution_configuration_must_match_fingerprint_record(self) -> None:
        execution = _execution("benjamin_feir")
        cell_id = BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id
        changed_record = execution.to_json_record()
        changed_record["horizon"] = {
            **changed_record["horizon"],
            "fixed_terminal_time": 0.24,
        }
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
            execution_record=changed_record,
        )
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            TrajectoryQuotaExecutor(
                run_spec=spec,
                execution=execution,
                constructor=MarkerConstructor(),
                arm_executor=FastArmExecutor(),
            )

        with self.assertRaisesRegex(ValueError, "must be labeled"):
            TrajectoryExecutionConfig(
                family="benjamin_feir",
                role="paper_corpus",
                numerical=_contract(),
                horizon=TrajectoryHorizonPolicy.fixed(0.16),
                stored_time_policy=execution.stored_time_policy,
                jonswap_quadrature_order=None,
            )

    def test_paper_role_rejects_injected_numerical_hooks(self) -> None:
        execution = TrajectoryExecutionConfig.paper("benjamin_feir")
        cell_id = BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id
        spec = _run_spec(
            self.root,
            execution,
            cell_ids=(cell_id,),
            targets=(1,),
            batch_size=1,
        )
        TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
        )
        with self.assertRaisesRegex(ValueError, "production constructor"):
            TrajectoryQuotaExecutor(
                run_spec=spec,
                execution=execution,
                constructor=MarkerConstructor(),
            )
        with self.assertRaisesRegex(ValueError, "production constructor"):
            TrajectoryQuotaExecutor(
                run_spec=spec,
                execution=execution,
                arm_executor=FastArmExecutor(),
            )


if __name__ == "__main__":
    unittest.main()
