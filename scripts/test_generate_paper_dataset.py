"""No-rollout tests for the paper-dataset generation launcher."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.generate_paper_dataset import (
    BOOTSTRAP_PLATFORM,
    FAMILY_CELL_IDS,
    GenerationRequest,
    build_chunk_config,
    incremental_simulation_targets,
    parse_args,
    preflight,
    request_from_args,
)
from solver.gen_data.pipeline.batch_storage import BatchPaths, save_batch_plan
from solver.gen_data.pipeline.simulation_allocation import (
    SplitId,
    balanced_simulation_targets,
    build_next_attempt_batch,
)
from solver.gen_data.pipeline.dataset_generation import (
    MAX_RETRIES_PER_PARAMETER_GROUP,
)
from solver.gen_data.pipeline.writer import build_batch_plan
from solver.gen_data.stokes_sampling import DEFAULT_MAXIMUM_URSELL_REDRAWS
from solver.gen_data.stokes_static_pipeline import (
    PAPER_STATIC_STOKES_CONTRACT,
)
from solver.gen_data.trajectory_batch_executor import (
    paper_trajectory_execution,
)


class PaperDatasetGenerationTests(unittest.TestCase):
    def test_learning_curve_chunks_are_nested_and_cumulatively_balanced(
        self,
    ) -> None:
        endpoints = (2_048, 4_096, 8_192, 16_384, 32_768)
        for family in FAMILY_CELL_IDS:
            with self.subTest(family=family):
                cells = FAMILY_CELL_IDS[family]
                cumulative = {cell_id: 0 for cell_id in cells}
                previous = 0
                for endpoint in endpoints:
                    chunk = incremental_simulation_targets(
                        cells,
                        accepted_simulations_before=previous,
                        simulation_count=endpoint - previous,
                    )
                    for target in chunk:
                        cumulative[target.cell_id] += target.simulation_count
                    expected = balanced_simulation_targets(
                        cells,
                        simulation_count=endpoint,
                    )
                    self.assertEqual(
                        cumulative,
                        {
                            target.cell_id: target.simulation_count
                            for target in expected
                        },
                    )
                    values = tuple(cumulative.values())
                    self.assertLessEqual(max(values) - min(values), 1)
                    previous = endpoint

    def test_exact_contract_and_balanced_quotas_for_every_family(self) -> None:
        accepted_by_family = {
            "stokes": 9,
            "tanaka": 11,
            "benjamin_feir": 68,
            "jonswap_tma": 29,
        }
        expected_revision_by_family = {
            "stokes": 2,
            "tanaka": 3,
            "benjamin_feir": 4,
            "jonswap_tma": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, accepted_simulations in accepted_by_family.items():
                with self.subTest(family=family):
                    request = GenerationRequest(
                        output_root=Path(directory) / family,
                        family=family,  # type: ignore[arg-type]
                        split=SplitId.VALIDATION,
                        accepted_simulations=accepted_simulations,
                        batch_size=3,
                        platform=BOOTSTRAP_PLATFORM,
                    )
                    chunk_config = build_chunk_config(request)
                    self.assertEqual(
                        chunk_config.revision_id,
                        expected_revision_by_family[family],
                    )
                    targets = tuple(
                        target.simulation_count
                        for target in chunk_config.simulation_targets
                    )
                    self.assertEqual(sum(targets), accepted_simulations)
                    self.assertLessEqual(max(targets) - min(targets), 1)
                    if family == "tanaka":
                        self.assertEqual(targets, (1,) * 11)
                    self.assertEqual(
                        tuple(
                            target.cell_id for target in chunk_config.simulation_targets
                        ),
                        FAMILY_CELL_IDS[family],  # type: ignore[index]
                    )
                    self.assertEqual(
                        chunk_config.configuration["execution_platform"],
                        BOOTSTRAP_PLATFORM,
                    )
                    if family == "benjamin_feir":
                        support = chunk_config.configuration["sampling_support"]
                        self.assertEqual(
                            support["schema"],
                            "paper_benjamin_feir_sampling_support_v1",
                        )
                        self.assertEqual(len(support["mode_pair_cells"]), 66)
                        self.assertEqual(
                            support["focused_steepness"]["upper_inclusive"],
                            (1.0 + np.sqrt(2.0)) / 10.0,
                        )
                        self.assertEqual(
                            support["sideband_to_carrier_ratio"]["upper_inclusive"],
                            0.10,
                        )
                    if family == "stokes":
                        self.assertEqual(
                            chunk_config.configuration["contract"],
                            PAPER_STATIC_STOKES_CONTRACT.to_json_record(),
                        )
                        self.assertEqual(
                            chunk_config.configuration["sampler"],
                            {
                                "maximum_ursell_redraws": (
                                    DEFAULT_MAXIMUM_URSELL_REDRAWS
                                )
                            },
                        )
                        continue

                    execution = chunk_config.configuration["trajectory_execution"]
                    self.assertEqual(
                        execution,
                        paper_trajectory_execution(
                            family  # type: ignore[arg-type]
                        ).to_json_record(),
                    )
                    numerical = execution["numerical"]
                    expected_evolution_order = 6 if family == "tanaka" else 4
                    expected_internal_nx = 2048 if family == "jonswap_tma" else 1024
                    expected_internal_cutoff = (
                        704.0 if family == "jonswap_tma" else 256.0
                    )
                    self.assertEqual(
                        (
                            numerical["nx"],
                            numerical["dno_order"],
                            numerical["pad_factor"],
                            numerical["maximum_wavenumber"],
                        ),
                        (
                            expected_internal_nx,
                            expected_evolution_order,
                            8,
                            expected_internal_cutoff,
                        ),
                    )
                    self.assertEqual(
                        (
                            numerical["target_nx"],
                            numerical["target_dno_order"],
                            numerical["target_maximum_wavenumber"],
                        ),
                        (1024, 6, 128.0),
                    )
                    self.assertEqual(
                        numerical["gl2_iteration_cap"],
                        5 if family == "jonswap_tma" else 4,
                    )
                    if family == "tanaka":
                        self.assertNotIn(
                            "internal_hamiltonian_drift_threshold",
                            numerical,
                        )
                    else:
                        self.assertEqual(
                            numerical["internal_hamiltonian_drift_threshold"],
                            1.0e-3,
                        )
                    self.assertEqual(numerical["dt"], 0.01)
                    self.assertNotIn("fine_dt", numerical)
                    self.assertNotIn("retry_dt", numerical)
                    stored = execution["frame_selection"]
                    self.assertEqual(
                        (
                            stored["tanaka_count"],
                            stored["benjamin_feir_count"],
                            stored["jonswap_tma_count"],
                        ),
                        (200, 200, 16),
                    )
                    horizon = execution["horizon"]
                    if family == "jonswap_tma":
                        self.assertEqual(
                            horizon["kind"],
                            "jonswap_peak_periods_floor_saved_grid",
                        )
                        self.assertEqual(horizon["period_count"], 16)
                        self.assertEqual(
                            execution["jonswap_quadrature_order"],
                            16,
                        )
                    elif family == "benjamin_feir":
                        self.assertEqual(
                            horizon["kind"],
                            "benjamin_feir_carrier_periods_floor_saved_grid",
                        )
                        self.assertEqual(horizon["period_count"], 100)
                        self.assertIsNone(execution["jonswap_quadrature_order"])
                    else:
                        self.assertEqual(
                            horizon["kind"],
                            "fixed_terminal_time",
                        )
                        self.assertEqual(
                            horizon["fixed_terminal_time"],
                            200.0,
                        )
                        self.assertIsNone(execution["jonswap_quadrature_order"])

    def test_default_preflight_is_read_only_and_reports_exact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            args = parse_args(
                (
                    "--family",
                    "tanaka",
                    "--split",
                    "validation",
                    "--accepted-simulations",
                    "5",
                    "--output-root",
                    str(output),
                    "--batch-size",
                    "2",
                )
            )
            self.assertFalse(args.execute)
            self.assertEqual(args.platform, "cpu")
            request = request_from_args(args)
            _, execution, state, plan = preflight(request)

            self.assertFalse(output.exists())
            self.assertFalse(state.complete)
            self.assertTrue(plan["no_numerical_generation_performed"])
            self.assertEqual(plan["revision_id"], 3)
            self.assertEqual(plan["run_spec"]["revision_id"], 3)
            allocation = plan["allocation"]
            self.assertEqual(allocation["cell_count"], 11)
            self.assertEqual(allocation["nonzero_quota_cell_count"], 5)
            self.assertEqual(allocation["quota_minimum"], 0)
            self.assertEqual(allocation["quota_maximum"], 1)
            self.assertEqual(allocation["accepted_simulations_before"], 0)
            self.assertEqual(allocation["accepted_simulations_after"], 5)
            self.assertEqual(
                allocation["maximum_retries_per_parameter_group"],
                MAX_RETRIES_PER_PARAMETER_GROUP,
            )
            self.assertEqual(
                [quota["maximum_attempts"] for quota in allocation["quotas"]],
                [33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0],
            )
            self.assertEqual(
                [quota["durable_attempted"] for quota in allocation["quotas"]],
                [0] * 11,
            )
            expected = plan["expected_output"]
            self.assertEqual(expected["stored_rows_per_accepted_simulation"], 200)
            self.assertEqual(expected["retained_rows"], 1_000)
            self.assertEqual(
                execution,
                paper_trajectory_execution("tanaka"),
            )
            json.dumps(plan, sort_keys=True, allow_nan=False)

    def test_base_preflight_rejects_unadjusted_jonswap_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = GenerationRequest(
                output_root=Path(directory) / "jonswap",
                family="jonswap_tma",
                split=SplitId.TEST,
                accepted_simulations=27,
                batch_size=27,
                platform=BOOTSTRAP_PLATFORM,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "generate_paper_dataset_jonswap.py",
            ):
                preflight(request)

    def test_stokes_preflight_reports_static_rows_and_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = GenerationRequest(
                output_root=Path(directory) / "stokes",
                family="stokes",
                split=SplitId.TRAIN,
                accepted_simulations=8,
                batch_size=4,
                platform=BOOTSTRAP_PLATFORM,
            )
            chunk_config, execution, state, plan = preflight(request)

            self.assertFalse(state.complete)
            self.assertEqual(execution, PAPER_STATIC_STOKES_CONTRACT)
            self.assertEqual(
                [target.simulation_count for target in chunk_config.simulation_targets],
                [2, 2, 2, 2],
            )
            expected = plan["expected_output"]
            self.assertEqual(expected["stored_rows_per_accepted_simulation"], 1)
            self.assertEqual(expected["retained_rows"], 8)
            self.assertEqual(expected["spatial_points_per_row"], 1024)
            self.assertEqual(plan["execution"]["dno_order"], 6)

    def test_pending_proposal_is_detected_without_rewriting_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resume"
            request = GenerationRequest(
                output_root=output,
                family="tanaka",
                split=SplitId.TEST,
                accepted_simulations=4,
                batch_size=1,
                platform=BOOTSTRAP_PLATFORM,
            )
            chunk_config = build_chunk_config(request)
            assignments = build_next_attempt_batch(
                chunk_config.simulation_targets,
                {},
                {},
                chunk_config.maximum_attempts_by_parameter_group,
                family_id=int(chunk_config.family_id),
                revision_id=chunk_config.revision_id,
                split_id=chunk_config.split_id,
                stream_id=chunk_config.stream_id,
                first_attempt_index=chunk_config.first_attempt_index,
                batch_size=chunk_config.batch_size,
            )
            records = tuple(
                {
                    "schema": "launcher_resume_test_v1",
                    "cell_id": assignment.cell_id,
                    "attempt_index": assignment.simulation_key.attempt_index,
                }
                for assignment in assignments
            )
            paths = BatchPaths.for_batch(
                chunk_config.root,
                family=chunk_config.family_name,
                split=chunk_config.split_id.value,
                batch_id=0,
            )
            save_batch_plan(
                paths,
                build_batch_plan(
                    assignments,
                    records,
                    cell_codes=chunk_config.cell_codes,
                    batch_id=0,
                    metadata={"test": "no_compute_resume_scan"},
                ),
            )

            _, _, state, plan = preflight(request)
            self.assertIsNotNone(state.pending_batch)
            self.assertEqual(plan["resume_state"]["pending_batch_id"], 0)


if __name__ == "__main__":
    unittest.main()
