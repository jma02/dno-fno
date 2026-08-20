"""No-rollout tests for the exact paper-corpus quota launcher."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_paper_corpus_quota import (
    BOOTSTRAP_PLATFORM,
    FAMILY_CELL_IDS,
    GenerationRequest,
    build_run_spec,
    incremental_cell_quotas,
    parse_args,
    preflight,
    request_from_args,
)
from solver.gen_data.pipeline.archive import ensure_proposal
from solver.gen_data.pipeline.production import (
    SplitId,
    balanced_cell_quotas,
    schedule_attempt_batch,
)
from solver.gen_data.pipeline.quota_driver import (
    DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE,
)
from solver.gen_data.pipeline.writer import (
    batch_paths_for_assignments,
    build_proposal_arrays,
)
from solver.gen_data.stokes_population import DEFAULT_MAXIMUM_URSELL_REDRAWS
from solver.gen_data.stokes_static_pipeline import (
    PAPER_STATIC_STOKES_CONTRACT,
)
from solver.gen_data.trajectory_quota_executor import (
    TrajectoryExecutionConfig,
)


class PaperCorpusQuotaLauncherTests(unittest.TestCase):
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
                    chunk = incremental_cell_quotas(
                        cells,
                        accepted_cases_before=previous,
                        accepted_case_count=endpoint - previous,
                    )
                    for quota in chunk:
                        cumulative[quota.cell_id] += quota.target_accepted
                    expected = balanced_cell_quotas(
                        cells,
                        accepted_case_count=endpoint,
                    )
                    self.assertEqual(
                        cumulative,
                        {
                            quota.cell_id: quota.target_accepted
                            for quota in expected
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
            for family, accepted_cases in accepted_by_family.items():
                with self.subTest(family=family):
                    request = GenerationRequest(
                        output_root=Path(directory) / family,
                        family=family,  # type: ignore[arg-type]
                        split=SplitId.VALIDATION,
                        accepted_cases=accepted_cases,
                        batch_size=3,
                        platform=BOOTSTRAP_PLATFORM,
                    )
                    spec = build_run_spec(request)
                    self.assertEqual(
                        spec.revision_id,
                        expected_revision_by_family[family],
                    )
                    targets = tuple(
                        quota.target_accepted for quota in spec.quotas
                    )
                    self.assertEqual(sum(targets), accepted_cases)
                    self.assertLessEqual(max(targets) - min(targets), 1)
                    if family == "tanaka":
                        self.assertEqual(targets, (1,) * 11)
                    self.assertEqual(
                        tuple(quota.cell_id for quota in spec.quotas),
                        FAMILY_CELL_IDS[family],  # type: ignore[index]
                    )
                    self.assertEqual(
                        spec.configuration["execution_platform"],
                        BOOTSTRAP_PLATFORM,
                    )
                    if family == "benjamin_feir":
                        support = spec.configuration["population_support"]
                        self.assertEqual(
                            support["schema"],
                            "paper_benjamin_feir_population_support_v1",
                        )
                        self.assertEqual(len(support["mode_pair_cells"]), 66)
                        self.assertEqual(
                            support["focused_steepness"]["upper_inclusive"],
                            (1.0 + np.sqrt(2.0)) / 10.0,
                        )
                        self.assertEqual(
                            support["sideband_to_carrier_ratio"][
                                "upper_inclusive"
                            ],
                            0.10,
                        )
                    dependency = spec.configuration["dependency_environment"]
                    self.assertEqual(
                        set(dependency["packages"]),
                        {"jax", "jaxlib", "numpy"},
                    )
                    self.assertEqual(
                        set(dependency["files_sha256"]),
                        {"pyproject.toml", "uv.lock"},
                    )
                    sources = spec.configuration["source_sha256"]
                    self.assertIn(
                        "scripts/run_paper_corpus_quota.py",
                        sources,
                    )
                    if family == "stokes":
                        self.assertEqual(
                            spec.configuration["contract"],
                            PAPER_STATIC_STOKES_CONTRACT.to_json_record(),
                        )
                        self.assertEqual(
                            spec.configuration["sampler"],
                            {
                                "maximum_ursell_redraws": (
                                    DEFAULT_MAXIMUM_URSELL_REDRAWS
                                )
                            },
                        )
                        self.assertIn(
                            "solver/gen_data/stokes_quota_executor.py",
                            sources,
                        )
                        self.assertNotIn(
                            "solver/gen_data/trajectory_quota_executor.py",
                            sources,
                        )
                        continue

                    execution = spec.configuration["trajectory_execution"]
                    self.assertEqual(
                        execution,
                        TrajectoryExecutionConfig.paper(
                            family  # type: ignore[arg-type]
                        ).to_json_record(),
                    )
                    numerical = execution["numerical"]
                    expected_evolution_order = (
                        6 if family == "tanaka" else 4
                    )
                    expected_internal_nx = (
                        2048 if family == "jonswap_tma" else 1024
                    )
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
                    if family == "jonswap_tma":
                        self.assertEqual(numerical["target_nx"], 1024)
                        self.assertEqual(numerical["gl2_iteration_cap"], 5)
                    else:
                        self.assertNotIn("target_nx", numerical)
                    if family == "tanaka":
                        self.assertEqual(
                            numerical["target_maximum_wavenumber"],
                            128.0,
                        )
                        self.assertEqual(
                            numerical["post_step_state_filter"],
                            "hou_li",
                        )
                        self.assertEqual(
                            numerical["post_step_maximum_wavenumber"],
                            256.0,
                        )
                        self.assertEqual(numerical["hou_li_coefficient"], 36.0)
                        self.assertEqual(numerical["hou_li_power"], 36)
                    else:
                        self.assertEqual(numerical["target_dno_order"], 6)
                        self.assertEqual(
                            numerical["target_maximum_wavenumber"],
                            128.0,
                        )
                        self.assertEqual(
                            numerical["internal_hamiltonian_drift_threshold"],
                            1.0e-3,
                        )
                        self.assertEqual(
                            numerical["post_step_state_filter"],
                            "sharp",
                        )
                    self.assertEqual(numerical["dt"], 0.01)
                    self.assertNotIn("fine_dt", numerical)
                    self.assertNotIn("retry_dt", numerical)
                    stored = execution["stored_time_policy"]
                    self.assertEqual(
                        (
                            stored["tanaka_count"],
                            stored["benjamin_feir_count"],
                            stored["random_sea_count"],
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
                    "--accepted-cases",
                    "5",
                    "--output-root",
                    str(output),
                    "--batch-size",
                    "2",
                )
            )
            self.assertFalse(args.execute)
            self.assertEqual(args.platform, "cpu")
            self.assertEqual(
                args.maximum_attempts_per_accepted_case,
                DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE,
            )
            overridden_args = parse_args(
                (
                    "--family",
                    "tanaka",
                    "--split",
                    "validation",
                    "--accepted-cases",
                    "5",
                    "--output-root",
                    str(output),
                    "--batch-size",
                    "2",
                    "--maximum-attempts-per-accepted-case",
                    "7",
                )
            )
            self.assertEqual(
                request_from_args(
                    overridden_args
                ).maximum_attempts_per_accepted_case,
                7,
            )
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
            self.assertEqual(allocation["accepted_cases_before"], 0)
            self.assertEqual(allocation["accepted_cases_after"], 5)
            self.assertEqual(
                allocation["maximum_attempts_per_accepted_case"],
                4,
            )
            self.assertEqual(
                [quota["attempt_ceiling"] for quota in allocation["quotas"]],
                [4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
            )
            self.assertEqual(
                [quota["durable_attempted"] for quota in allocation["quotas"]],
                [0] * 11,
            )
            expected = plan["expected_output"]
            self.assertEqual(expected["stored_rows_per_accepted_case"], 200)
            self.assertEqual(expected["retained_rows"], 1_000)
            self.assertEqual(
                execution,
                TrajectoryExecutionConfig.paper("tanaka"),
            )
            json.dumps(plan, sort_keys=True, allow_nan=False)

    def test_base_preflight_rejects_unadjusted_jonswap_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = GenerationRequest(
                output_root=Path(directory) / "jonswap",
                family="jonswap_tma",
                split=SplitId.TEST,
                accepted_cases=27,
                batch_size=27,
                platform=BOOTSTRAP_PLATFORM,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "run_paper_corpus_jonswap_bucketed.py",
            ):
                preflight(request)

    def test_stokes_preflight_reports_static_rows_and_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = GenerationRequest(
                output_root=Path(directory) / "stokes",
                family="stokes",
                split=SplitId.TRAIN,
                accepted_cases=8,
                batch_size=4,
                platform=BOOTSTRAP_PLATFORM,
            )
            spec, execution, state, plan = preflight(request)

            self.assertFalse(state.complete)
            self.assertEqual(execution, PAPER_STATIC_STOKES_CONTRACT)
            self.assertEqual(
                [quota.target_accepted for quota in spec.quotas],
                [2, 2, 2, 2],
            )
            expected = plan["expected_output"]
            self.assertEqual(expected["stored_rows_per_accepted_case"], 1)
            self.assertEqual(expected["retained_rows"], 8)
            self.assertEqual(expected["spatial_points_per_row"], 1024)
            self.assertEqual(plan["execution"]["dno_order"], 6)

    def test_pending_proposal_is_detected_and_changed_config_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resume"
            request = GenerationRequest(
                output_root=output,
                family="tanaka",
                split=SplitId.TEST,
                accepted_cases=4,
                batch_size=1,
                platform=BOOTSTRAP_PLATFORM,
            )
            spec = build_run_spec(request)
            assignments = schedule_attempt_batch(
                spec.quotas,
                {},
                family_id=int(spec.family_id),
                revision_id=spec.revision_id,
                split_id=spec.split_id,
                stream_id=spec.stream_id,
                first_attempt_index=spec.first_attempt_index,
                batch_size=spec.batch_size,
            )
            records = tuple(
                {
                    "schema": "launcher_resume_test_v1",
                    "cell_id": assignment.cell_id,
                    "attempt_index": assignment.case_key.attempt_index,
                }
                for assignment in assignments
            )
            paths = batch_paths_for_assignments(
                spec.root,
                assignments,
                family_name=spec.family_name,
                batch_id=0,
            )
            ensure_proposal(
                paths,
                build_proposal_arrays(
                    assignments,
                    records,
                    cell_codes=spec.cell_codes,
                    batch_id=0,
                    config_fingerprint=spec.config_fingerprint,
                    metadata={"test": "no_compute_resume_scan"},
                ),
            )

            _, _, state, plan = preflight(request)
            self.assertIsNotNone(state.pending)
            self.assertEqual(plan["resume_state"]["pending_batch_id"], 0)
            self.assertEqual(plan["resume_state"]["pending_status"], "proposed")

            changed = GenerationRequest(
                output_root=output,
                family="tanaka",
                split=SplitId.TEST,
                accepted_cases=5,
                batch_size=1,
                platform=BOOTSTRAP_PLATFORM,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "configuration fingerprint does not match",
            ):
                preflight(changed)

            changed_attempt_limit = GenerationRequest(
                output_root=output,
                family="tanaka",
                split=SplitId.TEST,
                accepted_cases=4,
                batch_size=1,
                platform=BOOTSTRAP_PLATFORM,
                maximum_attempts_per_accepted_case=5,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "configuration fingerprint does not match",
            ):
                preflight(changed_attempt_limit)

            with np.load(paths.proposal, allow_pickle=False) as archive:
                self.assertEqual(
                    str(archive["config_fingerprint"]),
                    spec.config_fingerprint,
                )


if __name__ == "__main__":
    unittest.main()
