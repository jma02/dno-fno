"""Focused fail-closed tests for the Tanaka revision-3 completion audit."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_completed_tanaka_revision3 import (  # noqa: E402
    AuditTotals,
    BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,
    EXPECTED_ACCEPTED_EVALUATED_BITS,
    EXPECTED_CHUNKS,
    EXPECTED_MAP_ARRAYS,
    EXPECTED_REQUIRED_BITS,
    Extrema,
    ProposedCase,
    _amplitude_inversion_record,
    _expected_chunk_quotas,
    _load_exact_chunks,
    _proposal_cases,
    _validate_case_record,
    _validate_map_array_schema,
    _validate_summary_cell_counts,
    _validate_shard,
)
from scripts.build_paper_dataset_view import TRAJECTORY_MAP_DTYPES  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_SAMPLE_CELL_IDS,
    sample_tanaka_case,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    sample_tanaka_trajectory_cases,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
)
from solver.tanaka_ICs import modified_tanaka as tanaka_solver  # noqa: E402


def _support_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "depth",
            "total_alpha",
            "crest_alpha",
            "physical_crest_amplitude",
            "resolution_ratio",
            "minimum_separation",
        )
    }


def _assignment() -> AttemptAssignment:
    return AttemptAssignment(
        CaseKey(
            family_id=2,
            revision_id=3,
            split_id=SplitId.TRAIN,
            stream_id=0,
            attempt_index=0,
        ),
        TANAKA_SAMPLE_CELL_IDS[0],
    )


def _proposed_case() -> ProposedCase:
    assignment = _assignment()
    sample = sample_tanaka_case(
        assignment,
        domain_length=TrajectoryExecutionConfig.paper("tanaka").numerical.length,
    )
    return ProposedCase(
        case_id=assignment.case_key.case_id,
        cell_code=0,
        cell_id=assignment.cell_id,
        depth=sample.depth,
        sample=sample,
    )


def _accepted_case() -> dict[str, object]:
    return {
        "accepted": True,
        "case_id": _assignment().case_key.case_id,
        "required_bits": EXPECTED_REQUIRED_BITS,
        "evaluated_bits": EXPECTED_ACCEPTED_EVALUATED_BITS,
        "failed_bits": 0,
        "first_row": 0,
        "row_count": 200,
        "metrics": {
            "accepted": True,
            "all_stages_solved": True,
            "complete_admissible_trajectory": True,
            "initial_internal_hamiltonian": None,
            "intended_terminal_time": 200.0,
            "internal_dno_finite": None,
            "internal_hamiltonian_drift_threshold": None,
            "internal_health_evaluated": False,
            "internal_state_finite": None,
            "maximum_internal_hamiltonian_drift": None,
            "maximum_stage_residual": 9.0e-9,
            "minimum_internal_water_column": None,
            "positive_water_column": True,
            "production_dt": 0.01,
            "realized_terminal_time": 200.0,
            "saved_time_count": 2_501,
            "state_finite": True,
            "target_finite": True,
        },
    }


class TanakaCompletionAuditTests(unittest.TestCase):
    def test_exact_six_chunk_plan_is_four_train_plus_validation_test(self) -> None:
        self.assertEqual(len(EXPECTED_CHUNKS), 6)
        self.assertEqual(
            [chunk.accepted_before for chunk in EXPECTED_CHUNKS[:4]],
            [0, 2_048, 4_096, 8_192],
        )
        self.assertEqual(
            [chunk.accepted_count for chunk in EXPECTED_CHUNKS[:4]],
            [2_048, 2_048, 4_096, 8_192],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exact six-chunk plan"):
                _load_exact_chunks(Path(temporary))

    def test_exact_proposal_replay_rejects_requested_crest_mutation(self) -> None:
        assignment = _assignment()
        sampled = sample_tanaka_trajectory_cases(
            (assignment,),
            contract=TrajectoryExecutionConfig.paper("tanaka").numerical,
        )
        record = json.loads(
            json.dumps(sampled.specification_records[0], allow_nan=False)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proposal.npz"

            def write(specification: dict[str, object]) -> None:
                np.savez(
                    path,
                    attempt_index=np.asarray(
                        [assignment.case_key.attempt_index],
                        dtype=np.uint64,
                    ),
                    batch_id=np.asarray(0, dtype=np.int64),
                    case_id=np.asarray(
                        [assignment.case_key.case_id],
                        dtype=np.int64,
                    ),
                    case_spec_json=np.asarray(
                        [json.dumps(specification, allow_nan=False)]
                    ),
                    cell_id=np.asarray([0], dtype=np.int32),
                    config_fingerprint=np.asarray("0" * 64),
                    family_id=np.asarray(2, dtype=np.int16),
                    metadata_json=np.asarray("{}"),
                    revision_id=np.asarray(3, dtype=np.int16),
                    root_seed=np.asarray(
                        [assignment.case_key.root_seed],
                        dtype=np.uint64,
                    ),
                    split_id=np.asarray(0, dtype=np.uint8),
                    stream_id=np.asarray(
                        [assignment.case_key.stream_id],
                        dtype=np.uint32,
                    ),
                )

            write(record)
            totals = AuditTotals()
            proposals = _proposal_cases(
                path,
                split=SplitId.TRAIN,
                cell_ids_by_code={0: assignment.cell_id},
                support_extrema=_support_extrema(),
                totals=totals,
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(totals.proposal_specs_checked, 1)
            self.assertEqual(totals.requested_crests_checked, 1)

            crests = record["crests"]
            assert isinstance(crests, list) and isinstance(crests[0], dict)
            crests[0]["alpha"] = float(crests[0]["alpha"]) + 1.0e-6
            write(record)
            with self.assertRaisesRegex(ValueError, "deterministic sampling"):
                _proposal_cases(
                    path,
                    split=SplitId.TRAIN,
                    cell_ids_by_code={0: assignment.cell_id},
                    support_extrema=_support_extrema(),
                    totals=AuditTotals(),
                )

    def test_case_audit_rejects_gate_terminal_and_row_mutations(self) -> None:
        proposal = _proposed_case()

        def validate(case: dict[str, object]) -> None:
            _validate_case_record(
                case,
                proposed=proposal,
                local_index=0,
                batch_id=0,
                residual_tolerance=1.0e-8,
                production_dt=0.01,
                saved_time_count=2_501,
                residual_extrema=Extrema(),
                rejection_reasons=Counter(),
            )

        validate(_accepted_case())

        residual_case = _accepted_case()
        residual_metrics = residual_case["metrics"]
        assert isinstance(residual_metrics, dict)
        residual_metrics["maximum_stage_residual"] = 1.1e-8
        with self.assertRaisesRegex(ValueError, "residual tolerance"):
            validate(residual_case)

        terminal_case = _accepted_case()
        terminal_metrics = terminal_case["metrics"]
        assert isinstance(terminal_metrics, dict)
        terminal_metrics["realized_terminal_time"] = 199.92
        with self.assertRaisesRegex(ValueError, "terminal time"):
            validate(terminal_case)

        row_case = _accepted_case()
        row_case["row_count"] = 199
        with self.assertRaisesRegex(ValueError, "exactly 200 rows"):
            validate(row_case)

        mask_case = _accepted_case()
        mask_case["required_bits"] = int(QualityReason.INCOMPLETE_TRAJECTORY)
        with self.assertRaisesRegex(ValueError, "current required mask"):
            validate(mask_case)

    def test_shard_rejects_dtype_and_dense_time_mutations(self) -> None:
        proposal = _proposed_case()
        audited = _validate_case_record(
            _accepted_case(),
            proposed=proposal,
            local_index=0,
            batch_id=0,
            residual_tolerance=1.0e-8,
            production_dt=0.01,
            saved_time_count=2_501,
            residual_extrema=Extrema(),
            rejection_reasons=Counter(),
        )
        dense_indices = np.linspace(0, 2_500, 200, dtype=np.int32)
        arrays: dict[str, np.ndarray] = {
            "case_local_index": np.zeros(200, dtype=np.int32),
            "config_fingerprint": np.asarray("f" * 64),
            "depth": np.full(200, proposal.depth, dtype=np.float64),
            "eta": np.zeros((200, 4), dtype=np.float32),
            "frame_index": np.arange(200, dtype=np.int32),
            "gxi": np.zeros((200, 4), dtype=np.float32),
            "proposal_sha256": np.asarray("a" * 64),
            "selected_dense_index": dense_indices,
            "time": 0.08 * dense_indices.astype(np.float64),
            "xi": np.zeros((200, 4), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.npz"

            def write() -> None:
                np.savez(path, **arrays)

            def validate() -> None:
                _validate_shard(
                    path,
                    proposal_sha256="a" * 64,
                    configuration_fingerprint="f" * 64,
                    proposed=(proposal,),
                    cases=(audited,),
                    delivered_nx=4,
                    dense_last_index=2_500,
                    saved_dt=0.08,
                    totals=AuditTotals(),
                )

            write()
            validate()

            arrays["gxi"] = arrays["gxi"].astype(np.float64)
            write()
            with self.assertRaisesRegex(TypeError, "gxi has the wrong dtype"):
                validate()

            arrays["gxi"] = np.zeros((200, 4), dtype=np.float32)
            arrays["time"][100] += 0.01
            write()
            with self.assertRaisesRegex(ValueError, "dense indices"):
                validate()

    def test_trajectory_map_rejects_equal_values_in_wrong_dtype(self) -> None:
        arrays = {
            name: (
                np.asarray(2, dtype=dtype)
                if name == "schema_version"
                else np.asarray([], dtype=dtype)
            )
            for name, dtype in TRAJECTORY_MAP_DTYPES.items()
        }
        self.assertEqual(frozenset(arrays), EXPECTED_MAP_ARRAYS)
        _validate_map_array_schema(arrays)
        arrays["trajectory_cell_id"] = np.asarray([], dtype=np.int64)
        with self.assertRaisesRegex(TypeError, "trajectory_cell_id"):
            _validate_map_array_schema(arrays)

    def test_corrected_amplitude_identity_and_below_floor_sentinel(self) -> None:
        identity = _amplitude_inversion_record(
            TrajectoryExecutionConfig.paper("tanaka")
        )
        limitation = identity["durable_artifact_limitation"]
        assert isinstance(limitation, dict)
        self.assertFalse(limitation["achieved_per_crest_amplitudes_persisted"])
        postcondition = identity["success_postcondition"]
        assert isinstance(postcondition, dict)
        self.assertEqual(postcondition["relative_tolerance"], 1.0e-6)
        self.assertEqual(postcondition["absolute_tolerance"], 1.0e-14)
        sentinel = identity["behavioral_sentinel"]
        assert isinstance(sentinel, dict)
        self.assertEqual(
            sentinel["requested_amplitude"],
            BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,
        )
        self.assertEqual(sentinel["validation_call_count"], 1)
        self.assertTrue(sentinel["deliberate_mismatch_rejected"])
        np.testing.assert_allclose(
            sentinel["achieved_amplitude"],
            BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,
            rtol=1.0e-6,
            atol=1.0e-14,
        )

    def test_amplitude_sentinel_rejects_no_op_validator(self) -> None:
        def no_op_validator(
            _eta_profile: object,
            _requested_amplitudes: object,
        ) -> None:
            return

        with (
            patch.object(
                tanaka_solver,
                "validate_solved_amplitudes",
                no_op_validator,
            ),
            self.assertRaisesRegex(
                RuntimeError, "accepted a deliberate crest mismatch"
            ),
        ):
            _amplitude_inversion_record(TrajectoryExecutionConfig.paper("tanaka"))

    def test_amplitude_sentinel_rejects_solver_that_skips_validator(self) -> None:
        requested = np.asarray(
            (BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,),
            dtype=np.float64,
        )
        template = tanaka_solver.make_default_tanaka_template(
            nx=32,
            dno_order=0,
            pad_factor=1,
        )
        solution = tanaka_solver.solve_modified_tanaka_batched(template, requested)
        with (
            patch.object(
                tanaka_solver,
                "solve_modified_tanaka_batched",
                return_value=solution,
            ),
            self.assertRaisesRegex(RuntimeError, "exactly once, observed 0 calls"),
        ):
            _amplitude_inversion_record(TrajectoryExecutionConfig.paper("tanaka"))

    def test_summary_cell_counts_bind_reconstructed_attempts_and_schema(self) -> None:
        quotas = _expected_chunk_quotas(EXPECTED_CHUNKS[0])
        accepted = dict(quotas)
        attempted = {cell_id: target + 2 for cell_id, target in quotas.items()}
        by_cell = {
            cell_id: {
                "target_accepted": target,
                "attempted": attempted[cell_id],
                "accepted": accepted[cell_id],
                "rejected": attempted[cell_id] - accepted[cell_id],
            }
            for cell_id, target in quotas.items()
        }
        summary: dict[str, object] = {
            "counts": {
                "attempted": sum(attempted.values()),
                "accepted": sum(accepted.values()),
                "rejected": sum(attempted.values()) - sum(accepted.values()),
                "by_cell": by_cell,
            }
        }
        _validate_summary_cell_counts(
            summary,
            expected_quotas=quotas,
            attempted_by_cell=attempted,
            accepted_by_cell=accepted,
        )

        cell_ids = tuple(quotas)
        redistributed = deepcopy(summary)
        redistributed_by_cell = redistributed["counts"]["by_cell"]
        assert isinstance(redistributed_by_cell, dict)
        for field, delta in (("attempted", 1), ("rejected", 1)):
            redistributed_by_cell[cell_ids[0]][field] += delta
            redistributed_by_cell[cell_ids[1]][field] -= delta
        with self.assertRaisesRegex(ValueError, "reconstructed transactions"):
            _validate_summary_cell_counts(
                redistributed,
                expected_quotas=quotas,
                attempted_by_cell=attempted,
                accepted_by_cell=accepted,
            )

        for name, mutation in (
            ("missing", lambda cells: cells.pop(cell_ids[0])),
            (
                "extra",
                lambda cells: cells.__setitem__("unexpected", dict(cells[cell_ids[0]])),
            ),
        ):
            with self.subTest(name=name):
                malformed = deepcopy(summary)
                malformed_by_cell = malformed["counts"]["by_cell"]
                assert isinstance(malformed_by_cell, dict)
                mutation(malformed_by_cell)
                with self.assertRaisesRegex(ValueError, "exact eleven-cell taxonomy"):
                    _validate_summary_cell_counts(
                        malformed,
                        expected_quotas=quotas,
                        attempted_by_cell=attempted,
                        accepted_by_cell=accepted,
                    )

        extra_field = deepcopy(summary)
        extra_field_by_cell = extra_field["counts"]["by_cell"]
        assert isinstance(extra_field_by_cell, dict)
        extra_field_by_cell[cell_ids[0]]["unbound"] = 0
        with self.assertRaisesRegex(ValueError, "exact count fields"):
            _validate_summary_cell_counts(
                extra_field,
                expected_quotas=quotas,
                attempted_by_cell=attempted,
                accepted_by_cell=accepted,
            )


if __name__ == "__main__":
    unittest.main()
