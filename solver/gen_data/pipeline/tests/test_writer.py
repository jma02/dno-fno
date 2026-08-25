"""CPU tests for common proposal and complete-case shard assembly."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.archive import BatchStatus, inspect_batch
from solver.gen_data.pipeline.manifest import build_dataset_view
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.writer import (
    AcceptedCaseRows,
    CaseOutcome,
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
)


FINGERPRINT = "e" * 64
REQUIRED = (
    QualityReason.NONFINITE_STATE
    | QualityReason.NONFINITE_TARGET
    | QualityReason.BOTTOM_CLEARANCE
)


def _assignments(split_id: SplitId) -> tuple[AttemptAssignment, ...]:
    return tuple(
        AttemptAssignment(
            case_key=CaseKey(
                family_id=3,
                revision_id=2,
                split_id=split_id,
                stream_id=7,
                attempt_index=index,
            ),
            cell_id=cell,
        )
        for index, cell in enumerate(("shallow", "deep"))
    )


def _decision(*, accepted: bool) -> QualityDecision:
    return QualityDecision(
        scope=QualityScope.TRAJECTORY,
        required=REQUIRED,
        evaluated=REQUIRED,
        failed=QualityReason.NONE if accepted else QualityReason.NONFINITE_TARGET,
    )


class CommonWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assignments = _assignments(SplitId.TRAIN)
        self.paths = batch_paths_for_assignments(
            self.root,
            self.assignments,
            family_name="jonswap_tma",
            batch_id=4,
        )
        self.proposal = build_proposal_arrays(
            self.assignments,
            (
                {"depth": 0.1, "phase_right": [0.1, 0.2]},
                {"depth": 5.0, "phase_right": [0.3, 0.4]},
            ),
            cell_codes={"shallow": 0, "deep": 1},
            batch_id=4,
            config_fingerprint=FINGERPRINT,
            metadata={"target": "order_6_pad_8_band_128"},
        )

    def test_proposal_preassigns_identity_and_commit_keeps_whole_cases(self) -> None:
        field = np.arange(12, dtype=np.float64).reshape(3, 4) / 100.0
        outcomes = (
            CaseOutcome(
                decision=_decision(accepted=True),
                rows=AcceptedCaseRows(
                    eta=field,
                    xi=field + 0.1,
                    gxi=field - 0.1,
                    depth=0.1,
                    time=np.asarray([0.0, 0.5, 1.0]),
                    selected_dense_index=np.asarray([0, 5, 10]),
                ),
                metrics={"maximum_error": 2.0e-5},
            ),
            CaseOutcome(
                decision=_decision(accepted=False),
                rows=None,
                metrics={"maximum_error": None},
            ),
        )

        commit_case_outcomes(
            self.paths,
            self.proposal,
            outcomes,
            metadata={"accepted_cases": 1},
        )

        self.assertEqual(
            inspect_batch(self.paths).status,
            BatchStatus.COMMITTED,
        )
        with np.load(self.paths.proposal, allow_pickle=False) as proposal:
            self.assertTrue(
                np.array_equal(
                    proposal["case_id"],
                    np.asarray(
                        [
                            assignment.case_key.case_id
                            for assignment in self.assignments
                        ],
                        dtype=np.int64,
                    ),
                )
            )
            self.assertTrue(
                np.array_equal(
                    proposal["root_seed"],
                    np.full(2, 2026072210, dtype=np.uint64),
                )
            )
            specifications = [
                json.loads(value) for value in proposal["case_spec_json"]
            ]
            self.assertEqual(specifications[0]["phase_right"], [0.1, 0.2])
        with np.load(self.paths.shard, allow_pickle=False) as shard:
            self.assertEqual(shard["eta"].shape, (3, 4))
            self.assertTrue(np.all(shard["case_local_index"] == 0))

        view = build_dataset_view(
            self.root,
            (self.paths,),
            expected_fingerprint=FINGERPRINT,
        )
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_accepted"],
                    np.asarray([True, False]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_required_bits"],
                    np.full(2, int(REQUIRED), dtype=np.uint32),
                )
            )

    def test_all_rejected_batch_commits_without_a_shard(self) -> None:
        outcomes = tuple(
            CaseOutcome(
                decision=_decision(accepted=False),
                rows=None,
                metrics={"maximum_error": None},
            )
            for _ in self.assignments
        )

        commit_case_outcomes(
            self.paths,
            self.proposal,
            outcomes,
            metadata={"accepted_cases": 0},
        )

        self.assertFalse(self.paths.shard.exists())
        self.assertEqual(inspect_batch(self.paths).status, BatchStatus.COMMITTED)

    def test_decision_and_row_presence_cannot_disagree(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have rows"):
            CaseOutcome(
                decision=_decision(accepted=True),
                rows=None,
                metrics={},
            )


if __name__ == "__main__":
    unittest.main()
