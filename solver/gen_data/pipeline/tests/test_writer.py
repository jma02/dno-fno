"""CPU tests for common batch-plan and complete-simulation shard assembly."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.types import DatasetSplit
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)


def _decision(*, accepted: bool) -> SimulationCheckResult:
    return SimulationCheckResult(
        accepted=accepted,
        nonfinite_target=not accepted,
    )


class CommonWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parameter_group_ids = ("shallow", "deep")
        self.path = batch_path(
            self.root,
            family="jonswap_tma",
            split=DatasetSplit.TRAIN.value,
            batch_id=4,
        )
        self.batch_plan = build_batch_plan(
            self.parameter_group_ids,
            (
                {"depth": 0.1, "phase_right": [0.1, 0.2]},
                {"depth": 5.0, "phase_right": [0.3, 0.4]},
            ),
            family_id=3,
            dataset_split=DatasetSplit.TRAIN,
        )

    def test_commit_keeps_whole_simulations(
        self,
    ) -> None:
        field = np.arange(12, dtype=np.float64).reshape(3, 4) / 100.0
        outcomes = (
            SimulationOutcome(
                decision=_decision(accepted=True),
                rows=AcceptedSimulationRows(
                    eta=field,
                    xi=field + 0.1,
                    gxi=field - 0.1,
                    depth=0.1,
                    time=np.asarray([0.0, 0.5, 1.0]),
                ),
                metrics={"maximum_error": 2.0e-5},
            ),
            SimulationOutcome(
                decision=_decision(accepted=False),
                rows=None,
                metrics={"maximum_error": None},
            ),
        )

        commit_simulation_outcomes(
            self.path,
            self.batch_plan,
            outcomes,
        )

        batch = load_completed_batch(self.path)
        self.assertEqual(batch.simulations[0].failed_checks, ())
        self.assertEqual(batch.simulations[1].failed_checks, ("nonfinite_target",))
        specifications = [
            json.loads(value) for value in batch.plan["simulation_spec_json"]
        ]
        self.assertEqual(specifications[0]["phase_right"], [0.1, 0.2])
        self.assertIsNotNone(batch.shard)
        assert batch.shard is not None
        self.assertEqual(batch.shard["eta"].shape, (3, 4))
        self.assertTrue(np.all(batch.shard["simulation_local_index"] == 0))

        view = build_dataset_view(
            self.root,
            (self.path,),
        )
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_accepted"],
                    np.asarray([True, False]),
                )
            )

    def test_all_rejected_batch_commits_without_a_shard(self) -> None:
        outcomes = tuple(
            SimulationOutcome(
                decision=_decision(accepted=False),
                rows=None,
                metrics={"maximum_error": None},
            )
            for _ in self.parameter_group_ids
        )

        commit_simulation_outcomes(
            self.path,
            self.batch_plan,
            outcomes,
        )

        self.assertIsNone(load_completed_batch(self.path).shard)

    def test_decision_and_row_presence_cannot_disagree(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have rows"):
            SimulationOutcome(
                decision=_decision(accepted=True),
                rows=None,
                metrics={},
            )


if __name__ == "__main__":
    unittest.main()
