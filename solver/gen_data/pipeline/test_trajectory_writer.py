"""CPU integration test from one GL2 production arm through a committed view."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.archive import ensure_proposal  # noqa: E402
from solver.gen_data.pipeline.manifest import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    ResidualControlledGL2Contract,
    execute_production_trajectory,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
    outcomes_from_production,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
)

jax.config.update("jax_enable_x64", True)


class TrajectoryWriterIntegrationTest(unittest.TestCase):
    def test_actual_single_arm_time_selection_commit_and_view(self) -> None:
        contract = ResidualControlledGL2Contract(
            nx=16,
            dno_order=0,
            pad_factor=1,
            maximum_wavenumber=3.0,
            production_dt=0.02,
            saved_dt=0.02,
            target_time_chunk_size=2,
        )
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        eta0 = np.stack(
            (0.001 * np.cos(x), 0.0015 * np.cos(2.0 * x))
        )
        xi0 = np.stack(
            (0.002 * np.sin(2.0 * x), 0.001 * np.sin(x))
        )
        depths = np.asarray([1.0, 1.5])
        assignments = tuple(
            AttemptAssignment(
                case_key=CaseKey(
                    family_id=4,
                    revision_id=1,
                    split_id=SplitId.TEST,
                    stream_id=2,
                    attempt_index=index,
                ),
                cell_id=cell,
            )
            for index, cell in enumerate(("finite_a", "finite_b"))
        )
        fingerprint = "a" * 64
        proposal = build_proposal_arrays(
            assignments,
            ({"amplitude": 0.001}, {"amplitude": 0.0015}),
            cell_codes={"finite_a": 0, "finite_b": 1},
            batch_id=0,
            config_fingerprint=fingerprint,
            metadata={"smoke": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = batch_paths_for_assignments(
                root,
                assignments,
                family_name="jonswap_tma",
                batch_id=0,
            )
            ensure_proposal(paths, proposal)
            self.assertTrue(paths.proposal.exists())

            execution = execute_production_trajectory(
                eta0,
                xi0,
                depths,
                np.asarray([0.0, 0.02, 0.04]),
                contract=contract,
            )
            outcomes = outcomes_from_production(
                execution,
                depths,
                family="jonswap_tma",
                length=contract.length,
                policy=StoredTimePolicy(
                    tanaka_count=2,
                    benjamin_feir_count=2,
                    random_sea_count=3,
                ),
            )
            self.assertTrue(
                all(outcome.decision.accepted for outcome in outcomes)
            )
            self.assertTrue(
                all(
                    outcome.metrics["production_dt"] == contract.production_dt
                    for outcome in outcomes
                )
            )
            self.assertTrue(
                all(
                    np.array_equal(
                        outcome.rows.selected_dense_index,
                        np.asarray([0, 1, 2]),
                    )
                    for outcome in outcomes
                    if outcome.rows is not None
                )
            )
            commit_case_outcomes(
                paths,
                proposal,
                outcomes,
                metadata={"accepted_cases": 2},
            )
            view = build_dataset_view(
                root,
                (paths,),
                expected_fingerprint=fingerprint,
            )
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([3, 3]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_split_id"],
                    np.asarray([2, 2], dtype=np.uint8),
                )


if __name__ == "__main__":
    unittest.main()
