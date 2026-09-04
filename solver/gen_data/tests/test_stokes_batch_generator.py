"""CPU tests for static Stokes batch generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from solver.gen_data.pipeline.batch_storage import load_completed_batch  # noqa: E402
from solver.gen_data.pipeline.simulation_checks import (  # noqa: E402
    SimulationCheckResult,
)
from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402
from solver.gen_data.pipeline.writer import (  # noqa: E402
    AcceptedSimulationRows,
    SimulationOutcome,
)
from solver.gen_data.stokes_batch_generator import (  # noqa: E402
    generate_static_stokes_batch,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    STOKES_PARAMETER_GROUP_IDS,
    StokesSample,
)


class StaticStokesBatchExecutionTests(unittest.TestCase):
    def test_rejected_and_accepted_attempts_remain_aligned_in_the_batch(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "batch.npz"
        sample = StokesSample("deep", 3, 4.0, 0.5, 0.005)
        field = np.zeros((1, 1024), dtype=np.float64)
        accepted = SimulationOutcome(
            decision=SimulationCheckResult(accepted=True),
            rows=AcceptedSimulationRows(
                eta=field,
                xi=field,
                gxi=field,
                depth=sample.depth,
                time=np.asarray([0.0], dtype=np.float64),
            ),
            metrics={},
        )
        parameter_groups = (
            STOKES_PARAMETER_GROUP_IDS[0],
            STOKES_PARAMETER_GROUP_IDS[2],
        )

        with (
            patch(
                "solver.gen_data.stokes_batch_generator.sample_stokes_simulation",
                side_effect=(None, sample),
            ) as sampler,
            patch(
                "solver.gen_data.stokes_batch_generator.evaluate_static_stokes_sample",
                return_value=accepted,
            ) as evaluate,
        ):
            generate_static_stokes_batch(
                parameter_groups,
                11,
                path,
                dataset_split=DatasetSplit.TEST,
            )

        self.assertEqual(
            sampler.call_args_list,
            [
                call(
                    parameter_groups[0],
                    dataset_split=DatasetSplit.TEST,
                    attempt_number=11,
                ),
                call(
                    parameter_groups[1],
                    dataset_split=DatasetSplit.TEST,
                    attempt_number=12,
                ),
            ],
        )
        evaluate.assert_called_once_with(sample)

        batch = load_completed_batch(path)
        specifications = tuple(
            json.loads(str(value)) for value in batch.plan["simulation_spec_json"]
        )
        self.assertEqual(specifications, ({}, sample._asdict()))
        self.assertEqual(
            tuple(simulation.failed_checks for simulation in batch.simulations),
            (("outside_support",), ()),
        )
        assert batch.shard is not None
        np.testing.assert_array_equal(
            batch.shard["simulation_local_index"],
            np.asarray([1], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
