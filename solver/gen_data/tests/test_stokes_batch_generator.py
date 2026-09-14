"""CPU tests for static Stokes batch generation."""

from __future__ import annotations

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
from solver.gen_data.pipeline.types import (  # noqa: E402
    SimulationRows,
)
from solver.gen_data.stokes_batch_generator import (  # noqa: E402
    generate_static_stokes_batch,
)
from solver.gen_data.stokes_sampling import StokesSample  # noqa: E402


class StaticStokesBatchExecutionTests(unittest.TestCase):
    def test_rejected_and_accepted_attempts_remain_aligned_in_the_batch(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "batch.npz"
        sample = StokesSample("deep", 3, 4.0, 0.5, 0.005)
        field = np.zeros((1, 1024), dtype=np.float64)
        accepted = SimulationRows(
            eta=field,
            xi=field,
            gxi=field,
            depth=sample.depth,
            time=np.asarray([0.0], dtype=np.float64),
        )
        parameter_groups = ("finite_low", "deep_low")

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
            accepted_count = generate_static_stokes_batch(
                parameter_groups,
                11,
                path,
                seed=2026072205,
            )
        self.assertEqual(accepted_count, 1)

        self.assertEqual(
            sampler.call_args_list,
            [
                call(
                    parameter_groups[0],
                    seed=2026072205,
                    attempt_number=11,
                ),
                call(
                    parameter_groups[1],
                    seed=2026072205,
                    attempt_number=12,
                ),
            ],
        )
        evaluate.assert_called_once_with(sample)

        batch = load_completed_batch(path)
        self.assertEqual(batch.parameter_group_ids, parameter_groups)
        np.testing.assert_array_equal(batch.accepted_simulations, (False, True))
        assert batch.shard is not None
        np.testing.assert_array_equal(
            batch.shard["simulation_local_index"],
            np.asarray([1], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
