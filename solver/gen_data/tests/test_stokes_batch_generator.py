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

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.batch_storage import load_completed_batch  # noqa: E402
from solver.gen_data.stokes_batch_generator import (  # noqa: E402
    PAPER_STATIC_STOKES_NX,
    generate_static_stokes_batch,
)
from solver.gen_data.stokes_sampling import StokesSample  # noqa: E402


class StaticStokesBatchExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_rejected_and_accepted_attempts_remain_aligned_in_the_batch(self) -> None:
        path = self.root / "batch.npz"
        sample = StokesSample("deep", 3, 4.0, 0.5, 0.005)
        field = np.zeros((1, 1024), dtype=np.float64)
        parameter_groups = ("finite_low", "deep_low")

        with (
            patch(
                "solver.gen_data.stokes_batch_generator.sample_stokes_simulation",
                side_effect=(None, sample),
            ) as sampler,
            patch(
                "solver.gen_data.stokes_batch_generator._construct_stokes_batch",
                return_value=(field, field),
            ) as construct,
            patch(
                "solver.gen_data.stokes_batch_generator.compute_dno_target",
                return_value=(field, field, field),
            ),
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
        construct.assert_called_once()

        batch = load_completed_batch(path)
        self.assertEqual(batch.parameter_group_ids, parameter_groups)
        np.testing.assert_array_equal(batch.accepted_simulations, (False, True))
        assert batch.shard is not None
        np.testing.assert_array_equal(
            batch.shard["simulation_local_index"],
            np.asarray([1], dtype=np.int32),
        )

    def test_compiled_batch_matches_eager_float64_evaluation(self) -> None:
        samples_by_group = {
            "finite_low": (
                StokesSample("finite", 20, 0.25, 0.3, 0.003),
                StokesSample("finite", 18, 0.3, 0.5, 0.002),
            ),
            "deep_low": (
                StokesSample("deep", 8, 5.0, 0.3, 0.01),
                StokesSample("deep", 4, 8.0, 0.5, 0.008),
            ),
        }
        for group, samples in samples_by_group.items():
            with self.subTest(group=group):
                eager_path = self.root / f"{group}_eager.npz"
                compiled_path = self.root / f"{group}_compiled.npz"
                with (
                    patch(
                        "solver.gen_data.stokes_batch_generator.sample_stokes_simulation",
                        side_effect=samples,
                    ),
                    jax.disable_jit(),
                ):
                    generate_static_stokes_batch(
                        (group,) * len(samples), 0, eager_path, seed=1
                    )
                with patch(
                    "solver.gen_data.stokes_batch_generator.sample_stokes_simulation",
                    side_effect=samples,
                ):
                    generate_static_stokes_batch(
                        (group,) * len(samples), 0, compiled_path, seed=1
                    )

                eager = load_completed_batch(eager_path)
                compiled = load_completed_batch(compiled_path)
                assert eager.shard is not None and compiled.shard is not None
                for name in ("eta", "xi", "gxi"):
                    np.testing.assert_allclose(
                        compiled.shard[name], eager.shard[name], rtol=1e-6, atol=1e-7
                    )
                self.assertEqual(
                    compiled.shard["eta"].shape, (2, PAPER_STATIC_STOKES_NX)
                )


if __name__ == "__main__":
    unittest.main()
