"""CPU tests for replay validation, exact old-row retention, and batch resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from scripts import replay_jonswap_dataset as replay
from solver.gen_data.pipeline.time_selection import select_uniform_times
from solver.gen_data.pipeline.trajectory_rollout import TrajectorySamples
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch


class JonswapReplayTest(unittest.TestCase):
    def test_resume_preserves_all_original_rows_and_rejects_a_mismatch(self) -> None:
        numerical = replay.PAPER_ROLLOUT_NUMERICS["jonswap_tma"]._replace(nx=8, target_nx=8)
        counts = np.asarray([271, 270], dtype=np.int32)
        trajectories = []
        for count in counts:
            times = 0.08 * np.arange(count)
            eta = 0.001 * (1 + times[:, None]) * (1 + np.arange(8)[None] / 10)
            trajectories.append(TrajectorySamples(times, eta, 2 * eta, 3 * eta))
        old_indices = np.stack([select_uniform_times(int(count), keep_samples=16) for count in counts])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            inputs = {
                "simulation_id": np.asarray([12, 29]), "dataset_split": np.asarray(["train", "test"]),
                "parameter_group_id": np.asarray(["finite__gamma_1__right_0", "shallow__gamma_1__right_1"]),
                "depth": np.asarray([1.0, 2.0]), "significant_height": np.full(2, 0.01),
                "peak_wavenumber": np.full(2, 4.0), "peak_enhancement": np.ones(2),
                "right_moving_fraction": np.ones(2), "phase_right": np.zeros((2, 128)),
                "phase_left": np.zeros((2, 128)), "production_time_count": counts,
                "adjustment_time_count": counts, "nonlinear_ramp_time": np.ones(2),
                "old_dense_indices": old_indices, "old_first_row": np.asarray([0, 16]),
                "old_times": np.stack([case.times[picks] for case, picks in zip(trajectories, old_indices)]),
                "comparison_indices": np.arange(2),
                "metadata_json": np.asarray(json.dumps({"current_numerical": numerical._asdict(), "numerical_differences": {}})),
            }
            for name in ("eta", "xi", "gxi"):
                old = np.stack([getattr(case, name)[picks] for case, picks in zip(trajectories, old_indices)]).astype(np.float32)
                inputs[f"comparison_{name}"] = old
                np.save(source / f"{name}.npy", old.reshape(32, 8))
            np.save(source / "depth.npy", np.repeat(inputs["depth"], 16))
            np.save(source / "time.npy", inputs["old_times"].reshape(-1))
            np.savez(root / "inputs.npz", **inputs)
            arguments = ["--inputs", str(root / "inputs.npz"), "--output-root", str(root / "out"), "--batch-size", "1"]
            with (
                mock.patch.dict(replay.PAPER_ROLLOUT_NUMERICS, {"jonswap_tma": numerical}),
                mock.patch.object(replay, "construct_jonswap_tma_trajectory_batch", side_effect=lambda samples, *a, **kw: (
                    TrajectoryInitialBatch(np.zeros((len(samples), 8)), np.zeros((len(samples), 8)), np.asarray([sample.parameters.depth for sample in samples])),
                    tuple(range(len(samples))),
                )),
                mock.patch.object(replay, "execute_adjustment_batch", side_effect=lambda eta, xi, *a, **kw: tuple(zip(eta, xi))),
                mock.patch.object(replay, "execute_trajectory_batch", side_effect=lambda eta, xi, depths, *a, **kw: tuple(trajectories[int(depth) - 1] for depth in depths)),
            ):
                replay.main([*arguments, "--pilot-only"])
                bulk = [*arguments, "--source-dataset", str(source)]
                replay.main([*bulk, "--max-batches", "1"])
                self.assertFalse((root / "out" / "complete.json").exists())
                replay.main(bulk)
                self.assertTrue((root / "out" / "complete.json").exists())
                times = np.load(root / "out" / "time.npy").reshape(2, 200)
                for index, case in enumerate(trajectories):
                    self.assertTrue(np.all(np.diff(times[index]) > 0))
                    np.testing.assert_array_equal(times[index, [0, -1]], case.times[[0, -1]])
                    positions = np.searchsorted(times[index], inputs["old_times"][index])
                    for name in ("eta", "xi", "gxi"):
                        actual = np.load(root / "out" / f"{name}.npy").reshape(2, 200, 8)
                        np.testing.assert_array_equal(actual[index, positions], inputs[f"comparison_{name}"][index])
                np.testing.assert_array_equal(np.load(root / "out" / "simulation_id.npy"), np.repeat([12, 29], 200))
                np.testing.assert_array_equal(np.load(root / "out" / "dataset_split.npy"), np.repeat(["train", "test"], 200))
                np.save(source / "eta.npy", np.load(source / "eta.npy") * 1.01)
                (root / "out" / "progress.json").unlink()
                (root / "out" / "complete.json").unlink()
                with self.assertRaisesRegex(RuntimeError, "Replay mismatch"):
                    replay.main(bulk)
                self.assertFalse((root / "out" / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
