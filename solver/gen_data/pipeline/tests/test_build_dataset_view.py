"""CPU tests for building a training view from committed batches."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    SimulationCommitRecord,
    commit_batch,
    save_batch_plan,
    save_shard,
)
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view


def _target(*, nx: int = 4, maximum_wavenumber: float = 1.0) -> dict[str, object]:
    return {
        "nx": nx,
        "length": 2.0 * math.pi,
        "gravity": 1.0,
        "dno_order": 2,
        "pad_factor": 2,
        "maximum_wavenumber": maximum_wavenumber,
        "dtype": "float64",
    }


def _static_metadata(*, maximum_wavenumber: float = 1.0) -> dict[str, object]:
    return {
        "family": "stokes",
        "simulation_type": "static",
        "contract": {
            "role": "test",
            **_target(maximum_wavenumber=maximum_wavenumber),
        },
    }


def _trajectory_metadata(
    family: str,
    *,
    evolution_nx: int = 4,
    target_nx: int | None = None,
    fine_dt: float = 0.01,
    maximum_wavenumber: float = 1.0,
    target_maximum_wavenumber: float | None = None,
) -> dict[str, object]:
    numerical = {
        **_target(nx=evolution_nx, maximum_wavenumber=maximum_wavenumber),
        "coarse_dt": 2.0 * fine_dt,
        "fine_dt": fine_dt,
        "retry_dt": 0.5 * fine_dt,
        "saved_dt": 0.02,
        "gl2_residual_tolerance": 1.0e-8,
        "gl2_iteration_cap": 8,
        "refinement_tolerance": 1.0e-3,
        "relative_floor": 1.0e-12,
        "target_time_chunk_size": 2,
    }
    if target_nx is not None:
        numerical["target_nx"] = target_nx
    if target_maximum_wavenumber is not None:
        numerical["target_maximum_wavenumber"] = target_maximum_wavenumber
    return {
        "family": family,
        "simulation_type": "trajectory",
        "trajectory_execution": {
            "family": family,
            "role": "test",
            "numerical": numerical,
            "horizon": {"kind": "fixed_terminal_time"},
            "frame_selection": {"count": 2},
        },
    }


def _proposal(
    *,
    family_id: int,
    revision_id: int,
    split_id: int,
    batch_id: int,
    simulation_ids: tuple[int, ...],
    metadata: dict[str, object],
) -> dict[str, np.ndarray]:
    return {
        "family_id": np.asarray(family_id, dtype=np.int16),
        "revision_id": np.asarray(revision_id, dtype=np.int16),
        "split_id": np.asarray(split_id, dtype=np.uint8),
        "batch_id": np.asarray(batch_id, dtype=np.int64),
        "simulation_id": np.asarray(simulation_ids, dtype=np.int64),
        "cell_id": np.arange(len(simulation_ids), dtype=np.int32),
        "root_seed": np.zeros(len(simulation_ids), dtype=np.uint64),
        "stream_id": np.zeros(len(simulation_ids), dtype=np.uint32),
        "attempt_index": np.arange(len(simulation_ids), dtype=np.uint64),
        "simulation_spec_json": np.asarray(
            [
                json.dumps({"simulation_id": simulation_id})
                for simulation_id in simulation_ids
            ]
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _write_batch(
    paths: BatchPaths,
    *,
    proposal: dict[str, np.ndarray],
    accepted_local_indices: tuple[int, ...],
    frames_per_simulation: int,
    spatial_size: int = 4,
) -> None:
    save_batch_plan(paths, proposal)
    simulation_local_index = np.repeat(
        np.asarray(accepted_local_indices, dtype=np.int32),
        frames_per_simulation,
    )
    frame_index = np.tile(
        np.arange(frames_per_simulation, dtype=np.int32),
        len(accepted_local_indices),
    )
    row_count = simulation_local_index.size
    field = np.arange(row_count * spatial_size, dtype=np.float32).reshape(
        row_count, spatial_size
    )
    save_shard(
        paths,
        {
            "eta": field,
            "xi": field + np.float32(0.1),
            "gxi": field - np.float32(0.1),
            "depth": np.ones(row_count, dtype=np.float64),
            "time": frame_index.astype(np.float64),
            "simulation_local_index": simulation_local_index,
            "frame_index": frame_index,
            "selected_dense_index": frame_index,
        },
    )
    blocks = {
        local_index: (position * frames_per_simulation, frames_per_simulation)
        for position, local_index in enumerate(accepted_local_indices)
    }
    commit_batch(
        paths,
        simulations=tuple(
            SimulationCommitRecord(
                simulation_id=int(simulation_id),
                accepted=local_index in blocks,
                required_bits=63,
                evaluated_bits=63,
                failed_bits=0 if local_index in blocks else 1,
                first_row=blocks.get(local_index, (-1, 0))[0],
                row_count=blocks.get(local_index, (-1, 0))[1],
                metrics={},
            )
            for local_index, simulation_id in enumerate(proposal["simulation_id"])
        ),
        metadata={},
    )


class DatasetViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_view_preserves_rejected_simulations_and_row_ownership(self) -> None:
        first = BatchPaths.for_batch(
            self.root, family="stokes", split="train", batch_id=0
        )
        second = BatchPaths.for_batch(
            self.root, family="tanaka", split="validation", batch_id=0
        )
        _write_batch(
            first,
            proposal=_proposal(
                family_id=0,
                revision_id=2,
                split_id=0,
                batch_id=0,
                simulation_ids=(10, 11),
                metadata=_static_metadata(),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=1,
        )
        _write_batch(
            second,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=1,
                batch_id=0,
                simulation_ids=(20,),
                metadata=_trajectory_metadata("tanaka"),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=2,
        )

        view = build_dataset_view(self.root, (first, second))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["n_rows"], 3)
        self.assertEqual(manifest["n_trajectories"], 3)
        with np.load(view.trajectory_map, allow_pickle=False) as trajectory_map:
            np.testing.assert_array_equal(
                trajectory_map["trajectory_accepted"],
                np.asarray([True, False, True]),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_first_row"],
                np.asarray([0, -1, 1], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                trajectory_map["trajectory_index"],
                np.asarray([0, 2, 2], dtype=np.int32),
            )

    def test_shared_target_contract_must_match(self) -> None:
        stokes = BatchPaths.for_batch(
            self.root, family="stokes", split="train", batch_id=0
        )
        tanaka = BatchPaths.for_batch(
            self.root, family="tanaka", split="validation", batch_id=0
        )
        _write_batch(
            stokes,
            proposal=_proposal(
                family_id=0,
                revision_id=2,
                split_id=0,
                batch_id=0,
                simulation_ids=(1,),
                metadata=_static_metadata(maximum_wavenumber=2.0),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=1,
        )
        _write_batch(
            tanaka,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=1,
                batch_id=0,
                simulation_ids=(2,),
                metadata=_trajectory_metadata("tanaka", maximum_wavenumber=1.0),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=1,
        )
        with self.assertRaisesRegex(ValueError, "DNO target contract"):
            build_dataset_view(self.root, (stokes, tanaka))

    def test_one_family_revision_cannot_mix_execution_contracts(self) -> None:
        batches = tuple(
            BatchPaths.for_batch(self.root, family="tanaka", split=split, batch_id=0)
            for split in ("train", "validation")
        )
        for index, (paths, fine_dt) in enumerate(zip(batches, (0.01, 0.005))):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=1,
                    revision_id=3,
                    split_id=index,
                    batch_id=0,
                    simulation_ids=(10 + index,),
                    metadata=_trajectory_metadata("tanaka", fine_dt=fine_dt),
                ),
                accepted_local_indices=(0,),
                frames_per_simulation=1,
            )
        with self.assertRaisesRegex(ValueError, "cannot mix execution contracts"):
            build_dataset_view(self.root, batches)

    def test_dual_grid_view_uses_target_grid(self) -> None:
        paths = BatchPaths.for_batch(
            self.root, family="jonswap_tma", split="test", batch_id=0
        )
        _write_batch(
            paths,
            proposal=_proposal(
                family_id=3,
                revision_id=4,
                split_id=2,
                batch_id=0,
                simulation_ids=(30,),
                metadata=_trajectory_metadata(
                    "jonswap_tma",
                    evolution_nx=8,
                    target_nx=4,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_simulation=2,
            spatial_size=4,
        )
        view = build_dataset_view(self.root, (paths,))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["grid"]["nx"], 4)
        self.assertEqual(manifest["dataset_contract"]["target"]["nx"], 4)

    def test_uncommitted_batch_cannot_enter_view(self) -> None:
        paths = BatchPaths.for_batch(
            self.root, family="stokes", split="test", batch_id=0
        )
        save_batch_plan(
            paths,
            _proposal(
                family_id=0,
                revision_id=2,
                split_id=2,
                batch_id=0,
                simulation_ids=(40,),
                metadata=_static_metadata(),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "committed batches"):
            build_dataset_view(self.root, (paths,))


if __name__ == "__main__":
    unittest.main()
