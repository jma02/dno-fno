"""Combine saved NPZ simulations into NumPy arrays with train/validation/test splits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from solver.gen_data.pipeline.batch_storage import load_completed_batch
from solver.gen_data.pipeline.types import PhysicalFamilyId


def select_simulation_rows(
    simulation_ids: np.ndarray,
    maximum: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep evenly spaced saved frames per simulation, including both endpoints."""
    _, starts, counts = np.unique(simulation_ids, return_index=True, return_counts=True)
    kept = counts if maximum is None else np.minimum(counts, maximum)
    rows = np.concatenate(
        [
            start + np.linspace(0, count - 1, keep, dtype=np.int64)
            for start, count, keep in zip(starts, counts, kept, strict=True)
        ]
    )
    frame_indices = np.arange(rows.size) - np.repeat(np.cumsum(kept) - kept, kept)
    return rows, frame_indices


def build_dataset(
    root: Path,
    batches: Sequence[Path],
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    length: float = 2.0 * math.pi,
    max_snapshots_per_simulation: int | None = None,
) -> Path:
    """Split simulations once and save their selected snapshots as training rows."""
    if not (
        0 <= validation_fraction <= 1
        and 0 <= test_fraction <= 1
        and validation_fraction + test_fraction <= 1
    ):
        raise ValueError(
            "validation/test fractions must be nonnegative and sum to at most one"
        )
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"dataset output already exists: {root}")
    batches = tuple(sorted(path.resolve() for path in batches))
    if len(set(batches)) != len(batches):
        raise ValueError("a completed batch must not appear twice")

    total_rows = 0
    total_simulations = 0
    spatial_size: int | None = None
    group_width = 0
    run_directories: dict[tuple[PhysicalFamilyId, int], Path] = {}
    # Count first so the final arrays can be filled one batch at a time.
    for path in batches:
        batch = load_completed_batch(path)
        # Copies of the same seeded run must not enter different dataset splits.
        run = (batch.family_id, batch.seed)
        if run_directories.setdefault(run, path.parent) != path.parent:
            raise ValueError(
                "separate generation directories must use distinct family/seed pairs"
            )
        group_width = max(group_width, max(map(len, batch.parameter_group_ids)))
        if batch.shard is None:
            continue
        selected, _ = select_simulation_rows(
            batch.shard["simulation_local_index"],
            max_snapshots_per_simulation,
        )
        nx = batch.shard["eta"].shape[1]
        if spatial_size is not None and nx != spatial_size:
            raise ValueError("all dataset batches must use the same spatial grid")
        spatial_size = nx
        total_rows += selected.size
        total_simulations += int(batch.accepted_simulations.sum())
        del batch
    if spatial_size is None:
        raise ValueError("a dataset must contain at least one accepted row")

    # Assign each simulation once. Its snapshots never cross split boundaries.
    order = np.random.default_rng(seed).permutation(total_simulations)
    validation_count = int(total_simulations * validation_fraction)
    test_count = int(total_simulations * test_fraction)
    train_count = total_simulations - validation_count - test_count
    splits = np.full(total_simulations, "train", dtype="U10")
    splits[order[train_count : train_count + validation_count]] = "validation"
    splits[order[train_count + validation_count :]] = "test"

    root.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only when every array has been written successfully.
    with TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        directory = Path(temporary)
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(total_rows, spatial_size)
                if name in {"eta", "xi", "gxi"}
                else (total_rows,),
            )
            for name, dtype in {
                "eta": "float32",
                "xi": "float32",
                "gxi": "float32",
                "depth": "float64",
                "time": "float64",
                "frame_index": "int32",
                "simulation_id": "int64",
                "family_id": "int16",
                "parameter_group_id": f"U{group_width}",
                "dataset_split": "U10",
            }.items()
        }
        next_simulation_id = 0
        first_row = 0
        for path in batches:
            batch = load_completed_batch(path)
            if batch.shard is None:
                continue
            shard = batch.shard
            selected, frame_indices = select_simulation_rows(
                shard["simulation_local_index"],
                max_snapshots_per_simulation,
            )
            row_count = selected.size
            rows = slice(first_row, first_row + row_count)
            for name in ("eta", "xi", "gxi", "depth", "time"):
                arrays[name][rows] = shard[name][selected]
            arrays["frame_index"][rows] = frame_indices
            local_index = shard["simulation_local_index"][selected]
            accepted_indices, row_simulation = np.unique(
                local_index, return_inverse=True
            )
            simulation_ids = next_simulation_id + row_simulation
            arrays["simulation_id"][rows] = simulation_ids
            arrays["family_id"][rows] = int(batch.family_id)
            arrays["dataset_split"][rows] = splits[simulation_ids]
            arrays["parameter_group_id"][rows] = np.asarray(batch.parameter_group_ids)[
                local_index
            ]
            first_row += row_count
            next_simulation_id += accepted_indices.size
            del batch, shard
        for array in arrays.values():
            array.flush()
        np.save(
            directory / "x.npy", np.linspace(0.0, length, spatial_size, endpoint=False)
        )
        directory.rename(root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Directory containing saved batch_*.npz files; repeat to combine directories.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-snapshots-per-simulation",
        type=int,
        help="Keep at most this many saved frames per simulation, including endpoints.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    batches = tuple(
        path
        for directory in args.input_root
        for path in directory.expanduser().resolve(strict=True).rglob("batch_*.npz")
    )
    print(
        build_dataset(
            args.output_root,
            batches,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            max_snapshots_per_simulation=args.max_snapshots_per_simulation,
        )
    )
