"""Join completed batches into NumPy arrays with simulation-level splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from solver.gen_data.pipeline.batch_storage import load_completed_batch
from solver.gen_data.pipeline.types import DatasetSplit


def build_dataset(
    root: Path,
    batches: Sequence[Path],
    *,
    length: float = 2.0 * math.pi,
) -> Path:
    """Save accepted rows and their family, group, simulation ID, and split.

    Wave data is already saved. This function does not generate it.
    Each simulation keeps the split assigned during generation. Rejected attempts
    stay in the batch files; they have no rows in the training dataset.
    """
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"dataset output already exists: {root}")

    total_rows = 0
    spatial_size: int | None = None
    group_width = 0
    # Count first so the final arrays can be filled one batch at a time.
    for path in batches:
        batch = load_completed_batch(path)
        group_width = max(group_width, max(map(len, batch.parameter_group_ids)))
        if batch.shard is None:
            continue
        row_count, nx = batch.shard["eta"].shape
        if spatial_size is not None and nx != spatial_size:
            raise ValueError("all dataset batches must use the same spatial grid")
        spatial_size = nx
        total_rows += row_count
        del batch
    if spatial_size is None:
        raise ValueError("a dataset must contain at least one accepted row")

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
        next_simulation_id: defaultdict[tuple[int, DatasetSplit], int] = defaultdict(
            int
        )
        first_row = 0
        for path in batches:
            batch = load_completed_batch(path)
            family_split = (int(batch.family_id), batch.dataset_split)
            first_simulation_id = next_simulation_id[family_split]
            next_simulation_id[family_split] += len(batch.parameter_group_ids)
            if batch.shard is None:
                continue
            shard = batch.shard
            row_count = shard["eta"].shape[0]
            rows = slice(first_row, first_row + row_count)
            for name in ("eta", "xi", "gxi", "depth", "time", "frame_index"):
                arrays[name][rows] = shard[name]
            local_index = shard["simulation_local_index"]
            arrays["simulation_id"][rows] = first_simulation_id + local_index.astype(
                np.int64
            )
            arrays["family_id"][rows] = int(batch.family_id)
            arrays["dataset_split"][rows] = batch.dataset_split.value
            arrays["parameter_group_id"][rows] = np.asarray(batch.parameter_group_ids)[
                local_index
            ]
            first_row += row_count
            del batch, shard
        for array in arrays.values():
            array.flush()
        np.save(
            directory / "x.npy", np.linspace(0.0, length, spatial_size, endpoint=False)
        )
        directory.rename(root)
    return root
