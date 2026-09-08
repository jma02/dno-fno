"""One-time export of the archived paper pool with a fresh simulation split.

This reads the archived manifest directly; it is not a production data loader.
Wave fields are copied exactly. Former holdouts are intentionally discarded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    report_path = output_root.with_name(f"{output_root.name}.conversion.json")
    if output_root.exists() or report_path.exists():
        raise FileExistsError("the output dataset and report must not already exist")
    manifest = json.loads(manifest_path.read_text())
    map_path = (manifest_path.parent / manifest["trajectory_map_npz"]).resolve(
        strict=True
    )
    source_records = {}
    for name, path in (
        ("manifest", manifest_path),
        ("trajectory_map", map_path),
        ("converter", Path(__file__).resolve()),
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source_records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    with np.load(map_path, allow_pickle=False) as archive:
        metadata = {
            name: archive[name]
            for name in (
                "trajectory_accepted",
                "trajectory_family_id",
                "trajectory_parameter_group_id",
                "trajectory_row_count",
                "trajectory_dataset_split",
                "trajectory_index",
                "frame_index",
                "shard_index",
                "shard_row",
            )
        }
    accepted = np.flatnonzero(metadata["trajectory_accepted"])
    total_simulations = len(accepted)
    total_rows = int(manifest["n_rows"])
    nx = int(manifest["grid"]["nx"])
    length = float(manifest["grid"]["length"])
    row_simulations = metadata["trajectory_index"]
    if (
        row_simulations.shape != (total_rows,)
        or not metadata["trajectory_accepted"][row_simulations].all()
        or np.any(row_simulations[1:] < row_simulations[:-1])
        or not np.array_equal(
            np.bincount(
                row_simulations, minlength=len(metadata["trajectory_accepted"])
            ),
            metadata["trajectory_row_count"],
        )
    ):
        raise ValueError("archived row ownership disagrees with accepted simulations")
    dense_ids = np.full(len(metadata["trajectory_accepted"]), -1, dtype=np.int64)
    dense_ids[accepted] = np.arange(total_simulations)

    # Match build_dataset: permutation, training prefix, then validation and test.
    seed = 42
    validation_count = int(total_simulations * 0.1)
    test_count = int(total_simulations * 0.1)
    train_count = total_simulations - validation_count - test_count
    order = np.random.default_rng(seed).permutation(total_simulations)
    splits = np.full(total_simulations, "train", dtype="U10")
    splits[order[train_count : train_count + validation_count]] = "validation"
    splits[order[train_count + validation_count :]] = "test"
    group_width = max(map(len, metadata["trajectory_parameter_group_id"]))
    dtypes = {
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
    }
    expected_bytes = (
        total_rows
        * sum(
            np.dtype(dtype).itemsize * (nx if name in {"eta", "xi", "gxi"} else 1)
            for name, dtype in dtypes.items()
        )
        + nx * 8
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < expected_bytes + 1024**3:
        raise OSError(f"need {expected_bytes:,} bytes plus 1 GiB; have {free_bytes:,}")
    print(
        f"{started_at}: {total_simulations:,} simulations, {total_rows:,} rows; "
        f"{expected_bytes:,} output bytes, {free_bytes:,} disk bytes free",
        flush=True,
    )
    shard_records = manifest["dataset_shards"]
    with TemporaryDirectory(
        prefix=f".{output_root.name}-", dir=output_root.parent
    ) as temporary:
        staging_root = Path(temporary)
        directory = staging_root / "arrays"
        directory.mkdir()
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(total_rows, nx)
                if name in {"eta", "xi", "gxi"}
                else (total_rows,),
            )
            for name, dtype in dtypes.items()
        }
        first_row = 0
        source_bytes = 0
        for shard_index, record in enumerate(shard_records):
            path = (manifest_path.parent / record["path"]).resolve(strict=True)
            row_count = int(record["n_rows"])
            rows = slice(first_row, first_row + row_count)
            if not np.all(
                metadata["shard_index"][rows] == shard_index
            ) or not np.array_equal(metadata["shard_row"][rows], np.arange(row_count)):
                raise ValueError(f"archived row pointers disagree with {path}")
            source_bytes += path.stat().st_size
            with np.load(path, allow_pickle=False) as archive:
                for name in ("eta", "xi", "gxi", "depth", "time", "frame_index"):
                    values = archive[name]
                    if (
                        values.dtype != arrays[name].dtype
                        or values.shape != arrays[name][rows].shape
                    ):
                        raise ValueError(f"wrong dtype or shape for {name} in {path}")
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite {name} in {path}")
                    if name == "frame_index" and not np.array_equal(
                        values, metadata[name][rows]
                    ):
                        raise ValueError(
                            f"frame indices disagree with the map in {path}"
                        )
                    arrays[name][rows] = values
                    if not np.array_equal(arrays[name][rows], values):
                        raise ValueError(f"copied {name} differs from {path}")
                    del values
            attempts = row_simulations[rows]
            simulation_ids = dense_ids[attempts]
            for name, values in {
                "simulation_id": simulation_ids,
                "dataset_split": splits[simulation_ids],
                "family_id": metadata["trajectory_family_id"][attempts],
                "parameter_group_id": metadata["trajectory_parameter_group_id"][
                    attempts
                ],
            }.items():
                arrays[name][rows] = values
                if not np.array_equal(arrays[name][rows], values):
                    raise ValueError(
                        f"copied {name} differs from its assigned metadata"
                    )
            first_row += row_count
            if (shard_index + 1) % 100 == 0 or shard_index + 1 == len(shard_records):
                print(
                    f"verified {shard_index + 1}/{len(shard_records)} shards; "
                    f"{first_row:,}/{total_rows:,} rows",
                    flush=True,
                )
        if first_row != total_rows:
            raise ValueError("copied row count differs from the manifest")
        print("Flushing verified arrays before publication", flush=True)
        for array in arrays.values():
            array.flush()
        x = np.linspace(0.0, length, nx, endpoint=False)
        np.save(directory / "x.npy", x)
        if not np.array_equal(np.load(directory / "x.npy"), x):
            raise ValueError("saved grid differs from the source domain")

        families = metadata["trajectory_family_id"][accepted]
        row_counts = metadata["trajectory_row_count"][accepted]
        population = {}
        family_population = {}
        for split in ("train", "validation", "test"):
            selected = splits == split
            population[split] = {
                "simulations": int(selected.sum()),
                "rows": int(row_counts[selected].sum()),
            }
            for family in range(1, 5):
                selected_family = selected & (families == family)
                family_population.setdefault(str(family), {})[split] = {
                    "simulations": int(selected_family.sum()),
                    "rows": int(row_counts[selected_family].sum()),
                }
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        report = {
            "status": "complete",
            "dataset": str(output_root),
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": time.perf_counter() - start,
            "command": shlex.join(["uv", "run", "--offline", "python", *sys.argv]),
            "sources": source_records,
            "source_shards": len(shard_records),
            "source_shard_bytes": source_bytes,
            "seed": seed,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
            "former_splits_discarded": True,
            "simulation_id_order": "dense accepted IDs in original trajectory-map order",
            "rows": total_rows,
            "simulations": total_simulations,
            "split_counts": population,
            "family_split_counts": family_population,
            "finite_source_fields": True,
            "exact_source_field_comparisons": True,
            "exact_row_metadata_comparisons": True,
            "one_split_per_simulation": True,
            "output_files": {
                path.name: {"bytes": path.stat().st_size}
                for path in sorted(directory.glob("*.npy"))
            },
        }
        staged_report = staging_root / "conversion.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        if output_root.exists() or report_path.exists():
            raise FileExistsError("output dataset or report appeared during conversion")
        directory.rename(output_root)
        staged_report.rename(report_path)
    print(json.dumps(report, indent=2), flush=True)
