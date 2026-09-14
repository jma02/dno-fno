"""Merge the full paper corpus, replayed JONSWAP, fresh Stokes, and old packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from solver.gen_data.pipeline.batch_storage import load_completed_batch


def write_rows(
    arrays: dict[str, np.memmap], first: int, values: dict[str, np.ndarray]
) -> int:
    """Copy one bounded block, checking finite fields and exact copied values."""
    stop = first + len(values["eta"])
    for name, value in values.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite source values in {name}")
        arrays[name][first:stop] = value
        if not np.array_equal(arrays[name][first:stop], value):
            raise ValueError(f"Copy mismatch in {name}, rows {first}:{stop}")
    return stop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("/data/outputs/paper_dataset/arrays"))
    parser.add_argument("--balanced", type=Path, default=Path("/data/outputs/paper_dataset_balanced_20260912/arrays"))
    parser.add_argument("--packets", type=Path, default=Path("/data/outputs/c27_mixed_wave_finetune_20260913_dataset"))
    parser.add_argument("--jonswap", type=Path, default=Path("/data/outputs/paper_dataset_regenerated_20260914/jonswap_arrays"))
    parser.add_argument("--new-stokes", type=Path, default=Path("/data/outputs/paper_dataset_regenerated_20260914/stokes"))
    parser.add_argument("--output", type=Path, default=Path("/data/outputs/paper_dataset_regenerated_20260914/arrays"))
    parser.add_argument("--seed", type=int, default=42, help="Split-assignment seed for newly generated Stokes simulations only")
    args = parser.parse_args()
    source_paths = (args.original, args.jonswap, args.balanced, args.packets)
    if args.output.exists():
        summary = json.loads((args.output / "complete.json").read_text())
        if summary["rows"] != 16_185_600 or summary["source_paths"] != list(map(str, source_paths)) or summary["new_stokes"] != str(args.new_stokes) or summary["new_stokes_split_seed"] != args.seed:
            raise ValueError("Completed dataset does not match this assembly request")
        for name, expected in summary["files"].items():
            path = args.output / name
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Completed dataset file changed: {path}")
        print(json.dumps(summary, indent=2), flush=True)
        raise SystemExit(0)
    if not (args.jonswap / "complete.json").is_file():
        raise ValueError("JONSWAP replay is not complete")

    names = ("eta", "xi", "gxi", "depth", "time", "frame_index", "simulation_id", "family_id", "parameter_group_id", "dataset_split")
    sources = [
        {name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in (*names, "x")}
        for path in source_paths
    ]
    original, jonswap, balanced, packets = sources
    expected_rows = (7_686_144, 3_686_400, 1_179_648, 1_440_000)
    for path, source, count in zip(source_paths, sources, expected_rows, strict=True):
        for name in names:
            expected_shape = (count, 1024) if name in {"eta", "xi", "gxi"} else (count,)
            if source[name].shape != expected_shape:
                raise ValueError(f"Unexpected {name} shape in {path}")
        grid = source["x"]
        if not (np.array_equal(grid, original["x"]) or np.array_equal(grid, original["x"].astype(np.float32).astype(np.float64))):
            raise ValueError(f"Different spatial grid in {path}")
        ids, starts, counts = np.unique(source["simulation_id"], return_index=True, return_counts=True)
        if not np.array_equal(source["simulation_id"], np.repeat(ids, counts)):
            raise ValueError(f"Simulation rows must be contiguous and ID-ordered in {path}")
        split = source["dataset_split"]
        if not np.isin(split, ("train", "validation", "test")).all() or not np.array_equal(split, np.repeat(split[starts], counts)):
            raise ValueError(f"Invalid or simulation-overlapping splits in {path}")
        if not np.array_equal(source["family_id"], np.repeat(source["family_id"][starts], counts)):
            raise ValueError(f"A simulation spans multiple families in {path}")

    original_ids, original_starts = np.unique(original["simulation_id"], return_index=True)
    if not np.array_equal(original_ids, np.arange(73_728)):
        raise ValueError("Original simulation IDs changed")
    jonswap_ids, jonswap_starts, jonswap_counts = np.unique(jonswap["simulation_id"], return_index=True, return_counts=True)
    old_jonswap = original_starts[original["family_id"][original_starts] == 4]
    if not np.array_equal(jonswap_ids, original["simulation_id"][old_jonswap]) or not np.all(jonswap_counts == 200):
        raise ValueError("JONSWAP must contain the original simulations with 200 rows each")
    for name in ("depth", "family_id", "parameter_group_id", "dataset_split"):
        if not np.array_equal(jonswap[name][jonswap_starts], original[name][old_jonswap]):
            raise ValueError(f"JONSWAP changed original {name}")
    old_rows = np.flatnonzero(original["family_id"] != 4)
    fresh_rows = np.flatnonzero((balanced["family_id"] == 1) & (balanced["simulation_id"] >= 73_728))
    if not np.array_equal(balanced["simulation_id"][fresh_rows], np.arange(73_728, 350_208)):
        raise ValueError("Existing fresh Stokes simulations changed")
    if not np.array_equal(np.unique(packets["simulation_id"]), np.arange(12_000)):
        raise ValueError("Packet simulation IDs changed")
    if not np.array_equal(np.unique(packets["family_id"], return_counts=True), (np.array([11, 12, 13]), np.array([480_000] * 3))):
        raise ValueError("Expected all three original packet groups")

    # Read small NPZ metadata first; validate the numerical fields while copying.
    batches = sorted(args.new_stokes.rglob("batch_*.npz"))
    new_count = 0
    run_directories: dict[int, Path] = {}
    for path in batches:
        with np.load(path, allow_pickle=False) as batch:
            if int(batch["family_id"]) != 1:
                raise ValueError(f"Not a Stokes batch: {path}")
            seed = int(batch["seed"])
            if run_directories.setdefault(seed, path.parent) != path.parent:
                raise ValueError("Duplicate seeded Stokes runs")
            if "simulation_local_index" in batch:
                local = batch["simulation_local_index"]
                if np.unique(local).size != local.size:
                    raise ValueError("New static Stokes simulations must have one row each")
                new_count += local.size
    if new_count != 3_391_488:
        raise ValueError(f"Expected 3391488 additional Stokes simulations, found {new_count}")
    new_splits = np.full(new_count, "train", dtype="U10")
    order = np.random.default_rng(args.seed).permutation(new_count)
    held_out = new_count // 10
    new_splits[order[-2 * held_out:-held_out]] = "validation"
    new_splits[order[-held_out:]] = "test"
    packet_offset = 350_208 + new_count
    total = 16_185_600
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        directory = Path(temporary)
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy", mode="w+",
                dtype=np.result_type(*(source[name].dtype for source in sources)),
                shape=(total, 1024) if name in {"eta", "xi", "gxi"} else (total,),
            ) for name in names
        }
        first = 0
        selections = (old_rows, np.arange(len(jonswap["eta"])), fresh_rows)
        for path, source, selected in zip(source_paths[:3], sources[:3], selections, strict=True):
            for start in range(0, selected.size, 8192):
                rows = selected[start:start + 8192]
                first = write_rows(arrays, first, {name: source[name][rows] for name in names})
            print(f"Copied {selected.size} rows from {path}", flush=True)

        next_stokes = 0
        for index, path in enumerate(batches):
            batch = load_completed_batch(path)
            if batch.shard is None:
                continue
            shard = batch.shard
            count = len(shard["eta"])
            values: dict[str, np.ndarray] = {name: shard[name] for name in ("eta", "xi", "gxi", "depth", "time", "frame_index")}
            values.update(
                simulation_id=350_208 + np.arange(next_stokes, next_stokes + count),
                family_id=np.full(count, 1, dtype=np.int16),
                parameter_group_id=np.asarray(batch.parameter_group_ids)[shard["simulation_local_index"]],
                dataset_split=new_splits[next_stokes:next_stokes + count],
            )
            first = write_rows(arrays, first, values)
            next_stokes += count
            if index % 1024 == 0:
                print(f"Copied {next_stokes}/{new_count} new Stokes simulations", flush=True)
        for start in range(0, len(packets["eta"]), 8192):
            values = {name: packets[name][start:start + 8192] for name in names}
            values["simulation_id"] = values["simulation_id"] + packet_offset
            first = write_rows(arrays, first, values)
        if first != total or next_stokes != new_count:
            raise ValueError("Final row count mismatch")
        family_ids, family_counts = np.unique(arrays["family_id"], return_counts=True)
        if not np.array_equal(family_ids, [1, 2, 3, 4, 11, 12, 13]) or not np.array_equal(family_counts, [3_686_400] * 4 + [480_000] * 3):
            raise ValueError("Final family counts mismatch")
        if not np.array_equal(np.unique(arrays["simulation_id"]), np.arange(packet_offset + 12_000)):
            raise ValueError("Final simulation IDs overlap or have missing simulations")
        for simulation_id in (16_471, 16_624):
            old = original_starts[simulation_id]
            new = np.flatnonzero(arrays["simulation_id"] == simulation_id)[0]
            if arrays["dataset_split"][new] != "test" or not all(np.array_equal(arrays[name][new], original[name][old]) for name in ("eta", "xi", "gxi", "depth", "time")):
                raise ValueError(f"Changed original failed IC {simulation_id}")
        for array in arrays.values():
            array.flush()
        np.save(directory / "x.npy", original["x"])
        summary = {
            "rows": total, "simulations": packet_offset + 12_000,
            "rows_per_family": dict(zip(map(str, family_ids), map(int, family_counts))),
            "rows_per_split": {split: int(np.count_nonzero(arrays["dataset_split"] == split)) for split in ("train", "validation", "test")},
            "original_ids_and_splits_preserved": True, "packet_simulation_id_offset": packet_offset,
            "new_stokes_split_seed": args.seed, "new_stokes_generation_seeds": sorted(run_directories), "additional_stokes_simulations": new_count,
            "source_paths": list(map(str, source_paths)), "new_stokes": str(args.new_stokes),
            "copied_fields_finite_and_exact": True,
            "files": {f"{name}.npy": {"shape": list(array.shape), "dtype": str(array.dtype), "bytes": (directory / f"{name}.npy").stat().st_size} for name, array in {**arrays, "x": original["x"]}.items()},
        }
        (directory / "complete.json").write_text(json.dumps(summary, indent=2) + "\n")
        directory.rename(args.output)
    print(json.dumps(summary, indent=2), flush=True)
