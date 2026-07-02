"""Stream a sharded Tanaka NPZ twice to build an exact subsampled dataset."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from .clean_tanaka_dataset import list_source_tags, read_npy, write_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact two-pass streaming subsample of a Tanaka dataset without loading the full archive."
    )
    parser.add_argument("--input", default="data/tanaka_1_clean.npz")
    parser.add_argument("--output", default=None, help="Output path (default: input stem + _sub{target}k.npz)")
    parser.add_argument("--target", type=int, default=500_000, help="Number of rows to keep after subsampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=50_000, help="Rows per output shard")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def count_rows_per_tag(zf: zipfile.ZipFile, tags: list[str]) -> list[int]:
    row_counts: list[int] = []
    for i, tag in enumerate(tags):
        case_ids = read_npy(zf, f"case_id_batch_{tag}.npy")
        row_counts.append(int(case_ids.shape[0]))
        if (i + 1) % 10 == 0 or i + 1 == len(tags):
            print(
                json.dumps(
                    {
                        "phase": "count_rows",
                        "processed_shards": i + 1,
                        "total_shards": len(tags),
                        "rows_seen": int(sum(row_counts)),
                    }
                ),
                flush=True,
            )
    return row_counts


def allocate_memmaps(tmp_dir: Path, target: int, nx: int) -> dict[str, np.memmap]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "eta": npy_format.open_memmap(tmp_dir / "eta.npy", mode="w+", dtype=np.float32, shape=(target, nx)),
        "xi": npy_format.open_memmap(tmp_dir / "xi.npy", mode="w+", dtype=np.float32, shape=(target, nx)),
        "gxi": npy_format.open_memmap(tmp_dir / "gxi.npy", mode="w+", dtype=np.float32, shape=(target, nx)),
        "time": npy_format.open_memmap(tmp_dir / "time.npy", mode="w+", dtype=np.float32, shape=(target,)),
        "case_id": npy_format.open_memmap(tmp_dir / "case_id.npy", mode="w+", dtype=np.int64, shape=(target,)),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    target = args.target
    suffix = f"_sub{target // 1000}k"
    output_path = Path(args.output).resolve() if args.output else input_path.with_name(f"{input_path.stem}{suffix}.npz")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace.")
    if output_path.exists():
        output_path.unlink()

    rng = np.random.default_rng(args.seed)
    tmp_dir = output_path.with_suffix(".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    with zipfile.ZipFile(input_path, mode="r") as zf:
        source_meta = json.loads(zf.read("meta.json"))
        tags = list_source_tags(zf)
        x = read_npy(zf, "x.npy").astype(np.float32)
        subsample_indices = read_npy(zf, "subsample_indices.npy") if "subsample_indices.npy" in zf.namelist() else None
        subsample_times = read_npy(zf, "subsample_times.npy") if "subsample_times.npy" in zf.namelist() else None
        row_counts = count_rows_per_tag(zf, tags)

    total_rows = int(sum(row_counts))
    if target > total_rows:
        raise ValueError(f"target ({target}) exceeds available rows ({total_rows})")

    chosen_global = rng.choice(total_rows, size=target, replace=False)
    sorted_order = np.argsort(chosen_global)
    chosen_sorted = chosen_global[sorted_order]
    output_positions = sorted_order.astype(np.int64, copy=False)

    memmaps = allocate_memmaps(tmp_dir, target, int(x.shape[0]))
    write_cursor = 0
    global_start = 0

    try:
        with zipfile.ZipFile(input_path, mode="r") as zf:
            for shard_idx, (tag, row_count) in enumerate(zip(tags, row_counts, strict=True)):
                global_end = global_start + row_count
                left = int(np.searchsorted(chosen_sorted, global_start, side="left"))
                right = int(np.searchsorted(chosen_sorted, global_end, side="left"))
                if right > left:
                    local_rows = (chosen_sorted[left:right] - global_start).astype(np.int64, copy=False)
                    dest_rows = output_positions[left:right]

                    eta = read_npy(zf, f"eta_batch_{tag}.npy").astype(np.float32, copy=False)
                    xi = read_npy(zf, f"xi_batch_{tag}.npy").astype(np.float32, copy=False)
                    gxi = read_npy(zf, f"gxi_batch_{tag}.npy").astype(np.float32, copy=False)
                    time = read_npy(zf, f"time_batch_{tag}.npy").astype(np.float32, copy=False)
                    case_id = read_npy(zf, f"case_id_batch_{tag}.npy").astype(np.int64, copy=False)

                    memmaps["eta"][dest_rows] = eta[local_rows]
                    memmaps["xi"][dest_rows] = xi[local_rows]
                    memmaps["gxi"][dest_rows] = gxi[local_rows]
                    memmaps["time"][dest_rows] = time[local_rows]
                    memmaps["case_id"][dest_rows] = case_id[local_rows]
                    write_cursor += int(dest_rows.shape[0])

                global_start = global_end
                if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == len(tags):
                    print(
                        json.dumps(
                            {
                                "phase": "stream_select",
                                "processed_shards": shard_idx + 1,
                                "total_shards": len(tags),
                                "rows_written": write_cursor,
                                "target_rows": target,
                            }
                        ),
                        flush=True,
                    )

        if write_cursor != target:
            raise RuntimeError(f"Selected rows written ({write_cursor}) did not match target ({target})")

        n_shards = (target + args.shard_size - 1) // args.shard_size
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            write_npy(zf, "x.npy", x)
            if subsample_indices is not None:
                write_npy(zf, "subsample_indices.npy", subsample_indices)
            if subsample_times is not None:
                write_npy(zf, "subsample_times.npy", subsample_times)

            for shard_id in range(n_shards):
                start = shard_id * args.shard_size
                end = min(start + args.shard_size, target)
                tag = f"{shard_id:04d}"
                write_npy(zf, f"eta_batch_{tag}.npy", memmaps["eta"][start:end])
                write_npy(zf, f"xi_batch_{tag}.npy", memmaps["xi"][start:end])
                write_npy(zf, f"gxi_batch_{tag}.npy", memmaps["gxi"][start:end])
                write_npy(zf, f"time_batch_{tag}.npy", memmaps["time"][start:end])
                write_npy(zf, f"case_id_batch_{tag}.npy", memmaps["case_id"][start:end])

            out_meta = dict(source_meta)
            out_meta.update(
                {
                    "source_dataset": str(input_path),
                    "target_samples": target,
                    "subsample_seed": args.seed,
                    "rows_before_subsample": total_rows,
                    "rows_after_subsample": target,
                    "n_batches_planned": n_shards,
                    "shard_size": args.shard_size,
                    "subsample_method": "exact_two_pass_streaming_without_replacement",
                    "subsample_order": "random_order_from_rng_choice",
                    "n_unique_case_ids": int(np.unique(np.asarray(memmaps["case_id"])).shape[0]),
                }
            )
            zf.writestr("meta.json", json.dumps(out_meta, indent=2))

        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "seed": args.seed,
            "rows_loaded": total_rows,
            "target": target,
            "n_shards": n_shards,
            "shard_size": args.shard_size,
            "n_unique_case_ids": int(np.unique(np.asarray(memmaps["case_id"])).shape[0]),
            "case_id_min": int(np.min(np.asarray(memmaps["case_id"]))),
            "case_id_max": int(np.max(np.asarray(memmaps["case_id"]))),
            "time_min": float(np.min(np.asarray(memmaps["time"]))),
            "time_max": float(np.max(np.asarray(memmaps["time"]))),
            "nx": int(x.shape[0]),
            "subsample_method": "exact_two_pass_streaming_without_replacement",
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        for array in memmaps.values():
            mmap_obj = getattr(array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        del memmaps
        gc.collect()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
