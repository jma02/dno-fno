"""Assemble combined_dataset_v8: v5 + shallow_steep_wide_v1 (additive, no trim).

v7_trim regressed cs_dno rollouts 2-5x vs v5 (memory: project-v7-trim-failed).
v8 keeps v5 intact and appends shallow_steep_wide_v1 as a new source id.

Usage:
    uv run python -m solver.gen_data.build_combined_v8 \\
        --v5 data/combined_dataset_v5.npz \\
        --shards data/shallow_steep_wide_v1_shard00.npz \\
                 data/shallow_steep_wide_v1_shard01.npz \\
        --output data/combined_dataset_v8.npz
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.lib import format as npy_format

from solver.gen_data.build_combined_v7_trim import (
    FIELDS_1D,
    FIELDS_2D,
    read_member_header,
    shard_batch_names,
    stream_member_payload,
    stream_rows_to_npy,
    write_npy_header,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v5", default="data/combined_dataset_v5.npz")
    p.add_argument(
        "--shards", nargs="+",
        default=[
            "data/shallow_steep_wide_v1_shard00.npz",
            "data/shallow_steep_wide_v1_shard01.npz",
        ],
    )
    p.add_argument("--output", default="data/combined_dataset_v8.npz")
    p.add_argument("--new_source_id", type=int, default=13)
    p.add_argument("--new_source_name", default="shallow_steep_wide")
    p.add_argument("--version", default="v8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v5_path = Path(args.v5)
    shard_paths = [Path(s) for s in args.shards]
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove first.")

    t0 = perf_counter()
    print(f"loading v5 source array ({v5_path})...", flush=True)
    with np.load(v5_path, mmap_mode="r") as d:
        src = np.asarray(d["source"])
    n_v5 = src.size
    print(f"  v5 rows={n_v5:,}", flush=True)

    print("inspecting v5 member shapes/dtypes...", flush=True)
    with zipfile.ZipFile(v5_path) as zv5:
        v5_shapes: dict[str, tuple[tuple[int, ...], np.dtype]] = {
            f: read_member_header(zv5, f"{f}.npy") for f in (*FIELDS_2D, *FIELDS_1D, "source")
        }
    nx = v5_shapes["eta"][0][1]

    print("inspecting shard rows...", flush=True)
    shard_info: list[dict[str, object]] = []
    n_new = 0
    for sp in shard_paths:
        with zipfile.ZipFile(sp) as zs:
            names = {f: shard_batch_names(zs, f) for f in (*FIELDS_2D, *FIELDS_1D)}
            rows = 0
            for nm in names["eta"]:
                shape, dtype = read_member_header(zs, nm)
                if shape[1] != nx or dtype != np.dtype("<f4"):
                    raise ValueError(f"{sp}:{nm} shape/dtype mismatch: {shape} {dtype}")
                rows += shape[0]
            meta = json.loads(zs.read("meta.json"))
        shard_info.append({"path": str(sp), "rows": rows, "names": names, "meta": meta})
        n_new += rows
    n_total = n_v5 + n_new
    print(f"  shards add {n_new:,} new rows  -> total {n_total:,}", flush=True)

    keep_indices = np.arange(n_v5, dtype=np.int64)

    print("writing output...", flush=True)
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
        for field in (*FIELDS_2D, *FIELDS_1D, "source"):
            is_2d = field in FIELDS_2D
            dtype = np.dtype("<i1") if field == "source" else np.dtype("<f4")
            shape = (n_total, nx) if is_2d else (n_total,)
            row_bytes = nx * 4 if is_2d else dtype.itemsize
            print(f"  field={field} (shape={shape}, row_bytes={row_bytes})", flush=True)
            with zout.open(f"{field}.npy", mode="w", force_zip64=True) as out:
                write_npy_header(out, shape, dtype)
                if field == "source":
                    out.write(src.astype(np.int8, copy=False).tobytes())
                else:
                    stream_rows_to_npy(v5_path, f"{field}.npy", keep_indices,
                                        out, dtype, row_bytes, desc=f"v5 {field}")
                if field == "source":
                    fill_size = min(n_new, 4_000_000)
                    fill = np.full(fill_size, args.new_source_id, dtype=np.int8).tobytes()
                    written = 0
                    while written < n_new:
                        take = min(n_new - written, len(fill))
                        out.write(fill[:take])
                        written += take
                else:
                    for info in shard_info:
                        with zipfile.ZipFile(Path(str(info["path"]))) as zs:
                            for nm in info["names"][field]:  # type: ignore[index]
                                stream_member_payload(zs, nm, out)

        with zipfile.ZipFile(v5_path) as zv5:
            with zv5.open("x.npy") as f:
                x = np.load(f, allow_pickle=False)
        with zout.open("x.npy", mode="w") as out:
            npy_format.write_array(out, np.asarray(x), allow_pickle=False)

    v5_meta = json.loads(v5_path.with_suffix(".meta.json").read_text())
    legend = dict(v5_meta["source_legend"])
    legend[str(args.new_source_id)] = args.new_source_name
    counts: dict[str, int] = dict(v5_meta.get("source_counts", {}))
    counts[args.new_source_name] = n_new
    meta = {
        **v5_meta,
        "kind": "combined_dataset",
        "version": args.version,
        "n_samples": n_total,
        "source_legend": legend,
        "source_counts": counts,
        "appended_unshuffled_tail": args.new_source_name,
        "v5_path": str(v5_path),
        "shards": [{"path": str(i["path"]), "rows": i["rows"], "meta": i["meta"]} for i in shard_info],
        "fields": {
            **{f: {"shape": [n_total, nx], "dtype": "float32"} for f in FIELDS_2D},
            **{f: {"shape": [n_total], "dtype": "float32"} for f in FIELDS_1D},
            "source": {"shape": [n_total], "dtype": "int8"},
        },
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print("verifying output...", flush=True)
    with zipfile.ZipFile(out_path) as zo:
        for field in (*FIELDS_2D, *FIELDS_1D, "source"):
            shape, _ = read_member_header(zo, f"{field}.npy")
            expect = (n_total, nx) if field in FIELDS_2D else (n_total,)
            if shape != expect:
                raise SystemExit(f"VERIFY FAIL: {field} shape {shape} != {expect}")
    print(f"verify OK: total rows = {n_total:,}", flush=True)
    print(f"wrote {out_path} in {perf_counter() - t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
