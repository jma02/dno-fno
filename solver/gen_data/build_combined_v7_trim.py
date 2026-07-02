"""Assemble combined_dataset_v7_trim: trimmed v5 + shallow_steep_wide.

Trim rules (set in TRIM_PLAN below):
  - DROP source ids in DROP_SIDS entirely.
  - For source ids in SUBSAMPLE_SIDS, keep every row with time==0 plus a
    seeded random 1/SUBSAMPLE_FRACTION of the t>0 rows. With FRACTION=5 on
    tanaka_g0/g1/bf_modal this cuts each trajectory to ~500 of 2501 timesteps
    (per-case coverage is uniform via random draw, NOT every-Nth-timestep,
    so no case is skipped).

Append a fresh shallow_steep_wide pack as a new source id at the tail.

Streams 2D field rows via random-access reads of v5's ZIP_STORED .npz members.
Writes a fresh v7_trim shuffled order would defeat the trainer's own
build_split_indices permutation, so we leave the trimmed-v5 region contiguous
and append the new pack unshuffled at the tail (matches v5's pattern).

Usage:
    uv run python -m solver.gen_data.build_combined_v7_trim \\
        --v5 data/combined_dataset_v5.npz \\
        --shards data/shallow_steep_wide_v1_shard00.npz \\
                 data/shallow_steep_wide_v1_shard01.npz \\
        --output data/combined_dataset_v7_trim.npz
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import numpy as np
from numpy.lib import format as npy_format

CHUNK_BYTES = 64 * 1024 * 1024
FIELDS_2D = ("eta", "xi", "gxi")
FIELDS_1D = ("depth", "time")

# Sources to remove entirely (5-cluster collapse: redundant in (h,kh,a/h)).
DROP_SIDS: set[int] = {0, 7, 8}  # random_sea_deep, bf_g0, bf_g1

# Sources to subsample temporally (keep all t=0 + random 1/N of t>0).
SUBSAMPLE_SIDS: dict[int, int] = {5: 5, 6: 5, 9: 5}  # tanaka_g0, tanaka_g1, bf_modal


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
    p.add_argument("--output", default="data/combined_dataset_v7_trim.npz")
    p.add_argument("--new_source_id", type=int, default=14)
    p.add_argument("--new_source_name", default="shallow_steep_wide")
    p.add_argument("--version", default="v7_trim")
    p.add_argument("--seed", type=int, default=20260617)
    return p.parse_args()


def read_member_header(zf: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype]:
    with zf.open(name) as f:
        version = npy_format.read_magic(f)
        reader = {(1, 0): npy_format.read_array_header_1_0,
                  (2, 0): npy_format.read_array_header_2_0}[version]
        shape, fortran, dtype = reader(f)
    if fortran:
        raise ValueError(f"{name}: fortran-order arrays unsupported")
    return shape, dtype


def write_npy_header(handle: BinaryIO, shape: tuple[int, ...], dtype: np.dtype) -> None:
    header = {"descr": npy_format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape}
    try:
        npy_format.write_array_header_1_0(handle, header)
    except ValueError:
        npy_format.write_array_header_2_0(handle, header)


def stream_member_payload(zf: zipfile.ZipFile, name: str, out: BinaryIO) -> int:
    copied = 0
    with zf.open(name) as f:
        version = npy_format.read_magic(f)
        reader = {(1, 0): npy_format.read_array_header_1_0,
                  (2, 0): npy_format.read_array_header_2_0}[version]
        reader(f)
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
    return copied


def shard_batch_names(zf: zipfile.ZipFile, field: str) -> list[str]:
    prefix = f"{field}_batch_"
    ids = sorted(int(n[len(prefix):-len(".npy")]) for n in zf.namelist()
                 if n.startswith(prefix) and n.endswith(".npy"))
    return [f"{prefix}{i:04d}.npy" for i in ids]


def member_data_offsets(zf: zipfile.ZipFile, name: str) -> tuple[int, np.dtype, tuple[int, ...]]:
    """Return (data_byte_offset_within_member, dtype, shape) for a ZIP_STORED npy member."""
    with zf.open(name) as f:
        version = npy_format.read_magic(f)
        reader = {(1, 0): npy_format.read_array_header_1_0,
                  (2, 0): npy_format.read_array_header_2_0}[version]
        shape, fortran, dtype = reader(f)
        data_start = f.tell()
        if fortran:
            raise ValueError("fortran-order arrays unsupported")
    return data_start, dtype, shape


def read_rows(path: Path, name: str, indices: np.ndarray, row_bytes: int,
              dtype: np.dtype) -> np.ndarray:
    """Random-access row read from a STORED npy member by seeking + reading row blocks."""
    n = indices.size
    if n == 0:
        return np.empty((0,), dtype=dtype)
    sample_shape: tuple[int, ...]
    with zipfile.ZipFile(path) as zf:
        data_start, member_dtype, shape = member_data_offsets(zf, name)
        if member_dtype != dtype:
            raise ValueError(f"{name}: dtype mismatch {member_dtype} != {dtype}")
        sample_shape = (n,) + tuple(shape[1:]) if len(shape) > 1 else (n,)
        out = np.empty(sample_shape, dtype=dtype)
        with zf.open(name) as f:
            current_pos = data_start
            f.read(0)  # noop; underlying _fileobj works through .seek but f doesn't seek directly
            # zipfile.ZipExtFile supports .seek for ZIP_STORED entries
            for i, ridx in enumerate(indices):
                pos = data_start + int(ridx) * row_bytes
                if pos != current_pos:
                    f.seek(pos)
                buf = f.read(row_bytes)
                if len(buf) != row_bytes:
                    raise IOError(f"{name}: short read at row {ridx}")
                out_view = out[i].view().reshape(-1) if out.ndim > 1 else out[i:i+1].view().reshape(-1)
                np.frombuffer(buf, dtype=dtype, count=row_bytes // dtype.itemsize, out=out_view)
                current_pos = pos + row_bytes
    return out


def stream_rows_to_npy(in_path: Path, in_name: str, indices: np.ndarray,
                      out_fp: BinaryIO, dtype: np.dtype, row_bytes: int,
                      desc: str = "") -> None:
    """Stream selected rows of a STORED npy member into an open output handle.

    indices: pre-sorted ascending row indices into the input member.
    out_fp: file-like positioned where to start writing row bytes (after npy header).
    """
    n = indices.size
    if n == 0:
        return
    t0 = perf_counter()
    with zipfile.ZipFile(in_path) as zf:
        data_start, member_dtype, _shape = member_data_offsets(zf, in_name)
        if member_dtype != dtype:
            raise ValueError(f"{in_name}: dtype mismatch {member_dtype} != {dtype}")
        with zf.open(in_name) as f:
            current = -1
            batch_step = 0
            for i, ridx in enumerate(indices):
                pos = data_start + int(ridx) * row_bytes
                if pos != current:
                    f.seek(pos)
                buf = f.read(row_bytes)
                if len(buf) != row_bytes:
                    raise IOError(f"{in_name}: short read at row {ridx}")
                out_fp.write(buf)
                current = pos + row_bytes
                batch_step += 1
                if batch_step % 250_000 == 0:
                    print(f"    {desc}: streamed {batch_step}/{n} rows ({100*batch_step/n:.1f}%)",
                          flush=True)
    print(f"  {desc}: {n} rows in {perf_counter()-t0:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    v5_path = Path(args.v5)
    shard_paths = [Path(s) for s in args.shards]
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove first.")

    t0 = perf_counter()
    print(f"loading v5 source + time arrays ({v5_path})...", flush=True)
    with np.load(v5_path, mmap_mode="r") as d:
        src = np.asarray(d["source"])  # int8 (small load)
        t = np.asarray(d["time"])      # float32 (~28MB)
    print(f"  v5 rows={src.size:,}", flush=True)

    print("building keep_mask...", flush=True)
    keep_mask = np.ones(src.size, dtype=bool)
    for sid in DROP_SIDS:
        keep_mask &= (src != sid)
    for sid, factor in SUBSAMPLE_SIDS.items():
        sid_mask = (src == sid)
        if not sid_mask.any():
            continue
        # t==0 always kept; t>0 randomly sampled at probability 1/factor
        t_pos = sid_mask & (t > 0.0)
        rand = rng.random(t_pos.sum())
        drop = np.zeros(src.size, dtype=bool)
        drop[t_pos] = rand >= (1.0 / factor)
        keep_mask &= ~drop
    keep_indices = np.flatnonzero(keep_mask).astype(np.int64)
    n_keep_v5 = keep_indices.size
    print(f"  trimmed v5: keep {n_keep_v5:,} of {src.size:,}", flush=True)
    # report per-source
    print("  per-source kept counts:")
    for sid in sorted(np.unique(src).tolist()):
        kept = int(((src == sid) & keep_mask).sum())
        total = int((src == sid).sum())
        print(f"    sid={sid:>3d}: kept {kept:>9,d} / {total:>9,d}  ({100*kept/max(total,1):5.1f}%)")

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
    n_total = n_keep_v5 + n_new
    print(f"  shards add {n_new:,} new rows  -> total {n_total:,}", flush=True)

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
                # 1. trimmed v5 region
                if field == "source":
                    src_chunk = src[keep_mask].astype(np.int8, copy=False)
                    out.write(src_chunk.tobytes())
                else:
                    stream_rows_to_npy(v5_path, f"{field}.npy", keep_indices,
                                        out, dtype, row_bytes, desc=f"v5 {field}")
                # 2. shard rows appended
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

        # x.npy from v5
        with zipfile.ZipFile(v5_path) as zv5:
            with zv5.open("x.npy") as f:
                x = np.load(f, allow_pickle=False)
        with zout.open("x.npy", mode="w") as out:
            npy_format.write_array(out, np.asarray(x), allow_pickle=False)

    # build meta
    v5_meta = json.loads(v5_path.with_suffix(".meta.json").read_text())
    legend = {k: v for k, v in v5_meta["source_legend"].items() if int(k) not in DROP_SIDS}
    legend[str(args.new_source_id)] = args.new_source_name
    # new counts: subsampled regimes get new counts; dropped removed; new family appended
    counts: dict[str, int] = {}
    for sid_str, name in legend.items():
        if name == args.new_source_name:
            counts[name] = n_new
            continue
        sid = int(sid_str)
        counts[name] = int(((src == sid) & keep_mask).sum())
    meta = {
        **v5_meta,
        "kind": "combined_dataset",
        "version": args.version,
        "n_samples": n_total,
        "source_legend": legend,
        "source_counts": counts,
        "appended_unshuffled_tail": args.new_source_name,
        "v5_path": str(v5_path),
        "trim_dropped_sids": sorted(DROP_SIDS),
        "trim_subsample_sids": {str(k): v for k, v in SUBSAMPLE_SIDS.items()},
        "shards": [{"path": str(i["path"]), "rows": i["rows"], "meta": i["meta"]} for i in shard_info],
        "fields": {
            **{f: {"shape": [n_total, nx], "dtype": "float32"} for f in FIELDS_2D},
            **{f: {"shape": [n_total], "dtype": "float32"} for f in FIELDS_1D},
            "source": {"shape": [n_total], "dtype": "int8"},
        },
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    # verify
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
