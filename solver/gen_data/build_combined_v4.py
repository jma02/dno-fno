"""Assemble combined_dataset_v4: v3 byte-identical + the shallow-steep family.

Streams payload bytes member-to-member (both archives are ZIP_STORED npy), so
peak RAM stays at the chunk size regardless of dataset size. New-family rows are
appended after v3's; no global reshuffle is needed because the trainer's
build_split_indices applies a seeded random permutation over all rows.

Usage:
    uv run python -m solver.gen_data.build_combined_v4 \\
        --v3 data/combined_dataset_v3.npz \\
        --shards data/shallow_steep_v1_shard00.npz data/shallow_steep_v1_shard01.npz \\
        --output data/combined_dataset_v4.npz
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3", default="data/combined_dataset_v3.npz")
    p.add_argument("--shards", nargs="+", default=[
        "data/shallow_steep_v1_shard00.npz",
        "data/shallow_steep_v1_shard01.npz",
    ])
    p.add_argument("--output", default="data/combined_dataset_v4.npz")
    p.add_argument("--new_source_id", type=int, default=11)
    p.add_argument("--new_source_name", default="shallow_steep")
    p.add_argument("--version", default="v4",
                   help="value written to meta['version'] of the output dataset")
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
    """Copy a member's array bytes (everything after its npy header) into out."""
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


def read_rows(path: Path, member: str, row_start: int, n_rows: int,
              row_bytes: int, dtype: np.dtype, header_skip_probe: bool = True) -> np.ndarray:
    """Random-access rows of a STORED npy member via seek (no full decompress)."""
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as f:
            version = npy_format.read_magic(f)
            reader = {(1, 0): npy_format.read_array_header_1_0,
                      (2, 0): npy_format.read_array_header_2_0}[version]
            shape, _, _ = reader(f)
            data_start = f.tell()
            f.seek(data_start + row_start * row_bytes)
            buf = f.read(n_rows * row_bytes)
    return np.frombuffer(buf, dtype=dtype).reshape(n_rows, -1)


def main() -> None:
    args = parse_args()
    v3_path = Path(args.v3)
    shard_paths = [Path(s) for s in args.shards]
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove it first.")

    t0 = perf_counter()
    v3_meta = json.loads(v3_path.with_suffix(".meta.json").read_text())
    with zipfile.ZipFile(v3_path) as zv3:
        v3_shapes = {f: read_member_header(zv3, f"{f}.npy") for f in (*FIELDS_2D, *FIELDS_1D, "source")}
    n_v3 = v3_shapes["eta"][0][0]
    nx = v3_shapes["eta"][0][1]

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
    n_total = n_v3 + n_new
    print(f"v3 rows={n_v3:,}  new rows={n_new:,}  total={n_total:,}", flush=True)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
        for field in (*FIELDS_2D, *FIELDS_1D, "source"):
            is_2d = field in FIELDS_2D
            dtype = np.dtype("<i1") if field == "source" else np.dtype("<f4")
            shape = (n_total, nx) if is_2d else (n_total,)
            t_field = perf_counter()
            with zout.open(f"{field}.npy", mode="w", force_zip64=True) as out:
                write_npy_header(out, shape, dtype)
                with zipfile.ZipFile(v3_path) as zv3:
                    stream_member_payload(zv3, f"{field}.npy", out)
                if field == "source":
                    fill = np.full(min(n_new, 4_000_000), args.new_source_id, dtype=np.int8).tobytes()
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
            print(f"  {field}: done in {perf_counter() - t_field:.1f}s", flush=True)

        with zipfile.ZipFile(v3_path) as zv3:
            with zv3.open("x.npy") as f:
                x = np.load(f, allow_pickle=False)
        with zout.open("x.npy", mode="w") as out:
            npy_format.write_array(out, np.asarray(x), allow_pickle=False)

    legend = dict(v3_meta["source_legend"])
    legend[str(args.new_source_id)] = args.new_source_name
    counts = dict(v3_meta["source_counts"])
    counts[args.new_source_name] = n_new
    meta = {
        **v3_meta,
        "kind": "combined_dataset",
        "version": args.version,
        "n_samples": n_total,
        "source_legend": legend,
        "source_counts": counts,
        "appended_unshuffled_tail": args.new_source_name,
        "v3_path": str(v3_path),
        "shards": [{"path": str(i["path"]), "rows": i["rows"], "meta": i["meta"]} for i in shard_info],
        "fields": {
            **{f: {"shape": [n_total, nx], "dtype": "float32"} for f in FIELDS_2D},
            **{f: {"shape": [n_total], "dtype": "float32"} for f in FIELDS_1D},
            "source": {"shape": [n_total], "dtype": "int8"},
        },
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    # --- verification: headers + row-level spot checks at both ends ---
    with zipfile.ZipFile(out_path) as zo:
        for field in (*FIELDS_2D, *FIELDS_1D, "source"):
            shape, _ = read_member_header(zo, f"{field}.npy")
            expect = (n_total, nx) if field in FIELDS_2D else (n_total,)
            if shape != expect:
                raise SystemExit(f"VERIFY FAIL: {field} shape {shape} != {expect}")

    rng = np.random.default_rng(0)
    row_bytes = nx * 4
    for ridx in rng.integers(0, n_v3, size=3):
        a = read_rows(out_path, "eta.npy", int(ridx), 1, row_bytes, np.dtype("<f4"))
        b = read_rows(v3_path, "eta.npy", int(ridx), 1, row_bytes, np.dtype("<f4"))
        if not np.array_equal(a, b):
            raise SystemExit(f"VERIFY FAIL: v3 region row {ridx} differs")
    first_info = shard_info[0]
    first_names: dict[str, list[str]] = first_info["names"]  # type: ignore[assignment]
    first_batch_shape, _ = read_member_header(zipfile.ZipFile(Path(str(first_info["path"]))), first_names["eta"][0])
    a = read_rows(out_path, "eta.npy", n_v3, 2, row_bytes, np.dtype("<f4"))
    b = read_rows(Path(str(first_info["path"])), first_names["eta"][0], 0, 2, row_bytes, np.dtype("<f4"))
    if not np.array_equal(a, b):
        raise SystemExit("VERIFY FAIL: first appended rows differ from shard00 batch0")
    src_tail = read_rows(out_path, "source.npy", n_total - 4, 4, 1, np.dtype("<i1"))
    if not np.all(src_tail == args.new_source_id):
        raise SystemExit(f"VERIFY FAIL: tail source ids {src_tail.ravel()} != {args.new_source_id}")
    src_head = read_rows(out_path, "source.npy", 0, 4, 1, np.dtype("<i1"))
    print(f"verify OK: shapes, v3 rows intact, appended rows intact, source head={src_head.ravel()} tail={src_tail.ravel()}")
    print(f"wrote {out_path} in {perf_counter() - t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
