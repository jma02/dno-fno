"""Build a balanced fine-tune dataset mixing v8 rows with precascade rows.

The base v8 dataset is a very large ZIP_STORED npz, so this script memory maps
its arrays in place and samples only the requested rows.  The output is a flat
npz plus sidecar metadata/stats compatible with train-jax-10m/util.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("data/combined_dataset_v8.npz"))
    p.add_argument("--target", type=Path, default=Path("data/tanaka_precascade_c2_v1.npz"))
    p.add_argument("--output", type=Path, default=Path("data/balanced_precascade_c2_v1_100k.npz"))
    p.add_argument("--n_base", type=int, default=90_000)
    p.add_argument("--n_target", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260708)
    p.add_argument("--target_source_id", type=int, default=50)
    return p.parse_args()


def _entry_payload_offset(npz_path: Path, member: str) -> int:
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member} is compressed; cannot mmap")
        header_offset = info.header_offset

    with npz_path.open("rb") as fh:
        fh.seek(header_offset)
        local_header = fh.read(30)
    if local_header[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local zip header for {npz_path}:{member}")
    filename_len, extra_len = struct.unpack("<HH", local_header[26:30])
    return header_offset + 30 + filename_len + extra_len


def _mmap_npy_member(npz_path: Path, member: str) -> np.memmap:
    payload_offset = _entry_payload_offset(npz_path, member)
    with npz_path.open("rb") as fh:
        fh.seek(payload_offset)
        version = npy_format.read_magic(fh)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"unsupported npy version {version} in {npz_path}:{member}")
        data_offset = fh.tell()
    if fortran_order:
        raise ValueError(f"{npz_path}:{member} is Fortran-order; unsupported")
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def _read_shard_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive, zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        n_shards = int(meta["n_batches_planned"])
        rows: dict[str, list[np.ndarray]] = {k: [] for k in ("eta", "xi", "gxi", "depth", "time")}
        for shard_id in range(n_shards):
            tag = f"{shard_id:04d}"
            rows["eta"].append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            rows["xi"].append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            rows["gxi"].append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
            rows["depth"].append(np.asarray(archive[f"depth_batch_{tag}"], dtype=np.float32))
            rows["time"].append(np.asarray(archive[f"time_batch_{tag}"], dtype=np.float32))
        x = np.asarray(archive["x"], dtype=np.float32)
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()} | {"x": x}


def _load_sidecar_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_meta_path = args.base.with_suffix(".meta.json")
    base_stats_path = args.base.with_suffix(".stats.json")
    if not base_meta_path.exists():
        raise FileNotFoundError(base_meta_path)
    if not base_stats_path.exists():
        raise FileNotFoundError(base_stats_path)

    base_meta = _load_sidecar_json(base_meta_path)
    base_n = int(base_meta["n_samples"])
    if args.n_base > base_n:
        raise ValueError(f"requested {args.n_base} base rows but only {base_n} available")

    base_idx = np.sort(rng.choice(base_n, size=args.n_base, replace=False))
    base_arrays = {
        "eta": _mmap_npy_member(args.base, "eta.npy"),
        "xi": _mmap_npy_member(args.base, "xi.npy"),
        "gxi": _mmap_npy_member(args.base, "gxi.npy"),
        "depth": _mmap_npy_member(args.base, "depth.npy"),
        "time": _mmap_npy_member(args.base, "time.npy"),
        "source": _mmap_npy_member(args.base, "source.npy"),
        "x": _mmap_npy_member(args.base, "x.npy"),
    }
    target = _read_shard_dataset(args.target)
    target_idx = rng.choice(target["eta"].shape[0], size=args.n_target, replace=True)

    eta = np.concatenate([np.asarray(base_arrays["eta"][base_idx]), target["eta"][target_idx]], axis=0)
    xi = np.concatenate([np.asarray(base_arrays["xi"][base_idx]), target["xi"][target_idx]], axis=0)
    gxi = np.concatenate([np.asarray(base_arrays["gxi"][base_idx]), target["gxi"][target_idx]], axis=0)
    depth = np.concatenate([np.asarray(base_arrays["depth"][base_idx]), target["depth"][target_idx]], axis=0)
    time = np.concatenate([np.asarray(base_arrays["time"][base_idx]), target["time"][target_idx]], axis=0)
    source = np.concatenate([
        np.asarray(base_arrays["source"][base_idx]),
        np.full((args.n_target,), args.target_source_id, dtype=np.int8),
    ])

    perm = rng.permutation(eta.shape[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        eta=eta[perm].astype(np.float32, copy=False),
        xi=xi[perm].astype(np.float32, copy=False),
        gxi=gxi[perm].astype(np.float32, copy=False),
        depth=depth[perm].astype(np.float32, copy=False),
        time=time[perm].astype(np.float32, copy=False),
        source=source[perm].astype(np.int8, copy=False),
        x=np.asarray(base_arrays["x"], dtype=np.float32),
    )

    out_meta = {
        "kind": "balanced_precascade_finetune",
        "nx": int(base_meta["nx"]),
        "length": float(base_meta.get("length", 2.0 * np.pi)),
        "n_samples": int(eta.shape[0]),
        "n_base": int(args.n_base),
        "n_target": int(args.n_target),
        "base": str(args.base),
        "target": str(args.target),
        "seed": int(args.seed),
        "target_source_id": int(args.target_source_id),
        "fields": {
            "eta": {"shape": list(eta.shape), "dtype": "float32"},
            "xi": {"shape": list(xi.shape), "dtype": "float32"},
            "gxi": {"shape": list(gxi.shape), "dtype": "float32"},
            "depth": {"shape": list(depth.shape), "dtype": "float32"},
            "time": {"shape": list(time.shape), "dtype": "float32"},
            "source": {"shape": list(source.shape), "dtype": "int8"},
        },
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    out_stats_path = args.output.with_suffix(".stats.json")
    shutil.copyfile(base_stats_path, out_stats_path)
    stats = _load_sidecar_json(out_stats_path)
    stats["dataset"] = args.output.name
    stats["num_examples"] = int(eta.shape[0])
    stats["stats_source"] = str(base_stats_path)
    out_stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output} with {args.n_base} base + {args.n_target} target rows")


if __name__ == "__main__":
    main()
