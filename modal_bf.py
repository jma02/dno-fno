"""Modal app to generate adaptive-sampled BF rollouts on A100s, sharded.

Workflow:
    modal token new                                    # once

    # Fire off both shards in parallel:
    modal run modal_bf.py::go_wide --n-shards 2

    # When done, merge into a single canonical npz on the volume:
    modal run modal_bf.py::merge_shards

    # Pull back to local data/:
    modal run modal_bf.py::download

    # Peek at progress:
    modal run modal_bf.py::status

Seeding is disjoint from the local g0 (seed=42, rng_stream_id=0,
case_id_offset=0) and g1 (seed=7, rng_stream_id=1, case_id_offset=1_000_000)
runs: we use seed=13, rng_stream_id=99, case_id_offset=2_000_000.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "dno-bf"
VOLUME_NAME = "dno-bf-data"
GPU_TYPE = "A100-40GB"  # RTX 6000 Ada peaks ~37GB; envelope signal avoids the FFT-temp OOM, so 40GB fits.
TIMEOUT_SECONDS = 14 * 3600

REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy>=2.0.2",
        "scipy>=1.13",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "jax[cuda12]>=0.7",
    )
    .add_local_dir(REPO_ROOT / "solver", remote_path="/repo/solver")
    .add_local_file(REPO_ROOT / "pyproject.toml", remote_path="/repo/pyproject.toml")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOLUME_MOUNT = "/data"

TARGET_SAMPLES = 1_000_000
BATCH_SIZE = 256
KEEP_SAMPLES = 200
SAMPLES_PER_BATCH = BATCH_SIZE * KEEP_SAMPLES  # 51200
TOTAL_BATCHES = 20  # ceil(1_000_000 / 51_200)

# Seeding disjoint from local g0/g1.
MODAL_SEED = 13
MODAL_RNG_STREAM_ID = 99
MODAL_CASE_ID_OFFSET = 2_000_000

# Adaptive params — matches the local BF runs (envelope + magnitude + power=4).
ADAPTIVE_ALPHA = 0.85
ADAPTIVE_SMOOTH_SIGMA = 20.0
ADAPTIVE_POWER = 4.0


def _ensure_repo_on_path() -> None:
    import os
    import sys

    os.environ["PYTHONPATH"] = "/repo"
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if "/repo" not in sys.path:
        sys.path.insert(0, "/repo")


def _gen_argv(output: str, target_samples: int) -> list[str]:
    return [
        "generate_bf_dataset",
        "--output", output,
        "--target_samples", str(target_samples),
        "--batch_size", str(BATCH_SIZE),
        "--keep_samples", str(KEEP_SAMPLES),
        "--dt", "0.08",
        "--tmax", "200.0",
        "--seed", str(MODAL_SEED),
        "--rng_stream_id", str(MODAL_RNG_STREAM_ID),
        "--case_id_offset", str(MODAL_CASE_ID_OFFSET),
        "--rollout_dtype", "float64",
        "--adaptive",
        "--adaptive_signal", "envelope",
        "--adaptive_density_mode", "magnitude",
        "--adaptive_alpha", str(ADAPTIVE_ALPHA),
        "--adaptive_smooth_sigma", str(ADAPTIVE_SMOOTH_SIGMA),
        "--adaptive_power", str(ADAPTIVE_POWER),
    ]


def _preinit_shard_zip(output_path: str, start_batch: int, end_batch: int) -> None:
    """Pre-create the zip + state file so the generator resumes at start_batch.

    With state.json.next_batch=start_batch the generator skips ahead to
    start_batch and stops at n_batches_planned = end_batch. RNG is still
    keyed by (seed, rng_stream_id, batch_idx) → disjoint across shards as
    long as their batch ranges don't overlap.
    """
    import json
    import math
    import zipfile
    import numpy as np
    from numpy.lib import format as npy_format
    from solver.solvers.dno_series_jax import build_grid

    out = Path(output_path)
    state_path = out.with_suffix(".state.json")
    if state_path.exists():
        return  # resume from existing state

    x_grid, _ = build_grid(1024, 2.0 * math.pi)
    meta = {
        "shard_kind": "bf_adaptive_shard",
        "start_batch": start_batch,
        "n_batches_planned": end_batch,
        "batch_size": BATCH_SIZE,
        "keep_samples": KEEP_SAMPLES,
        "length": 2.0 * math.pi,
        "nx": 1024,
        "dt": 0.08,
        "tmax": 200.0,
        "seed": MODAL_SEED,
        "rng_stream_id": MODAL_RNG_STREAM_ID,
        "case_id_offset": MODAL_CASE_ID_OFFSET,
        "adaptive_sampling": True,
        "adaptive_signal": "envelope",
        "adaptive_density_mode": "magnitude",
        "adaptive_alpha": ADAPTIVE_ALPHA,
        "adaptive_smooth_sigma": ADAPTIVE_SMOOTH_SIGMA,
        "adaptive_power": ADAPTIVE_POWER,
    }
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open("x.npy", mode="w", force_zip64=True) as h:
            npy_format.write_array(h, np.asarray(x_grid, dtype=np.float32), allow_pickle=False)
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    state = {
        "output_path": str(out),
        "samples_written": start_batch * SAMPLES_PER_BATCH,
        "next_batch": start_batch,
        "n_batches_planned": end_batch,
        "complete": False,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


@app.function(
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    cpu=4,
    memory=32 * 1024,
)
def run_shard(shard_id: int, start_batch: int, end_batch: int) -> None:
    """One worker handling batches [start_batch, end_batch). Resume-safe."""
    import sys

    _ensure_repo_on_path()
    output = f"{VOLUME_MOUNT}/bf_2_adaptive_modal_shard_{shard_id:02d}.npz"

    # The generator stops at n_batches_planned = ceil(target_samples / samples_per_batch),
    # which we steer to end_batch via target_samples.
    target_samples = end_batch * SAMPLES_PER_BATCH

    _preinit_shard_zip(output, start_batch=start_batch, end_batch=end_batch)
    volume.commit()

    sys.argv = _gen_argv(output=output, target_samples=target_samples)
    from solver.gen_data.generate_bf_dataset import main as gen_main

    gen_main()
    volume.commit()


def _split_remaining(start_batch: int, total_batches: int, n_shards: int) -> list[tuple[int, int]]:
    remaining = total_batches - start_batch
    base = remaining // n_shards
    extra = remaining % n_shards
    slices: list[tuple[int, int]] = []
    cursor = start_batch
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue
        slices.append((cursor, cursor + size))
        cursor += size
    return slices


@app.local_entrypoint()
def go_wide(n_shards: int = 2) -> None:
    slices = _split_remaining(0, TOTAL_BATCHES, n_shards)
    print(f"Dispatching {len(slices)} BF shards on {GPU_TYPE}:")
    for i, (s, e) in enumerate(slices):
        print(f"  shard {i:02d}: batches [{s}, {e})  ({e-s} batches, ~{(e-s)*SAMPLES_PER_BATCH:,} samples)")
    args = [(i, s, e) for i, (s, e) in enumerate(slices)]
    list(run_shard.starmap(args))
    print("all shards finished.")


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=2 * 3600, cpu=4, memory=32 * 1024)
def merge_shards() -> None:
    """Concatenate every shard's batches into bf_2_adaptive_modal.npz on the volume."""
    import json
    import os
    import zipfile

    canonical = f"{VOLUME_MOUNT}/bf_2_adaptive_modal.npz"
    canonical_state = f"{VOLUME_MOUNT}/bf_2_adaptive_modal.state.json"

    shard_files = sorted(
        p for p in os.listdir(VOLUME_MOUNT)
        if p.startswith("bf_2_adaptive_modal_shard_") and p.endswith(".npz")
    )
    if not shard_files:
        raise RuntimeError("no shard npz files found on volume")
    print(f"found {len(shard_files)} shards: {shard_files}")

    samples_total = 0
    next_batch_max = 0
    seeded_canonical = False

    if Path(canonical).exists():
        os.remove(canonical)

    with zipfile.ZipFile(canonical, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as out_zf:
        existing: set[str] = set()
        for shard in shard_files:
            with zipfile.ZipFile(f"{VOLUME_MOUNT}/{shard}") as in_zf:
                for name in in_zf.namelist():
                    if name in ("x.npy", "meta.json"):
                        if not seeded_canonical:
                            out_zf.writestr(name, in_zf.read(name))
                            existing.add(name)
                            if name == "meta.json":
                                seeded_canonical = True
                        continue
                    if name in existing:
                        print(f"  skip dup {shard}::{name}")
                        continue
                    data = in_zf.read(name)
                    out_zf.writestr(name, data)
                    existing.add(name)
                    if name.startswith("eta_batch_") and name.endswith(".npy"):
                        import io
                        import numpy as np

                        arr = np.load(io.BytesIO(data))
                        samples_total += int(arr.shape[0])
                        bidx_str = name[len("eta_batch_"):-len(".npy")]
                        try:
                            next_batch_max = max(next_batch_max, int(bidx_str) + 1)
                        except ValueError:
                            pass

    state = {
        "output_path": canonical,
        "samples_written": samples_total,
        "next_batch": next_batch_max,
        "n_batches_planned": TOTAL_BATCHES,
        "complete": samples_total >= TARGET_SAMPLES,
        "merged_from_shards": shard_files,
    }
    Path(canonical_state).write_text(json.dumps(state, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps(state, indent=2))


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=600)
def status() -> dict:
    import json
    import os

    out: dict[str, dict] = {}
    for name in sorted(os.listdir(VOLUME_MOUNT)):
        if name.endswith(".state.json"):
            with open(f"{VOLUME_MOUNT}/{name}") as f:
                out[name] = json.load(f)
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def download(target_dir: str = "data", pattern: str = "bf_2_adaptive_modal") -> None:
    out = Path(target_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading volume {VOLUME_NAME!r} -> {out}/  (filter: {pattern!r})")
    for entry in volume.iterdir("/"):
        name = entry.path.lstrip("/")
        if pattern and pattern not in name:
            continue
        if not (name.endswith(".npz") or name.endswith(".state.json") or name.endswith(".json")):
            continue
        local = out / name
        local.parent.mkdir(parents=True, exist_ok=True)
        size_gb = (entry.size or 0) / 1e9
        print(f"  pulling {name}  ({size_gb:.2f} GB) -> {local}")
        with open(local, "wb") as f:
            for chunk in volume.read_file(name):
                f.write(chunk)
    print("done.")
