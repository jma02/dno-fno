"""Modal app to run random-sea dataset generation on a remote GPU, sharded wide.

Workflow:
    # 0. Authenticate once (uses ~/.modal.toml):
    modal token new

    # 1. Push your existing partial files into the Modal Volume (one-time):
    modal run modal_random_sea.py::upload_existing

    # 2. Fan out 10 H100 workers per regime to finish the remaining batches.
    # NB: Modal account is capped at 10 concurrent GPU workers, so run
    # the two regimes sequentially (not in parallel).
    modal run modal_random_sea.py::go_wide_deep --n-shards 10
    modal run modal_random_sea.py::go_wide_finite --n-shards 10

    # 3. Merge shards back into the canonical /data/random_sea_<regime>.npz:
    modal run modal_random_sea.py::merge_shards --regime deep
    modal run modal_random_sea.py::merge_shards --regime finite

    # 4. Pull merged outputs back to local data/:
    modal run modal_random_sea.py::download

    # Quick progress peek:
    modal run modal_random_sea.py::status

Resume safety: each shard has its own .state.json, so re-running go_wide_*
just resumes whatever didn't finish. RNG is seeded by (seed, rng_stream_id,
batch_idx), so disjoint batch_idx ranges across workers give disjoint ICs.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "dno-random-sea"
VOLUME_NAME = "dno-random-sea-data"
GPU_TYPE = "A100-40GB"
TIMEOUT_SECONDS = 12 * 3600

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
TOTAL_BATCHES = 391           # ceil(1_000_000 / (256 * 10))
TARGET_SAMPLES = 1_000_000
BATCH_SIZE = 256
KEEP_SAMPLES = 10
SAMPLES_PER_BATCH = BATCH_SIZE * KEEP_SAMPLES   # 2560


def _regime_params(regime: str) -> tuple[float, float, int]:
    if regime == "deep":
        return 5.0, 25.0, 100
    if regime == "finite":
        return 0.1, 1.5, 200
    raise ValueError(f"unknown regime: {regime}")


def _ensure_repo_on_path() -> None:
    import os, sys
    os.environ["PYTHONPATH"] = "/repo"
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if "/repo" not in sys.path:
        sys.path.insert(0, "/repo")


def _gen_argv(*, output: str, target_samples: int, depth_min: float, depth_max: float,
              rng_stream_id: int, case_id_offset: int = 0,
              batch_size: int = BATCH_SIZE, keep_samples: int = KEEP_SAMPLES) -> list[str]:
    return [
        "generate_random_sea_dataset",
        "--output", output,
        "--target_samples", str(target_samples),
        "--batch_size", str(batch_size),
        "--keep_samples", str(keep_samples),
        "--length", "6.283185307179586",
        "--nx", "1024",
        "--dt", "0.08",
        "--tmax", "20.0",
        "--Hs_lo", "0.005", "--Hs_hi", "0.03",
        "--kp_lo", "2.0", "--kp_hi", "12.0",
        "--bw_lo", "1.5", "--bw_hi", "5.0",
        "--depth_min", str(depth_min), "--depth_max", str(depth_max),
        "--seed", "42",
        "--rng_stream_id", str(rng_stream_id),
        "--case_id_offset", str(case_id_offset),
        "--rollout_dtype", "float64",
    ]


def _preinit_shard_zip(output_path: str, start_batch: int, samples_per_batch: int,
                       n_batches_planned: int, regime: str, rng_stream_id: int,
                       depth_min: float, depth_max: float) -> None:
    """Pre-create the zip + state file so the generator's main() resumes at start_batch."""
    import json
    import math
    import zipfile
    import numpy as np
    from numpy.lib import format as npy_format
    from solver.solvers.dno_series_jax import build_grid

    out = Path(output_path)
    state_path = out.with_suffix(".state.json")
    if state_path.exists():
        return  # nothing to do, generator will resume from existing state

    x_grid, _ = build_grid(1024, 2.0 * math.pi)
    meta = {
        "shard_kind": "random_sea_shard",
        "regime": regime,
        "rng_stream_id": rng_stream_id,
        "depth_min": depth_min, "depth_max": depth_max,
        "start_batch": start_batch,
        "n_batches_planned": n_batches_planned,
        "batch_size": BATCH_SIZE, "keep_samples": KEEP_SAMPLES,
        "length": 2.0 * math.pi, "nx": 1024, "dt": 0.08, "tmax": 20.0,
        "Hs_lo": 0.005, "Hs_hi": 0.03, "kp_lo": 2.0, "kp_hi": 12.0,
        "bw_lo": 1.5, "bw_hi": 5.0,
        "seed": 42,
    }
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open("x.npy", mode="w", force_zip64=True) as h:
            npy_format.write_array(h, np.asarray(x_grid, dtype=np.float32), allow_pickle=False)
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    state = {
        "output_path": str(out),
        "samples_written": start_batch * samples_per_batch,
        "next_batch": start_batch,
        "n_batches_planned": n_batches_planned,
        "complete": False,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


@app.function(
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    cpu=4,
    memory=16 * 1024,
)
def run_shard(regime: str, shard_id: int, start_batch: int, end_batch: int) -> None:
    """One worker handling batches [start_batch, end_batch). Resume-safe."""
    import sys
    _ensure_repo_on_path()
    depth_min, depth_max, rng_stream_id = _regime_params(regime)
    output = f"{VOLUME_MOUNT}/random_sea_{regime}_shard_{shard_id:02d}.npz"

    # planned n_batches inside the generator = ceil(target_samples / samples_per_batch).
    # We pick target_samples so n_batches_planned == end_batch.
    target_samples = end_batch * SAMPLES_PER_BATCH

    _preinit_shard_zip(
        output_path=output, start_batch=start_batch,
        samples_per_batch=SAMPLES_PER_BATCH, n_batches_planned=end_batch,
        regime=regime, rng_stream_id=rng_stream_id,
        depth_min=depth_min, depth_max=depth_max,
    )
    volume.commit()

    sys.argv = _gen_argv(
        output=output, target_samples=target_samples,
        depth_min=depth_min, depth_max=depth_max, rng_stream_id=rng_stream_id,
        case_id_offset=0,
    )
    from solver.gen_data.generate_random_sea_dataset import main as gen_main
    gen_main()
    volume.commit()


def _split_remaining(start_batch: int, total_batches: int, n_shards: int) -> list[tuple[int, int]]:
    """Divide [start_batch, total_batches) into n_shards roughly-equal contiguous slices."""
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


def _resume_start_batch(regime: str) -> int:
    """Read the canonical /data/random_sea_<regime>.state.json on the volume, if present."""
    import json
    state_path = f"{VOLUME_MOUNT}/random_sea_{regime}.state.json"
    if not Path(state_path).exists():
        return 0
    with open(state_path) as f:
        state = json.load(f)
    return int(state.get("next_batch", 0))


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=300)
def plan_shards(regime: str, n_shards: int) -> dict:
    """Returns the list of (start_batch, end_batch) the workers will run."""
    start = _resume_start_batch(regime)
    slices = _split_remaining(start, TOTAL_BATCHES, n_shards)
    plan = {"regime": regime, "n_shards": len(slices), "main_done_through_batch": start, "slices": slices}
    print(plan)
    return plan


@app.local_entrypoint()
def go_wide_deep(n_shards: int = 10) -> None:
    _go_wide("deep", n_shards)


@app.local_entrypoint()
def go_wide_finite(n_shards: int = 10) -> None:
    _go_wide("finite", n_shards)


def _go_wide(regime: str, n_shards: int) -> None:
    plan = plan_shards.remote(regime, n_shards)
    slices = plan["slices"]
    print(f"[{regime}] dispatching {len(slices)} shards on {GPU_TYPE} workers:")
    for i, (s, e) in enumerate(slices):
        print(f"  shard {i:02d}: batches [{s}, {e})  ({e-s} batches, ~{(e-s)*SAMPLES_PER_BATCH:,} samples)")
    args = [(regime, i, s, e) for i, (s, e) in enumerate(slices)]
    list(run_shard.starmap(args))
    print(f"[{regime}] all shards finished.")


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=2 * 3600, cpu=4, memory=32 * 1024)
def merge_shards(regime: str) -> None:
    """Append every shard's batches into the canonical /data/random_sea_<regime>.npz."""
    import json
    import os
    import shutil
    import zipfile

    canonical = f"{VOLUME_MOUNT}/random_sea_{regime}.npz"
    canonical_state = f"{VOLUME_MOUNT}/random_sea_{regime}.state.json"
    backup = canonical + ".pre_merge.bak"

    if not Path(canonical).exists():
        raise RuntimeError(f"missing canonical npz: {canonical} (upload_existing first?)")

    if not Path(backup).exists():
        print(f"backing up canonical -> {backup}")
        shutil.copy2(canonical, backup)

    # Discover shards
    shard_files = sorted(p for p in os.listdir(VOLUME_MOUNT)
                         if p.startswith(f"random_sea_{regime}_shard_") and p.endswith(".npz"))
    print(f"found {len(shard_files)} shards: {shard_files}")

    samples_total = 0
    next_batch_max = 0
    if Path(canonical_state).exists():
        with open(canonical_state) as f:
            cstate = json.load(f)
        samples_total = int(cstate.get("samples_written", 0))
        next_batch_max = int(cstate.get("next_batch", 0))

    with zipfile.ZipFile(canonical, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as out_zf:
        existing = set(out_zf.namelist())
        for shard in shard_files:
            shard_path = f"{VOLUME_MOUNT}/{shard}"
            with zipfile.ZipFile(shard_path) as in_zf:
                for name in in_zf.namelist():
                    if name == "x.npy" or name == "meta.json":
                        continue
                    if name in existing:
                        print(f"  skip dup: {shard}::{name}")
                        continue
                    with in_zf.open(name) as src:
                        data = src.read()
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


@app.function(
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
    cpu=4,
    memory=16 * 1024,
)
def benchmark(regime: str = "deep", n_batches: int = 2,
              batch_size: int = BATCH_SIZE, keep_samples: int = KEEP_SAMPLES) -> dict:
    """Time `n_batches` batches end-to-end on a fresh worker, reporting both
    JIT-cold (first batch) and JIT-warm (subsequent) wall times.
    """
    import json
    import os
    import sys
    from time import perf_counter
    _ensure_repo_on_path()
    depth_min, depth_max, rng_stream_id = _regime_params(regime)
    output = f"{VOLUME_MOUNT}/_bench_{regime}_b{batch_size}.npz"
    state = output.replace(".npz", ".state.json")
    for p in (output, state):
        if Path(p).exists():
            os.remove(p)

    samples_per_batch = batch_size * keep_samples
    target_samples = n_batches * samples_per_batch
    sys.argv = _gen_argv(
        output=output, target_samples=target_samples,
        depth_min=depth_min, depth_max=depth_max, rng_stream_id=999,
        batch_size=batch_size, keep_samples=keep_samples,
    )
    # Capture per-batch timings via the state file the generator writes after each batch.
    from solver.gen_data.generate_random_sea_dataset import main as gen_main
    t0 = perf_counter()
    gen_main()
    total = perf_counter() - t0

    with open(state) as f:
        final_state = json.load(f)
    last_batch = float(final_state.get("last_batch_seconds", 0.0))

    # nvidia-smi snapshot for sanity
    import subprocess
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        gpu = f"(nvidia-smi failed: {exc})"

    samples_per_sec = (n_batches * samples_per_batch) / total if total > 0 else 0.0
    sec_per_sample = last_batch / samples_per_batch if samples_per_batch > 0 else 0.0
    print(json.dumps({
        "regime": regime, "n_batches": n_batches,
        "batch_size": batch_size, "keep_samples": keep_samples,
        "samples_per_batch": samples_per_batch,
        "gpu_type_requested": GPU_TYPE, "gpu": gpu,
        "total_wall_seconds": total,
        "last_batch_seconds": last_batch,
        "implied_jit_cold_seconds": total - last_batch * (n_batches - 1) if n_batches > 1 else total,
        "warm_seconds_per_sample": sec_per_sample,
        "warm_samples_per_sec": samples_per_batch / last_batch if last_batch > 0 else 0.0,
        "throughput_overall_samples_per_sec": samples_per_sec,
        "samples_written": final_state.get("samples_written"),
    }, indent=2))

    # Clean up bench artifacts so they don't pollute the volume
    for p in (output, state):
        if Path(p).exists():
            os.remove(p)
    volume.commit()

    return {
        "total_wall_seconds": total,
        "last_batch_seconds": last_batch,
        "n_batches": n_batches,
    }


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=600)
def status() -> dict:
    """Print every .state.json on the volume."""
    import json, os
    out: dict[str, dict] = {}
    for name in sorted(os.listdir(VOLUME_MOUNT)):
        if name.endswith(".state.json"):
            with open(f"{VOLUME_MOUNT}/{name}") as f:
                out[name] = json.load(f)
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def upload_existing(local_dir: str = "data") -> None:
    """Push local random_sea_*.npz and .state.json into the Modal volume."""
    src = Path(local_dir).resolve()
    targets: list[tuple[Path, str]] = []
    for regime in ("deep", "finite"):
        for suffix in (".npz", ".state.json"):
            p = src / f"random_sea_{regime}{suffix}"
            if not p.exists():
                print(f"skip (missing): {p}")
                continue
            targets.append((p, f"random_sea_{regime}{suffix}"))
    if not targets:
        print("nothing to upload")
        return
    total = sum(p.stat().st_size for p, _ in targets)
    print(f"Uploading {len(targets)} files ({total / 1e9:.2f} GB) to volume {VOLUME_NAME!r}")
    with volume.batch_upload(force=True) as batch:
        for path, remote_name in targets:
            print(f"  {path.name}  ({path.stat().st_size / 1e9:.2f} GB) -> /{remote_name}")
            batch.put_file(str(path), f"/{remote_name}")
    print("upload complete.")


@app.local_entrypoint()
def download(target_dir: str = "data", pattern: str = "") -> None:
    """Download all .npz / .state.json from the volume to <target_dir>/."""
    out = Path(target_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading volume {VOLUME_NAME!r} -> {out}/")
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
