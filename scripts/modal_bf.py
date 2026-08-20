"""Modal app to generate deep-water JCP09-form BF rollouts on A100s.

Workflow:
    modal token new                                    # once

    # Fire off both shards in parallel:
    modal run scripts/modal_bf.py::go_wide --n-shards 2

    # When done, merge into a single canonical npz on the volume:
    modal run scripts/modal_bf.py::merge_shards

    # Pull back to local data/:
    modal run scripts/modal_bf.py::download

    # Peek at progress:
    modal run scripts/modal_bf.py::status

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

REPO_ROOT = Path(__file__).resolve().parent.parent

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
TOTAL_BATCHES = (
    TARGET_SAMPLES + SAMPLES_PER_BATCH - 1
) // SAMPLES_PER_BATCH

# Seeding disjoint from local g0/g1.
MODAL_SEED = 13
MODAL_RNG_STREAM_ID = 99
MODAL_CASE_ID_OFFSET = 2_000_000

# Adaptive params — matches the local BF runs (envelope + magnitude + power=4).
ADAPTIVE_ALPHA = 0.85
ADAPTIVE_SMOOTH_SIGMA = 20.0
ADAPTIVE_POWER = 4.0


def _target_through_batch(end_batch: int) -> int:
    """Return cumulative rows through ``end_batch``, capped at the corpus target."""

    return min(end_batch * SAMPLES_PER_BATCH, TARGET_SAMPLES)


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
    from solver.gen_data.generate_bf_dataset import (
        archive_contract_fields,
        build_archive_state,
        build_generation_configuration,
        parse_args,
        save_state,
        validate_archive_contract,
    )
    from solver.solvers.dno_series_jax import build_grid

    if not 0 <= start_batch < end_batch:
        raise ValueError("Modal BF shard interval must satisfy 0 <= start < end")
    out = Path(output_path)
    state_path = out.with_suffix(".state.json")
    target_samples = _target_through_batch(end_batch)
    args = parse_args(_gen_argv(output_path, target_samples)[1:])
    configuration = build_generation_configuration(args)
    contract_fields = archive_contract_fields(configuration)
    if state_path.exists():
        validated = validate_archive_contract(
            out,
            state_path,
            expected_configuration=configuration,
        )
        if validated.metadata.get("modal_shard") != {
            "start_batch": start_batch,
            "end_batch": end_batch,
        }:
            raise ValueError(
                "existing BF Modal shard interval does not match this request"
            )
        return
    if out.exists():
        raise ValueError(
            "BF Modal shard exists without a resumable v2 state sidecar"
        )

    x_grid, _ = build_grid(1024, 2.0 * math.pi)
    meta = {
        **contract_fields,
        "shard_kind": "bf_jcp09_deep_adaptive_shard_v2",
        "modal_shard": {
            "start_batch": start_batch,
            "end_batch": end_batch,
        },
    }
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open("x.npy", mode="w", force_zip64=True) as h:
            npy_format.write_array(h, np.asarray(x_grid, dtype=np.float32), allow_pickle=False)
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    save_state(
        state_path,
        build_archive_state(
            output_path=out,
            configuration=configuration,
            start_batch=start_batch,
            next_batch=start_batch,
        ),
    )


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
    output = (
        f"{VOLUME_MOUNT}/bf_jcp09_deep_adaptive_modal_shard_{shard_id:02d}.npz"
    )

    # The generator stops at n_batches_planned = ceil(target_samples / samples_per_batch),
    # which we steer to end_batch via target_samples.
    target_samples = _target_through_batch(end_batch)

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
        shard_rows = _target_through_batch(e) - _target_through_batch(s)
        print(
            f"  shard {i:02d}: batches [{s}, {e})  "
            f"({e-s} batches, {shard_rows:,} samples)"
        )
    args = [(i, s, e) for i, (s, e) in enumerate(slices)]
    list(run_shard.starmap(args))
    print("all shards finished.")


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=2 * 3600, cpu=4, memory=32 * 1024)
def merge_shards() -> None:
    """Concatenate every JCP09-form shard into one canonical archive."""
    import json
    import os
    import zipfile
    from solver.gen_data.generate_bf_dataset import (
        archive_contract_fields,
        build_archive_state,
        build_generation_configuration,
        parse_args,
        save_state,
        validate_archive_contract,
    )

    canonical = f"{VOLUME_MOUNT}/bf_jcp09_deep_adaptive_modal.npz"
    canonical_state = f"{VOLUME_MOUNT}/bf_jcp09_deep_adaptive_modal.state.json"

    shard_files = sorted(
        p for p in os.listdir(VOLUME_MOUNT)
        if p.startswith("bf_jcp09_deep_adaptive_modal_shard_")
        and p.endswith(".npz")
    )
    if not shard_files:
        raise RuntimeError("no shard npz files found on volume")
    print(f"found {len(shard_files)} shards: {shard_files}")

    validated: list[tuple[str, int, int]] = []
    expected_corpus_fingerprint: str | None = None
    reference_x: bytes | None = None
    for shard in shard_files:
        shard_path = Path(VOLUME_MOUNT) / shard
        state_path = shard_path.with_suffix(".state.json")
        validated_contract = validate_archive_contract(shard_path, state_path)
        meta = validated_contract.metadata
        if meta.get("shard_kind") != "bf_jcp09_deep_adaptive_shard_v2":
            raise RuntimeError(f"{shard} is not a corrected v2 Modal BF shard")
        shard_record = meta.get("modal_shard")
        if not isinstance(shard_record, dict):
            raise RuntimeError(f"{shard} lacks its Modal batch interval")
        start_batch = shard_record.get("start_batch")
        end_batch = shard_record.get("end_batch")
        if (
            isinstance(start_batch, bool)
            or not isinstance(start_batch, int)
            or isinstance(end_batch, bool)
            or not isinstance(end_batch, int)
            or not 0 <= start_batch < end_batch <= TOTAL_BATCHES
        ):
            raise RuntimeError(f"{shard} has an invalid Modal batch interval")
        expected_args = parse_args(
            _gen_argv(shard_path.as_posix(), _target_through_batch(end_batch))[1:]
        )
        expected_configuration = build_generation_configuration(expected_args)
        if validated_contract.metadata.get("configuration") != expected_configuration:
            raise RuntimeError(
                f"{shard} configuration does not match the current Modal run"
            )
        state = validated_contract.state
        expected_shard_target = _target_through_batch(end_batch)
        if (
            not bool(state.get("complete"))
            or state.get("next_batch") != end_batch
            or state.get("samples_written") != expected_shard_target
        ):
            raise RuntimeError(f"{shard} is not a complete Modal batch interval")
        with zipfile.ZipFile(shard_path) as archive:
            members = archive.namelist()
            names = set(members)
            if len(names) != len(members):
                raise RuntimeError(f"{shard} contains duplicate archive members")
            expected_eta_members = {
                f"eta_batch_{batch_index:04d}.npy"
                for batch_index in range(start_batch, end_batch)
            }
            actual_eta_members = {
                name
                for name in names
                if name.startswith("eta_batch_") and name.endswith(".npy")
            }
            if actual_eta_members != expected_eta_members:
                raise RuntimeError(
                    f"{shard} contains data outside its declared batch interval"
                )
            for batch_index in range(start_batch, end_batch):
                tag = f"batch_{batch_index:04d}"
                required = {
                    f"eta_{tag}.npy",
                    f"xi_{tag}.npy",
                    f"gxi_{tag}.npy",
                    f"time_{tag}.npy",
                    f"case_id_{tag}.npy",
                    f"depth_{tag}.npy",
                    f"subsample_indices_{tag}.npy",
                    f"specs_{tag}.json",
                }
                missing = sorted(required - names)
                if missing:
                    raise RuntimeError(
                        f"{shard} lacks complete batch {batch_index}: {missing}"
                    )
            x_bytes = archive.read("x.npy")
        if reference_x is None:
            reference_x = x_bytes
        elif x_bytes != reference_x:
            raise RuntimeError("Modal BF shards use different spatial grids")
        corpus_fingerprint = meta["corpus_configuration_fingerprint"]
        if expected_corpus_fingerprint is None:
            expected_corpus_fingerprint = str(corpus_fingerprint)
        elif corpus_fingerprint != expected_corpus_fingerprint:
            raise RuntimeError("Modal BF shards use different corpus configurations")
        validated.append((shard, start_batch, end_batch))

    validated.sort(key=lambda item: item[1])
    cursor = 0
    for shard, start_batch, end_batch in validated:
        if start_batch != cursor:
            raise RuntimeError(
                f"Modal BF shard intervals overlap or leave a gap before {shard}"
            )
        cursor = end_batch
    if cursor != TOTAL_BATCHES:
        raise RuntimeError("Modal BF shards do not cover the planned batch range")

    samples_total = 0
    next_batch_max = 0

    merged_target = TARGET_SAMPLES
    merged_args = parse_args(_gen_argv(canonical, merged_target)[1:])
    merged_configuration = build_generation_configuration(merged_args)
    merged_contract = archive_contract_fields(merged_configuration)
    merged_meta = {
        **merged_contract,
        "archive_kind": "bf_jcp09_deep_adaptive_modal_merged_v2",
        "merged_from_shards": [item[0] for item in validated],
    }
    staged_archive = Path(f"{canonical}.staging")
    staged_state = Path(f"{canonical_state}.staging")
    with zipfile.ZipFile(staged_archive, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as out_zf:
        assert reference_x is not None
        out_zf.writestr("x.npy", reference_x)
        out_zf.writestr("meta.json", json.dumps(merged_meta, indent=2))
        existing: set[str] = {"x.npy", "meta.json"}
        for shard, _, _ in validated:
            with zipfile.ZipFile(f"{VOLUME_MOUNT}/{shard}") as in_zf:
                for name in in_zf.namelist():
                    if name in ("x.npy", "meta.json"):
                        continue
                    if name in existing:
                        raise RuntimeError(f"duplicate Modal BF member {shard}::{name}")
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

    if samples_total != merged_target or next_batch_max != TOTAL_BATCHES:
        raise RuntimeError(
            "merged BF payload does not contain the exact planned rows/batches"
        )
    state = build_archive_state(
        output_path=staged_archive,
        configuration=merged_configuration,
        start_batch=0,
        next_batch=TOTAL_BATCHES,
    )
    state["output_path"] = canonical
    state["merged_from_shards"] = [item[0] for item in validated]
    save_state(staged_state, state)
    validate_archive_contract(
        staged_archive,
        staged_state,
        expected_configuration=merged_configuration,
    )
    staged_archive.replace(canonical)
    staged_state.replace(canonical_state)
    validate_archive_contract(
        Path(canonical),
        Path(canonical_state),
        expected_configuration=merged_configuration,
    )
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
def download(
    target_dir: str = "data",
    pattern: str = "bf_jcp09_deep_adaptive_modal",
) -> None:
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
