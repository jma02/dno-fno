"""Modal H100 benchmark for the exact revision-2 paper-corpus generator.

The first entrypoint deliberately reproduces the isolated local cap-four
Tanaka replay.  It does not write into the production corpus namespace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = "dno-paper-corpus-cap4"
VOLUME_NAME = "dno-paper-corpus-cap4-revision2"
VOLUME_MOUNT = "/data"
GPU_TYPE = "H100"
BENCHMARK_TIMEOUT_SECONDS = 2 * 3600
REPO_ROOT = Path(__file__).resolve().parent

BENCHMARK_RELATIVE_ROOT = (
    "noncorpus_cap4_h100_b256_replay_20260729/tanaka_b256"
)
BENCHMARK_EXPECTED_FINGERPRINT = (
    "6922e80500f132884adc9019fd4c81aa34cc96c65cdc8e2045f0a2741d25c501"
)

image = (
    modal.Image.from_registry("python:3.11.13-slim-bookworm")
    .pip_install(
        "jax[cuda12]==0.9.2",
        "matplotlib==3.10.8",
        "numpy==2.4.2",
        "scipy==1.17.0",
        "tqdm==4.67.3",
    )
    .add_local_dir(REPO_ROOT / "solver", remote_path="/repo/solver")
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/repo/scripts")
    .add_local_file(REPO_ROOT / "pyproject.toml", remote_path="/repo/pyproject.toml")
    .add_local_file(REPO_ROOT / "uv.lock", remote_path="/repo/uv.lock")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _validated_relative_root(relative_root: str) -> str:
    path = PurePosixPath(relative_root)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("relative_root must stay inside the Modal volume")
    return str(path)


def _quota_argv(
    *,
    relative_root: str,
    execute: bool,
) -> list[str]:
    output_root = f"{VOLUME_MOUNT}/{_validated_relative_root(relative_root)}"
    return [
        "python",
        "/repo/scripts/run_paper_corpus_quota.py",
        "--family",
        "tanaka",
        "--split",
        "validation",
        "--accepted-cases",
        "256",
        "--accepted-cases-before",
        "0",
        "--batch-size",
        "256",
        "--maximum-attempts-per-accepted-case",
        "4",
        "--stream-id",
        "910",
        "--first-attempt-index",
        "0",
        "--output-root",
        output_root,
        "--platform",
        "gpu",
        "--execute" if execute else "--dry-run",
    ]


def _subprocess_environment() -> dict[str, str]:
    return {
        **os.environ,
        "DNO_TANAKA_DTYPE": "float64",
        "JAX_ENABLE_X64": "true",
        "JAX_PLATFORMS": "cuda",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "PYTHONPATH": "/repo",
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


@app.function(
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    cpu=8,
    memory=64 * 1024,
)
def probe_tanaka_b256_preflight(
    *,
    relative_root: str = BENCHMARK_RELATIVE_ROOT,
) -> dict[str, Any]:
    """Return the remote exact-contract identity without generating data."""
    import subprocess

    normalized_root = _validated_relative_root(relative_root)
    process = subprocess.run(
        _quota_argv(relative_root=normalized_root, execute=False),
        cwd="/repo",
        env=_subprocess_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=True,
    )
    preflight = json.loads(process.stdout)
    configuration = preflight["run_spec"]["configuration"]
    result = {
        "configuration_fingerprint": preflight["configuration_fingerprint"],
        "dependency_environment": configuration["dependency_environment"],
        "source_sha256": configuration["source_sha256"],
        "runtime": preflight["runtime"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    cpu=8,
    memory=64 * 1024,
)
def run_tanaka_b256_benchmark(
    *,
    relative_root: str = BENCHMARK_RELATIVE_ROOT,
    expected_fingerprint: str = BENCHMARK_EXPECTED_FINGERPRINT,
) -> dict[str, Any]:
    """Run the exact same-input cap-four Tanaka quota on one H100."""
    import subprocess
    from time import perf_counter

    normalized_root = _validated_relative_root(relative_root)
    output_root = Path(VOLUME_MOUNT) / normalized_root
    if output_root.exists():
        raise RuntimeError(
            f"benchmark root already exists and will not be resumed: {output_root}"
        )

    environment = _subprocess_environment()
    preflight_process = subprocess.run(
        _quota_argv(relative_root=normalized_root, execute=False),
        cwd="/repo",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=True,
    )
    preflight = json.loads(preflight_process.stdout)
    actual_fingerprint = preflight.get("configuration_fingerprint")
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Modal preflight fingerprint mismatch: "
            f"{actual_fingerprint!r} != {expected_fingerprint!r}"
        )

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    started = perf_counter()
    execution_process = subprocess.run(
        _quota_argv(relative_root=normalized_root, execute=True),
        cwd="/repo",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=True,
    )
    wall_seconds = perf_counter() - started
    execution = json.loads(execution_process.stdout)
    if execution.get("status") != "complete":
        raise RuntimeError(f"Modal benchmark did not complete: {execution!r}")

    volume.commit()
    return {
        "schema": "paper_corpus_modal_h100_benchmark_v1",
        "gpu": gpu,
        "requested_gpu": GPU_TYPE,
        "relative_root": normalized_root,
        "configuration_fingerprint": actual_fingerprint,
        "wall_seconds": wall_seconds,
        "preflight": preflight,
        "execution": execution,
    }


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=300)
def benchmark_status() -> dict[str, Any]:
    """Return the durable benchmark artifact inventory."""
    root = Path(VOLUME_MOUNT) / BENCHMARK_RELATIVE_ROOT
    files = (
        {
            str(path.relative_to(root)): path.stat().st_size
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        if root.is_dir()
        else {}
    )
    result = {
        "root_exists": root.exists(),
        "relative_root": BENCHMARK_RELATIVE_ROOT,
        "files": files,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def benchmark() -> None:
    """Launch the isolated H100 benchmark and print its audited summary."""
    result = run_tanaka_b256_benchmark.remote()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
