"""Compare one full-horizon JONSWAP batch on Modal H100 and H200 GPUs.

Run both commands from the same clean commit, stopping if H100 fails::

    MODAL_GPU=H100! modal run experiments/modal_jonswap_hardware_benchmark.py::main \
        --expected-source-sha COMMIT --pad-factor 8
    MODAL_GPU=H200 modal run experiments/modal_jonswap_hardware_benchmark.py::main \
        --expected-source-sha COMMIT --pad-factor 8
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Protocol, cast

import modal  # pyright: ignore[reportMissingImports]


APP_NAME = "dno-fno-jonswap-hardware-benchmark"
GPU_SPEC = os.environ.get("MODAL_GPU", "H100!")
REPLAY_SHA256 = "03450cb94c30333e697e051985e61acf11647c78979bdc946efe388570f675b1"
REPO_ROOT = Path(__file__).resolve().parent.parent
VOLUME_MOUNT = "/data"


class _RemoteBenchmark(Protocol):
    def remote(self, **kwargs: object) -> dict[str, object]: ...


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.4.2",
        "scipy==1.17.0",
        "matplotlib>=3.8",
        "jax[cuda12]==0.9.2",
    )
    .add_local_dir(REPO_ROOT / "solver", remote_path="/repo/solver")
)
app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("dno-fno-train-data", create_if_missing=False)


@app.function(
    gpu=GPU_SPEC,
    volumes={VOLUME_MOUNT: volume},
    timeout=780,
    cpu=8,
    memory=64 * 1024,
    retries=0,
)
def run_benchmark(
    *,
    expected_source_sha: str,
    expected_solver_sha256: str,
    pad_factor: int,
    repeats: int,
) -> dict[str, object]:
    import json
    import sys
    from time import perf_counter

    os.environ.update(
        PYTHONPATH="/repo",
        DNO_TANAKA_DTYPE="float64",
        NCCL_P2P_LEVEL="PHB",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    sys.path.insert(0, "/repo")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from solver.gen_data.jonswap_tma import JonswapTmaParameters, ResolvedBand
    from solver.gen_data.jonswap_tma_sampling import JonswapTmaSample
    from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
    from solver.gen_data.pipeline.trajectory_rollout import (
        execute_adjustment_batch,
        execute_trajectory_batch,
    )
    from solver.gen_data.trajectory_family_adapters import (
        construct_jonswap_tma_trajectory_batch,
    )

    volume.reload()
    jax.config.update("jax_enable_x64", True)
    expected_gpu = GPU_SPEC.split(":", maxsplit=1)[0].removesuffix("!")
    device = jax.local_devices()[0]
    if expected_gpu not in device.device_kind.upper():
        raise RuntimeError(f"requested {expected_gpu}, received {device.device_kind}")
    if pad_factor not in (2, 8):
        raise ValueError("pad factor must be 2 or 8")
    if _python_tree_sha256(Path("/repo/solver")) != expected_solver_sha256:
        raise RuntimeError("remote solver differs from the local clean commit")
    if jnp.ones((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError("JAX float64 is not enabled")

    replay_path = (
        Path(VOLUME_MOUNT)
        / "outputs/paper_dataset_regenerated_20260914/replay_inputs.npz"
    )
    with replay_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != REPLAY_SHA256:
            raise RuntimeError("replay input hash differs from the benchmark fixture")
    with np.load(replay_path, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}

    base_numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
    metadata = json.loads(str(inputs["metadata_json"].item()))
    if (
        metadata["current_numerical"] != base_numerical._asdict()
        or metadata["numerical_differences"]
    ):
        raise RuntimeError("replay inputs do not match current rollout numerics")
    numerical = base_numerical._replace(pad_factor=pad_factor)

    production_counts = inputs["production_time_count"]
    median_count = np.median(production_counts)
    indices = np.argsort(np.abs(production_counts - median_count), kind="stable")[:128]
    time_grids = {
        phase: tuple(
            numerical.saved_dt
            * np.arange(int(inputs[f"{phase}_time_count"][index]), dtype=np.float64)
            for index in indices
        )
        for phase in ("adjustment", "production")
    }

    output = (
        Path(VOLUME_MOUNT)
        / "outputs/jonswap_hardware_benchmark"
        / expected_source_sha
        / f"{expected_gpu.lower()}_pad{pad_factor}.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    report: dict[str, object] = {
        "source_sha": expected_source_sha,
        "solver_sha256": expected_solver_sha256,
        "replay_sha256": REPLAY_SHA256,
        "requested_gpu": GPU_SPEC,
        "device": device.device_kind,
        "pad_factor": pad_factor,
        "numerical": numerical._asdict(),
        "simulation_ids": inputs["simulation_id"][indices].tolist(),
        "maximum_saved_time_counts": {
            phase: max(map(len, grids)) for phase, grids in time_grids.items()
        },
        "complete": False,
        "runs": runs,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()

    for repeat in range(repeats):
        result: dict[str, object] = {"repeat": repeat + 1, "status": "running"}
        runs.append(result)
        output.write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        started = perf_counter()
        try:
            initial, valid = construct_jonswap_tma_trajectory_batch(
                tuple(
                    JonswapTmaSample(
                        JonswapTmaParameters(
                            *(
                                float(inputs[name][index])
                                for name in JonswapTmaParameters._fields
                            )
                        ),
                        inputs["phase_right"][index],
                        inputs["phase_left"][index],
                    )
                    for index in indices
                ),
                numerical,
                band=ResolvedBand(
                    numerical.length, numerical.target_maximum_wavenumber
                ),
            )
            result["constructor_seconds"] = perf_counter() - started
            if initial is None or valid != tuple(range(128)):
                raise RuntimeError("initial-condition gate rejected a benchmark case")

            phase_started = perf_counter()
            endpoints = execute_adjustment_batch(
                initial.eta0,
                initial.xi0,
                initial.depths,
                time_grids["adjustment"],
                nonlinear_ramp_times=inputs["nonlinear_ramp_time"][indices],
                nonlinear_ramp_order=4,
                config=numerical,
            )
            result["adjustment_seconds"] = perf_counter() - phase_started
            result["valid_adjustment"] = sum(x is not None for x in endpoints)
            if result["valid_adjustment"] != 128:
                raise RuntimeError("adjustment convergence/state gate failed")

            phase_started = perf_counter()
            trajectories = execute_trajectory_batch(
                np.stack([x[0] for x in endpoints if x is not None]),
                np.stack([x[1] for x in endpoints if x is not None]),
                initial.depths,
                time_grids["production"],
                config=numerical,
            )
            completed = perf_counter()
            result["production_and_labels_seconds"] = completed - phase_started
            elapsed = completed - started
            result["valid_production"] = sum(x is not None for x in trajectories)
            if result["valid_production"] != 128:
                raise RuntimeError("production numerical-health gate failed")

            padded_intervals = 128 * sum(
                max(map(len, grids)) - 1 for grids in time_grids.values()
            )
            memory = device.memory_stats() or {}
            result.update(
                status="passed",
                total_seconds=elapsed,
                case_saved_intervals_per_second=padded_intervals / elapsed,
                allocator_memory_bytes={
                    name: int(memory[name])
                    for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                    if name in memory
                },
            )
        except Exception as error:
            result.update(status="failed", error=repr(error))
            output.write_text(json.dumps(report, indent=2) + "\n")
            volume.commit()
            raise
        output.write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        print(json.dumps(result), flush=True)
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(expected_source_sha: str, pad_factor: int = 8, repeats: int = 2) -> None:
    source_sha = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
    ).strip()
    if source_sha != expected_source_sha:
        raise RuntimeError(f"HEAD is {source_sha}, expected {expected_source_sha}")
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=REPO_ROOT, text=True
    ):
        raise RuntimeError("benchmark must be launched from a clean worktree")
    report = cast(_RemoteBenchmark, run_benchmark).remote(
        expected_source_sha=expected_source_sha,
        expected_solver_sha256=_python_tree_sha256(REPO_ROOT / "solver"),
        pad_factor=pad_factor,
        repeats=repeats,
    )
    print(report)
