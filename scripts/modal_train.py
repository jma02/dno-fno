"""Modal app to run the JAX DNO/FNO trainer on remote GPUs.

Workflow:
    # 0. Log in once:
    modal token new

    # 1. Upload a training dataset directory:
    modal run scripts/modal_train.py::upload_dataset --dataset outputs/paper_dataset/arrays

    # 2. Train with the C27 recipe (two A100-80GB GPUs by default):
    bash scripts/launch_c27_paper_dataset_modal.sh

    # Optional GPU shape override for that same recipe:
    GPU_SPEC=H100:4 bash scripts/launch_c27_paper_dataset_modal.sh

    # 3. Download a finished run dir to local outputs/:
    modal run scripts/modal_train.py::download_run --run-name YOUR_RUN_NAME

    # Quick peek at what's on the volume:
    modal run scripts/modal_train.py::status

The volume keeps both input files and run outputs, so later runs reuse the upload.
Use the lower-level ``train`` entrypoint directly only when configuring a custom run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, cast

import modal  # pyright: ignore[reportMissingImports]

APP_NAME = "dno-fno-train"
VOLUME_NAME = "dno-fno-train-data"
# GPU shape is fixed at decorator time in modal>=1.x (no with_options on Function);
# expose an env var so the caller can pick a shape without editing the file.
DEFAULT_GPU = os.environ.get("MODAL_GPU", "H100:1")
TIMEOUT_SECONDS = 24 * 3600
VOLUME_MOUNT = "/data"


class _ModalFunctionCall(Protocol):
    object_id: str


class _ModalTrainingFunction(Protocol):
    def spawn(self, **kwargs: object) -> _ModalFunctionCall: ...

    def remote(self, **kwargs: object) -> object: ...


REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy==2.4.2",
        "scipy==1.17.0",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "flax==0.12.6",
        "optax==0.2.5",
        "orbax-checkpoint==0.11.33",
        "jax[cuda12]==0.9.2",
    )
    .add_local_dir(REPO_ROOT / "train-jax-10m", remote_path="/repo/train-jax-10m")
    .add_local_dir(REPO_ROOT / "models" / "fno-jax", remote_path="/repo/models/fno-jax")
    .add_local_dir(REPO_ROOT / "models" / "dno-net", remote_path="/repo/models/dno-net")
    .add_local_dir(REPO_ROOT / "solver", remote_path="/repo/solver")
    .add_local_file(
        REPO_ROOT / "scripts/generate_paper_dataset.py",
        remote_path="/repo/scripts/generate_paper_dataset.py",
    )
    .add_local_file(
        REPO_ROOT / "scripts/replay_jonswap_dataset.py",
        remote_path="/repo/scripts/replay_jonswap_dataset.py",
    )
    .add_local_file(
        REPO_ROOT / "scripts/assemble_regenerated_dataset.py",
        remote_path="/repo/scripts/assemble_regenerated_dataset.py",
    )
    .add_local_file(
        REPO_ROOT / "scripts/benchmark_jonswap_replay.py",
        remote_path="/repo/scripts/benchmark_jonswap_replay.py",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=600)
def status() -> dict[str, list[str]]:
    """List dataset and run files currently on the volume."""
    import json
    import os

    out_dir = f"{VOLUME_MOUNT}/outputs"
    summary = {
        "datasets": [
            str(path.parent.relative_to(VOLUME_MOUNT))
            for path in sorted(Path(VOLUME_MOUNT).rglob("eta.npy"))
        ],
        "runs": sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else [],
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def upload_dataset(dataset: str) -> None:
    """Upload the dataset's NPY arrays while preserving repo-relative paths."""
    dataset_path = Path(dataset).resolve()
    targets = sorted(dataset_path.glob("*.npy"))
    if not targets:
        raise ValueError(f"dataset directory contains no NPY arrays: {dataset_path}")
    repo_root = REPO_ROOT.resolve()
    relative_targets = [
        (path, f"/{path.relative_to(repo_root).as_posix()}") for path in targets
    ]

    total_bytes = sum(path.stat().st_size for path, _ in relative_targets)
    print(
        f"Uploading {len(relative_targets):,} files "
        f"({total_bytes / 1e9:.2f} GB) -> volume {VOLUME_NAME!r}"
    )
    with volume.batch_upload(force=True) as batch:
        for path, remote in relative_targets:
            batch.put_file(str(path), remote)
    print("dataset upload complete.")


@app.function(
    gpu=DEFAULT_GPU,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    cpu=8,
    memory=256 * 1024,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=0.0),
)
def run_training(
    *,
    dataset: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    modes: int,
    width: int,
    n_blocks: int,
    norm: str,
    model_kind: str,
    seed: int,
    latent: int,
    cs_mult_hidden: int,
    trainer_args: str,
) -> dict[str, object]:
    import json
    import os
    import shlex
    import subprocess
    import sys
    import time
    from time import perf_counter

    repo = Path("/repo")
    data_link = repo / "data"
    out_link = repo / "outputs"
    out_dir = Path(VOLUME_MOUNT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_link.exists():
        data_link.symlink_to(VOLUME_MOUNT)
    if not out_link.exists():
        out_link.symlink_to(out_dir)

    os.environ["PYTHONPATH"] = (
        "/repo:/repo/models/fno-jax:/repo/models/dno-net:/repo/train-jax-10m"
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for repo_path in (
        "/repo",
        "/repo/models/fno-jax",
        "/repo/models/dno-net",
        "/repo/train-jax-10m",
    ):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    options = {
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "width": width,
        "n_blocks": n_blocks,
        "norm": norm,
        "model": model_kind,
        "seed": seed,
        "run_name": run_name,
        "latent": latent,
        "cs_mult_hidden": cs_mult_hidden,
    }
    cmd = [
        sys.executable,
        "/repo/train-jax-10m/1d_dno_fno_jax.py",
        *(part for key, value in options.items() for part in (f"--{key}", str(value))),
    ]
    if model_kind == "fno":
        cmd.extend(["--modes", str(modes)])
    cmd.extend(shlex.split(trainer_args))

    print("$", " ".join(cmd))
    started = perf_counter()
    # Run the trainer as a child so we can periodically commit the volume — without
    # this, preemption can lose all writes since the last function exit (the trainer
    # writes per-epoch ckpts to /data/outputs/<run>/latest_ckpt; we need those flushed
    # to the volume so the next retry actually sees them and resumes).
    proc = subprocess.Popen(cmd, cwd="/repo", env=os.environ.copy())
    commit_interval_s = 300.0
    next_commit = time.monotonic() + commit_interval_s
    try:
        while True:
            try:
                proc.wait(timeout=30.0)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= next_commit:
                    try:
                        volume.commit()
                    except Exception as e:
                        print(f"[modal] periodic volume.commit() failed: {e!r}")
                    next_commit = time.monotonic() + commit_interval_s
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            volume.commit()
        except Exception as e:
            print(f"[modal] final volume.commit() failed: {e!r}")
    completed_returncode = proc.returncode if proc.returncode is not None else -1
    elapsed = perf_counter() - started

    summary_path = Path(VOLUME_MOUNT) / "outputs" / run_name / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    print(f"training finished in {elapsed:.0f}s, returncode={completed_returncode}")
    if completed_returncode != 0:
        # Raise so Modal's Retries policy resurrects the function with the same args;
        # the trainer's auto-resume from latest_ckpt picks up where the crash left off.
        raise RuntimeError(
            f"trainer exited with returncode={completed_returncode} "
            f"(run_name={run_name}, elapsed={elapsed:.0f}s)"
        )
    return {
        "returncode": completed_returncode,
        "runtime_seconds": elapsed,
        "run_name": run_name,
        "run_dir_on_volume": f"/outputs/{run_name}",
        "summary": summary,
    }


@app.function(
    gpu="H100:1",
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    cpu=8,
    memory=256 * 1024,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=0.0),
)
def regenerate_dataset(training_options: dict[str, object]) -> dict[str, object]:
    """Replay JONSWAP, add independent Stokes states, then start fresh training."""
    import json
    import shlex
    import subprocess
    import sys
    import time

    volume.reload()
    root = Path(str(training_options["dataset"])).parent
    root.mkdir(parents=True, exist_ok=True)
    training_call_path = root / "training_call.json"
    if training_call_path.exists():
        return json.loads(training_call_path.read_text())
    os.environ.update(PYTHONPATH="/repo", DNO_TANAKA_DTYPE="float64")
    os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
    )
    quoted_root = shlex.quote(str(root))
    replay_command = (
        f"scripts.replay_jonswap_dataset --inputs {quoted_root}/replay_inputs.npz "
        f"--output-root {quoted_root}/jonswap_arrays "
        "--source-dataset /data/outputs/paper_dataset/arrays"
    )
    commands = (
        ("pilot", replay_command + " --batch-size 4 --pilot-only"),
        (
            "longest128",
            f"scripts.benchmark_jonswap_replay --inputs {quoted_root}/replay_inputs.npz "
            f"--output {quoted_root}/longest128.json "
            "--batch-size 128 --full-horizon --longest --repeats 1",
        ),
        ("jonswap", replay_command + " --batch-size 128"),
        (
            "stokes",
            "scripts.generate_paper_dataset --family stokes --seed 2026091400 "
            "--num-simulations 3391488 --batch-size 256 --gpu "
            f"--output-root {quoted_root}/stokes",
        ),
        (
            "assembly",
            f"scripts.assemble_regenerated_dataset --jonswap {quoted_root}/jonswap_arrays "
            f"--new-stokes {quoted_root}/stokes --output {quoted_root}/arrays",
        ),
    )
    for stage, arguments in commands:
        print(f"Starting {stage}: {arguments}", flush=True)
        with (root / f"{stage}.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", *shlex.split(arguments)],
                cwd="/repo",
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            next_commit = time.monotonic() + 300
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= next_commit:
                            volume.commit()
                            next_commit = time.monotonic() + 300
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                volume.commit()
        print(f"Finished {stage}: returncode={process.returncode}", flush=True)
        if process.returncode:
            # A failed numerical check must stop the workflow, not trigger retries.
            failure = {
                "status": "failed",
                "stage": stage,
                "returncode": process.returncode,
            }
            (root / "failure.json").write_text(json.dumps(failure) + "\n")
            volume.commit()
            print((root / f"{stage}.log").read_text()[-8000:], flush=True)
            return failure

    # If interrupted during dispatch, stop for inspection instead of launching twice.
    training_call_path.write_text(
        json.dumps({"status": "training_dispatch_incomplete", **training_options})
        + "\n"
    )
    volume.commit()
    call = cast(_ModalTrainingFunction, run_training).spawn(**training_options)
    result = {
        "status": "training_started",
        "function_call_id": call.object_id,
        **training_options,
    }
    training_call_path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    print(json.dumps(result), flush=True)
    return result


@app.function(
    gpu="H100:1",
    volumes={VOLUME_MOUNT: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=256 * 1024,
)
def benchmark_jonswap(
    batch_sizes: str = "32,64,128,256",
    full_horizon: bool = False,
    longest: bool = False,
    repeats: int = 2,
) -> dict[str, object]:
    """Benchmark one H100 only; never generate a dataset or dispatch training."""
    import csv
    import json
    import subprocess
    import sys

    volume.reload()
    root = Path("/data/outputs/paper_dataset_regenerated_20260914")
    mode = "full_longest" if longest else "full_median" if full_horizon else "short"
    directory = root / "benchmarks" / mode
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONPATH="/repo", DNO_TANAKA_DTYPE="float64", JAX_LOG_COMPILES="1")
    os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    results: dict[str, object] = {}
    for batch_size in map(int, batch_sizes.split(",")):
        output = directory / f"batch_{batch_size}.json"
        if output.exists():
            previous = json.loads(output.read_text())
            if len(previous["runs"]) == repeats and all(run["status"] == "passed" for run in previous["runs"]):
                results[str(batch_size)] = previous
                continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.benchmark_jonswap_replay",
            "--inputs",
            str(root / "replay_inputs.npz"),
            "--output",
            str(output),
            "--batch-size",
            str(batch_size),
            "--repeats",
            str(repeats),
        ]
        command += [
            flag
            for flag, enabled in (
                ("--full-horizon", full_horizon),
                ("--longest", longest),
            )
            if enabled
        ]
        print(f"Benchmark batch={batch_size}, mode={mode}", flush=True)
        with (
            output.with_suffix(".csv").open("w") as metrics,
            output.with_suffix(".log").open("w") as log,
        ):
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=250",
                ],
                stdout=metrics,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd="/repo",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            finally:
                monitor.terminate()
                monitor.wait()
                volume.commit()
        if completed.returncode:
            print(output.with_suffix(".log").read_text()[-8000:], flush=True)
            results[str(batch_size)] = {"returncode": completed.returncode}
            break
        result = json.loads(output.read_text())
        with output.with_suffix(".csv").open() as metrics:
            samples = [
                [float(value) for value in row]
                for row in csv.reader(metrics)
                if len(row) == 3
            ]
        result["sampled_peak_gpu_memory_mib"] = max(row[0] for row in samples)
        result["sampled_mean_gpu_busy_percent"] = sum(row[1] for row in samples) / len(
            samples
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
        results[str(batch_size)] = result
        print(json.dumps(result), flush=True)
    return results


@app.local_entrypoint()
def train(
    dataset: str,
    run_name: str = "",
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    modes: int = 64,
    width: int = 64,
    n_blocks: int = 4,
    norm: str = "scale",
    model_kind: str = "fno",
    seed: int = 0,
    latent: int = 64,
    cs_mult_hidden: int = 32,
    trainer_args: str = "",
    spawn: bool = False,
    regenerate: bool = False,
) -> None:
    """Local entrypoint: dispatch a training run on a Modal GPU worker.

    GPU shape is taken from the ``MODAL_GPU`` env var at module import time
    (defaults to ``H100:1``). Set it before invoking modal CLI, e.g.
    ``MODAL_GPU=H100:2 modal run scripts/modal_train.py::train ...``.
    """
    from datetime import datetime

    if not run_name:
        run_name = datetime.now().strftime("fno_jax_10m_%Y%m%d_%H%M%S")
    print(f"run_name = {run_name}")
    print(f"gpu      = {DEFAULT_GPU}")

    call_kwargs: dict[str, object] = dict(
        dataset=dataset,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        modes=modes,
        width=width,
        n_blocks=n_blocks,
        norm=norm,
        model_kind=model_kind,
        seed=seed,
        latent=latent,
        cs_mult_hidden=cs_mult_hidden,
        trainer_args=trainer_args,
    )
    remote_training = cast(
        _ModalTrainingFunction, regenerate_dataset if regenerate else run_training
    )
    if regenerate:
        call_kwargs = {"training_options": call_kwargs}
    if spawn:
        call = remote_training.spawn(**call_kwargs)
        print(f"function_call_id = {call.object_id}")
        return

    result = remote_training.remote(**call_kwargs)
    import json

    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def download_run(run_name: str, target_dir: str = "outputs") -> None:
    """Pull a single run's artifacts back from the volume."""
    out = Path(target_dir).resolve() / run_name
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"/outputs/{run_name}"
    print(f"Downloading {prefix} -> {out}/")
    pulled = 0
    for entry in volume.iterdir(prefix):
        rel = entry.path[len(prefix) :].lstrip("/")
        local = out / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        size_gb = (entry.size or 0) / 1e9
        print(f"  {entry.path}  ({size_gb:.3f} GB) -> {local}")
        with open(local, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        pulled += 1
    print(f"done; pulled {pulled} files.")


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=300)
def tail_log(run_name: str, lines: int = 60) -> str:
    """Print the tail of train_log.jsonl for a given run."""
    p = Path(VOLUME_MOUNT) / "outputs" / run_name / "train_log.jsonl"
    if not p.exists():
        msg = f"no log at {p}"
        print(msg)
        return msg
    text = p.read_text(encoding="utf-8").splitlines()
    tail = "\n".join(text[-lines:])
    print(tail)
    return tail
