"""Modal app to run the train-jax-10m FNO trainer on a remote GPU.

Workflow:
    # 0. Authenticate once:
    modal token new

    # 1. Upload the combined training dataset + the rescaled DNO test set:
    modal run modal_train.py::upload_data

    # 2. Train (single-GPU H100 by default; pass --gpu-spec to override):
    modal run modal_train.py::train \\
        --epochs 30 --batch-size 1024 --modes 64 --width 64 --n-blocks 4

    # multi-GPU example:
    modal run modal_train.py::train --gpu-spec "H100:4" --batch-size 4096

    # 3. Download a finished run dir to local outputs/:
    modal run modal_train.py::download_run --run-name fno_jax_10m_20260507_103000

    # Quick peek at what's on the volume:
    modal run modal_train.py::status

The volume keeps both the input npz files and the run outputs, so re-running
``train`` re-uses the same dataset upload.
"""
from __future__ import annotations

from pathlib import Path

import modal


import os

APP_NAME = "dno-fno-train"
VOLUME_NAME = "dno-fno-train-data"
# GPU shape is fixed at decorator time in modal>=1.x (no with_options on Function);
# expose an env var so the caller can pick a shape without editing the file.
DEFAULT_GPU = os.environ.get("MODAL_GPU", "H100:1")
TIMEOUT_SECONDS = 24 * 3600
VOLUME_MOUNT = "/data"

REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy>=2.0.2",
        "scipy>=1.13",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "flax>=0.11",
        "optax>=0.2.5",
        "orbax-checkpoint>=0.11.6",
        "jax[cuda12]>=0.7",
    )
    .add_local_dir(REPO_ROOT / "train-jax-10m", remote_path="/repo/train-jax-10m")
    .add_local_dir(REPO_ROOT / "models" / "fno-jax", remote_path="/repo/models/fno-jax")
    .add_local_dir(REPO_ROOT / "models" / "dno-net", remote_path="/repo/models/dno-net")
    .add_local_dir(REPO_ROOT / "solver", remote_path="/repo/solver")
    .add_local_file(REPO_ROOT / "jax_training_util.py", remote_path="/repo/jax_training_util.py")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _setup_repo_layout() -> Path:
    """Symlink /repo/data -> /data and /repo/outputs -> /data/outputs so the trainer's
    REPO_ROOT-relative paths land on the persistent volume."""
    import os
    import sys

    repo = Path("/repo")
    data_link = repo / "data"
    out_link = repo / "outputs"
    out_dir = Path(VOLUME_MOUNT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_link.exists():
        data_link.symlink_to(VOLUME_MOUNT)
    if not out_link.exists():
        out_link.symlink_to(out_dir)

    os.environ["PYTHONPATH"] = "/repo:/repo/models/fno-jax:/repo/models/dno-net:/repo/train-jax-10m"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for p in ("/repo", "/repo/models/fno-jax", "/repo/models/dno-net", "/repo/train-jax-10m"):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=600)
def status() -> dict:
    """List dataset and run files currently on the volume."""
    import json
    import os

    summary: dict[str, list[str]] = {"datasets": [], "runs": []}
    for name in sorted(os.listdir(VOLUME_MOUNT)):
        if name.endswith(".npz") or name.endswith(".meta.json"):
            sz = os.path.getsize(f"{VOLUME_MOUNT}/{name}") / 1e9
            summary["datasets"].append(f"{name}  ({sz:.2f} GB)")
    out_dir = f"{VOLUME_MOUNT}/outputs"
    if os.path.isdir(out_dir):
        for name in sorted(os.listdir(out_dir)):
            summary["runs"].append(name)
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def upload_data(
    train_dataset: str = "combined_dataset.npz",
    test_dataset: str = "test_dno_rescaled.npz",
    local_dir: str = "data",
) -> None:
    """Push training + test datasets (and their .meta.json sidecars) onto the volume."""
    src = Path(local_dir).resolve()
    targets: list[tuple[Path, str]] = []
    for name in (train_dataset, test_dataset):
        npz = src / name
        if not npz.exists():
            raise FileNotFoundError(f"missing local file: {npz}")
        targets.append((npz, f"/{name}"))
        sidecar = npz.with_suffix(".meta.json")
        if sidecar.exists():
            targets.append((sidecar, f"/{sidecar.name}"))

    total = sum(p.stat().st_size for p, _ in targets) / 1e9
    print(f"Uploading {len(targets)} files ({total:.2f} GB) -> volume {VOLUME_NAME!r}")
    with volume.batch_upload(force=True) as batch:
        for path, remote in targets:
            print(f"  {path.name}  ({path.stat().st_size / 1e9:.2f} GB) -> {remote}")
            batch.put_file(str(path), remote)
    print("upload complete.")


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
    test_dataset: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    modes: int,
    width: int,
    n_blocks: int,
    sobolev_k: int,
    norm: str,
    model_kind: str,
    seed: int,
    skip_dno_eval: bool,
    latent: int,
    cs_mult_hidden: int,
    cs_use_g1_baseline: bool,
    cs_g1_k_cut: int,
    cs_tie_xi_out_mult: bool,
    hamiltonian_weight: float,
    hamiltonian_dt: float,
    hamiltonian_warmup_steps: int,
    hamiltonian_clip: float,
    pushforward_steps: int,
    pushforward_weight: float,
    pushforward_dt: float,
    pushforward_order: int,
    pushforward_pad_factor: int,
    psd_hinge_weight: float,
    psd_hinge_warmup_steps: int,
    precision: str,
    resume_from: str,
    total_epochs: int,
    reset_opt_state: bool,
    keep_schedule_step: bool,
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import time
    from time import perf_counter

    _setup_repo_layout()
    cmd = [
        sys.executable, "/repo/train-jax-10m/1d_dno_fno_jax.py",
        "--dataset", dataset,
        "--dno_eval_dataset", test_dataset,
        "--precision", precision,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--modes", str(modes),
        "--width", str(width),
        "--n_blocks", str(n_blocks),
        "--sobolev_k", str(sobolev_k),
        "--norm", norm,
        "--model", model_kind,
        "--seed", str(seed),
        "--run_name", run_name,
        "--latent", str(latent),
        "--cs_mult_hidden", str(cs_mult_hidden),
        "--cs_g1_k_cut", str(cs_g1_k_cut),
        "--hamiltonian_weight", str(hamiltonian_weight),
        "--hamiltonian_dt", str(hamiltonian_dt),
        "--hamiltonian_warmup_steps", str(hamiltonian_warmup_steps),
        "--hamiltonian_clip", str(hamiltonian_clip),
        "--pushforward_steps", str(pushforward_steps),
        "--pushforward_weight", str(pushforward_weight),
        "--pushforward_dt", str(pushforward_dt),
        "--pushforward_order", str(pushforward_order),
        "--pushforward_pad_factor", str(pushforward_pad_factor),
        "--psd_hinge_weight", str(psd_hinge_weight),
        "--psd_hinge_warmup_steps", str(psd_hinge_warmup_steps),
    ]
    if skip_dno_eval:
        cmd.append("--skip_dno_eval")
    if cs_use_g1_baseline:
        cmd.append("--cs_use_g1_baseline")
    if cs_tie_xi_out_mult:
        cmd.append("--cs_tie_xi_out_mult")
    if resume_from:
        cmd.extend(["--resume_from", resume_from])
    if total_epochs > 0:
        cmd.extend(["--total_epochs", str(total_epochs)])
    if reset_opt_state:
        cmd.append("--reset_opt_state")
    if keep_schedule_step:
        cmd.append("--keep_schedule_step")

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
                rc = proc.wait(timeout=30.0)
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


@app.local_entrypoint()
def train(
    dataset: str = "combined_dataset.npz",
    test_dataset: str = "test_dno_rescaled.npz",
    run_name: str = "",
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    modes: int = 64,
    width: int = 64,
    n_blocks: int = 4,
    sobolev_k: int = 1,
    norm: str = "scale",
    model_kind: str = "fno",
    seed: int = 0,
    skip_dno_eval: bool = False,
    latent: int = 64,
    cs_mult_hidden: int = 32,
    cs_use_g1_baseline: bool = False,
    cs_g1_k_cut: int = 128,
    cs_tie_xi_out_mult: bool = False,
    hamiltonian_weight: float = 0.0,
    hamiltonian_dt: float = 0.01,
    hamiltonian_warmup_steps: int = 10000,
    hamiltonian_clip: float = 10.0,
    pushforward_steps: int = 0,
    pushforward_weight: float = 1.0,
    pushforward_dt: float = 0.01,
    pushforward_order: int = 4,
    pushforward_pad_factor: int = 4,
    psd_hinge_weight: float = 0.0,
    psd_hinge_warmup_steps: int = 10000,
    precision: str = "fp32",
    resume_from: str = "",
    total_epochs: int = 0,
    reset_opt_state: bool = False,
    keep_schedule_step: bool = False,
) -> None:
    """Local entrypoint: dispatch a training run on a Modal GPU worker.

    GPU shape is taken from the ``MODAL_GPU`` env var at module import time
    (defaults to ``H100:1``). Set it before invoking modal CLI, e.g.
    ``MODAL_GPU=H100:2 modal run modal_train.py::train ...``.
    """
    from datetime import datetime

    if not run_name:
        run_name = datetime.now().strftime(f"fno_jax_10m_%Y%m%d_%H%M%S")
    print(f"run_name = {run_name}")
    print(f"gpu      = {DEFAULT_GPU}")

    result = run_training.remote(
        dataset=dataset,
        test_dataset=test_dataset,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        modes=modes,
        width=width,
        n_blocks=n_blocks,
        sobolev_k=sobolev_k,
        norm=norm,
        model_kind=model_kind,
        seed=seed,
        skip_dno_eval=skip_dno_eval,
        latent=latent,
        cs_mult_hidden=cs_mult_hidden,
        cs_use_g1_baseline=cs_use_g1_baseline,
        cs_g1_k_cut=cs_g1_k_cut,
        cs_tie_xi_out_mult=cs_tie_xi_out_mult,
        hamiltonian_weight=hamiltonian_weight,
        hamiltonian_dt=hamiltonian_dt,
        hamiltonian_warmup_steps=hamiltonian_warmup_steps,
        hamiltonian_clip=hamiltonian_clip,
        pushforward_steps=pushforward_steps,
        pushforward_weight=pushforward_weight,
        pushforward_dt=pushforward_dt,
        pushforward_order=pushforward_order,
        pushforward_pad_factor=pushforward_pad_factor,
        psd_hinge_weight=psd_hinge_weight,
        psd_hinge_warmup_steps=psd_hinge_warmup_steps,
        precision=precision,
        resume_from=resume_from,
        total_epochs=total_epochs,
        reset_opt_state=reset_opt_state,
        keep_schedule_step=keep_schedule_step,
    )
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
        rel = entry.path[len(prefix):].lstrip("/")
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
