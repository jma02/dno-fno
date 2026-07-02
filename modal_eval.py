"""Run `solver.evals.eval_suite` on Modal against a checkpoint stored on the volume.

Usage:
    # One-time: push tanaka trajectories to the volume (shared with modal_train).
    modal run modal_eval.py::upload_tanaka

    # One-time per model: push the run's config + best_val_ckpt.
    modal run modal_eval.py::upload_run \
        --local-run-dir outputs/cs_dno_w512b8_l256_v8_jacreg_20260629_142544

    # Run tanaka_g0 eval on A100.
    modal run modal_eval.py::eval_tanaka \
        --run-name cs_dno_w512b8_l256_v8_jacreg_20260629_142544

Downloads the resulting summary JSON back to the local run dir on completion.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "dno-fno-eval"
VOLUME_NAME = "dno-fno-train-data"
VOLUME_MOUNT = "/data"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100-40GB:1")
TIMEOUT_SECONDS = 6 * 3600
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


@app.local_entrypoint()
def upload_tanaka(g0: bool = True, g1: bool = False, local_dir: str = "data") -> None:
    """Push tanaka_2_adaptive_g{0,1}.npz onto the volume."""
    src = Path(local_dir).resolve()
    names: list[str] = []
    if g0:
        names.append("tanaka_2_adaptive_g0.npz")
    if g1:
        names.append("tanaka_2_adaptive_g1.npz")
    targets: list[tuple[Path, str]] = []
    for name in names:
        npz = src / name
        if not npz.exists():
            raise FileNotFoundError(npz)
        targets.append((npz, f"/{name}"))
    total = sum(p.stat().st_size for p, _ in targets) / 1e9
    print(f"Uploading {len(targets)} files ({total:.2f} GB) -> volume {VOLUME_NAME!r}")
    with volume.batch_upload(force=True) as batch:
        for path, remote in targets:
            print(f"  {path.name}  ({path.stat().st_size / 1e9:.2f} GB) -> {remote}")
            batch.put_file(str(path), remote)
    print("tanaka upload complete.")


@app.local_entrypoint()
def upload_run(local_run_dir: str) -> None:
    """Push config.json + best_val_ckpt/ from a local run to /data/outputs/<name>/."""
    src = Path(local_run_dir).resolve()
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"no config.json in {src}")
    if not (src / "best_val_ckpt").is_dir():
        raise FileNotFoundError(f"no best_val_ckpt/ in {src}")
    run_name = src.name
    remote_base = f"/outputs/{run_name}"
    targets: list[tuple[Path, str]] = [(src / "config.json", f"{remote_base}/config.json")]
    for p in (src / "best_val_ckpt").rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            targets.append((p, f"{remote_base}/{rel.as_posix()}"))
    total = sum(p.stat().st_size for p, _ in targets) / 1e6
    print(f"Uploading {len(targets)} files ({total:.1f} MB) -> {remote_base}")
    with volume.batch_upload(force=True) as batch:
        for path, remote in targets:
            batch.put_file(str(path), remote)
    print("run upload complete.")


@app.function(
    gpu=DEFAULT_GPU,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    cpu=4,
    memory=64 * 1024,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0, initial_delay=0.0),
)
def run_eval(
    *,
    run_name: str,
    regimes: list[str],
    checkpoint: str = "best",
    n_ics: int = 16,
    f64_harness: bool = False,
    filter_gxi: bool = False,
    filter_shape: str = "hard",
    houli_a: float = 36.0,
    houli_m: float = 36.0,
    filter_fraction: float = 1.0,
    output_tag: str = "modal_a100",
) -> dict:
    import json
    import subprocess
    import sys

    _setup_repo_layout()
    run_dir = f"/repo/outputs/{run_name}"
    output_dir = f"{run_dir}/eval_{output_tag}"
    cmd = [
        sys.executable, "-m", "solver.evals.eval_suite",
        "--run_dir", run_dir,
        "--checkpoint", checkpoint,
        "--regimes", *regimes,
        "--n_ics", str(n_ics),
        "--output_dir", output_dir,
        "--gpu",
    ]
    if f64_harness:
        cmd.append("--f64_harness")
    if filter_gxi:
        cmd += [
            "--filter_gxi",
            "--filter_shape", filter_shape,
            "--houli_a", str(houli_a),
            "--houli_m", str(houli_m),
            "--filter_fraction", str(filter_fraction),
        ]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd="/repo")

    summaries: dict[str, dict] = {}
    for regime in regimes:
        sp = Path(output_dir) / f"{regime}_summary.json"
        if sp.exists():
            summaries[regime] = json.loads(sp.read_text())
    return {"output_dir": output_dir, "summaries": summaries}


@app.local_entrypoint()
def eval_tanaka(
    run_name: str,
    regime: str = "tanaka_g0",
    checkpoint: str = "best",
    f64_harness: bool = False,
    filter_gxi: bool = False,
    houli_m: float = 36.0,
    filter_fraction: float = 1.0,
    output_tag: str = "modal_a100",
) -> None:
    """Run tanaka rollout eval on Modal A100 (single GPU)."""
    import json

    result = run_eval.remote(
        run_name=run_name,
        regimes=[regime],
        checkpoint=checkpoint,
        f64_harness=f64_harness,
        filter_gxi=filter_gxi,
        houli_m=houli_m,
        filter_fraction=filter_fraction,
        output_tag=output_tag,
    )
    print(json.dumps(result, indent=2))

    local_out = REPO_ROOT / "outputs" / run_name / f"eval_{output_tag}"
    local_out.mkdir(parents=True, exist_ok=True)
    for regime_name, summary in result.get("summaries", {}).items():
        (local_out / f"{regime_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summaries written to {local_out}")
