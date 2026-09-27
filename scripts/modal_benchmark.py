"""Run an explicit benchmark command on the personal jma02 Modal account.

MODAL_PROFILE=jma02 modal run scripts/modal_benchmark.py --command \
  'python scripts/benchmark_dno_dense.py --output outputs/modal_result/dense.json'
Only listed code, the candidate checkpoint, and four reference cases/family upload.
"""
from __future__ import annotations

from datetime import datetime
import io
import os
from pathlib import Path
import shlex
import subprocess
import zipfile

os.environ["MODAL_PROFILE"] = "jma02"
import modal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("curl", "ca-certificates")
    .pip_install("jax[cuda12]==0.9.2", "flax==0.12.6", "optax==0.2.5",
                 "orbax-checkpoint==0.11.33", "numpy==2.4.2", "scipy", "matplotlib", "tqdm")
    .run_commands(
        "mkdir -p /repo/outputs/cufftdx_20260925",
        "curl -fL https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/"
        "nvidia-mathdx-25.12.1-cuda12.tar.gz | tar -xz -C /repo/outputs/cufftdx_20260925",
    )
    .env({"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
          "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
    .workdir("/repo")
)
if os.environ.get("DNO_MODAL_CUDA13") == "1":
    image = image.run_commands(
        "python -m venv /opt/cuda13",
        "/opt/cuda13/bin/pip install 'jax[cuda13]==0.9.2' flax==0.12.6 optax==0.2.5 "
        "orbax-checkpoint==0.11.33 numpy==2.4.2 scipy matplotlib tqdm "
        "nvidia-cufft==12.1.0.31 nvidia-nvjitlink==13.1.80 nvidia-cuda-nvcc==13.1.80 "
        "nvidia-cuda-nvrtc==13.1.80 nvidia-cuda-runtime==13.1.80 nvidia-cuda-cupti==13.1.75",
    )
app = modal.App("dno-kernel-benchmarks", include_source=False)


@app.function(image=image, gpu=os.environ.get("DNO_MODAL_GPU", "RTX-PRO-6000"),
              timeout=3600, retries=0, max_containers=1, scaledown_window=5,
              cpu=4, memory=16384, serialized=True, include_source=False)
def benchmark(payload: bytes, command: str, result_dir: str) -> tuple[int, dict[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall("/repo")
    completed = subprocess.run(shlex.split(command), cwd="/repo", check=False)
    results = Path("/repo") / result_dir
    files = {str(path.relative_to(results)): path.read_bytes() for path in results.rglob("*")
             if path.is_file() and path.suffix in (".json", ".npz", ".png", ".pdf", ".hlo", ".log", ".gz", ".pb")}
    return completed.returncode, files


@app.local_entrypoint()
def main(command: str, result_dir: str = "outputs/modal_result", output: str = "", extra_files: str = "") -> None:
    destination = ROOT / (output or f"outputs/modal_{datetime.now():%Y%m%d_%H%M%S}")
    destination.mkdir(parents=True, exist_ok=False)
    code = (
        "scripts/benchmark_cufftdx.py", "scripts/cufftdx_fft.cu", "scripts/benchmark_dno_fusion.py",
        "scripts/benchmark_surrogate_rhs.py",
        "scripts/benchmark_dno_dense.py", "scripts/time_single_rollouts.py", "scripts/profile_dno.py",
        "models/dno-net/dno_net_v2.py", "models/fno-jax/fno1d.py",
        "solver/__init__.py", "solver/solvers/__init__.py", "solver/solvers/dno_series_jax.py",
        "solver/solvers/time_integrator.py", "solver/evals/__init__.py", "solver/evals/model_rollout.py",
        "solver/evals/eval_suite.py", "solver/evals/compare_rollout_jax.py", "solver/evals/render_rollout_movie.py",
        "solver/reference_solutions/__init__.py", "solver/reference_solutions/solitary_wave.py",
        "solver/reference_solutions/stokes_wave.py", "solver/gen_data/__init__.py",
        "solver/gen_data/pipeline/__init__.py", "solver/gen_data/pipeline/types.py",
    )
    run = ROOT / "outputs/c27_w320_b4_h80_tanaka_hard128_20260924"
    selected = [*(ROOT / name for name in code), run / "config.json",
                *(p for p in (run / "best_val_ckpt").rglob("*") if p.is_file()),
                *(ROOT / name for name in shlex.split(extra_files))]
    # Modal's isolated CLI environment need not contain NumPy; use the project interpreter.
    trimmed = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", """
import io, sys, zipfile
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
    for family in ('stokes', 'tanaka', 'benjamin_feir', 'jonswap_tma'):
        path = next((root / 'outputs/c27_tanaka_hard128_full_equal_local_20260918').glob(
            f'eval_best_current_test_stratified_n32*/{family}_trajs.npz'))
        with np.load(path) as data:
            fields = {key: data[key][:, :4] for key in ('truth_eta', 'truth_xi', 'truth_gxi')}
            fields.update(times=data['times'], depths=data['depths'][:4], simulation_ids=data['simulation_ids'][:4])
            contents = io.BytesIO()
            np.savez_compressed(contents, **fields)
            archive.writestr(str(path.relative_to(root)), contents.getvalue())
sys.stdout.buffer.write(buffer.getvalue())
""", str(ROOT)], check=True, capture_output=True)
    payload = io.BytesIO(trimmed.stdout)
    with zipfile.ZipFile(payload, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, str(path.relative_to(ROOT)))
        print(f"Uploading {len(archive.namelist())} selected files, {payload.getbuffer().nbytes / 1e6:.1f} MB", flush=True)
    returncode, files = benchmark.remote(payload.getvalue(), command, result_dir)
    for name, contents in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    print(f"Saved {len(files)} result files to {destination}; command exit status {returncode}", flush=True)
    if returncode:
        raise SystemExit(returncode)
