"""Short four-family rollout of the saved epoch-one model on one L4."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/torch_epoch1_rollouts_20260926"
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_file(OUTPUT / "initial_conditions.npz", "/input.npz"))
app = modal.App("torch-epoch1-rollouts")


@app.function(image=image, volumes={"/data": volume}, gpu="L4", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run() -> bytes:
    import sys

    sys.path.insert(0, "/repo/torch-attention")
    from rollout_eval import main

    output = Path("/data/experiments/torch_epoch1_rollouts_20260926/predictions.npz")
    sys.argv = ["rollout_eval.py", "--checkpoint",
                "/data/experiments/torch_spectral_width256_full_epoch_20260926/checkpoint.pt",
                "--input", "/input.npz", "--out", str(output), "--device", "cuda", "--substeps", "8"]
    main()
    volume.commit()
    return output.read_bytes()


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    with (OUTPUT / "predictions.npz").open("xb") as handle:
        handle.write(result)
    print(f"Saved {OUTPUT / 'predictions.npz'}")
