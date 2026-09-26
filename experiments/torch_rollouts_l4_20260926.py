"""Screen saved model rollouts on one L4."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/torch_epoch1_rollouts_20260926"
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention"))
app = modal.App("torch-epoch1-rollouts")


@app.function(image=image, volumes={"/data": volume}, gpu="L4", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(checkpoint_run: str, output_name: str, input_bytes: bytes) -> bytes:
    import sys

    sys.path.insert(0, "/repo/torch-attention")
    from rollout_eval import main

    Path("/tmp/input.npz").write_bytes(input_bytes)
    output = Path("/data/experiments") / output_name / "predictions.npz"
    sys.argv = ["rollout_eval.py", "--checkpoint",
                f"/data/experiments/{checkpoint_run}/checkpoint.pt",
                "--input", "/tmp/input.npz", "--out", str(output), "--device", "cuda", "--substeps", "8"]
    main()
    volume.commit()
    return output.read_bytes()


@app.local_entrypoint()
def main(checkpoint_run: str = "torch_spectral_width256_full_epoch_20260926",
         output_name: str = "torch_epoch1_rollouts_20260926",
         input_file: str = str(OUTPUT / "initial_conditions.npz")) -> None:
    import shutil

    destination = ROOT / "outputs" / output_name
    if (destination / "predictions.npz").exists():
        raise FileExistsError(destination / "predictions.npz")
    destination.mkdir(parents=True, exist_ok=True)
    if Path(input_file).resolve() != (destination / "initial_conditions.npz").resolve():
        shutil.copyfile(input_file, destination / "initial_conditions.npz")
    result = run.remote(checkpoint_run, output_name, Path(input_file).read_bytes())
    with (destination / "predictions.npz").open("xb") as handle:
        handle.write(result)
    print(f"Saved {destination / 'predictions.npz'}")
