"""Read-only CPU extraction of a balanced, split-preserving local dataset sample."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import modal

app = modal.App("dno-local-validation-subset")
volume = modal.Volume.from_name("dno-fno-train-data").with_mount_options(read_only=True)


@app.function(image=modal.Image.debian_slim().pip_install("numpy==2.4.2"),
              volumes={"/data": volume}, cpu=4, memory=8192, timeout=600, retries=0)
def extract() -> bytes:
    import numpy as np

    root = Path("/data/outputs/paper_dataset_full_equal_20260916/arrays")
    splits = np.load(root / "dataset_split.npy", mmap_mode="r")
    families = np.load(root / "family_id.npy", mmap_mode="r")
    simulations = np.load(root / "simulation_id.npy", mmap_mode="r")
    rng = np.random.default_rng(24)
    selected = []
    counts = {}
    for split, per_family in (("train", 1024), ("validation", 256)):
        for family in (1, 2, 3, 4):
            candidates = np.flatnonzero((splits == split) & (families == family))
            rows = rng.choice(candidates, per_family, replace=False)
            selected.append(rows)
            counts[f"{split}_{family}"] = per_family
    rows = np.sort(np.concatenate(selected))
    train_simulations = np.unique(simulations[rows[splits[rows] == "train"]])
    validation_simulations = np.unique(simulations[rows[splits[rows] == "validation"]])
    assert not np.intersect1d(train_simulations, validation_simulations).size
    manifest = {"source_volume": "dno-fno-train-data", "source": str(root), "seed": 24,
                "sampling": "Uniform rows without replacement within each original split/family",
                "family_counts": counts, "rows": len(rows),
                "train_simulations": len(train_simulations),
                "validation_simulations": len(validation_simulations),
                "train_validation_simulation_overlap": 0, "arrays": {}}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in ("eta", "xi", "gxi", "depth", "family_id", "simulation_id", "dataset_split", "x", "source_indices"):
            if name == "source_indices":
                values = rows
            else:
                source = np.load(root / f"{name}.npy", mmap_mode="r")
                values = np.asarray(source if name == "x" else source[rows])
            if np.issubdtype(values.dtype, np.number):
                assert np.isfinite(values).all(), name
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            payload = buffer.getvalue()
            archive.writestr(f"{name}.npy", payload)
            manifest["arrays"][name] = {"shape": list(values.shape), "dtype": str(values.dtype),
                                          "sha256": hashlib.sha256(payload).hexdigest()}
            print(f"Extracted {name}: {values.shape}, {values.dtype}", flush=True)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    return output.getvalue()


@app.local_entrypoint()
def main(destination: str) -> None:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    payload = extract.remote()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, info in manifest["arrays"].items():
            data = archive.read(f"{name}.npy")
            assert hashlib.sha256(data).hexdigest() == info["sha256"], name
            (target / f"{name}.npy").write_bytes(data)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"destination": str(target), "download_bytes": len(payload), **manifest}, indent=2))
