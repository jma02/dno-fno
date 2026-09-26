"""Inventory the full validation rows and simulation groups without attaching a GPU."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
app = modal.App("audit-validation-rollout-scope")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.4.2")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=300)
def audit() -> dict:
    import numpy as np
    from time import perf_counter

    started = perf_counter()
    root = Path("/data/outputs/paper_dataset_full_equal_20260916/arrays")
    files = {p.name: p.stat().st_size for p in root.iterdir() if p.is_file()}
    split = np.load(root / "dataset_split.npy", mmap_mode="r")
    rows = np.flatnonzero(split == "validation")
    ids = np.load(root / "simulation_id.npy", mmap_mode="r")[rows]
    families = np.load(root / "family_id.npy", mmap_mode="r")[rows]
    result = {"source": str(root), "files": files, "validation_rows": len(rows), "families": {}}
    for f, name in enumerate(("stokes", "tanaka", "benjamin_feir", "jonswap_tma"), start=1):
        mask = families == f
        unique, counts = np.unique(ids[mask], return_counts=True)
        result["families"][name] = {"rows": int(mask.sum()), "unique_simulations": len(unique),
                                     "rows_per_simulation_min": int(counts.min()),
                                     "rows_per_simulation_median": float(np.median(counts)),
                                     "rows_per_simulation_max": int(counts.max())}
    result["seconds"] = perf_counter() - started
    return result


@app.local_entrypoint()
def main() -> None:
    result = audit.remote()
    (ROOT / "experiments/audit_validation_rollout_scope_20260926.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
