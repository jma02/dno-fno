from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


NX_VALS = [64, 128, 256, 512, 1024]
M_VALS = [4, 5, 6, 7, 8]
ICHOI_VALS = [0, 1]


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def save_manifest(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def find_run_dir_for_combo(outputs_dir: Path, nx: int, m: int, ichoi: int, epochs: int, campaign_tag: str) -> str | None:
    pattern = f"fno_Nx{nx}_M{m}_ichoi{ichoi}_p*_ep{epochs}_{campaign_tag}"
    candidates = sorted(outputs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FNO grid search over pre-generated Stokes datasets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--plot-every", type=int, default=100)
    parser.add_argument("--campaign-tag", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    grid_data_dir = repo_root / "grid_data"
    processed_dir = repo_root / "data"
    outputs_dir = repo_root / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    campaign_tag = args.campaign_tag.strip() or datetime.now().strftime("gridfno_%Y%m%d_%H%M%S")
    manifest_path = outputs_dir / f"{campaign_tag}_manifest.json"
    manifest: Dict[str, Any] = {
        "campaign_tag": campaign_tag,
        "created_at": utc_now(),
        "settings": {
            "device": args.device,
            "epochs": args.epochs,
            "plot_every": args.plot_every,
            "nx_vals": NX_VALS,
            "m_vals": M_VALS,
            "ichoi_vals": ICHOI_VALS,
        },
        "records": [],
    }
    save_manifest(manifest_path, manifest)
    print(f"[Campaign] {campaign_tag}")
    print(f"[Manifest] {manifest_path}")

    python_exe = sys.executable
    process_script = repo_root / "train" / "process_stokes.py"
    train_script = repo_root / "train" / "1d_dno_fno.py"

    for m in M_VALS:
        m_dir = grid_data_dir / f"M{m}"
        for nx in NX_VALS:
            for ichoi in ICHOI_VALS:
                mat_filename = f"stokes_dno_training_Nx{nx}_M{m}_ichoi{ichoi}.mat"
                mat_path = m_dir / mat_filename
                npz_filename = mat_path.stem + ".npz"
                npz_path = processed_dir / npz_filename

                record: Dict[str, Any] = {
                    "timestamp": utc_now(),
                    "campaign_tag": campaign_tag,
                    "combo": {"Nx": nx, "M": m, "ichoi": ichoi},
                    "mat_path": str(mat_path),
                    "npz_path": str(npz_path),
                    "status": "started",
                    "run_dir": None,
                }

                print(f"\n=== M={m} Nx={nx} ichoi={ichoi} ===")

                if not mat_path.exists():
                    msg = f"Missing input file: {mat_path}"
                    print(f"[Skip] {msg}")
                    record["status"] = "missing_input"
                    record["error"] = msg
                    manifest["records"].append(record)
                    save_manifest(manifest_path, manifest)
                    continue

                if not npz_path.exists():
                    process_cmd = [
                        python_exe,
                        str(process_script),
                        "--mat-file",
                        str(mat_path),
                        "--out-dir",
                        str(processed_dir),
                    ]
                    print(f"[Preprocess] {' '.join(process_cmd)}")
                    try:
                        subprocess.run(process_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[Error] Preprocessing failed (exit={e.returncode})")
                        record["status"] = "preprocess_failed"
                        record["returncode"] = e.returncode
                        record["stage"] = "preprocess"
                        manifest["records"].append(record)
                        save_manifest(manifest_path, manifest)
                        continue
                else:
                    print(f"[Preprocess] Reusing {npz_filename}")

                train_cmd = [
                    python_exe,
                    str(train_script),
                    "--dataset",
                    npz_filename,
                    "--device",
                    args.device,
                    "--epochs",
                    str(args.epochs),
                    "--plot-every",
                    str(args.plot_every),
                    "--campaign-tag",
                    campaign_tag,
                ]
                print(f"[Train] {' '.join(train_cmd)}")
                try:
                    subprocess.run(train_cmd, check=True)
                    record["status"] = "ok"
                    record["run_dir"] = find_run_dir_for_combo(outputs_dir, nx, m, ichoi, args.epochs, campaign_tag)
                    if record["run_dir"] is None:
                        record["warning"] = "training_succeeded_but_run_dir_not_found"
                except subprocess.CalledProcessError as e:
                    print(f"[Error] Training failed (exit={e.returncode})")
                    record["status"] = "train_failed"
                    record["returncode"] = e.returncode
                    record["stage"] = "train"

                manifest["records"].append(record)
                save_manifest(manifest_path, manifest)

    counts = Counter(rec["status"] for rec in manifest["records"])
    manifest["finished_at"] = utc_now()
    manifest["counts"] = dict(sorted(counts.items()))
    save_manifest(manifest_path, manifest)
    total = len(manifest["records"])
    print(f"\n[Summary] total={total} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[Manifest] {manifest_path}")


if __name__ == "__main__":
    main()
