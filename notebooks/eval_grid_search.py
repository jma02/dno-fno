import argparse
import csv
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from models.fno import FNO1d, count_params

fno_train = importlib.import_module("train.1d_dno_fno")
load_processed_dataset = fno_train.load_processed_dataset
build_feature_target_tensors = fno_train.build_feature_target_tensors
normalize_to_range = fno_train.normalize_to_range
evaluate = fno_train.evaluate
build_dataloader = fno_train.build_dataloader


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    state_path = run_dir / "best_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {run_dir}")
    config["state_path"] = state_path
    return config


def eval_single_run(run_dir: Path, data_dir: Path, device: str = "cuda:0"):
    print(f"Evaluating {run_dir.name}...")
    try:
        config = load_run_config(run_dir)

        import re
        match = re.search(r"Nx(\d+)_M(\d+)_ichoi(\d+)", run_dir.name)
        if not match:
            print(f"Could not parse params from {run_dir.name}")
            return None

        nx, m, ichoi = match.groups()
        dataset_name = f"stokes_dno_training_Nx{nx}_M{m}_ichoi{ichoi}.npz"
        dataset_path = data_dir / dataset_name

        if not dataset_path.exists():
            print(f"Dataset {dataset_name} missing for {run_dir.name}")
            return None

        dataset_np = load_processed_dataset(dataset_path)
        features_raw, targets_raw = build_feature_target_tensors(dataset_np)

        features, _ = normalize_to_range(features_raw)
        targets, _ = normalize_to_range(targets_raw)

        base_dataset = TensorDataset(features, targets)
        n_samples = len(base_dataset)

        test_fraction = 0.1
        indices = np.arange(n_samples)
        test_size = int(n_samples * test_fraction)
        test_idx = indices[:test_size]
        test_loader = build_dataloader(base_dataset, test_idx, batch_size=128, shuffle=False)

        n_blocks = config.get("n_blocks", 10)
        model = FNO1d(
            config["modes"],
            config["width"],
            n_blocks=n_blocks,
            mode_sampling=config.get("mode_sampling", "low"),
            high_mode_frac=config.get("high_mode_frac", 0.5),
        ).to(device)
        model.load_state_dict(torch.load(config["state_path"], map_location=device))

        test_loss = evaluate(model, test_loader, torch.device(device), config.get("loss", "sobolev"))

        return {
            "Nx": int(nx),
            "M": int(m),
            "ichoi": int(ichoi),
            "run_dir": run_dir.name,
            "test_loss": test_loss,
            "param_count": count_params(model),
        }
    except Exception as e:
        print(f"Error evaluating {run_dir.name}: {e}")
        return None


def write_csv_fallback(rows: list[dict], csv_path: Path) -> None:
    fieldnames = ["Nx", "M", "ichoi", "run_dir", "test_loss", "param_count"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Evaluate FNO grid-search runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name-contains", default="")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    data_dir = Path(args.data_dir)

    results = []
    for run_dir in outputs_dir.glob("fno_*"):
        if not run_dir.is_dir():
            continue
        if args.name_contains and args.name_contains not in run_dir.name:
            continue
        if not (run_dir / "best_model.pt").exists():
            continue
        res = eval_single_run(run_dir, data_dir, device=args.device)
        if res:
            results.append(res)

    if not results:
        print("No valid runs found to evaluate.")
        return

    results = sorted(results, key=lambda r: (r["M"], r["Nx"], r["ichoi"]))
    if args.csv_out:
        csv_path = Path(args.csv_out)
    elif args.name_contains:
        csv_path = outputs_dir / f"{args.name_contains}_results.csv"
    else:
        csv_path = outputs_dir / "grid_search_results.csv"

    if pd is not None:
        df = pd.DataFrame(results)
        print("\n=== Grid Search Evaluation Results ===")
        print(df.to_string(index=False))
        df.to_csv(csv_path, index=False)
    else:
        print("\n=== Grid Search Evaluation Results ===")
        for row in results:
            print(row)
        write_csv_fallback(results, csv_path)

    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
