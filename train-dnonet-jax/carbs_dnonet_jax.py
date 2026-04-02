from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from carbs.carbs import CARBS
from carbs.utils import CARBSParams, LinearSpace, LogSpace, ObservationInParam, Param

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_BEST_SEARCH_CENTER = {
    "lr": 1.2784948703599842e-4,
    "weight_decay": 2.976304745294121e-4,
    "batch_size": 64,
    "epochs": 386,
    "rank": 256,
    "width": 112,
    "spectral_width": 112,
    "spectral_floor": 1e-3,
    "modes": 64,
    "n_blocks": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARBS over the JAX DNONet trainer")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default="outputs/carbs_dnonet_jax")
    parser.add_argument("--num_random_samples", type=int, default=4)
    parser.add_argument("--max_suggestion_cost", type=float, default=None)
    parser.add_argument("--rank_max", type=int, default=1024)
    parser.add_argument("--width_max", type=int, default=256)
    parser.add_argument("--epoch_max", type=int, default=600)
    return parser.parse_args()


def resolve_dataset_path(dataset: str) -> Path:
    dataset_path = Path(dataset)
    if dataset_path.is_absolute():
        return dataset_path
    return REPO_ROOT / "data" / dataset


def infer_grid_size(dataset_path: Path, sources: str) -> int:
    selected_sources = [
        source.strip().lower()
        for source in sources.split(",")
        if source.strip()
    ]
    if not selected_sources or selected_sources == ["all"]:
        selected_sources = ["soliton", "stokes", "linear"]

    with np.load(dataset_path) as npz:
        first_source = selected_sources[0]
        eta_key = f"{first_source}_eta"
        if eta_key not in npz.files:
            raise KeyError(f"Dataset missing key {eta_key}")
        return int(npz[eta_key].shape[1])


def build_carbs(
    seed: int,
    num_random_samples: int,
    max_suggestion_cost: float | None,
    rank_max: int,
    width_max: int,
    epoch_max: int,
) -> CARBS:
    rank_min = min(64, rank_max)
    param_spaces = [
        Param("lr", LogSpace(min=3e-5, max=3e-3), search_center=V2_BEST_SEARCH_CENTER["lr"]),
        Param(
            "weight_decay",
            LogSpace(min=1e-6, max=1e-3),
            search_center=V2_BEST_SEARCH_CENTER["weight_decay"],
        ),
        Param(
            "batch_size",
            LinearSpace(min=16, max=128, scale=16, is_integer=True, rounding_factor=16),
            search_center=V2_BEST_SEARCH_CENTER["batch_size"],
        ),
        Param(
            "epochs",
            LogSpace(min=25, max=epoch_max, is_integer=True),
            search_center=min(V2_BEST_SEARCH_CENTER["epochs"], epoch_max),
        ),
        Param(
            "rank",
            LinearSpace(min=rank_min, max=rank_max, scale=64, is_integer=True, rounding_factor=32),
            search_center=min(max(rank_min, V2_BEST_SEARCH_CENTER["rank"]), rank_max),
        ),
        Param(
            "width",
            LinearSpace(min=32, max=width_max, scale=32, is_integer=True, rounding_factor=16),
            search_center=min(V2_BEST_SEARCH_CENTER["width"], width_max),
        ),
        Param(
            "spectral_width",
            LinearSpace(min=32, max=width_max, scale=32, is_integer=True, rounding_factor=16),
            search_center=min(V2_BEST_SEARCH_CENTER["spectral_width"], width_max),
        ),
        Param(
            "spectral_floor",
            LogSpace(min=1e-6, max=1e-2),
            search_center=V2_BEST_SEARCH_CENTER["spectral_floor"],
        ),
        Param(
            "modes",
            LinearSpace(min=32, max=128, scale=16, is_integer=True, rounding_factor=16),
            search_center=V2_BEST_SEARCH_CENTER["modes"],
        ),
        Param(
            "n_blocks",
            LinearSpace(min=4, max=10, scale=2, is_integer=True),
            search_center=V2_BEST_SEARCH_CENTER["n_blocks"],
        ),
    ]
    carbs_params = CARBSParams(
        better_direction_sign=-1,
        seed=seed,
        is_wandb_logging_enabled=False,
        is_saved_on_every_observation=False,
        resample_frequency=0,
        num_random_samples=num_random_samples,
        max_suggestion_cost=max_suggestion_cost,
    )
    return CARBS(carbs_params, param_spaces)


def run_trial(
    suggestion: dict[str, float | int],
    dataset: str,
    sources: str,
    output_root: str,
    trial_idx: int,
) -> dict[str, object]:
    run_name = f"carbs_dnonet_jax_trial_{trial_idx:03d}"
    run_dir = REPO_ROOT / output_root / run_name
    command = [
        "uv",
        "run",
        "--python",
        "3.11",
        "python",
        "train-dnonet-jax/1d_dno_dnonet_jax.py",
        "--dataset",
        dataset,
        "--sources",
        sources,
        "--lr",
        str(suggestion["lr"]),
        "--weight_decay",
        str(suggestion["weight_decay"]),
        "--batch_size",
        str(suggestion["batch_size"]),
        "--epochs",
        str(suggestion["epochs"]),
        "--rank",
        str(suggestion["rank"]),
        "--modes",
        str(suggestion["modes"]),
        "--width",
        str(suggestion["width"]),
        "--spectral_width",
        str(suggestion["spectral_width"]),
        "--spectral_floor",
        str(suggestion["spectral_floor"]),
        "--n_blocks",
        str(suggestion["n_blocks"]),
        "--output_root",
        output_root,
        "--run_name",
        run_name,
        "--skip_plots",
    ]

    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = perf_counter() - started

    if completed.returncode != 0:
        return {
            "success": False,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "runtime_seconds": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    with open(run_dir / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["success"] = True
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(REPO_ROOT / args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset_path(args.dataset)
    grid_size = infer_grid_size(dataset_path, args.sources)
    rank_max = min(grid_size, max(64, args.rank_max))
    width_max = max(32, args.width_max)
    epoch_max = max(25, args.epoch_max)

    carbs = build_carbs(
        seed=args.seed,
        num_random_samples=args.num_random_samples,
        max_suggestion_cost=args.max_suggestion_cost,
        rank_max=rank_max,
        width_max=width_max,
        epoch_max=epoch_max,
    )

    trial_records: list[dict[str, object]] = []
    for trial_idx in range(args.trials):
        suggestion = carbs.suggest().suggestion
        result = run_trial(
            suggestion=suggestion,
            dataset=args.dataset,
            sources=args.sources,
            output_root=args.output_root,
            trial_idx=trial_idx,
        )
        trial_records.append(
            {
                "trial_idx": trial_idx,
                "suggestion": suggestion,
                "result": result,
            }
        )

        if result["success"]:
            carbs.observe(
                ObservationInParam(
                    input=suggestion,
                    output=float(result["best_val_loss"]),
                    cost=float(result["runtime_seconds"]),
                )
            )
        else:
            carbs.observe(
                ObservationInParam(
                    input=suggestion,
                    output=float("inf"),
                    cost=float(result["runtime_seconds"]),
                    is_failure=True,
                )
            )

        with open(output_root / "carbs_history.json", "w", encoding="utf-8") as handle:
            json.dump(trial_records, handle, indent=2)

        best_successes = [record for record in trial_records if record["result"]["success"]]
        if best_successes:
            best_record = min(
                best_successes,
                key=lambda record: float(record["result"]["best_val_loss"]),
            )
            with open(output_root / "best_result.json", "w", encoding="utf-8") as handle:
                json.dump(best_record, handle, indent=2)

        print(
            json.dumps(
                {
                    "trial_idx": trial_idx,
                    "suggestion": suggestion,
                    "success": result["success"],
                    "best_val_loss": result.get("best_val_loss"),
                    "runtime_seconds": result["runtime_seconds"],
                }
            )
        )


if __name__ == "__main__":
    main()
