from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

from carbs.carbs import CARBS
from carbs.utils import CARBSParams, LinearSpace, LogSpace, ObservationInParam, Param

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARBS over the JAX FNO trainer")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default="outputs/carbs_jax")
    parser.add_argument("--num_random_samples", type=int, default=4)
    parser.add_argument("--max_suggestion_cost", type=float, default=None)
    return parser.parse_args()


def build_carbs() -> CARBS:
    param_spaces = [
        Param("lr", LogSpace(min=1e-5, max=1e-2), search_center=5e-3),
        Param("weight_decay", LogSpace(min=1e-6, max=1e-2), search_center=1e-4),
        Param(
            "batch_size",
            LinearSpace(min=32, max=256, scale=32, is_integer=True, rounding_factor=32),
            search_center=128,
        ),
        Param(
            "epochs",
            LogSpace(min=5, max=400, is_integer=True),
            search_center=100,
        ),
        Param(
            "modes",
            LinearSpace(min=16, max=128, scale=16, is_integer=True, rounding_factor=16),
            search_center=64,
        ),
        Param(
            "width",
            LinearSpace(min=16, max=128, scale=16, is_integer=True, rounding_factor=8),
            search_center=64,
        ),
        Param(
            "n_blocks",
            LinearSpace(min=2, max=12, scale=4, is_integer=True),
            search_center=6,
        ),
    ]
    carbs_params = CARBSParams(
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        is_saved_on_every_observation=False,
        resample_frequency=0,
        num_random_samples=4,
    )
    return CARBS(carbs_params, param_spaces)


def run_trial(
    suggestion: dict[str, float | int],
    dataset: str,
    sources: str,
    output_root: str,
    trial_idx: int,
) -> dict[str, object]:
    run_name = f"carbs_jax_trial_{trial_idx:03d}"
    run_dir = REPO_ROOT / output_root / run_name
    command = [
        "uv",
        "run",
        "--python",
        "3.11",
        "python",
        "train-jax/1d_dno_fno_jax.py",
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
        "--modes",
        str(suggestion["modes"]),
        "--width",
        str(suggestion["width"]),
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

    carbs = build_carbs()
    carbs.config.seed = args.seed
    carbs.config.num_random_samples = args.num_random_samples
    carbs.config.max_suggestion_cost = args.max_suggestion_cost

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
