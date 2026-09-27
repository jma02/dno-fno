"""Score saved PyTorch rollouts against the current held-out truth trajectories.

Usage::

    uv run scripts/summarize_torch_rollouts.py \
        --predictions-dir outputs/torch_rollouts \
        --output-dir outputs/torch_rollouts/evaluation

Each FAMILY_trajs.npz input contains times, depths, simulation_ids,
pred_eta/pred_xi/pred_gxi, and a scalar rollout_metadata_json string. Metadata
must include the full-rollout timing in seconds. No rollouts are run here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from solver.evals.eval_suite import (  # noqa: E402
    FAMILY_CONFIGS,
    _write_json,
    compute_macro_summary,
    compute_metrics,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument(
        "--reference-dirs", type=Path, nargs="+",
        default=sorted((ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918").glob(
            "eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu*",
        )),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for family in args.families:
        references = [
            directory / f"{family}_trajs.npz"
            for directory in args.reference_dirs
            if (directory / f"{family}_trajs.npz").is_file()
        ]
        if len(references) != 1:
            raise ValueError(f"{family}: expected one reference trajectory file, found {references}")
        reference_path = references[0]
        reference_summary = json.loads(reference_path.with_name(f"{family}_summary.json").read_text())
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            args.predictions_dir / f"{family}_trajs.npz", allow_pickle=False,
        ) as prediction:
            coordinates = {name: prediction[name] for name in ("times", "depths", "simulation_ids")}
            for name, values in coordinates.items():
                if not np.array_equal(values, reference[name]):
                    raise ValueError(f"{family}: prediction {name} do not match the reference")
            truth = {field: reference[f"truth_{field}"] for field in ("eta", "xi", "gxi")}
            pred = {field: prediction[f"pred_{field}"] for field in truth}
            expected_shape = (
                coordinates["times"].size, coordinates["simulation_ids"].size,
                reference_summary["nx"],
            )
            for field in truth:
                if truth[field].shape != expected_shape or pred[field].shape != expected_shape:
                    raise ValueError(f"{family}: {field} trajectories must have shape {expected_shape}")
            metadata_json = prediction["rollout_metadata_json"]
            metadata = json.loads(str(metadata_json.item()))
            protocol_json = reference["truth_protocol_json"]

        metrics = compute_metrics(truth, pred, coordinates["times"], reference_summary["length"])
        np.savez_compressed(
            args.output_dir / f"{family}_trajs.npz",
            **coordinates,
            **{f"truth_{field}": values for field, values in truth.items()},
            **{f"pred_{field}": values for field, values in pred.items()},
            **metrics.pop("_arrays"),
            truth_protocol_json=protocol_json,
            rollout_metadata_json=metadata_json,
        )
        summary = {
            **{key: reference_summary[key] for key in (
                "family", "dt", "tmax", "length", "nx", "n_ics", "depths",
                "simulation_ids", "evaluation_source", "truth_wall_s", "truth_protocol",
            )},
            **metrics,
            "surrogate_wall_s": metadata["seconds"],
            "rollout_metadata": metadata,
        }
        summaries[family] = summary
        _write_json(args.output_dir / f"{family}_summary.json", summary)
        print(
            f"{family}: truth-valid={summary['n_truth_valid']}/{summary['n_ics_attempted']}, "
            f"nonfinite={summary['model_nonfinite_any_count_truth_valid']}, "
            f"eta median={summary['rel_l2_eta_median_conditional_finite_tfinal']:.6g}",
            flush=True,
        )
    _write_json(args.output_dir / "macro_summary.json", compute_macro_summary(summaries))
