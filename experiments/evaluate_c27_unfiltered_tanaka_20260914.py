"""Roll out the original C27 checkpoint on its fixed Tanaka panel without low-pass filtering."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.eval_suite import FAMILY_CONFIGS, IC, _write_json, compute_metrics
from solver.evals.model_rollout import build_predict_gxi_batched, load_run
from solver.evals.eval_suite import surrogate_rollout_batched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("tanaka_g0", "tanaka_g1"), required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    source_path = Path(f"data/tanaka_2_adaptive_{args.regime[-2:]}.npz").resolve()
    truth_path = Path(
        "outputs/c27_h1_to_l2_full_20260717_212550/"
        "eval_final_soliton_spectral_guard_20260719_191228/"
        f"{args.regime}/{args.regime}_trajs.npz"
    ).resolve()
    output_dir = args.output_root.resolve() / args.regime
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source_path, allow_pickle=False) as source:
        case_ids = np.asarray(source["case_id_batch_0000"])
        first_rows = np.flatnonzero(np.r_[True, case_ids[1:] != case_ids[:-1]])[:32]
        if args.case_index is not None:
            first_rows = first_rows[args.case_index:args.case_index + 1]
        ics = [
            IC(
                eta=np.asarray(source["eta_batch_0000"][row], dtype=np.float64),
                xi=np.asarray(source["xi_batch_0000"][row], dtype=np.float64),
                depth=float(source["depth_batch_0000"][row]),
                simulation_id=int(case_ids[row]),
                meta={"source_row": int(row)},
            )
            for row in first_rows
        ]
    with np.load(truth_path, allow_pickle=False) as archive:
        selection = slice(None) if args.case_index is None else slice(args.case_index, args.case_index + 1)
        expected_ids = np.asarray(archive["case_ids"][selection], dtype=np.int64)
        times = np.asarray(archive["times"], dtype=np.float64)
        truth = {
            name: np.asarray(archive[f"truth_{name}"][:, selection], dtype=np.float64)
            for name in ("eta", "xi", "gxi")
        }
    np.testing.assert_array_equal([ic.simulation_id for ic in ics], expected_ids)
    np.testing.assert_allclose(times, np.arange(0.0, 200.0 + 0.4, 0.8), rtol=0.0, atol=1e-5)

    run_dir = Path("outputs/c27_h1_to_l2_full_20260717_212550").resolve()
    loaded = load_run(run_dir, checkpoint="final")
    predictor = build_predict_gxi_batched(loaded)
    # A fraction of one makes both rollout low-pass calls exact no-ops.
    config = replace(FAMILY_CONFIGS["tanaka"], filter_fraction=1.0)
    started = perf_counter()
    prediction = surrogate_rollout_batched(
        ics,
        jnp.asarray(times, dtype=jnp.float64),
        1024,
        2.0 * np.pi,
        config,
        predictor,
    )
    wall_seconds = perf_counter() - started
    predicted = {name: np.asarray(prediction[name], dtype=np.float64) for name in truth}
    metrics = compute_metrics(truth, predicted, times, 2.0 * np.pi)
    arrays = metrics.pop("_arrays")
    first_nonfinite_times = {}
    for index, case_id in enumerate(expected_ids):
        finite = np.logical_and.reduce(tuple(
            np.isfinite(predicted[name][:, index]).all(axis=-1) for name in predicted
        ))
        first_nonfinite_times[str(int(case_id))] = (
            None if finite.all() else float(times[np.flatnonzero(~finite)[0]])
        )

    np.savez(
        output_dir / "trajectories.npz",
        x=2.0 * np.pi * np.arange(1024) / 1024,
        times=times,
        case_ids=expected_ids,
        depths=np.asarray([ic.depth for ic in ics]),
        **{f"truth_{name}": values for name, values in truth.items()},
        **{f"pred_{name}": values for name, values in predicted.items()},
        **arrays,
    )
    summary = {
        **metrics,
        "regime": args.regime,
        "checkpoint": str(run_dir / "final_ckpt"),
        "checkpoint_epoch": loaded.epoch,
        "source": str(source_path),
        "case_ids": expected_ids.tolist(),
        "source_rows": first_rows.tolist(),
        "wall_seconds": wall_seconds,
        "filter_fraction": config.filter_fraction,
        "adaptive_guard": False,
        "precision": "float64 model, integration, and saved predictions",
        "first_nonfinite_saved_time": first_nonfinite_times,
        "reference": "historical C27 fixed-panel truth; no new truth integration",
        "unfiltered_scope": (
            "No rollout RHS or post-step low-pass and no adaptive damping. "
            "The trained model, quadratic dealiasing, Nyquist convention, and mean-zero gauges are unchanged."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "regime": args.regime,
        "wall_seconds": wall_seconds,
        "nonfinite_cases": metrics["model_nonfinite_any_count_attempted"],
        "median_final_eta_error": metrics["rel_l2_eta_median_conditional_finite_tfinal"],
        "output": str(output_dir),
    }), flush=True)


if __name__ == "__main__":
    main()
