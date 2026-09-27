"""Probe the two old Tanaka rollout failures with the hard-P128 retrained model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.evals import eval_suite as ev  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_id", type=int, choices=(16471, 16624))
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parent.parent
    run = root / "outputs/c27_tanaka_hard128_full_equal_local_20260918"
    dataset = root / "outputs/paper_dataset/arrays"
    reference = root / "outputs/c27_paper_dataset_20260908_141423/eval_final_n32/tanaka_trajs.npz"
    output = run / "eval_old_tanaka_ids_fp64_20260922"
    cfg = ev.FAMILY_CONFIGS["tanaka"]
    panel, _, nx, length = ev._load_paper_dataset_ics(dataset, "tanaka", 32)
    ic = next(case for case in panel if case.simulation_id == args.simulation_id)

    with np.load(reference, allow_pickle=False) as archive:
        index = archive["simulation_ids"].tolist().index(ic.simulation_id)
        truth_eta = np.asarray(archive["truth_eta"][:, index], dtype=np.float64)
    times = np.arange(0.0, cfg.tmax + cfg.dt / 2, cfg.dt)

    loaded = ev.load_run(run, checkpoint="best")
    prediction = ev.surrogate_rollout_batched(
        [ic], jnp.asarray(times), nx, length, cfg, ev.build_predict_gxi_batched(loaded)
    )
    fields = {
        name: np.asarray(prediction[name])[:, 0] for name in ("eta", "xi", "gxi")
    }
    finite = np.logical_and.reduce(
        [np.isfinite(values).all(axis=1) for values in fields.values()]
    )
    first_bad = np.flatnonzero(~finite)
    with np.errstate(over="ignore", invalid="ignore"):
        rel_l2_eta = ev._rel_l2(fields["eta"], truth_eta)
    final_error = float(rel_l2_eta[-1])

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"simulation_{ic.simulation_id}.npz",
        times=times,
        depth=ic.depth,
        initial_eta=ic.eta,
        initial_xi=ic.xi,
        rel_l2_eta=rel_l2_eta,
        **{f"pred_{name}": values for name, values in fields.items()},
    )
    summary = {
        "family": "tanaka",
        "simulation_id": ic.simulation_id,
        "dataset_row": ic.meta["dataset_row"],
        "depth": ic.depth,
        "checkpoint": str(run / "best_val_ckpt"),
        "epoch": loaded.epoch,
        "initial_conditions": str(dataset),
        "reference": str(reference),
        "reference_storage_dtype": "float32",
        "reference_generation": False,
        "model_and_integration_dtype": "float64",
        "method": "gl2_if",
        "nx": nx,
        "length": length,
        "save_dt": cfg.dt,
        "substeps": cfg.substeps,
        "internal_dt": cfg.dt / cfg.substeps,
        "tmax": cfg.tmax,
        "picard_iterations": ev.GL2_ITERATIONS,
        "filter_fraction": cfg.filter_fraction,
        "cutoff_mode": cfg.filter_fraction * nx / 2,
        "zero_mean_xi": True,
        "adaptive_damping": False,
        "all_saved_values_finite": bool(finite.all()),
        "first_nonfinite_saved_time": (
            float(times[first_bad[0]]) if first_bad.size else None
        ),
        "final_rel_l2_eta_vs_archived_fp32_reference": (
            final_error if np.isfinite(final_error) else None
        ),
        "wall_seconds": float(prediction["wall_s"]),
    }
    ev._write_json(output / f"simulation_{ic.simulation_id}.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
