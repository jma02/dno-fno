"""Compare saved networks with M3 on identical states and fresh FP64 M6 targets.

Defaults to CPU; no training or trajectory integration. Also evaluate saved reference
trajectory snapshots, without feeding predictions back into the integrator.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.evals.model_rollout import build_predict_gxi_batched, load_run  # noqa: E402
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402

RUNS = {
    "old_ema": "c27_w320_b4_h80_ema_grad09_20260927",
    "rollout_teacher": "c27_rollout_teacher_m6_ep120_gpu0_20261001",
    "full_loss_teacher": "c27_rollout_full_loss_ep120_2gpu_20261003",
    "rms_context": "c27_surface_rms_context8_rollout_ep120_gpu0_20261006",
}
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")


def band(field: np.ndarray, *, zero_mean: bool = True) -> np.ndarray:
    coefficients = np.fft.rfft(field, axis=-1)
    if zero_mean:
        coefficients[..., 0] = 0
    coefficients[..., 129:] = 0
    return np.fft.irfft(coefficients, n=field.shape[-1], axis=-1)


def chunked(
    function: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    eta: np.ndarray, xi: np.ndarray, depth: np.ndarray,
) -> np.ndarray:
    return np.concatenate([
        np.asarray(function(*(jnp.asarray(field[start:start + 32]) for field in (eta, xi, depth))))
        for start in range(0, len(eta), 32)
    ])


def statistics(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)), "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512, help="Validation rows per family.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {name: load_run(ROOT / "outputs" / path, checkpoint="latest") for name, path in RUNS.items()}
    predictors = {name: build_predict_gxi_batched(run) for name, run in loaded.items()}
    print({name: run.epoch for name, run in loaded.items()}, flush=True)
    dataset = ROOT / loaded["rms_context"].config["dataset"]
    arrays = {name: np.load(dataset / f"{name}.npy", mmap_mode="r") for name in
              ("eta", "xi", "gxi", "depth", "family_id", "dataset_split")}
    nx = arrays["eta"].shape[1]
    rng = np.random.default_rng(0)
    results = {
        "metric": "per-state physical relative L2; primary outputs projected to modes 1..128",
        "reference": "fresh FP64 M6, pad8, actual depth, projected identical inputs",
        "network_precision": "generic production FP32 predictor, checkpoint baseline settings unchanged",
        "models": {name: {"run": path, "epoch": loaded[name].epoch, "checkpoint": "latest"}
                   for name, path in RUNS.items()},
        "cohorts": {},
    }
    for family_id, family in enumerate(FAMILIES, 1):
        candidates = np.flatnonzero((arrays["dataset_split"] == "validation") &
                                   (arrays["family_id"] == family_id))
        rows = np.sort(rng.choice(candidates, args.samples, replace=False))
        group = 0 if family in ("stokes", "tanaka") else 1
        archive_path = (ROOT / "outputs" / RUNS["rms_context"] /
                        f"eval_epoch120_test_stratified_n32_gpu{group}" / f"{family}_trajs.npz")
        with np.load(archive_path) as archive:
            frames = np.linspace(0, len(archive["times"]) - 1, 11, dtype=int)
            trajectory_fields = {name: archive[f"truth_{name}"][frames].reshape(-1, nx)
                                 for name in ("eta", "xi", "gxi")}
            trajectory_fields["depth"] = np.tile(archive["depths"], len(frames))
            trajectory_selection = {"simulation_ids": archive["simulation_ids"].tolist(),
                                    "times": archive["times"][frames].tolist()}
        for cohort in ("validation", "reference_trajectory"):
            fields = ({name: np.asarray(arrays[name][rows], dtype=np.float64)
                       for name in ("eta", "xi", "gxi", "depth")}
                      if cohort == "validation" else trajectory_fields)
            eta, xi = band(fields["eta"], zero_mean=False), band(fields["xi"])
            depth = fields["depth"].reshape(-1)
            classical = {}
            for order in (2, 3, 6):
                classical[f"M{order}"] = band(chunked(
                    lambda e, x, h: compute_dno_target(
                        e, x, h[:, None], nx=nx, length=2 * np.pi, dno_order=order,
                        pad_factor=8, maximum_wavenumber=128.0,
                    )[2], eta, xi, depth,
                ))
            reference = classical.pop("M6")
            denominator = np.linalg.norm(reference, axis=-1)
            m3_error = np.linalg.norm(classical["M3"] - reference, axis=-1) / denominator
            summary = {"count": len(eta),
                       "selection": {"rows": rows.tolist()} if cohort == "validation" else trajectory_selection,
                       "saved_target_vs_fresh_M6": statistics(
                           np.linalg.norm(band(fields["gxi"]) - reference, axis=-1) / denominator),
                       "methods": {}}
            for name in (*classical, *predictors):
                prediction = (classical[name] if name in classical else
                              chunked(predictors[name], eta, xi, np.log(depth)))
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f"Nonfinite prediction: {cohort}/{family}/{name}")
                retained_prediction = prediction if name in classical else band(prediction)
                error = np.linalg.norm(retained_prediction - reference, axis=-1) / denominator
                summary["methods"][name] = {
                    "retained_band": statistics(error),
                    "full_output_against_projected_reference": statistics(
                        np.linalg.norm(prediction - reference, axis=-1) / denominator),
                    "fraction_beating_M3": float(np.mean(error < m3_error)),
                    "paired_ratio_to_M3": statistics(error / np.maximum(m3_error, 1e-30)),
                    "per_state_relative_error": error.tolist(),
                }
                print(f"{cohort}/{family} {name}: median={np.median(error):.6g} "
                      f"mean={np.mean(error):.6g} beats_M3={np.mean(error < m3_error):.1%}", flush=True)
            results["cohorts"].setdefault(cohort, {})[family] = summary
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
