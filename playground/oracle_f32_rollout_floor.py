"""Floor of the f32 surrogate rollout harness, measured with the analytic DNO.

Runs the order-6 Craig-Sulem DNO (the same operator the truth uses) through
the f32 per-IC surrogate rollout path (IF-GL2, substeps, state filter) and
compares against the saved f64 truth trajectories. A perfect model cannot
beat this number: any gap to the f64 truth here is harness error (f32
precision + integrator wrapper), not model error.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -u -m playground.oracle_f32_rollout_floor \
        --regime tanaka_g0 \
        --truth_npz outputs/cs_dno_w512b8_l256_v3_20260605_032608/eval_suite_gxifilt/tanaka_g0_trajs.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="tanaka_g0")
    parser.add_argument("--truth_npz", required=True)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--output", default="outputs/truth_convergence")
    args = parser.parse_args()

    import jax.numpy as jnp

    from solver.evals.eval_suite import (
        REGISTRY, build_ics_and_truth_targets, surrogate_rollout_per_ic,
    )
    from solver.solvers.dno_series_jax import build_grid, dno_series_eval

    truth = np.load(args.truth_npz)
    truth_eta = truth["truth_eta"].astype(np.float64)  # (n_t, NB, nx)
    times_np = truth["times"].astype(np.float64)
    n_t, NB, nx = truth_eta.shape
    assert nx == args.nx

    cfg = REGISTRY[args.regime]
    cfg = dataclasses.replace(cfg, n_ics=NB)
    ics, dt, tmax, _, _ = build_ics_and_truth_targets(cfg)
    eta0 = np.stack([ic.eta for ic in ics])
    drift0 = float(np.max(np.abs(eta0 - truth_eta[0])))
    print(f"{args.regime}: NB={NB} n_t={n_t} dt={dt} tmax={tmax}; "
          f"IC match vs saved truth: max|d_eta0|={drift0:.3e}", flush=True)
    assert drift0 < 1e-6, "saved truth was generated from different ICs"

    _, k_np = build_grid(args.nx, args.length)
    k_f32 = jnp.asarray(k_np, dtype=jnp.float32)

    def predict_oracle(eta: jnp.ndarray, xi: jnp.ndarray, log_h: jnp.ndarray) -> jnp.ndarray:
        return dno_series_eval(
            eta, xi, k_f32, jnp.exp(log_h), args.order, pad_factor=args.pad_factor,
        )

    out = surrogate_rollout_per_ic(
        ics, jnp.asarray(times_np, dtype=jnp.float32), args.nx, args.length,
        cfg, None, predict_oracle,
    )
    print(f"oracle f32 rollout wall = {out['wall_s']:.1f}s", flush=True)

    pred_eta = out["eta"].astype(np.float64)
    num = np.linalg.norm(pred_eta - truth_eta, axis=-1)
    den = np.linalg.norm(truth_eta, axis=-1) + 1e-30
    rel = num / den  # (n_t, NB)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.regime}_oracle_f32_floor.npz",
        times=times_np, rel=rel,
        case_ids=np.asarray([ic.case_id for ic in ics]),
        depths=np.asarray([ic.depth for ic in ics]),
    )

    n_nan = int((~np.isfinite(rel[-1])).sum())
    report = {"n_ics": NB, "n_nonfinite_tfinal": n_nan}
    for frac, label in ((0.1, "t10p"), (0.5, "t50p"), (1.0, "tfinal")):
        i = min(n_t - 1, int(round(frac * (n_t - 1))))
        report[label] = {
            "t": float(times_np[i]),
            "median": float(np.nanmedian(rel[i])),
            "mean": float(np.nanmean(rel[i])),
            "p95": float(np.nanquantile(rel[i], 0.95)),
        }
        print(f"  t={times_np[i]:6.1f}  oracle-f32 vs truth rel-L2 (finite ICs): "
              f"med={report[label]['median']:.4f} mean={report[label]['mean']:.4f} "
              f"p95={report[label]['p95']:.4f}", flush=True)
    print(f"  non-finite ICs at tfinal: {n_nan}/{NB}", flush=True)
    (out_dir / f"{args.regime}_oracle_f32_floor.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir / (args.regime + '_oracle_f32_floor.npz')}", flush=True)


if __name__ == "__main__":
    main()
