"""Truth self-convergence at long horizon: substeps=80 vs substeps=160.

Runs the eval_suite truth protocol (order-6 Craig-Sulem, GL2, filter 0.25, f64)
twice on the same ICs with inner dt halved, and reports the rel-L2 eta
disagreement between the two truths over time. This is the floor of the
long-horizon rollout eval: a surrogate cannot meaningfully beat the truth's
own temporal-discretization phase error.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -m playground.truth_convergence_check \
        --regime tanaka_g0 --n_ics 8
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
    parser.add_argument("--n_ics", type=int, default=8)
    parser.add_argument("--substeps_hi", type=int, default=160)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--output", default="outputs/truth_convergence")
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from solver.evals.eval_suite import (
        REGISTRY, build_ics_and_truth_targets, truth_rollout_batched,
    )

    cfg = REGISTRY[args.regime]
    cfg = dataclasses.replace(cfg, n_ics=args.n_ics)
    ics, dt, tmax, _, _ = build_ics_and_truth_targets(cfg)
    n_t = int(round(tmax / dt)) + 1
    times = jnp.asarray(np.linspace(0.0, tmax, n_t), dtype=jnp.float64)
    print(f"{args.regime}: {len(ics)} ICs, dt={dt}, tmax={tmax}, n_t={n_t}")

    out_lo = truth_rollout_batched(ics, times, args.nx, args.length, cfg)
    print(f"substeps={cfg.substeps}: wall={out_lo['wall_s']:.1f}s")
    cfg_hi = dataclasses.replace(cfg, substeps=args.substeps_hi)
    out_hi = truth_rollout_batched(ics, times, args.nx, args.length, cfg_hi)
    print(f"substeps={args.substeps_hi}: wall={out_hi['wall_s']:.1f}s")

    eta_lo = out_lo["eta"].astype(np.float64)   # (n_t, NB, nx)
    eta_hi = out_hi["eta"].astype(np.float64)
    num = np.linalg.norm(eta_lo - eta_hi, axis=-1)
    den = np.linalg.norm(eta_hi, axis=-1) + 1e-30
    rel = num / den                              # (n_t, NB)

    t_axis = np.asarray(times)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.regime}_truth_conv.npz",
        times=t_axis, rel=rel,
        case_ids=np.asarray([ic.case_id for ic in ics]),
        depths=np.asarray([ic.depth for ic in ics]),
    )

    report = {}
    for frac, label in ((0.1, "t10p"), (0.5, "t50p"), (1.0, "tfinal")):
        i = min(n_t - 1, int(round(frac * (n_t - 1))))
        report[label] = {
            "t": float(t_axis[i]),
            "median": float(np.median(rel[i])),
            "mean": float(np.mean(rel[i])),
            "p95": float(np.quantile(rel[i], 0.95)),
        }
        print(f"  t={t_axis[i]:6.1f}  truth-vs-truth rel-L2: "
              f"med={report[label]['median']:.4f} mean={report[label]['mean']:.4f} "
              f"p95={report[label]['p95']:.4f}")
    (out_dir / f"{args.regime}_truth_conv.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir / (args.regime + '_truth_conv.npz')}")


if __name__ == "__main__":
    main()
