"""Rollout a trained FNO/DNO surrogate against analytic Stokes wave evolution.

Truth: closed-form Stokes solution from solver.data.stokes_truth_jax.
Surrogate: trained model wrapped via solver.evals.model_rollout.

Usage:
    uv run python -m solver.evals.rollout_stokes \\
        --run_dir outputs/fno_prod_50ep_4xl40s_20260507_060820 \\
        --depth 1.0 --ichoi 1 --n0 14 --a0 0.1 --tmax 20.0 --dt 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp

from solver.data.stokes_truth_jax import stokes_eta_xi
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import build_grid
from solver.evals.model_rollout import (
    build_predict_gxi,
    save_rollout_gif,
    compare_rollouts,
    load_run,
    rollout_surrogate,
    save_comparison_plot,
    save_paper_error_plot,
    save_rollout_npz,
    truth_gxi_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout FNO surrogate against Stokes truth.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ichoi", type=int, choices=(0, 1), default=1, help="0 deep, 1 finite-depth")
    p.add_argument("--depth", type=float, default=1.0)
    p.add_argument("--n0", type=int, default=14)
    p.add_argument("--a0", type=float, default=0.1)
    p.add_argument("--length", type=float, default=2.0 * float(jnp.pi))
    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--gif", action="store_true", help="Also render a GIF rollout movie.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "rollout_eval" / "stokes"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    # ichoi=0 (deep) historically hardcoded h=1000; honor --depth if the user passes a non-default.
    physical_depth = float(args.depth) if (args.ichoi == 1 or args.depth != 1.0) else 1000.0
    predict_gxi = build_predict_gxi(loaded, physical_depth)

    x, k = build_grid(args.nx, args.length)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=jnp.float32)

    truth_eta, truth_xi = jax.vmap(
        lambda tau: stokes_eta_xi(x, tau, args.n0, args.a0, args.length, physical_depth, 1.0, ichoi=args.ichoi)
    )(times)
    truth_gxi = truth_gxi_from_state(truth_eta, truth_xi, k, physical_depth)

    solver_params = ti.make_solver_params(
        nx=args.nx, length=args.length, depth=physical_depth,
        gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
    )
    initial = ti.State(eta=truth_eta[0], xi=truth_xi[0])
    pred = rollout_surrogate(initial, times, solver_params, predict_gxi, substeps=args.substeps, zero_mean_xi=True)

    pred_gxi = jnp.stack([predict_gxi(pred["eta"][i], pred["xi"][i]) for i in range(pred["eta"].shape[0])])

    pred_eta_np = np.asarray(pred["eta"])
    pred_xi_np = np.asarray(pred["xi"])
    truth_eta_np = np.asarray(truth_eta)
    truth_xi_np = np.asarray(truth_xi)
    truth_gxi_np = np.asarray(truth_gxi)
    pred_gxi_np = np.asarray(pred_gxi)
    metrics = compare_rollouts(pred_eta_np, pred_xi_np, truth_eta_np, truth_xi_np, truth_gxi_np, pred_gxi_np)

    regime = "finite" if args.ichoi == 1 else "deep"
    stem = f"stokes_{regime}_n{args.n0:02d}_a{args.a0:.3f}_h{physical_depth:g}"
    save_comparison_plot(
        out_dir / f"{stem}.png",
        title=f"Stokes ({regime} depth h={physical_depth:g}) — n0={args.n0}, a0={args.a0}",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        metrics=metrics,
    )

    summary = {
        "regime": "stokes",
        "ichoi": args.ichoi,
        "depth": physical_depth,
        "n0": args.n0, "a0": args.a0,
        "epoch": loaded.epoch,
        "metrics": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in metrics.items()},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    save_paper_error_plot(
        out_dir / f"{stem}_error.png",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        truth_gxi=truth_gxi_np, pred_gxi=pred_gxi_np,
        title=f"Stokes {regime} (h={physical_depth:g}, n0={args.n0}, a0={args.a0})",
    )
    save_rollout_npz(
        out_dir / f"{stem}_arrays.npz",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        truth_gxi=truth_gxi_np, pred_gxi=pred_gxi_np,
        depth=physical_depth,
    )
    print(json.dumps({k: v for k, v in summary["metrics"].items() if not isinstance(v, list)}, indent=2))
    print(f"saved -> {out_dir / (stem + '.png')}")
    print(f"saved -> {out_dir / (stem + '_error.png')}")
    if args.gif:
        gif_path = save_rollout_gif(
            out_dir / f"{stem}.gif",
            title=f"Stokes ({regime} depth h={physical_depth:g}) — n0={args.n0}, a0={args.a0}",
            x=np.asarray(x), t=np.asarray(times),
            truth_eta=truth_eta_np, pred_eta=pred_eta_np,
            truth_xi=truth_xi_np, pred_xi=pred_xi_np,
            truth_gxi=truth_gxi_np, pred_gxi=pred_gxi_np,
            fps=args.fps, n_frames=args.n_frames, dpi=args.dpi,
        )
        print(f"saved -> {gif_path}")


if __name__ == "__main__":
    main()
