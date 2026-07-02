"""Rollout a trained surrogate against a Tanaka soliton trajectory.

Truth: a Tanaka steady-soliton trajectory loaded from disk (or, if the file
isn't available, the analytic-DNO-driven time-stepper applied to the soliton
IC). Surrogate: trained model wrapped via solver.evals.model_rollout.

Usage:
    uv run python -m solver.evals.rollout_tanaka \\
        --run_dir outputs/fno_prod_50ep_4xl40s_20260507_060820 \\
        --soliton /home/johnma/dno_locl/soliton_data/<file> \\
        --depth 1.0 --substeps 8
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

from solver.data.solitary_loader_jax import (
    DEFAULT_SOLITON_ROOT,
    list_soliton_files,
    load_soliton_file,
    EXPECTED_LENGTH,
)
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import build_grid
from solver.gen_data.multi_crest import (
    CrestSpec,
    build_multi_crest_initial_condition,
)
from solver.evals.model_rollout import (
    build_predict_gxi,
    compare_rollouts,
    load_run,
    rollout_surrogate,
    save_comparison_plot,
    save_paper_error_plot,
    save_rollout_gif,
    save_rollout_npz,
    truth_gxi_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout surrogate against Tanaka soliton truth.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--soliton", default=None,
                   help="Path to a Tanaka soliton text file. Defaults to first file in DEFAULT_SOLITON_ROOT.")
    p.add_argument("--npz", default=None,
                   help="Path to a tanaka_2_g*.npz file (L=2π). Overrides --soliton when set.")
    p.add_argument("--case_id", type=int, default=0)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--depth", type=float, default=1.0)
    p.add_argument("--length", type=float, default=EXPECTED_LENGTH)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--max_t", type=float, default=None,
                   help="Truncate truth trajectory to t<=max_t (default: full file).")
    p.add_argument("--ic_only", action="store_true",
                   help="Use only the t=0 frame of the soliton file as IC, rescale to L=2pi "
                        "Zakharov coords, generate truth via our analytic-DNO solver at "
                        "--solver_dt and --solver_tmax. Ignores --max_t and the saved times.")
    p.add_argument("--collide", action="store_true",
                   help="Build a 2-soliton head-on collision IC directly in L=2pi Zakharov "
                        "coords via build_multi_crest_initial_condition, then truth-roll our "
                        "solver and FNO from it. Overrides --soliton / --npz / --ic_only.")
    p.add_argument("--collide_amp", type=float, default=0.012,
                   help="Per-crest amplitude (Zakharov-rescaled) for the collision IC.")
    p.add_argument("--collide_nx", type=int, default=1024)
    p.add_argument("--solver_dt", type=float, default=0.05,
                   help="dt used by our truth solver in rescaled time (only with --ic_only/--collide).")
    p.add_argument("--solver_tmax", type=float, default=10.0,
                   help="Total time horizon in rescaled units (only with --ic_only/--collide).")
    p.add_argument("--gpu", action="store_true")
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
    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "rollout_eval" / "tanaka"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.collide:
        # Build a 2-soliton head-on collision IC directly in L=2π Zakharov coords.
        # Two crests with opposite directions, placed at L/4 and 3L/4.
        args.length = 2.0 * float(jnp.pi)
        nx = int(args.collide_nx)
        specs = [
            CrestSpec(amplitude=float(args.collide_amp), center=args.length * 0.25, direction=+1),
            CrestSpec(amplitude=float(args.collide_amp), center=args.length * 0.75, direction=-1),
        ]
        ic = build_multi_crest_initial_condition(
            specs, nx=nx, length=args.length, depth=float(args.depth),
            gravity=1.0, dno_order=6, pad_factor=8, zero_mean_xi=True,
        )
        x = jnp.asarray(ic["x"], dtype=jnp.float32)
        times = jnp.asarray(np.arange(0.0, args.solver_tmax + 0.5 * args.solver_dt, args.solver_dt), dtype=jnp.float32)
        eta0 = np.asarray(ic["eta"], dtype=np.float64)
        xi0 = np.asarray(ic["xi"], dtype=np.float64)
        soliton_path = Path(f"collide_2soliton_amp{args.collide_amp:.4f}_h{args.depth:g}")
        truth_eta = None
    elif args.npz is not None:
        npz_path = Path(args.npz).resolve()
        with np.load(npz_path) as d:
            x_np = np.asarray(d["x"], dtype=np.float32)
            tag = f"batch_{args.batch_idx:04d}"
            eta_b = np.asarray(d[f"eta_{tag}"], dtype=np.float32)
            xi_b = np.asarray(d[f"xi_{tag}"], dtype=np.float32)
            gxi_b = np.asarray(d[f"gxi_{tag}"], dtype=np.float32)
            t_b = np.asarray(d[f"time_{tag}"], dtype=np.float32)
            case_b = np.asarray(d[f"case_id_{tag}"])
            depth_b = np.asarray(d[f"depth_{tag}"], dtype=np.float32)
        mask = case_b == int(args.case_id)
        if not mask.any():
            raise SystemExit(f"case_id={args.case_id} not present in {npz_path.name} {tag}")
        order = np.argsort(t_b[mask])
        x = jnp.asarray(x_np, dtype=jnp.float32)
        nx = int(x.shape[0])
        case_depth = float(depth_b[mask][0])
        if args.length == EXPECTED_LENGTH:
            args.length = float(2.0 * float(jnp.pi))
        soliton_path = npz_path.with_name(f"{npz_path.stem}_case{args.case_id:04d}")
        if args.depth == 1.0:
            args.depth = case_depth  # honor file's per-case depth unless user overrode
        if args.ic_only:
            # Take only the t=0 frame as IC; run our solver as truth over [0, solver_tmax].
            eta0 = np.asarray(eta_b[mask][order][0], dtype=np.float64)
            xi0 = np.asarray(xi_b[mask][order][0], dtype=np.float64)
            times = jnp.asarray(np.arange(0.0, args.solver_tmax + 0.5 * args.solver_dt, args.solver_dt), dtype=jnp.float32)
            truth_eta = None
        else:
            truth_eta = jnp.asarray(eta_b[mask][order], dtype=jnp.float32)
            truth_xi = jnp.asarray(xi_b[mask][order], dtype=jnp.float32)
            truth_gxi_loaded = jnp.asarray(gxi_b[mask][order], dtype=jnp.float32)
            times = jnp.asarray(t_b[mask][order], dtype=jnp.float32)
            if args.max_t is not None:
                m2 = np.asarray(times) <= args.max_t
                truth_eta = truth_eta[m2]; truth_xi = truth_xi[m2]
                truth_gxi_loaded = truth_gxi_loaded[m2]; times = times[m2]
    else:
        soliton_path = Path(args.soliton).resolve() if args.soliton else list_soliton_files(DEFAULT_SOLITON_ROOT)[0]
        soliton = load_soliton_file(soliton_path)
        nx = int(soliton["nx"])
        if args.ic_only:
            # Rescale t=0 frame from MATLAB units (L≈164) to Zakharov L=2π coords.
            # alpha = L_matlab / (2π) ≈ 26.1; eta/=α, xi/=α^1.5, gxi/=α^0.5, h/=α.
            length_matlab = float(soliton["length"])
            alpha = length_matlab / (2.0 * float(jnp.pi))
            eta0 = np.asarray(soliton["eta"][0], dtype=np.float64) / alpha
            xi0 = np.asarray(soliton["xi"][0], dtype=np.float64) / (alpha ** 1.5)
            args.length = 2.0 * float(jnp.pi)
            args.depth = float(args.depth) / alpha  # default depth=1 in MATLAB -> ≈0.038 rescaled
            times = jnp.asarray(np.arange(0.0, args.solver_tmax + 0.5 * args.solver_dt, args.solver_dt), dtype=jnp.float32)
            x = jnp.asarray(np.linspace(0.0, args.length, nx, endpoint=False), dtype=jnp.float32)
            truth_eta = None  # filled below via ti.rollout
        else:
            truth_eta = jnp.asarray(soliton["eta"], dtype=jnp.float32)
            truth_xi = jnp.asarray(soliton["xi"], dtype=jnp.float32)
            truth_gxi_loaded = jnp.asarray(soliton["gxi"], dtype=jnp.float32)
            times = jnp.asarray(soliton["t"], dtype=jnp.float32)
            if args.max_t is not None:
                mask = np.asarray(times) <= args.max_t
                truth_eta = truth_eta[mask]; truth_xi = truth_xi[mask]
                truth_gxi_loaded = truth_gxi_loaded[mask]; times = times[mask]
            x = jnp.asarray(soliton["x"], dtype=jnp.float32)

    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    predict_gxi = build_predict_gxi(loaded, float(args.depth))

    _, k = build_grid(nx, float(args.length))
    solver_params = ti.make_solver_params(
        nx=nx, length=float(args.length), depth=float(args.depth),
        gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
    )

    if (args.ic_only or args.collide) and truth_eta is None:
        # Generate truth by rolling out our analytic-DNO solver from the rescaled IC.
        ic_state = ti.State(eta=jnp.asarray(eta0, dtype=jnp.float32),
                            xi=jnp.asarray(xi0, dtype=jnp.float32))
        truth = ti.rollout(ic_state, times, solver_params, substeps_per_interval=args.substeps, zero_mean_xi=True)
        truth_eta = truth["eta"]
        truth_xi = truth["xi"]
        truth_gxi_loaded = truth_gxi_from_state(truth_eta, truth_xi, k, float(args.depth))

    pred = rollout_surrogate(
        ti.State(eta=truth_eta[0], xi=truth_xi[0]),
        times, solver_params, predict_gxi, substeps=args.substeps, zero_mean_xi=True,
    )
    pred_gxi = jnp.stack([predict_gxi(pred["eta"][i], pred["xi"][i]) for i in range(pred["eta"].shape[0])])

    pred_eta_np = np.asarray(pred["eta"]); pred_xi_np = np.asarray(pred["xi"])
    truth_eta_np = np.asarray(truth_eta); truth_xi_np = np.asarray(truth_xi)
    metrics = compare_rollouts(
        pred_eta_np, pred_xi_np, truth_eta_np, truth_xi_np,
        np.asarray(truth_gxi_loaded), np.asarray(pred_gxi),
    )

    stem = f"tanaka_{soliton_path.stem}_h{float(args.depth):g}"
    save_comparison_plot(
        out_dir / f"{stem}.png",
        title=f"Tanaka soliton {soliton_path.name} (h={float(args.depth):g})",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        metrics=metrics,
    )
    summary = {
        "regime": "tanaka",
        "soliton": soliton_path.name,
        "depth": float(args.depth),
        "epoch": loaded.epoch,
        "metrics": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in metrics.items()},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    save_paper_error_plot(
        out_dir / f"{stem}_error.png",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        truth_gxi=np.asarray(truth_gxi_loaded), pred_gxi=np.asarray(pred_gxi),
        title=f"Tanaka {soliton_path.name} (h={float(args.depth):g})",
    )
    save_rollout_npz(
        out_dir / f"{stem}_arrays.npz",
        x=np.asarray(x), t=np.asarray(times),
        truth_eta=truth_eta_np, pred_eta=pred_eta_np,
        truth_xi=truth_xi_np, pred_xi=pred_xi_np,
        truth_gxi=np.asarray(truth_gxi_loaded), pred_gxi=np.asarray(pred_gxi),
        depth=float(args.depth),
    )
    print(json.dumps({k: v for k, v in summary["metrics"].items() if not isinstance(v, list)}, indent=2))
    print(f"saved -> {out_dir / (stem + '.png')}")
    print(f"saved -> {out_dir / (stem + '_error.png')}")
    if args.gif:
        gif_path = save_rollout_gif(
            out_dir / f"{stem}.gif",
            title=f"Tanaka soliton {soliton_path.name} (h={float(args.depth):g})",
            x=np.asarray(x), t=np.asarray(times),
            truth_eta=truth_eta_np, pred_eta=pred_eta_np,
            truth_xi=truth_xi_np, pred_xi=pred_xi_np,
            truth_gxi=np.asarray(truth_gxi_loaded), pred_gxi=np.asarray(pred_gxi),
            fps=args.fps, n_frames=args.n_frames, dpi=args.dpi,
        )
        print(f"saved -> {gif_path}")


if __name__ == "__main__":
    main()
