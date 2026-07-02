"""Smoke test: roll out a 2-crest Tanaka case and visualize adaptive sampling.

Plots ``S(t) = ||η_x(t)||^2`` together with the activity signal
``|dS/dt| / S`` and marks the K time-steps chosen by the adaptive sampler
against a uniform-K baseline. Run on CPU is fine; tmax shortened for speed.

Usage:
    uv run python -m solver.evals.adaptive_subsample_demo \
        --keep_samples 26 --tmax 40 --alpha 0.5 --depth 0.05 \
        --out outputs/adaptive_demo.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from solver.gen_data.adaptive_sampling import (
    adaptive_indices_from_energy,
    surface_gradient_energy,
)
from solver.gen_data.generate_tanaka_dataset_v2 import (
    _grad_energy_traj,
    build_per_case_initial_conditions,
)
from solver.gen_data.multi_crest import CrestSpec
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)
from solver.tanaka_ICs.modified_tanaka import make_default_tanaka_template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keep_samples", type=int, default=26)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--smooth_sigma_steps", type=float, default=50.0)
    p.add_argument("--depth", type=float, default=0.05)
    p.add_argument("--amp_per_crest", type=float, default=0.06)
    p.add_argument("--center_a", type=float, default=2.0)
    p.add_argument("--center_b", type=float, default=4.5)
    p.add_argument("--direction_a", type=int, default=+1)
    p.add_argument("--direction_b", type=int, default=-1)
    p.add_argument("--length", type=float, default=2.0 * math.pi)
    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmax", type=float, default=40.0)
    p.add_argument("--gravity", type=float, default=1.0)
    p.add_argument("--out", default="outputs/adaptive_demo.png")
    p.add_argument("--cpu", action="store_true", help="Force CPU; otherwise use whatever JAX picks (GPU when available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    rollout_defaults = make_normalized_rollout_settings()
    times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float64)

    template_params = make_default_tanaka_template(
        depth=1.0,
        gravity=args.gravity,
        direction=1,
        nx=args.nx,
        length=args.length,
        center=0.0,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
    )

    case_specs = [[
        CrestSpec(amplitude=args.amp_per_crest, center=args.center_a, direction=args.direction_a),
        CrestSpec(amplitude=args.amp_per_crest, center=args.center_b, direction=args.direction_b),
    ]]
    case_h_ref = np.array([args.depth], dtype=np.float64)

    initial_eta, initial_xi = build_per_case_initial_conditions(
        template_params=template_params,
        case_h_ref=case_h_ref,
        case_specs=case_specs,
        length=args.length,
        nx=args.nx,
        gravity=args.gravity,
    )
    initial_eta = apply_lowpass(initial_eta, jnp.asarray(build_grid(args.nx, args.length)[1]),
                                rollout_defaults.filter_fraction)
    initial_xi = apply_lowpass(initial_xi, jnp.asarray(build_grid(args.nx, args.length)[1]),
                               rollout_defaults.filter_fraction)
    initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

    _, k_grid = build_grid(args.nx, args.length)
    k_grid = jnp.asarray(k_grid, dtype=jnp.float64)
    depth_2d = jnp.asarray(case_h_ref, dtype=jnp.float64)[:, None]
    solver_params = SolverParams(
        nx=args.nx, length=args.length, depth=depth_2d, gravity=args.gravity,
        dno_order=rollout_defaults.dno_order, pad_factor=rollout_defaults.pad_factor,
        filter_fraction=rollout_defaults.filter_fraction, k=k_grid,
        g0=make_linear_dno_symbol(k_grid, depth_2d),
    )
    solver_params = cast_solver_params_dtype(solver_params, jnp.float64)

    rollout_times = jnp.asarray(times, dtype=jnp.float64)
    payload = batched_rollout(
        cast_state_dtype(State(eta=initial_eta, xi=initial_xi), jnp.float64),
        rollout_times, solver_params, save_gxi=False,
        substeps_per_interval=rollout_defaults.substeps_per_interval,
        method=rollout_defaults.method,
        implicit_iterations=rollout_defaults.implicit_iterations,
        implicit_relaxation=rollout_defaults.implicit_relaxation,
        zero_mean_xi=rollout_defaults.zero_mean_xi,
    )
    jax.block_until_ready(payload["eta"])

    eta_BTN = payload["eta"]  # (T, B, N)
    grad_energy_BT = np.asarray(jax.device_get(_grad_energy_traj(eta_BTN, float(args.length))))
    s = grad_energy_BT[0]  # (T,)
    activity = np.abs(np.gradient(s)) / (s + 1e-12)

    idx_adaptive = adaptive_indices_from_energy(
        grad_energy_BT,
        keep_samples=int(args.keep_samples),
        alpha=float(args.alpha),
        smooth_sigma_steps=float(args.smooth_sigma_steps),
    )[0]
    idx_uniform = np.linspace(0, times.shape[0] - 1, args.keep_samples, dtype=np.int32)

    eta_traj = np.asarray(jax.device_get(eta_BTN))[:, 0, :]  # (T, N)
    xi_traj = np.asarray(jax.device_get(payload["xi"]))[:, 0, :]
    x_grid = np.asarray(build_grid(args.nx, args.length)[0])
    vmax_eta = float(np.max(np.abs(eta_traj)))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, args.keep_samples, height_ratios=[1.6, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.05)

    # Row 0: η(x, t) heatmap with adaptive (red) and uniform (grey) sample times.
    ax_hm = fig.add_subplot(gs[0, :])
    extent = [times[0], times[-1], x_grid[0], x_grid[-1]]
    ax_hm.imshow(eta_traj.T, aspect="auto", origin="lower", extent=extent,
                 cmap="RdBu_r", vmin=-vmax_eta, vmax=vmax_eta)
    for t_idx in idx_uniform:
        ax_hm.axvline(times[t_idx], color="#444444", lw=0.7, alpha=0.55)
    for t_idx in idx_adaptive:
        ax_hm.axvline(times[t_idx], color="#d62728", lw=0.9, alpha=0.9)
    ax_hm.set_xlabel("t")
    ax_hm.set_ylabel("x")
    ax_hm.set_title(
        f"η(x, t) — grey = uniform K={args.keep_samples}, red = adaptive (α={args.alpha}); "
        f"h={args.depth}, amp/crest={args.amp_per_crest}"
    )

    # Row 1: S(t) with sample markers (sanity).
    ax_s = fig.add_subplot(gs[1, :])
    ax_s.plot(times, s, color="#1f77b4", lw=1.1)
    ax_s.scatter(times[idx_uniform], s[idx_uniform], color="#444444", s=20, marker="o",
                 label=f"uniform K={args.keep_samples}")
    ax_s.scatter(times[idx_adaptive], s[idx_adaptive], color="#d62728", s=30, marker="x",
                 label=f"adaptive (α={args.alpha})")
    ax_s.set_ylabel(r"$\|\eta_x\|^2$")
    ax_s.legend(loc="upper right", fontsize=8)
    ax_s.grid(alpha=0.25)

    # Rows 2-3: the actual saved (η, ξ) at each adaptive sample.
    eta_max = float(np.max(np.abs(eta_traj))) * 1.05
    xi_max = float(np.max(np.abs(xi_traj))) * 1.05
    for col, t_idx in enumerate(idx_adaptive):
        ax_e = fig.add_subplot(gs[2, col])
        ax_x = fig.add_subplot(gs[3, col])
        ax_e.plot(x_grid, eta_traj[t_idx], color="#d62728", lw=0.8)
        ax_x.plot(x_grid, xi_traj[t_idx], color="#1f77b4", lw=0.8)
        ax_e.set_ylim(-eta_max, eta_max)
        ax_x.set_ylim(-xi_max, xi_max)
        ax_e.set_xticks([]); ax_x.set_xticks([])
        ax_e.set_yticks([]); ax_x.set_yticks([])
        ax_e.set_title(f"t={times[t_idx]:.1f}", fontsize=7, pad=1)
        if col == 0:
            ax_e.set_ylabel("η", rotation=0, labelpad=8, fontsize=9)
            ax_x.set_ylabel("ξ", rotation=0, labelpad=8, fontsize=9)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print(f"adaptive indices (T={times.shape[0]}): {idx_adaptive.tolist()}")
    print(f"adaptive times: {[f'{times[i]:.2f}' for i in idx_adaptive]}")
    print(f"uniform indices: {idx_uniform.tolist()}")


if __name__ == "__main__":
    main()
