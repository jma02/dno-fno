"""BF version of adaptive_subsample_demo: roll out a Benjamin-Feir IC and
visualize which time-steps the adaptive sampler picks vs. uniform.

Uses the same IC builder as the BF dataset generator. Default parameters
correspond to a canonical BF setup (n_carr=10, eps≈0.11, side_offset=3) that
exhibits clear envelope recurrence within tmax=80 s.

Usage:
    uv run python -m solver.evals.adaptive_subsample_demo_bf \\
        --keep_samples 24 --tmax 80 --dt 0.08 --alpha 0.5 --smooth_sigma_steps 10
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
    _gaussian_smooth_1d,
    adaptive_indices_from_signal,
)
from solver.gen_data.generate_bf_dataset import (
    build_bf_initial_conditions_batched,
    _grad_energy_traj,
    _envelope_peak_traj,
)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keep_samples", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--smooth_sigma_steps", type=float, default=10.0)
    p.add_argument("--depth", type=float, default=1.5)
    p.add_argument("--n_carr", type=int, default=10)
    p.add_argument("--side_offset", type=int, default=3)
    p.add_argument("--eps_carrier", type=float, default=0.11)
    p.add_argument("--eps_pert", type=float, default=0.1)
    p.add_argument("--phase_l", type=float, default=0.0)
    p.add_argument("--phase_r", type=float, default=0.0)
    p.add_argument("--length", type=float, default=2.0 * math.pi)
    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.08)
    p.add_argument("--tmax", type=float, default=80.0)
    p.add_argument("--gravity", type=float, default=1.0)
    p.add_argument("--bf_2nd_order", action="store_true", default=True)
    p.add_argument("--out", default="outputs/adaptive_demo_bf.png")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--signal", choices=("envelope", "grad_energy"), default="envelope")
    p.add_argument("--density_mode", choices=("rate", "magnitude"), default="magnitude",
                   help="rate = |dS/dt|/S (good for localized events), magnitude = S directly (good for envelope peaks)")
    p.add_argument("--power", type=float, default=1.0,
                   help="Sharpen contrast by raising the (signal - min) to this power. Useful for shallow modulation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    rollout_defaults = make_normalized_rollout_settings()._replace(filter_fraction=0.25)
    times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float64)
    x_grid_np, k_grid_np = build_grid(args.nx, args.length)
    x_grid = jnp.asarray(x_grid_np, dtype=jnp.float64)
    k_grid = jnp.asarray(k_grid_np, dtype=jnp.float64)

    params = {
        "n_carr": np.array([args.n_carr], dtype=np.int32),
        "n_l": np.array([args.n_carr - args.side_offset], dtype=np.int32),
        "n_r": np.array([args.n_carr + args.side_offset], dtype=np.int32),
        "side_offset": np.array([args.side_offset], dtype=np.int32),
        "eps_carrier": np.array([args.eps_carrier], dtype=np.float64),
        "eps_pert_l": np.array([args.eps_pert], dtype=np.float64),
        "eps_pert_r": np.array([args.eps_pert], dtype=np.float64),
        "phase_l": np.array([args.phase_l], dtype=np.float64),
        "phase_r": np.array([args.phase_r], dtype=np.float64),
        "depth": np.array([args.depth], dtype=np.float64),
    }

    initial_eta, initial_xi = build_bf_initial_conditions_batched(
        x=x_grid, params=params, length=args.length, gravity=args.gravity,
        bf_2nd_order=args.bf_2nd_order, dtype=jnp.float64,
    )
    initial_eta = apply_lowpass(initial_eta, k_grid, rollout_defaults.filter_fraction)
    initial_xi = apply_lowpass(initial_xi, k_grid, rollout_defaults.filter_fraction)
    if rollout_defaults.zero_mean_xi:
        initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

    depth_2d = jnp.asarray(params["depth"], dtype=jnp.float64)[:, None]
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

    eta_BTN = payload["eta"]  # (T, B=1, N)
    if args.signal == "envelope":
        signal_BT = np.asarray(jax.device_get(_envelope_peak_traj(eta_BTN)))
        signal_label = r"$\|\eta\|_\infty$"
    else:
        signal_BT = np.asarray(jax.device_get(_grad_energy_traj(eta_BTN, float(args.length))))
        signal_label = r"$\|\eta_x\|^2$"
    s = signal_BT[0]
    activity = np.abs(np.gradient(s)) / (s + 1e-12)

    idx_adaptive = adaptive_indices_from_signal(
        signal_BT, keep_samples=int(args.keep_samples),
        alpha=float(args.alpha), smooth_sigma_steps=float(args.smooth_sigma_steps),
        density_mode=args.density_mode, power=float(args.power),
    )[0]
    idx_uniform = np.linspace(0, times.shape[0] - 1, args.keep_samples, dtype=np.int32)

    eta_traj = np.asarray(jax.device_get(eta_BTN))[:, 0, :]
    xi_traj = np.asarray(jax.device_get(payload["xi"]))[:, 0, :]
    vmax_eta = float(np.max(np.abs(eta_traj)))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, args.keep_samples, height_ratios=[1.6, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.05)

    ax_hm = fig.add_subplot(gs[0, :])
    extent = [times[0], times[-1], x_grid_np[0], x_grid_np[-1]]
    ax_hm.imshow(eta_traj.T, aspect="auto", origin="lower", extent=extent,
                 cmap="RdBu_r", vmin=-vmax_eta, vmax=vmax_eta)
    for t_idx in idx_uniform:
        ax_hm.axvline(times[t_idx], color="#444444", lw=0.7, alpha=0.55)
    for t_idx in idx_adaptive:
        ax_hm.axvline(times[t_idx], color="#d62728", lw=0.9, alpha=0.9)
    ax_hm.set_xlabel("t")
    ax_hm.set_ylabel("x")
    ax_hm.set_title(
        f"BF η(x, t) — grey = uniform K={args.keep_samples}, red = adaptive (α={args.alpha}); "
        f"h={args.depth}, n_carr={args.n_carr}, ε={args.eps_carrier}"
    )

    ax_s = fig.add_subplot(gs[1, :])
    ax_s.plot(times, s, color="#1f77b4", lw=1.1)
    smoothed = _gaussian_smooth_1d(activity, args.smooth_sigma_steps)
    ax_s.scatter(times[idx_uniform], s[idx_uniform], color="#444444", s=20, marker="o", label="uniform")
    ax_s.scatter(times[idx_adaptive], s[idx_adaptive], color="#d62728", s=30, marker="x",
                 label=f"adaptive (α={args.alpha}, σ={args.smooth_sigma_steps:g})")
    ax_s.set_ylabel(signal_label)
    ax_s.legend(loc="upper right", fontsize=8)
    ax_s.grid(alpha=0.25)

    eta_max = float(np.max(np.abs(eta_traj))) * 1.05
    xi_max = float(np.max(np.abs(xi_traj))) * 1.05
    for col, t_idx in enumerate(idx_adaptive):
        ax_e = fig.add_subplot(gs[2, col])
        ax_x = fig.add_subplot(gs[3, col])
        ax_e.plot(x_grid_np, eta_traj[t_idx], color="#d62728", lw=0.8)
        ax_x.plot(x_grid_np, xi_traj[t_idx], color="#1f77b4", lw=0.8)
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
    print(f"adaptive times: {[f'{times[i]:.2f}' for i in idx_adaptive]}")
    print(f"uniform times:  {[f'{times[i]:.2f}' for i in idx_uniform]}")


if __name__ == "__main__":
    main()
