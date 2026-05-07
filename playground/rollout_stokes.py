"""Rollout a trained FNO as surrogate DNO on Stokes wave ICs and compare to truth.

Usage:
    uv run python playground/rollout_stokes.py --run_dir playground/runs/combined_fno --ichoi 1 --depth 1 --n0 14 --a0 0.1
    uv run python playground/rollout_stokes.py --run_dir playground/runs/combined_fno --ichoi 0 --depth 1000 --n0 10 --a0 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from fno_jax.fno1d import FNO1d
from solver.solvers import time_integrator as ti
from solver.data.stokes_truth_jax import stokes_eta_xi


# ---------------------------------------------------------------------------
# Surrogate rollout (FNO replaces dno_series_eval)
# ---------------------------------------------------------------------------

def build_predict_gxi(
    model: FNO1d, params: dict, ns: dict, log_depth: float,
) -> callable:
    absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
    absmax = jnp.where(absmax > 0, absmax, 1.0)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    depth_arr = jnp.full((1, 1), log_depth)

    @jax.jit
    def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        # eta, xi: (N,) single sample
        stacked = jnp.stack((eta, xi), axis=-1)[None, :, :]  # (1, N, 2)
        inp = stacked / absmax
        pred_norm = model.apply({"params": params}, inp, depth_arr)
        return (pred_norm[0, :, 0] * ta)

    return predict


def rhs_nonlinear_surrogate(
    state: ti.State, params: ti.SolverParams, predict_gxi: callable,
) -> ti.State:
    eta_x = ti.spectral_dx(state.eta, params.k)
    xi_x = ti.spectral_dx(state.xi, params.k)
    gxi = predict_gxi(state.eta, state.xi)
    linear_gxi = ti.linear_dno_action(state.xi, params.g0)
    eta_t = gxi - linear_gxi
    numerator = gxi + eta_x * xi_x
    xi_t = -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    return ti.State(eta=eta_t, xi=xi_t)


def rhs_nonlinear_if_surrogate(
    v_hat: ti.SpectralState, t: float, params: ti.SolverParams, predict_gxi: callable,
) -> ti.SpectralState:
    physical_hat = ti.apply_linear_flow_hat(v_hat, t, params)
    physical_state = ti.State(eta=ti.myifft(physical_hat.eta_hat), xi=ti.myifft(physical_hat.xi_hat))
    nl = rhs_nonlinear_surrogate(physical_state, params, predict_gxi)
    nl_hat = ti.SpectralState(eta_hat=ti.myfft(nl.eta, params.nx), xi_hat=ti.myfft(nl.xi, params.nx))
    return ti.apply_linear_flow_hat(nl_hat, -t, params)


def gl2_if_step_surrogate(
    state: ti.State, t: float, dt: float, params: ti.SolverParams,
    predict_gxi: callable, iterations: int = 4,
) -> ti.State:
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = ti.SpectralState(eta_hat=ti.myfft(state.eta, params.nx), xi_hat=ti.myfft(state.xi, params.nx))
    v0 = ti.apply_linear_flow_hat(state_hat, -t, params)

    _add = lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y, a, b)
    _scale = lambda a, s: jax.tree_util.tree_map(lambda x: s * x, a)

    def body_fn(_: int, stages: tuple[ti.SpectralState, ti.SpectralState]) -> tuple[ti.SpectralState, ti.SpectralState]:
        stage1, stage2 = stages
        f1 = rhs_nonlinear_if_surrogate(stage1, t + c1 * dt, params, predict_gxi)
        f2 = rhs_nonlinear_if_surrogate(stage2, t + c2 * dt, params, predict_gxi)
        candidate1 = _add(v0, _scale(_add(_scale(f1, a11), _scale(f2, a12)), dt))
        candidate2 = _add(v0, _scale(_add(_scale(f1, a21), _scale(f2, a22)), dt))
        return candidate1, candidate2

    stage1, stage2 = jax.lax.fori_loop(0, iterations, body_fn, (v0, v0))
    f1 = rhs_nonlinear_if_surrogate(stage1, t + c1 * dt, params, predict_gxi)
    f2 = rhs_nonlinear_if_surrogate(stage2, t + c2 * dt, params, predict_gxi)
    v1 = _add(v0, _scale(_add(f1, f2), 0.5 * dt))

    next_hat = ti.apply_linear_flow_hat(v1, t + dt, params)
    next_state = ti.State(eta=ti.myifft(next_hat.eta_hat), xi=ti.myifft(next_hat.xi_hat))
    if params.filter_fraction < 1.0:
        next_state = ti.State(
            eta=ti.apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=ti.apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def rollout_surrogate(
    initial: ti.State, times: jnp.ndarray, params: ti.SolverParams,
    predict_gxi: callable, substeps: int = 8, zero_mean_xi: bool = True,
    gl2_iterations: int = 4,
) -> dict[str, jnp.ndarray]:
    state = ti.State(eta=jnp.asarray(initial.eta), xi=jnp.asarray(initial.xi))
    if zero_mean_xi:
        state = ti.project_zero_mean_xi(state)

    dts = times[1:] - times[:-1]

    def step_fn(carry: ti.State, data: tuple[jnp.ndarray, jnp.ndarray]):
        current_t, dt_interval = data
        dt_sub = dt_interval / substeps

        def body_fn(sub_idx: int, subcarry: ti.State) -> ti.State:
            sub_t = current_t + dt_sub * sub_idx
            ns = gl2_if_step_surrogate(
                subcarry, sub_t, dt_sub, params, predict_gxi,
                iterations=gl2_iterations,
            )
            if zero_mean_xi:
                ns = ti.project_zero_mean_xi(ns)
            return ns

        next_state = jax.lax.fori_loop(0, substeps, body_fn, carry)
        return next_state, next_state

    _, saved = jax.lax.scan(step_fn, state, (times[:-1], dts))
    eta = jnp.concatenate((state.eta[None, :], saved.eta), axis=0)
    xi = jnp.concatenate((state.xi[None, :], saved.xi), axis=0)
    return {"times": times, "eta": eta, "xi": xi}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout FNO surrogate on Stokes ICs.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ichoi", type=int, default=1, choices=[0, 1])
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--n0", type=int, default=14)
    parser.add_argument("--a0", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        os.environ.pop("JAX_PLATFORMS", None)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))
    log_depth = float(np.log(args.depth))
    physical_depth = args.depth if args.ichoi == 1 else 1000.0

    # ---- Load model ----
    fa = np.asarray(ns["feature_absmax"], dtype=np.float32)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    model = FNO1d(
        modes=config["modes"], width=config["width"], n_blocks=config["n_blocks"],
        xi_scale=float(fa[1]), target_scale=ta,
    )
    with np.load(run_dir / "best_params.npz") as f:
        flat_arrays = [jnp.asarray(f[k]) for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    with open(run_dir / "tree_def.pkl", "rb") as f:
        tree_def = pickle.load(f)
    params = tree_def.unflatten(flat_arrays)

    predict_gxi = build_predict_gxi(model, params, ns, log_depth)

    # ---- Generate Stokes truth ----
    from solver.solvers.dno_series_jax import build_grid, dno_series_eval
    x, k = build_grid(args.nx, args.length)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=jnp.float32)

    print(f"Generating Stokes truth: ichoi={args.ichoi}, h={physical_depth}, n0={args.n0}, a0={args.a0}", flush=True)
    truth_eta, truth_xi = jax.vmap(
        lambda tau: stokes_eta_xi(x, tau, args.n0, args.a0, args.length, physical_depth, 1.0, ichoi=args.ichoi)
    )(times)

    truth_gxi = jax.vmap(
        lambda e, xi_row: dno_series_eval(e, xi_row, k, physical_depth, args.dno_order, pad_factor=8)
    )(truth_eta, truth_xi)

    # ---- Truth rollout (DNO solver) ----
    print("Running truth rollout (DNO solver)...", flush=True)
    solver_params = ti.make_solver_params(
        nx=args.nx, length=args.length, depth=physical_depth, gravity=1.0,
        dno_order=args.dno_order, pad_factor=8, filter_fraction=2.0 / 3.0,
    )
    truth_rollout = ti.rollout(
        ti.State(eta=truth_eta[0], xi=truth_xi[0]), times, solver_params,
        save_gxi=True, substeps_per_interval=args.substeps, method="gl2_if", zero_mean_xi=True,
    )

    # ---- Surrogate rollout (FNO) ----
    print("Running surrogate rollout (FNO)...", flush=True)
    pred_rollout = rollout_surrogate(
        ti.State(eta=truth_eta[0], xi=truth_xi[0]), times, solver_params,
        predict_gxi, substeps=args.substeps, zero_mean_xi=True,
    )

    # ---- Compute gxi for both rollouts ----
    from solver.solvers.dno_series_jax import dno_series_eval
    print("Computing gxi for rollouts...", flush=True)
    truth_r_gxi = np.asarray(jax.vmap(
        lambda e, xi_row: dno_series_eval(e, xi_row, k, physical_depth, args.dno_order, pad_factor=8)
    )(truth_rollout["eta"], truth_rollout["xi"]))

    pred_r_gxi_parts = []
    for i in range(pred_rollout["eta"].shape[0]):
        pred_r_gxi_parts.append(np.asarray(predict_gxi(pred_rollout["eta"][i], pred_rollout["xi"][i])))
    pred_r_gxi = np.stack(pred_r_gxi_parts)

    # ---- Compare ----
    truth_r_eta = np.asarray(truth_rollout["eta"])
    truth_r_xi = np.asarray(truth_rollout["xi"])
    pred_r_eta = np.asarray(pred_rollout["eta"])
    pred_r_xi = np.asarray(pred_rollout["xi"])
    t_arr = np.asarray(times)
    x_arr = np.asarray(x)

    def rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pred - truth, axis=-1) / (np.linalg.norm(truth, axis=-1) + 1e-12)

    eta_err = rel_l2(pred_r_eta, truth_r_eta)
    xi_err = rel_l2(pred_r_xi, truth_r_xi)

    regime = "finite" if args.ichoi == 1 else "deep"
    tag = f"rollout_{regime}_n{args.n0}_a{args.a0:.3f}"
    print(f"\n{tag}:", flush=True)
    print(f"  eta final rel_l2 = {eta_err[-1]:.6f}", flush=True)
    print(f"  xi  final rel_l2 = {xi_err[-1]:.6f}", flush=True)
    print(f"  eta mean  rel_l2 = {np.mean(eta_err):.6f}", flush=True)
    print(f"  xi  mean  rel_l2 = {np.mean(xi_err):.6f}", flush=True)

    # ---- Render GIF ----
    from solver.evals.render_rollout_movie import render_rollout_gif

    gif_payload = {
        "x": x_arr,
        "t": t_arr,
        "truth_eta": truth_r_eta,
        "truth_xi": truth_r_xi,
        "truth_gxi": truth_r_gxi,
        "pred_eta": pred_r_eta,
        "pred_xi": pred_r_xi,
        "pred_gxi": pred_r_gxi,
    }
    gif_path = run_dir / f"{tag}.gif"
    title = f"{regime} Stokes n0={args.n0} a0={args.a0} | true=blue, FNO=red"
    print("Rendering GIF...", flush=True)
    render_rollout_gif(gif_payload, output_path=gif_path, title=title, fps=30, n_frames=200, dpi=120)
    print(f"GIF -> {gif_path}", flush=True)

    # ---- Static plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = len(t_arr) // 2
    final = len(t_arr) - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].semilogy(t_arr, eta_err + 1e-16, label="eta", linewidth=2.0)
    axes[0, 0].semilogy(t_arr, xi_err + 1e-16, label="xi", linewidth=2.0)
    axes[0, 0].set_title(f"Relative L2 Error: {regime} n0={args.n0} a0={args.a0}")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x_arr, truth_r_eta[mid], label="truth", linewidth=2.0)
    axes[0, 1].plot(x_arr, pred_r_eta[mid], label="FNO", linewidth=1.5, linestyle="--")
    axes[0, 1].set_title(f"eta(x) at t={t_arr[mid]:.1f}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x_arr, truth_r_eta[final], label="truth", linewidth=2.0)
    axes[1, 0].plot(x_arr, pred_r_eta[final], label="FNO", linewidth=1.5, linestyle="--")
    axes[1, 0].set_title(f"eta(x) at t={t_arr[final]:.1f}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x_arr, truth_r_xi[final], label="truth", linewidth=2.0)
    axes[1, 1].plot(x_arr, pred_r_xi[final], label="FNO", linewidth=1.5, linestyle="--")
    axes[1, 1].set_title(f"xi(x) at t={t_arr[final]:.1f}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    plot_path = run_dir / f"{tag}.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"Plot -> {plot_path}", flush=True)

    summary = {
        "regime": regime, "ichoi": args.ichoi, "depth": physical_depth,
        "n0": args.n0, "a0": args.a0, "tmax": args.tmax, "dt": args.dt,
        "substeps": args.substeps,
        "eta_final_rel_l2": float(eta_err[-1]),
        "xi_final_rel_l2": float(xi_err[-1]),
        "eta_mean_rel_l2": float(np.mean(eta_err)),
        "xi_mean_rel_l2": float(np.mean(xi_err)),
    }
    json_path = run_dir / f"{tag}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
