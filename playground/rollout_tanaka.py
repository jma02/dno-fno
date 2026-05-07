"""Rollout a trained FNO as surrogate DNO on soliton ICs from dno_dataset.

Picks soliton snapshots as ICs, rolls out with both the true DNO solver and
the FNO surrogate, and renders comparison GIFs.

Usage:
    uv run python playground/rollout_tanaka.py --run_dir playground/runs/combined_fno --case 0 --n_cases 3
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
from solver.solvers.dno_series_jax import build_grid, dno_series_eval
from solver.evals.render_rollout_movie import render_rollout_gif


def build_predict_gxi(
    model: FNO1d, params: dict, ns: dict, log_depth: float,
) -> callable:
    absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
    absmax = jnp.where(absmax > 0, absmax, 1.0)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    depth_arr = jnp.full((1, 1), log_depth)

    @jax.jit
    def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        stacked = jnp.stack((eta, xi), axis=-1)[None, :, :]
        inp = stacked / absmax
        pred_norm = model.apply({"params": params}, inp, depth_arr)
        return pred_norm[0, :, 0] * ta

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
    predict_gxi: callable, substeps: int = 8, gl2_iterations: int = 4,
) -> dict[str, jnp.ndarray]:
    state = ti.State(eta=jnp.asarray(initial.eta), xi=jnp.asarray(initial.xi))
    state = ti.project_zero_mean_xi(state)
    dts = times[1:] - times[:-1]

    def step_fn(carry: ti.State, data: tuple[jnp.ndarray, jnp.ndarray]):
        current_t, dt_interval = data
        dt_sub = dt_interval / substeps

        def body_fn(sub_idx: int, subcarry: ti.State) -> ti.State:
            ns = gl2_if_step_surrogate(
                subcarry, current_t + dt_sub * sub_idx, dt_sub, params,
                predict_gxi, iterations=gl2_iterations,
            )
            return ti.project_zero_mean_xi(ns)

        next_state = jax.lax.fori_loop(0, substeps, body_fn, carry)
        return next_state, next_state

    _, saved = jax.lax.scan(step_fn, state, (times[:-1], dts))
    eta = jnp.concatenate((state.eta[None, :], saved.eta), axis=0)
    xi = jnp.concatenate((state.xi[None, :], saved.xi), axis=0)
    return {"times": times, "eta": eta, "xi": xi}


def load_tanaka_ics(
    path: str, cases: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (eta, xi, x) for specific sample indices from the sharded Tanaka npz."""
    import json as _json, zipfile
    with zipfile.ZipFile(path, mode="r") as zf:
        meta = _json.loads(zf.read("meta.json"))
    batch_size = int(meta["batch_size"]) * int(meta.get("keep_samples", meta.get("n_time_samples_subsampled", 200)))
    with np.load(path) as f:
        x = np.asarray(f["x"], dtype=np.float32)
        eta_list, xi_list = [], []
        for ci in cases:
            shard = ci // batch_size
            local = ci % batch_size
            tag = f"{shard:04d}"
            eta_list.append(np.asarray(f[f"eta_batch_{tag}"][local], dtype=np.float32))
            xi_list.append(np.asarray(f[f"xi_batch_{tag}"][local], dtype=np.float32))
    return np.stack(eta_list), np.stack(xi_list), x


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout FNO surrogate on soliton ICs.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/dno_dataset.npz")
    parser.add_argument("--tanaka_data", type=str, default="data/tanaka_1_clean_sub500k.npz",
                        help="Tanaka sharded npz (used with --tanaka)")
    parser.add_argument("--tanaka", action="store_true", help="Load ICs from Tanaka dataset")
    parser.add_argument("--case", type=int, nargs="+", default=[0],
                        help="Index(es) into snapshots to use as IC")
    parser.add_argument("--tag", type=str, default=None, help="Override output file tag")
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        os.environ.pop("JAX_PLATFORMS", None)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))
    log_depth = float(np.log(args.depth))

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
    fno_params = tree_def.unflatten(flat_arrays)
    predict_gxi = build_predict_gxi(model, fno_params, ns, log_depth)

    # ---- Load ICs ----
    if args.tanaka:
        all_eta, all_xi, x = load_tanaka_ics(args.tanaka_data, args.case)
    else:
        with np.load(args.data) as f:
            full_eta = np.asarray(f["soliton_eta"], dtype=np.float32)
            full_xi = np.asarray(f["soliton_xi"], dtype=np.float32)
            x = np.asarray(f["x"], dtype=np.float32)
        all_eta = full_eta[args.case]
        all_xi = full_xi[args.case]

    nx = x.shape[0]
    length = float(x[1] - x[0]) * nx
    _, k = build_grid(nx, length)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=jnp.float32)
    n_steps = times.shape[0]

    solver_params = ti.make_solver_params(
        nx=nx, length=length, depth=args.depth, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
    )

    for local_i in range(all_eta.shape[0]):
        ci = args.case[local_i] if isinstance(args.case, list) else args.case + local_i
        eta0 = jnp.asarray(all_eta[local_i])
        xi0 = jnp.asarray(all_xi[local_i])
        xi0 = xi0 - jnp.mean(xi0)
        amp = float(jnp.max(jnp.abs(eta0)))

        print(f"\nCase {ci}: max|eta0|={amp:.4f}", flush=True)

        # Truth rollout
        print("Running truth rollout (DNO solver)...", flush=True)
        truth = ti.rollout(
            ti.State(eta=eta0, xi=xi0), times, solver_params,
            save_gxi=True, substeps_per_interval=args.substeps,
            method="gl2_if", zero_mean_xi=True,
        )

        # FNO surrogate rollout
        print("Running FNO surrogate rollout...", flush=True)
        pred = rollout_surrogate(
            ti.State(eta=eta0, xi=xi0), times, solver_params,
            predict_gxi, substeps=args.substeps,
        )

        truth_eta = np.asarray(truth["eta"])
        truth_xi = np.asarray(truth["xi"])
        truth_gxi = np.asarray(truth["gxi"])
        pred_eta = np.asarray(pred["eta"])
        pred_xi = np.asarray(pred["xi"])

        # Compute pred gxi
        pred_gxi_parts = [np.asarray(predict_gxi(pred["eta"][i], pred["xi"][i])) for i in range(n_steps)]
        pred_gxi = np.stack(pred_gxi_parts)

        def rel_l2(p: np.ndarray, t: np.ndarray) -> np.ndarray:
            return np.linalg.norm(p - t, axis=-1) / (np.linalg.norm(t, axis=-1) + 1e-12)

        eta_err = rel_l2(pred_eta, truth_eta)
        xi_err = rel_l2(pred_xi, truth_xi)

        print(f"  eta mean rel_l2 = {np.mean(eta_err):.6f}, final = {eta_err[-1]:.6f}", flush=True)
        print(f"  xi  mean rel_l2 = {np.mean(xi_err):.6f}, final = {xi_err[-1]:.6f}", flush=True)

        tag = args.tag or f"rollout_soliton_case{ci}"
        if all_eta.shape[0] > 1 and args.tag:
            tag = f"{args.tag}_{local_i}"

        gif_payload = {
            "x": x, "t": np.asarray(times),
            "truth_eta": truth_eta, "truth_xi": truth_xi, "truth_gxi": truth_gxi,
            "pred_eta": pred_eta, "pred_xi": pred_xi, "pred_gxi": pred_gxi,
        }
        gif_path = run_dir / f"{tag}.gif"
        title = f"Soliton case {ci} (amp={amp:.2f}) | true=blue, FNO=red | h={args.depth}"
        print("Rendering GIF...", flush=True)
        render_rollout_gif(gif_payload, output_path=gif_path, title=title, fps=30, n_frames=200, dpi=120)
        print(f"GIF -> {gif_path}", flush=True)

        summary = {
            "case": ci, "depth": args.depth, "amplitude": amp,
            "dt": args.dt, "tmax": args.tmax,
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
