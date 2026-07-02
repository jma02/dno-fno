from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from util import NormStats, require_jax_devices
from jax_training_util import build_source_labels, load_training_arrays, parse_sources, split_indices
from solver.evals.render_rollout_movie import render_rollout_gif
from solver.solvers import time_integrator as ti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out dno_dataset initial conditions using a trained surrogate DNO model and render GIFs."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--dataset", default="dno_dataset.npz")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--shuffle_cases", action="store_true")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--method", default="gl2_if")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--implicit_iterations", type=int, default=4)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--keep_xi_mean", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def load_checkpoint(run_dir: Path, checkpoint_name: str):
    checkpoint_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    metadata_path = checkpoint_dir / "metadata.json"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return checkpoint_dir, params, metadata


def infer_length(x: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    return float(dx * x.shape[0])


def build_model(config: dict[str, object]):
    if config.get("model", "fno") == "spectral_dno":
        return SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", 2.0 * np.pi)),
            xi_scale=float(config.get("xi_scale", 1.0)),
            target_scale=float(config.get("target_scale", 1.0)),
        )
    return FNO1d(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
        domain_length=float(config.get("domain_length", 2.0 * np.pi)),
        xi_scale=float(config.get("xi_scale", 1.0)),
        target_scale=float(config.get("target_scale", 1.0)),
    )


def build_predict_gxi_fn(
    model,
    params,
    ns: NormStats,
    depth: float,
):
    feature_min = jnp.asarray(ns.feature_min, dtype=jnp.float32)
    feature_max = jnp.asarray(ns.feature_max, dtype=jnp.float32)
    feature_absmax = jnp.asarray(ns.feature_absmax, dtype=jnp.float32)
    target_min = jnp.asarray(ns.target_min, dtype=jnp.float32)
    target_max = jnp.asarray(ns.target_max, dtype=jnp.float32)
    target_absmax = jnp.asarray(ns.target_absmax, dtype=jnp.float32)
    log_depth = jnp.asarray(np.log(max(depth, 1e-12)), dtype=jnp.float32)
    mode = ns.mode

    def normalize_features_jax(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        stacked = jnp.stack((eta, xi), axis=-1)
        if mode == "scale":
            return stacked / feature_absmax
        scaled = (stacked - feature_min) / (feature_max - feature_min + 1e-8)
        return scaled * 2.0 - 1.0

    def denormalize_targets_jax(predicted_norm: jnp.ndarray) -> jnp.ndarray:
        if mode == "scale":
            return predicted_norm * target_absmax
        return ((predicted_norm + 1.0) * 0.5) * (target_max - target_min + 1e-8) + target_min

    @jax.jit
    def predict_gxi_batch(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        inputs = normalize_features_jax(eta, xi)
        batch_depth = jnp.full((eta.shape[0], 1), log_depth, dtype=jnp.float32)
        predicted_norm = model.apply({"params": params}, inputs, batch_depth)
        return denormalize_targets_jax(predicted_norm)[..., 0]

    return predict_gxi_batch


def rhs_nonlinear_surrogate(
    state: ti.State,
    params: ti.SolverParams,
    predict_gxi_batch,
) -> ti.State:
    eta_x = ti.spectral_dx(state.eta, params.k)
    xi_x = ti.spectral_dx(state.xi, params.k)
    gxi = predict_gxi_batch(state.eta[jnp.newaxis, :], state.xi[jnp.newaxis, :])[0]
    linear_gxi = ti.linear_dno_action(state.xi, params.g0)
    eta_t = gxi - linear_gxi
    numerator = gxi + eta_x * xi_x
    xi_t = -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    return ti.State(eta=eta_t, xi=xi_t)


def rhs_nonlinear_if_surrogate(
    v_hat: ti.SpectralState,
    t: float | jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi_batch,
) -> ti.SpectralState:
    physical_hat = ti.apply_linear_flow_hat(v_hat, t, params)
    physical_state = ti.State(
        eta=ti.myifft(physical_hat.eta_hat),
        xi=ti.myifft(physical_hat.xi_hat),
    )
    nonlinear_state = rhs_nonlinear_surrogate(physical_state, params, predict_gxi_batch)
    nonlinear_hat = ti.SpectralState(
        eta_hat=ti.myfft(nonlinear_state.eta, params.nx),
        xi_hat=ti.myfft(nonlinear_state.xi, params.nx),
    )
    return ti.apply_linear_flow_hat(nonlinear_hat, -t, params)


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, scale: float):
    return jax.tree_util.tree_map(lambda x: scale * x, a)


def _tree_axpy(base, scale: float, delta):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, base, delta)


def take_step_surrogate(
    state: ti.State,
    t: float,
    dt: float,
    params: ti.SolverParams,
    predict_gxi_batch,
    *,
    method: str,
    implicit_iterations: int,
    implicit_relaxation: float,
) -> ti.State:
    if method == "rk4_if":
        state_hat = ti.SpectralState(
            eta_hat=ti.myfft(state.eta, params.nx),
            xi_hat=ti.myfft(state.xi, params.nx),
        )
        v0 = ti.apply_linear_flow_hat(state_hat, -t, params)
        k1 = rhs_nonlinear_if_surrogate(v0, t, params, predict_gxi_batch)
        k2 = rhs_nonlinear_if_surrogate(_tree_axpy(v0, 0.5 * dt, k1), t + 0.5 * dt, params, predict_gxi_batch)
        k3 = rhs_nonlinear_if_surrogate(_tree_axpy(v0, 0.5 * dt, k2), t + 0.5 * dt, params, predict_gxi_batch)
        k4 = rhs_nonlinear_if_surrogate(_tree_axpy(v0, dt, k3), t + dt, params, predict_gxi_batch)
        v1 = _tree_add(
            v0,
            _tree_scale(
                _tree_add(
                    _tree_add(k1, _tree_scale(k2, 2.0)),
                    _tree_add(_tree_scale(k3, 2.0), k4),
                ),
                dt / 6.0,
            ),
        )
        next_hat = ti.apply_linear_flow_hat(v1, t + dt, params)
        next_state = ti.State(eta=ti.myifft(next_hat.eta_hat), xi=ti.myifft(next_hat.xi_hat))
    elif method == "implicit_midpoint_if":
        state_hat = ti.SpectralState(
            eta_hat=ti.myfft(state.eta, params.nx),
            xi_hat=ti.myfft(state.xi, params.nx),
        )
        v0 = ti.apply_linear_flow_hat(state_hat, -t, params)
        midpoint_time = t + 0.5 * dt

        def body_fn(_: int, stage: ti.SpectralState) -> ti.SpectralState:
            rhs = rhs_nonlinear_if_surrogate(stage, midpoint_time, params, predict_gxi_batch)
            candidate = _tree_axpy(v0, 0.5 * dt, rhs)
            if implicit_relaxation == 1.0:
                return candidate
            return jax.tree_util.tree_map(
                lambda s, c: (1.0 - implicit_relaxation) * s + implicit_relaxation * c,
                stage,
                candidate,
            )

        stage = jax.lax.fori_loop(0, implicit_iterations, body_fn, v0)
        v1 = jax.tree_util.tree_map(lambda mid, start: 2.0 * mid - start, stage, v0)
        next_hat = ti.apply_linear_flow_hat(v1, t + dt, params)
        next_state = ti.State(eta=ti.myifft(next_hat.eta_hat), xi=ti.myifft(next_hat.xi_hat))
    elif method == "gl2_if":
        sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
        c1 = 0.5 - sqrt3 / 6.0
        c2 = 0.5 + sqrt3 / 6.0
        a11 = 0.25
        a12 = 0.25 - sqrt3 / 6.0
        a21 = 0.25 + sqrt3 / 6.0
        a22 = 0.25

        state_hat = ti.SpectralState(
            eta_hat=ti.myfft(state.eta, params.nx),
            xi_hat=ti.myfft(state.xi, params.nx),
        )
        v0 = ti.apply_linear_flow_hat(state_hat, -t, params)

        def body_fn(_: int, stages: tuple[ti.SpectralState, ti.SpectralState]):
            stage1, stage2 = stages
            f1 = rhs_nonlinear_if_surrogate(stage1, t + c1 * dt, params, predict_gxi_batch)
            f2 = rhs_nonlinear_if_surrogate(stage2, t + c2 * dt, params, predict_gxi_batch)
            candidate1 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, a11), _tree_scale(f2, a12)), dt))
            candidate2 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, a21), _tree_scale(f2, a22)), dt))
            if implicit_relaxation == 1.0:
                return candidate1, candidate2
            next_stage1 = jax.tree_util.tree_map(
                lambda s, c: (1.0 - implicit_relaxation) * s + implicit_relaxation * c,
                stage1,
                candidate1,
            )
            next_stage2 = jax.tree_util.tree_map(
                lambda s, c: (1.0 - implicit_relaxation) * s + implicit_relaxation * c,
                stage2,
                candidate2,
            )
            return next_stage1, next_stage2

        stage1, stage2 = jax.lax.fori_loop(0, implicit_iterations, body_fn, (v0, v0))
        f1 = rhs_nonlinear_if_surrogate(stage1, t + c1 * dt, params, predict_gxi_batch)
        f2 = rhs_nonlinear_if_surrogate(stage2, t + c2 * dt, params, predict_gxi_batch)
        v1 = _tree_add(v0, _tree_scale(_tree_add(f1, f2), 0.5 * dt))
        next_hat = ti.apply_linear_flow_hat(v1, t + dt, params)
        next_state = ti.State(eta=ti.myifft(next_hat.eta_hat), xi=ti.myifft(next_hat.xi_hat))
    else:
        raise ValueError(f"Unknown time-stepping method {method!r}")

    if params.filter_fraction < 1.0:
        next_state = ti.State(
            eta=ti.apply_lowpass(next_state.eta, params.k, params.filter_fraction),
            xi=ti.apply_lowpass(next_state.xi, params.k, params.filter_fraction),
        )
    return next_state


def rollout_surrogate(
    initial_state: ti.State,
    times: jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi_batch,
    *,
    method: str,
    substeps_per_interval: int,
    implicit_iterations: int,
    implicit_relaxation: float,
    zero_mean_xi: bool,
):
    times = jnp.asarray(times)
    state0 = ti.State(eta=jnp.asarray(initial_state.eta), xi=jnp.asarray(initial_state.xi))
    if zero_mean_xi:
        state0 = ti.project_zero_mean_xi(state0)

    if times.shape[0] == 1:
        eta = state0.eta[jnp.newaxis, :]
        xi = state0.xi[jnp.newaxis, :]
    else:
        dts = times[1:] - times[:-1]

        def step_fn(carry: ti.State, data: tuple[jnp.ndarray, jnp.ndarray]):
            current_t, dt = data
            if substeps_per_interval == 1:
                next_state = take_step_surrogate(
                    carry,
                    current_t,
                    dt,
                    params,
                    predict_gxi_batch,
                    method=method,
                    implicit_iterations=implicit_iterations,
                    implicit_relaxation=implicit_relaxation,
                )
                if zero_mean_xi:
                    next_state = ti.project_zero_mean_xi(next_state)
            else:
                dt_sub = dt / substeps_per_interval

                def body_fn(sub_idx: int, subcarry: ti.State) -> ti.State:
                    sub_t = current_t + dt_sub * sub_idx
                    next_state = take_step_surrogate(
                        subcarry,
                        sub_t,
                        dt_sub,
                        params,
                        predict_gxi_batch,
                        method=method,
                        implicit_iterations=implicit_iterations,
                        implicit_relaxation=implicit_relaxation,
                    )
                    if zero_mean_xi:
                        next_state = ti.project_zero_mean_xi(next_state)
                    return next_state

                next_state = jax.lax.fori_loop(0, substeps_per_interval, body_fn, carry)
            return next_state, next_state

        _, saved_states = jax.lax.scan(step_fn, state0, (times[:-1], dts))
        eta = jnp.concatenate((state0.eta[jnp.newaxis, :], saved_states.eta), axis=0)
        xi = jnp.concatenate((state0.xi[jnp.newaxis, :], saved_states.xi), axis=0)

    gxi = predict_gxi_batch(eta, xi)
    return {"times": times, "eta": eta, "xi": xi, "gxi": gxi}


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint_dir, params, metadata = load_checkpoint(run_dir, args.checkpoint)

    dataset_path = REPO_ROOT / "data" / args.dataset
    selected_sources = parse_sources(args.sources)
    arrays = load_training_arrays(dataset_path, selected_sources)
    x = np.asarray(arrays["x"], dtype=np.float32)
    eta = np.asarray(arrays["eta"], dtype=np.float32)
    xi = np.asarray(arrays["xi"], dtype=np.float32)
    source_labels = build_source_labels(dataset_path, arrays["source_names"])

    _, val_indices, test_indices = split_indices(int(eta.shape[0]), args.seed)
    chosen_indices = val_indices if args.split == "val" else test_indices
    eta = eta[chosen_indices]
    xi = xi[chosen_indices]
    source_labels = source_labels[chosen_indices]
    global_indices = chosen_indices

    if not args.keep_xi_mean:
        xi = xi - np.mean(xi, axis=1, keepdims=True)

    if args.shuffle_cases:
        order = np.random.default_rng(args.seed).permutation(eta.shape[0])
        eta = eta[order]
        xi = xi[order]
        source_labels = source_labels[order]
        global_indices = global_indices[order]

    if args.max_cases is not None:
        eta = eta[: args.max_cases]
        xi = xi[: args.max_cases]
        source_labels = source_labels[: args.max_cases]
        global_indices = global_indices[: args.max_cases]

    if eta.shape[0] == 0:
        raise ValueError("No rollout cases selected after filtering.")

    train_stats = metadata.get("stats")
    if train_stats is None:
        raise ValueError(f"Checkpoint metadata at {checkpoint_dir} has no 'stats' entry.")
    ns = NormStats.from_dict(train_stats, mode=str(config.get("norm", "minmax")))
    if config.get("model", "fno") == "fno" and bool(
        config.get("linear_baseline", config.get("linear_hotpath", config.get("predict_residual", False)))
    ):
        raise ValueError("Checkpoint uses removed FNO linear-baseline/residual path.")

    model = build_model(config)
    predict_gxi_batch = build_predict_gxi_fn(model, params, ns, args.depth)
    length = infer_length(x)
    solver_params = ti.make_solver_params(
        nx=int(x.shape[0]),
        length=length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )
    rollout_times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float32)

    @jax.jit
    def rollout_pred_single_case(eta0: jnp.ndarray, xi0: jnp.ndarray):
        return rollout_surrogate(
            ti.State(eta=eta0, xi=xi0),
            jnp.asarray(rollout_times),
            solver_params,
            predict_gxi_batch,
            method=args.method,
            substeps_per_interval=args.substeps,
            implicit_iterations=args.implicit_iterations,
            implicit_relaxation=args.implicit_relaxation,
            zero_mean_xi=not args.keep_xi_mean,
        )

    @jax.jit
    def rollout_truth_single_case(eta0: jnp.ndarray, xi0: jnp.ndarray):
        return ti.rollout(
            ti.State(eta=eta0, xi=xi0),
            jnp.asarray(rollout_times),
            solver_params,
            save_gxi=True,
            substeps_per_interval=args.substeps,
            method=args.method,
            implicit_iterations=args.implicit_iterations,
            implicit_relaxation=args.implicit_relaxation,
            zero_mean_xi=not args.keep_xi_mean,
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / f"dno_dataset_rollouts_{args.checkpoint}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_dir),
        "checkpoint_epoch": int(metadata["epoch"]),
        "trained_on_dataset": config["dataset"],
        "evaluated_on_dataset": dataset_path.name,
        "backend": backend,
        "model": config.get("model", "fno"),
        "norm": config.get("norm", "minmax"),
        "deriv_features": bool(config.get("deriv_features", False)),
        "sources": list(selected_sources),
        "split": args.split,
        "split_seed": args.seed,
        "dt": args.dt,
        "tmax": args.tmax,
        "depth": args.depth,
        "gravity": args.gravity,
        "method": args.method,
        "substeps": args.substeps,
        "implicit_iterations": args.implicit_iterations,
        "implicit_relaxation": args.implicit_relaxation,
        "zero_mean_xi": not args.keep_xi_mean,
        "num_cases": int(eta.shape[0]),
        "cases": [],
        "per_source_timings": {},
    }

    total_start = perf_counter()
    for case_idx in range(eta.shape[0]):
        source = str(source_labels[case_idx])
        global_idx = int(global_indices[case_idx])
        stem = f"{case_idx:04d}_{source}_idx{global_idx:05d}"
        gif_path = output_dir / f"{stem}.gif"
        npz_path = output_dir / f"{stem}.npz"

        start = perf_counter()
        truth_start = perf_counter()
        truth_payload = rollout_truth_single_case(jnp.asarray(eta[case_idx]), jnp.asarray(xi[case_idx]))
        truth_payload = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), truth_payload)
        truth_elapsed = perf_counter() - truth_start

        pred_start = perf_counter()
        pred_payload = rollout_pred_single_case(jnp.asarray(eta[case_idx]), jnp.asarray(xi[case_idx]))
        pred_payload = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), pred_payload)
        pred_elapsed = perf_counter() - pred_start

        payload = {
            "x": x,
            "t": pred_payload["times"],
            "truth_eta": truth_payload["eta"],
            "truth_xi": truth_payload["xi"],
            "truth_gxi": truth_payload["gxi"],
            "pred_eta": pred_payload["eta"],
            "pred_xi": pred_payload["xi"],
            "pred_gxi": pred_payload["gxi"],
        }
        title = (
            f"{source} idx={global_idx} | true=blue, ours=red | "
            f"{args.method} | substeps={args.substeps}"
            + (" | xi-zero-mean" if not args.keep_xi_mean else "")
        )
        render_start = perf_counter()
        render_rollout_gif(
            payload,
            output_path=gif_path,
            title=title,
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )
        render_elapsed = perf_counter() - render_start
        if args.save_npz:
            np.savez_compressed(npz_path, **payload)

        elapsed = perf_counter() - start
        case_summary = {
            "case_idx_in_selected_split": case_idx,
            "global_idx": global_idx,
            "source": source,
            "gif_path": str(gif_path),
            "npz_path": str(npz_path) if args.save_npz else None,
            "truth_rollout_seconds": truth_elapsed,
            "pred_rollout_seconds": pred_elapsed,
            "render_seconds": render_elapsed,
            "runtime_seconds": elapsed,
        }
        summary["cases"].append(case_summary)
        print(json.dumps(case_summary), flush=True)

    summary["runtime_seconds"] = perf_counter() - total_start
    per_source_cases: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in summary["cases"]:
        per_source_cases[str(case["source"])].append(case)
    for source, cases in per_source_cases.items():
        truth_times = np.asarray([float(case["truth_rollout_seconds"]) for case in cases], dtype=np.float64)
        pred_times = np.asarray([float(case["pred_rollout_seconds"]) for case in cases], dtype=np.float64)
        render_times = np.asarray([float(case["render_seconds"]) for case in cases], dtype=np.float64)
        total_times = np.asarray([float(case["runtime_seconds"]) for case in cases], dtype=np.float64)
        summary["per_source_timings"][source] = {
            "num_cases": int(len(cases)),
            "mean_truth_rollout_seconds": float(np.mean(truth_times)),
            "median_truth_rollout_seconds": float(np.median(truth_times)),
            "mean_pred_rollout_seconds": float(np.mean(pred_times)),
            "median_pred_rollout_seconds": float(np.median(pred_times)),
            "mean_render_seconds": float(np.mean(render_times)),
            "median_render_seconds": float(np.median(render_times)),
            "mean_total_case_seconds": float(np.mean(total_times)),
            "median_total_case_seconds": float(np.median(total_times)),
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved rollout artifacts to {output_dir}")


if __name__ == "__main__":
    main()
