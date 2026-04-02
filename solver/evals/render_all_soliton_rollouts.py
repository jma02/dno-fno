from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
import jax.numpy as jnp

from .. import State, batched_rollout, load_soliton_dataset, make_solver_params, rollout
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout comparisons for all soliton trajectories and save GIFs.")
    parser.add_argument("--output_dir", default="soliton_rollouts")
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=1.0)
    parser.add_argument("--default_method", default="gl2_if")
    parser.add_argument("--fast_method", default="gl2_if")
    parser.add_argument("--default_substeps", type=int, default=2)
    parser.add_argument("--fast_substeps", type=int, default=16)
    parser.add_argument("--implicit_iterations", type=int, default=8)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--zero_mean_xi", action="store_true")
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def substeps_for_case(case_name: str, default_substeps: int, fast_substeps: int) -> int:
    if case_name.startswith("coll.anim_f"):
        return fast_substeps
    return default_substeps


def method_for_case(case_name: str, default_method: str, fast_method: str) -> str:
    if case_name.startswith("coll.anim_f"):
        return fast_method
    return default_method


def zero_mean_time_series(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def make_payload(
    trajectory: dict,
    pred_eta: np.ndarray,
    pred_xi: np.ndarray,
    pred_gxi: np.ndarray,
    zero_mean_xi: bool,
) -> dict[str, np.ndarray]:
    truth_xi = np.asarray(trajectory["xi"])
    if zero_mean_xi:
        truth_xi = zero_mean_time_series(truth_xi)
        pred_xi = zero_mean_time_series(pred_xi)
    return {
        "x": np.asarray(trajectory["x"]),
        "t": np.asarray(trajectory["t"]),
        "truth_eta": np.asarray(trajectory["eta"]),
        "truth_xi": truth_xi,
        "truth_gxi": np.asarray(trajectory["gxi"]),
        "pred_eta": pred_eta,
        "pred_xi": pred_xi,
        "pred_gxi": pred_gxi,
    }


def trajectory_group_key(
    trajectory: dict,
    default_method: str,
    fast_method: str,
    default_substeps: int,
    fast_substeps: int,
) -> tuple[str, int, int, bytes]:
    case_name = str(trajectory["name"])
    method = method_for_case(case_name, default_method, fast_method)
    substeps = substeps_for_case(case_name, default_substeps, fast_substeps)
    times = np.asarray(trajectory["t"], dtype=np.float32)
    return method, substeps, times.shape[0], times.tobytes()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_soliton_dataset()
    params = make_solver_params(
        nx=int(dataset[0]["nx"]),
        length=float(dataset[0]["length"]),
        depth=1.0,
        gravity=1.0,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )

    total_start = perf_counter()
    if args.batched:
        groups: OrderedDict[tuple[str, int, int, bytes], dict] = OrderedDict()
        for trajectory in dataset:
            key = trajectory_group_key(
                trajectory,
                args.default_method,
                args.fast_method,
                args.default_substeps,
                args.fast_substeps,
            )
            if key not in groups:
                method, substeps, _, _ = key
                groups[key] = {
                    "method": method,
                    "substeps": substeps,
                    "times": np.asarray(trajectory["t"], dtype=np.float32),
                    "trajectories": [],
                }
            groups[key]["trajectories"].append(trajectory)

        processed = 0
        for group_idx, group in enumerate(groups.values(), start=1):
            trajectories = group["trajectories"]
            method = group["method"]
            substeps = group["substeps"]
            print(
                f"[batch {group_idx}/{len(groups)}] {len(trajectories)} cases | {method} | substeps={substeps}",
                flush=True,
            )
            start = perf_counter()
            initial_eta = np.stack([np.asarray(trajectory["eta"][0]) for trajectory in trajectories], axis=0)
            initial_xi = np.stack([np.asarray(trajectory["xi"][0]) for trajectory in trajectories], axis=0)
            prediction = batched_rollout(
                State(eta=jnp.asarray(initial_eta), xi=jnp.asarray(initial_xi)),
                jnp.asarray(group["times"]),
                params,
                save_gxi=True,
                substeps_per_interval=substeps,
                method=method,
                implicit_iterations=args.implicit_iterations,
                implicit_relaxation=args.implicit_relaxation,
                zero_mean_xi=args.zero_mean_xi,
            )
            pred_eta = np.asarray(prediction["eta"])
            pred_xi = np.asarray(prediction["xi"])
            pred_gxi = np.asarray(prediction["gxi"])

            for batch_idx, trajectory in enumerate(trajectories, start=1):
                case_name = str(trajectory["name"])
                gif_path = output_dir / f"{case_name}.gif"
                payload = make_payload(
                    trajectory,
                    pred_eta[:, batch_idx - 1],
                    pred_xi[:, batch_idx - 1],
                    pred_gxi[:, batch_idx - 1],
                    args.zero_mean_xi,
                )
                render_rollout_gif(
                    payload,
                    output_path=gif_path,
                    title=f"{case_name} | true=blue, ours=red | {method} | substeps={substeps}" + (" | xi-zero-mean" if args.zero_mean_xi else ""),
                    fps=args.fps,
                    n_frames=args.n_frames,
                    dpi=args.dpi,
                )
                processed += 1
                print(f"  [{processed}/{len(dataset)}] -> {gif_path}", flush=True)

            elapsed = perf_counter() - start
            print(f"  batch runtime {elapsed:.2f}s", flush=True)
    else:
        for idx, trajectory in enumerate(dataset, start=1):
            case_name = str(trajectory["name"])
            gif_path = output_dir / f"{case_name}.gif"
            print(f"[{idx}/{len(dataset)}] {case_name}", flush=True)

            start = perf_counter()
            substeps = substeps_for_case(case_name, args.default_substeps, args.fast_substeps)
            method = method_for_case(case_name, args.default_method, args.fast_method)
            prediction = rollout(
                State(eta=jnp.asarray(trajectory["eta"][0]), xi=jnp.asarray(trajectory["xi"][0])),
                jnp.asarray(trajectory["t"]),
                params,
                save_gxi=True,
                substeps_per_interval=substeps,
                method=method,
                implicit_iterations=args.implicit_iterations,
                implicit_relaxation=args.implicit_relaxation,
                zero_mean_xi=args.zero_mean_xi,
            )
            payload = make_payload(
                trajectory,
                np.asarray(prediction["eta"]),
                np.asarray(prediction["xi"]),
                np.asarray(prediction["gxi"]),
                args.zero_mean_xi,
            )
            render_rollout_gif(
                payload,
                output_path=gif_path,
                title=f"{case_name} | true=blue, ours=red | {method} | substeps={substeps}" + (" | xi-zero-mean" if args.zero_mean_xi else ""),
                fps=args.fps,
                n_frames=args.n_frames,
                dpi=args.dpi,
            )
            elapsed = perf_counter() - start
            print(f"  -> {gif_path} ({elapsed:.2f}s)", flush=True)

    print(f"done in {perf_counter() - total_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
