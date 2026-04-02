from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..gen_data.multi_crest import build_multi_crest_preset
from ..solvers.time_integrator import State, batched_rollout, make_solver_params
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-crest rollout GIFs in the same style as the rollout comparison movies.")
    parser.add_argument("--output_dir", default="test_multicrest_rollouts")
    parser.add_argument("--presets", nargs="+", default=["head_on_2v2", "head_on_3v2", "head_on_3v3"])
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=30.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--method", default="gl2_if")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--implicit_iterations", type=int, default=4)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--zero_mean_xi", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt)

    payloads = [
        build_multi_crest_preset(
            preset_name,
            nx=args.nx,
            length=args.length,
            depth=args.depth,
            dno_order=args.dno_order,
            pad_factor=args.pad_factor,
            zero_mean_xi=args.zero_mean_xi,
        )
        for preset_name in args.presets
    ]

    initial_eta = np.stack([np.asarray(payload["eta"]) for payload in payloads], axis=0)
    initial_xi = np.stack([np.asarray(payload["xi"]) for payload in payloads], axis=0)

    rollout_payload = batched_rollout(
        State(eta=jnp.asarray(initial_eta), xi=jnp.asarray(initial_xi)),
        times,
        params,
        save_gxi=True,
        substeps_per_interval=args.substeps,
        method=args.method,
        implicit_iterations=args.implicit_iterations,
        implicit_relaxation=args.implicit_relaxation,
        zero_mean_xi=args.zero_mean_xi,
    )

    predicted_eta = np.asarray(rollout_payload["eta"])
    predicted_xi = np.asarray(rollout_payload["xi"])
    predicted_gxi = np.asarray(rollout_payload["gxi"])
    times_np = np.asarray(times)

    for batch_idx, payload in enumerate(payloads):
        preset_name = str(payload["name"])
        stem = f"{preset_name}_{args.method}_sub{args.substeps}"
        npz_path = output_dir / f"{stem}.npz"
        gif_path = output_dir / f"{stem}.gif"

        single_payload = {
            "x": np.asarray(payload["x"]),
            "t": times_np,
            "pred_eta": predicted_eta[:, batch_idx],
            "pred_xi": predicted_xi[:, batch_idx],
            "pred_gxi": predicted_gxi[:, batch_idx],
            "eta0": np.asarray(payload["eta"]),
            "xi0": np.asarray(payload["xi"]),
            "gxi0": np.asarray(payload["gxi"]),
        }
        np.savez_compressed(npz_path, **single_payload)
        render_rollout_gif(
            single_payload,
            output_path=gif_path,
            title=f"{preset_name} | {args.method} | substeps={args.substeps}" + (" | xi-zero-mean" if args.zero_mean_xi else ""),
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )
        print(gif_path)


if __name__ == "__main__":
    main()
