from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..gen_data.multi_crest import build_multi_crest_preset
from ..solvers.time_integrator import State, batched_rollout, make_solver_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollout plots for multi-crest solitary-wave initial conditions.")
    parser.add_argument("--output_dir", default="test_multicrest_rollouts")
    parser.add_argument("--presets", nargs="+", default=["head_on_2v2", "head_on_3v2", "head_on_3v3"])
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--method", default="gl2_if")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--implicit_iterations", type=int, default=4)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--zero_mean_xi", action="store_true")
    parser.add_argument("--snapshots", type=int, default=6)
    return parser.parse_args()


def _plot_snapshots(
    axis: plt.Axes,
    x: np.ndarray,
    series: np.ndarray,
    times: np.ndarray,
    n_snapshots: int,
    title: str,
) -> None:
    indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
    for color, idx in zip(colors, indices, strict=True):
        axis.plot(x, series[idx], color=color, linewidth=1.8, label=f"t={times[idx]:.1f}")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)


def _plot_xt(axis: plt.Axes, x: np.ndarray, times: np.ndarray, eta: np.ndarray, title: str) -> None:
    vmax = float(np.max(np.abs(eta)))
    image = axis.imshow(
        eta,
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(times[0]), float(times[-1])],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def save_rollout_plot(
    output_path: Path,
    preset_name: str,
    x: np.ndarray,
    times: np.ndarray,
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    n_snapshots: int,
    method: str,
    substeps: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_xt(axes[0, 0], x, times, eta, r"$\eta(x,t)$")
    _plot_snapshots(axes[0, 1], x, eta, times, n_snapshots, r"$\eta(x)$ snapshots")
    _plot_snapshots(axes[1, 0], x, xi, times, n_snapshots, r"$\xi(x)$ snapshots")
    _plot_snapshots(axes[1, 1], x, gxi, times, n_snapshots, r"$G(\eta)\xi(x)$ snapshots")
    fig.suptitle(f"{preset_name} | {method} | substeps={substeps}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
        save_rollout_plot(
            output_dir / f"{stem}.png",
            preset_name,
            np.asarray(payload["x"]),
            times_np,
            predicted_eta[:, batch_idx],
            predicted_xi[:, batch_idx],
            predicted_gxi[:, batch_idx],
            args.snapshots,
            args.method,
            args.substeps,
        )
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            x=np.asarray(payload["x"]),
            t=times_np,
            eta=predicted_eta[:, batch_idx],
            xi=predicted_xi[:, batch_idx],
            gxi=predicted_gxi[:, batch_idx],
            eta0=np.asarray(payload["eta"]),
            xi0=np.asarray(payload["xi"]),
            gxi0=np.asarray(payload["gxi"]),
        )
        print(output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
