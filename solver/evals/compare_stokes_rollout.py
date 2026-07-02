from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .. import State, compare_rollout_to_truth, make_solver_params, rollout, stokes_truth_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the JAX PDE rollout against closed-form Stokes dynamics.")
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tmax", type=float, default=20.0)
    parser.add_argument("--n0", type=int, default=14)
    parser.add_argument("--a0", type=float, default=0.1)
    parser.add_argument("--ichoi", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=1.0)
    parser.add_argument("--output_dir", default="outputs/solver_rollouts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = stokes_truth_trajectory(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dt=args.dt,
        tmax=args.tmax,
        n0=args.n0,
        a0=args.a0,
        ichoi=args.ichoi,
        gxi_order=args.dno_order,
        pad_factor=args.pad_factor,
    )

    params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth if args.ichoi == 1 else 1000.0,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )

    initial = State(eta=truth["eta"][0], xi=truth["xi"][0])
    start = perf_counter()
    prediction = rollout(initial, truth["t"], params, save_gxi=True)
    metrics = compare_rollout_to_truth(truth, params)
    runtime = perf_counter() - start

    truth_eta = np.asarray(truth["eta"])
    truth_xi = np.asarray(truth["xi"])
    truth_gxi = np.asarray(truth["gxi"])
    pred_eta = np.asarray(prediction["eta"])
    pred_xi = np.asarray(prediction["xi"])
    pred_gxi = np.asarray(prediction["gxi"])
    x = np.asarray(truth["x"])
    t = np.asarray(truth["t"])
    eta_err = np.asarray(metrics["eta_rel_l2"])
    xi_err = np.asarray(metrics["xi_rel_l2"])
    gxi_err = np.asarray(metrics["gxi_rel_l2"])

    regime = "finite" if args.ichoi == 1 else "deep"
    stem = f"stokes_rollout_{regime}_n{args.n0:02d}_a{args.a0:.3f}_order{args.dno_order}"
    png_path = output_dir / f"{stem}.png"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    mid = len(t) // 2
    final = len(t) - 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].semilogy(t, eta_err + 1e-16, label="eta", linewidth=2.0)
    axes[0, 0].semilogy(t, xi_err + 1e-16, label="xi", linewidth=2.0)
    axes[0, 0].semilogy(t, gxi_err + 1e-16, label="Gxi", linewidth=2.0)
    axes[0, 0].set_title(f"Relative L2 Error: {truth['name']}")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, truth_eta[mid], label="truth", linewidth=2.0)
    axes[0, 1].plot(x, pred_eta[mid], label="rollout", linewidth=1.5, linestyle="--")
    axes[0, 1].set_title(f"eta(x) at t={t[mid]:.2f}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x, truth_eta[final], label="truth", linewidth=2.0)
    axes[1, 0].plot(x, pred_eta[final], label="rollout", linewidth=1.5, linestyle="--")
    axes[1, 0].set_title(f"eta(x) at t={t[final]:.2f}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, truth_xi[final], label="truth", linewidth=2.0)
    axes[1, 1].plot(x, pred_xi[final], label="rollout", linewidth=1.5, linestyle="--")
    axes[1, 1].set_title(f"xi(x) at t={t[final]:.2f}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        x=x,
        t=t,
        truth_eta=truth_eta,
        truth_xi=truth_xi,
        truth_gxi=truth_gxi,
        pred_eta=pred_eta,
        pred_xi=pred_xi,
        pred_gxi=pred_gxi,
        eta_rel_l2=eta_err,
        xi_rel_l2=xi_err,
        gxi_rel_l2=gxi_err,
    )

    summary = {
        "name": truth["name"],
        "backend": "jax",
        "regime": regime,
        "n0": args.n0,
        "a0": args.a0,
        "ichoi": args.ichoi,
        "dno_order": args.dno_order,
        "runtime_seconds": runtime,
        "eta_final_rel_l2": float(metrics["eta_final_rel_l2"]),
        "xi_final_rel_l2": float(metrics["xi_final_rel_l2"]),
        "gxi_final_rel_l2": float(metrics["gxi_final_rel_l2"]),
        "eta_mean_rel_l2": float(metrics["eta_mean_rel_l2"]),
        "xi_mean_rel_l2": float(metrics["xi_mean_rel_l2"]),
        "gxi_mean_rel_l2": float(metrics["gxi_mean_rel_l2"]),
        "png_path": str(png_path),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
