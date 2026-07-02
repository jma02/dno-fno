"""Experiment B — Snapshot rank of R_M(eta(t), xi(t)) over rollout time.

For each saved <regime>_trajs.npz in an eval_suite directory, evaluate the
residual

    R_M(eta(t), xi(t)) = G_M(eta(t)) xi(t) - G_0(h) xi(t)

at every saved time step for a single initial condition. Stack the outputs
into a (n_t, nx) matrix and SVD it. The reported singular spectrum is the
quantity the proofs in the note actually refer to: the dimension of the span
of the residual over an *evolving* (eta, xi) trajectory.

Expectations:
  - Tanaka single soliton: rank ≈ Fourier bandwidth of one translation orbit
    (Prop. 9 + lab-frame Prop. 10) — sharp cliff at small index.
  - BF carrier+sidebands: rank tracks active sideband count (Prop. 13/14).
  - Random / stokes_deep: smooth decay, no rank bound from M alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
    myfft,
    myifft,
)

DEFAULT_REGIMES = (
    "linear",
    "stokes_finite",
    "stokes_deep",
    "tanaka_g0",
    "tanaka_g1",
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "random_sea_finite",
    "random_sea_deep",
)


def select_ic(rel_l2_eta: np.ndarray, ic_select: str) -> int:
    final = rel_l2_eta[-1]
    order_ic = np.argsort(np.where(np.isfinite(final), final, np.inf))
    if ic_select == "best":
        return int(order_ic[0])
    if ic_select == "worst":
        return int(order_ic[-1])
    if ic_select == "median":
        return int(order_ic[len(order_ic) // 2])
    if ic_select.startswith("idx="):
        return int(ic_select.split("=", 1)[1])
    return int(ic_select)


def residual_stack(
    eta_t: np.ndarray,
    xi_t: np.ndarray,
    depth: float,
    length: float,
    order: int,
    pad_factor: int,
    batch_size: int,
) -> np.ndarray:
    """Return R_M evaluated at every time step. Shape (n_t, nx)."""
    nx = int(eta_t.shape[-1])
    _, k = build_grid(nx, length)
    g0 = make_linear_dno_symbol(k, float(depth))

    @jax.jit
    def step(eta_batch: jax.Array, xi_batch: jax.Array) -> jax.Array:
        full = dno_series_eval(eta_batch, xi_batch, k, float(depth), order, pad_factor=pad_factor)
        linear = myifft(myfft(xi_batch, nx) * g0)
        return full - linear

    n_t = int(eta_t.shape[0])
    out = np.empty((n_t, nx), dtype=np.float32)
    for start in range(0, n_t, batch_size):
        stop = min(n_t, start + batch_size)
        e = jnp.asarray(eta_t[start:stop])
        x = jnp.asarray(xi_t[start:stop])
        out[start:stop] = np.asarray(jax.device_get(step(e, x)), dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_suite_dir", required=True,
                        help="Path to eval_suite directory containing <regime>_trajs.npz.")
    parser.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=4)
    parser.add_argument("--ic_select", default="median",
                        help="'best', 'worst', 'median', or 'idx=K'.")
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="playground/runs/dno_snapshot_rank")
    args = parser.parse_args()

    eval_dir = Path(args.eval_suite_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    results: list[dict[str, object]] = []

    for regime in regimes:
        npz_path = eval_dir / f"{regime}_trajs.npz"
        if not npz_path.exists():
            print(f"[{regime}] skipping: {npz_path} not found", flush=True)
            continue
        print(f"[{regime}] loading {npz_path.name}", flush=True)
        with np.load(npz_path) as d:
            truth_eta = np.asarray(d["truth_eta"])
            truth_xi = np.asarray(d["truth_xi"])
            depths = np.asarray(d["depths"])
            rel_l2_eta = np.asarray(d["rel_l2_eta"])
            case_ids = np.asarray(d["case_ids"]) if "case_ids" in d.files else np.arange(depths.shape[0])

        n_t, n_b, nx = truth_eta.shape
        j = select_ic(rel_l2_eta, args.ic_select)
        eta_t = truth_eta[:, j, :].astype(np.float32)
        xi_t = truth_xi[:, j, :].astype(np.float32)
        depth = float(depths[j])
        case_id = int(case_ids[j])
        print(f"  IC idx={j} case={case_id} depth={depth:g} n_t={n_t} nx={nx}", flush=True)

        t0 = time.time()
        residual = residual_stack(eta_t, xi_t, depth, args.length, args.order, args.pad_factor, args.batch_size)
        elapsed = float(time.time() - t0)

        sigma = np.linalg.svd(residual, compute_uv=False)
        norms = np.linalg.norm(residual, axis=-1)
        ranks = {
            f"rel_tol_{tol:.0e}": int(np.sum(sigma > tol * max(sigma[0], 1e-30)))
            for tol in (1e-2, 1e-3, 1e-4, 1e-6)
        }
        print(
            f"  sigma_1={sigma[0]:.3e} ranks={ranks} "
            f"final_residual_norm={float(norms[-1]):.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )

        results.append({
            "regime": regime,
            "ic_idx": j,
            "case_id": case_id,
            "depth": depth,
            "n_t": int(n_t),
            "nx": int(nx),
            "singular_values": sigma.astype(float).tolist(),
            "residual_norms": norms.astype(float).tolist(),
            "ranks": ranks,
            "elapsed_seconds": elapsed,
        })

    summary = {
        "eval_suite_dir": str(eval_dir),
        "order": int(args.order),
        "pad_factor": int(args.pad_factor),
        "ic_select": args.ic_select,
        "length": float(args.length),
        "target_operator": "R_M(eta(t), xi(t)) = G_M(eta) xi - G_0(h) xi over rollout time",
        "backend": jax.default_backend(),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not results:
        print("no regimes produced output; exiting.", flush=True)
        return

    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes_flat, results):
        s = np.asarray(r["singular_values"], dtype=np.float64)
        rel = s / max(s[0], 1e-30)
        ax.semilogy(np.arange(1, s.shape[0] + 1), np.maximum(rel, 1e-16))
        ax.set_title(f"{r['regime']} (case={r['case_id']}, h={r['depth']:.2f})")
        ax.set_xlabel("singular index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    fig.savefig(output_dir / "snapshot_rank_per_regime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for r in results:
        s = np.asarray(r["singular_values"], dtype=np.float64)
        rel = s / max(s[0], 1e-30)
        ax.semilogy(np.arange(1, s.shape[0] + 1), np.maximum(rel, 1e-16),
                    label=f"{r['regime']} (rk@1e-4={r['ranks']['rel_tol_1e-04']})", alpha=0.8)
    ax.set_xlabel("singular index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(f"Snapshot rank of $R_M$ over rollout time (M={args.order})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "snapshot_rank_overlay.png", dpi=160)
    plt.close(fig)

    print(f"wrote {output_dir / 'snapshot_rank_per_regime.png'}", flush=True)
    print(f"wrote {output_dir / 'snapshot_rank_overlay.png'}", flush=True)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
