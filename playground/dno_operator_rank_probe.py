"""Experiment A — Global operator rank of R_M(eta) on the full Fourier basis.

At a fixed eta snapshot the residual operator R_M(eta) = G_M(eta) - G_0 is a
linear map xi |-> R_M(eta) xi. Apply it to *every* Fourier basis vector on a
grid of size N to build the N x N matrix, then SVD. Sweep N by spectrally
truncating eta.

Prop. 3 of the note states that fixed-M does NOT bound the global operator
rank. We expect:
  - numerical rank ~ N (the operator is essentially full rank);
  - rank-at-tolerance grows with N;
  - the spectrum is much flatter than the snapshot/restricted spectra.
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


def spectral_resample(field: np.ndarray, n_target: int) -> np.ndarray:
    """Spectral resampling of a 1D periodic field from len = n_src to n_target."""
    n_src = field.shape[-1]
    if n_target == n_src:
        return field.astype(np.float32)
    f = np.fft.fft(field)
    if n_target > n_src:
        pad = np.zeros(field.shape[:-1] + (n_target,), dtype=f.dtype)
        half = n_src // 2
        pad[..., :half] = f[..., :half]
        pad[..., -half + 1:] = f[..., -half + 1:]
        if n_src % 2 == 0:
            pad[..., half] = 0.5 * f[..., half]
            pad[..., -half] = 0.5 * f[..., half]
        resampled = np.fft.ifft(pad).real * (n_target / n_src)
    else:
        half = n_target // 2
        trunc = np.empty(field.shape[:-1] + (n_target,), dtype=f.dtype)
        trunc[..., :half] = f[..., :half]
        trunc[..., -half + 1:] = f[..., -half + 1:]
        if n_target % 2 == 0:
            trunc[..., half] = f[..., half] + f[..., -half]
        resampled = np.fft.ifft(trunc).real * (n_target / n_src)
    return resampled.astype(np.float32)


def operator_matrix(
    eta: np.ndarray,
    depth: float,
    length: float,
    order: int,
    pad_factor: int,
    mode_batch_size: int,
) -> np.ndarray:
    """Build the N x N matrix of R_M(eta) by applying it to each Fourier basis vector."""
    nx = int(eta.shape[-1])
    _, k = build_grid(nx, length)
    g0 = make_linear_dno_symbol(k, float(depth))

    @jax.jit
    def apply_op(eta_batch: jax.Array, xi_batch: jax.Array) -> jax.Array:
        full = dno_series_eval(eta_batch, xi_batch, k, float(depth), order, pad_factor=pad_factor)
        linear = myifft(myfft(xi_batch, nx) * g0)
        return full - linear

    basis = np.eye(nx, dtype=np.float32)
    out = np.empty((nx, nx), dtype=np.float32)
    eta_const = np.broadcast_to(eta.astype(np.float32), (mode_batch_size, nx))
    for start in range(0, nx, mode_batch_size):
        stop = min(nx, start + mode_batch_size)
        bs = stop - start
        eta_chunk = eta_const[:bs] if bs == mode_batch_size else np.broadcast_to(eta.astype(np.float32), (bs, nx))
        xi_chunk = basis[start:stop]
        result = apply_op(jnp.asarray(eta_chunk), jnp.asarray(xi_chunk))
        out[start:stop] = np.asarray(jax.device_get(result), dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_suite_dir", required=True,
                        help="eval_suite directory containing <regime>_trajs.npz to pull eta from.")
    parser.add_argument("--regimes", default="tanaka_g0,bf_g0,random_sea_deep,stokes_finite,linear",
                        help="Comma-separated regime names.")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=4)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--time_frac", type=float, default=0.5,
                        help="Fraction of rollout time at which to sample eta (0=start, 1=end).")
    parser.add_argument("--ic_select", default="median",
                        help="'best', 'worst', 'median', or 'idx=K'.")
    parser.add_argument("--n_sweep", default="128,256,512,1024",
                        help="Comma-separated grid sizes to sweep.")
    parser.add_argument("--mode_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="playground/runs/dno_operator_rank")
    args = parser.parse_args()

    eval_dir = Path(args.eval_suite_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    n_sweep = tuple(int(n) for n in args.n_sweep.split(",") if n.strip())
    results: list[dict[str, object]] = []

    for regime in regimes:
        npz_path = eval_dir / f"{regime}_trajs.npz"
        if not npz_path.exists():
            print(f"[{regime}] skipping: {npz_path} not found", flush=True)
            continue
        print(f"[{regime}] loading {npz_path.name}", flush=True)
        with np.load(npz_path) as d:
            truth_eta = np.asarray(d["truth_eta"])
            depths = np.asarray(d["depths"])
            rel_l2_eta = np.asarray(d["rel_l2_eta"])
            case_ids = np.asarray(d["case_ids"]) if "case_ids" in d.files else np.arange(depths.shape[0])

        n_t, n_b, nx_native = truth_eta.shape
        final = rel_l2_eta[-1]
        order_ic = np.argsort(np.where(np.isfinite(final), final, np.inf))
        sel = args.ic_select
        if sel == "best":
            j = int(order_ic[0])
        elif sel == "worst":
            j = int(order_ic[-1])
        elif sel == "median":
            j = int(order_ic[len(order_ic) // 2])
        elif sel.startswith("idx="):
            j = int(sel.split("=", 1)[1])
        else:
            j = int(sel)
        t_idx = int(round(args.time_frac * (n_t - 1)))
        eta_native = truth_eta[t_idx, j, :].astype(np.float32)
        depth = float(depths[j])
        case_id = int(case_ids[j])
        print(f"  IC idx={j} case={case_id} depth={depth:g} t_idx={t_idx} nx_native={nx_native}", flush=True)

        per_regime: list[dict[str, object]] = []
        for n_target in n_sweep:
            if n_target > nx_native:
                print(f"  skipping N={n_target} (> native nx={nx_native})", flush=True)
                continue
            eta_n = spectral_resample(eta_native, n_target)
            t0 = time.time()
            mat = operator_matrix(eta_n, depth, args.length, args.order, args.pad_factor, args.mode_batch_size)
            sigma = np.linalg.svd(mat, compute_uv=False)
            elapsed = float(time.time() - t0)
            ranks = {
                f"rel_tol_{tol:.0e}": int(np.sum(sigma > tol * max(sigma[0], 1e-30)))
                for tol in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
            }
            print(
                f"  N={n_target}: sigma_1={sigma[0]:.3e} sigma_N={sigma[-1]:.3e} ranks={ranks} elapsed={elapsed:.1f}s",
                flush=True,
            )
            per_regime.append({
                "N": int(n_target),
                "singular_values": sigma.astype(float).tolist(),
                "ranks": ranks,
                "elapsed_seconds": elapsed,
            })

        results.append({
            "regime": regime,
            "ic_idx": j,
            "case_id": case_id,
            "depth": depth,
            "time_frac": float(args.time_frac),
            "nx_native": int(nx_native),
            "by_N": per_regime,
        })

    summary = {
        "eval_suite_dir": str(eval_dir),
        "order": int(args.order),
        "pad_factor": int(args.pad_factor),
        "length": float(args.length),
        "n_sweep": list(n_sweep),
        "ic_select": args.ic_select,
        "time_frac": float(args.time_frac),
        "target_operator": "R_M(eta) viewed as N x N matrix on Fourier basis",
        "backend": jax.default_backend(),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not results:
        return

    n_regimes = len(results)
    cols = min(3, n_regimes)
    rows = int(np.ceil(n_regimes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    for ax, r in zip(axes_flat, results):
        entries = r["by_N"]
        for entry, color_t in zip(entries, np.linspace(0.1, 0.9, max(len(entries), 1))):
            s = np.asarray(entry["singular_values"], dtype=np.float64)
            rel = s / max(s[0], 1e-30)
            ax.semilogy(np.arange(1, s.shape[0] + 1) / s.shape[0], np.maximum(rel, 1e-16),
                        color=cmap(color_t), label=f"N={entry['N']}")
        ax.set_title(f"{r['regime']} (case={r['case_id']}, h={r['depth']:.2f})")
        ax.set_xlabel("singular index / N")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    fig.savefig(output_dir / "operator_rank_per_regime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for r in results:
        ns = [int(e["N"]) for e in r["by_N"]]
        rk = [int(e["ranks"]["rel_tol_1e-04"]) for e in r["by_N"]]
        ax.plot(ns, rk, marker="o", label=r["regime"])
    ax.plot(n_sweep, n_sweep, "k--", alpha=0.4, label="rank = N")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("N (grid size)")
    ax.set_ylabel(r"rank @ $\sigma > 10^{-4}\sigma_1$")
    ax.set_title("Operator rank vs grid size N")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "operator_rank_vs_N.png", dpi=160)
    plt.close(fig)

    print(f"wrote {output_dir / 'operator_rank_per_regime.png'}", flush=True)
    print(f"wrote {output_dir / 'operator_rank_vs_N.png'}", flush=True)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
