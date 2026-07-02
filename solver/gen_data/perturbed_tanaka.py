"""Perturbed-soliton training data for the tanaka families.

Rollout diagnosis (2026-06-09): the CS-DNO baseline is near-exact on true
tanaka trajectories (1-step rel err ~2e-4) yet diverges on the steepest
solitons (a/h >~ 0.3) and is 10-70x worse at depth <~ 0.02. The training set
contains only exact soliton states, so the off-manifold neighborhood that
rollouts actually visit is unrepresented. This script samples tanaka_g0/g1
base states from the v3 TRAIN split (val stays honest), adds band-limited
perturbations to (eta, xi), and labels them with the order-6 Craig-Sulem
series in f64, rejecting draws whose series tail indicates non-convergence.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -u -m solver.gen_data.perturbed_tanaka \
        --n_draws 600000 --output data/tanaka_perturbed_v1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train-jax-10m"))

TANAKA_SOURCES = (5, 6)
PERTURBED_SOURCE_ID = 10


def dno_series_with_tail(
    eta: "jnp.ndarray", xi: "jnp.ndarray", k: "jnp.ndarray", depth: "jnp.ndarray",
    order: int, pad_factor: int,
) -> tuple["jnp.ndarray", "jnp.ndarray"]:
    """Order-`order` CS series sum and its last term, batched over rows.

    The last term is the convergence diagnostic: for a geometrically
    converging series the truncation error is bounded by ~|last term|.
    """
    import jax.numpy as jnp
    from solver.solvers.dno_series_jax import (
        compute_gm_term, make_linear_dno_symbol, multiply, myfft, myifft,
    )

    nx = int(eta.shape[-1])
    g0 = make_linear_dno_symbol(k, depth)
    etam = [jnp.ones_like(eta)]
    for m in range(1, order + 1):
        etam.append(multiply(eta, etam[m - 1], nx, pad_factor=pad_factor) / m)
    fxi = myfft(xi, nx)
    xi_x = myifft(1j * k * fxi)
    gm_terms = [myifft(g0 * fxi)]
    for m in range(1, order + 1):
        gm_terms.append(
            compute_gm_term(m, etam, gm_terms, xi_x, k, g0, nx, pad_factor=pad_factor)
        )
    total = gm_terms[0]
    for term in gm_terms[1:]:
        total = total + term
    return total, gm_terms[-1]


def band_limited_noise(rng: np.random.Generator, n: int, nx: int, k_keep: int) -> np.ndarray:
    """Unit-RMS random fields with content only in 1 <= k_index <= k_keep."""
    spec = np.fft.rfft(rng.standard_normal((n, nx)), axis=-1)
    spec[:, 0] = 0.0
    spec[:, k_keep + 1:] = 0.0
    out = np.fft.irfft(spec, n=nx, axis=-1)
    rms = np.sqrt(np.mean(out**2, axis=-1, keepdims=True))
    return out / np.maximum(rms, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="combined_dataset_v3.npz")
    parser.add_argument("--n_draws", type=int, default=600000)
    parser.add_argument("--output", default="data/tanaka_perturbed_v1.npz")
    parser.add_argument("--k_keep", type=int, default=128)
    parser.add_argument("--r_min", type=float, default=1e-3)
    parser.add_argument("--r_max", type=float, default=1e-1)
    parser.add_argument("--tail_tol", type=float, default=1e-4,
                        help="Reject if ||order-6 term|| / ||G xi|| exceeds this (series not converged).")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from solver.solvers.dno_series_jax import build_grid
    from util import build_split_indices

    t0 = time.time()
    print(f"loading tanaka train rows from {args.dataset} ...", flush=True)
    with np.load(REPO_ROOT / "data" / args.dataset) as d:
        src = np.asarray(d["source"])
        depth_all = np.asarray(d["depth"], dtype=np.float64)
        x_grid = np.asarray(d["x"], dtype=np.float32)
        train_idx, _, _ = build_split_indices(src.shape[0], 0)
        tanaka_train = train_idx[np.isin(src[train_idx], TANAKA_SOURCES)]
        eta_base = d["eta"][tanaka_train].astype(np.float64)
        xi_base = d["xi"][tanaka_train].astype(np.float64)
    depth_base = depth_all[tanaka_train]
    src_base = src[tanaka_train]
    nx = eta_base.shape[-1]
    length = 2.0 * np.pi
    _, k_np = build_grid(nx, length)
    k_grid = jnp.asarray(k_np, dtype=jnp.float64)
    label_fn = jax.jit(lambda e, x, d: dno_series_with_tail(e, x, k_grid, d, 6, 8))
    amax_eta = np.abs(eta_base).max(axis=-1)
    amax_xi = np.abs(xi_base).max(axis=-1)
    a_over_h = amax_eta / depth_base
    print(f"  {tanaka_train.size} base rows in {time.time()-t0:.0f}s; "
          f"depth [{depth_base.min():.3f},{depth_base.max():.3f}], "
          f"a/h p99={np.quantile(a_over_h, 0.99):.3f} max={a_over_h.max():.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    n_all = np.arange(tanaka_train.size)
    buckets = {
        "shallow(h<0.06)": (n_all[depth_base < 0.06], int(0.4 * args.n_draws)),
        "steep(a/h>0.2)": (n_all[a_over_h > 0.2], int(0.3 * args.n_draws)),
        "uniform": (n_all, args.n_draws - int(0.4 * args.n_draws) - int(0.3 * args.n_draws)),
    }
    draw_rows_list = []
    for name, (pool, n) in buckets.items():
        print(f"  bucket {name:16s}: pool={pool.size:8d} draws={n}", flush=True)
        draw_rows_list.append(rng.choice(pool, size=n, replace=True))
    draw_rows = np.concatenate(draw_rows_list)
    rng.shuffle(draw_rows)

    acc_eta, acc_xi, acc_gxi, acc_depth, acc_src = [], [], [], [], []
    n_rej_tail = n_rej_phys = 0
    tails = []
    t0 = time.time()
    for i in range(0, draw_rows.size, args.chunk):
        rows = draw_rows[i:i + args.chunk]
        n = rows.size
        eta0, xi0 = eta_base[rows], xi_base[rows]
        h = depth_base[rows]
        r_eta = np.exp(rng.uniform(np.log(args.r_min), np.log(args.r_max), size=(n, 1)))
        r_xi = np.exp(rng.uniform(np.log(args.r_min), np.log(args.r_max), size=(n, 1)))
        eta_p = eta0 + band_limited_noise(rng, n, nx, args.k_keep) * (r_eta * amax_eta[rows, None])
        xi_p = xi0 + band_limited_noise(rng, n, nx, args.k_keep) * (r_xi * amax_xi[rows, None])
        xi_p = xi_p - xi_p.mean(axis=-1, keepdims=True)

        phys_ok = eta_p.min(axis=-1) > -0.9 * h
        gxi_p, tail = label_fn(jnp.asarray(eta_p), jnp.asarray(xi_p), jnp.asarray(h[:, None]))
        gxi_p, tail = np.asarray(gxi_p), np.asarray(tail)
        tail_rel = (np.linalg.norm(tail, axis=-1)
                    / (np.linalg.norm(gxi_p, axis=-1) + 1e-30))
        finite = np.isfinite(gxi_p).all(axis=-1)
        keep = phys_ok & finite & (tail_rel < args.tail_tol)
        n_rej_phys += int((~phys_ok).sum())
        n_rej_tail += int((phys_ok & finite & ~keep).sum())
        tails.append(tail_rel)
        acc_eta.append(eta_p[keep].astype(np.float32))
        acc_xi.append(xi_p[keep].astype(np.float32))
        acc_gxi.append(gxi_p[keep].astype(np.float32))
        acc_depth.append(h[keep].astype(np.float32))
        acc_src.append(src_base[rows[keep]])
        if (i // args.chunk) % 50 == 0:
            done = i + n
            print(f"  {done}/{draw_rows.size} drawn, {sum(a.shape[0] for a in acc_eta)} kept, "
                  f"{(time.time()-t0)/max(done,1)*1e3:.1f} ms/draw", flush=True)

    eta_out = np.concatenate(acc_eta)
    xi_out = np.concatenate(acc_xi)
    gxi_out = np.concatenate(acc_gxi)
    depth_out = np.concatenate(acc_depth)
    base_src_out = np.concatenate(acc_src)
    tail_all = np.concatenate(tails)
    n_kept = eta_out.shape[0]
    print(f"kept {n_kept}/{draw_rows.size} "
          f"(rej tail={n_rej_tail}, rej phys={n_rej_phys}); "
          f"tail_rel med={np.median(tail_all):.2e} p99={np.quantile(tail_all, 0.99):.2e}", flush=True)
    print(f"absmax: eta={np.abs(eta_out).max():.4f} xi={np.abs(xi_out).max():.4f} "
          f"gxi={np.abs(gxi_out).max():.4f}", flush=True)

    out_path = REPO_ROOT / args.output
    np.savez(
        out_path,
        eta=eta_out, xi=xi_out, gxi=gxi_out, depth=depth_out,
        time=np.zeros(n_kept, dtype=np.float32),
        source=np.full(n_kept, PERTURBED_SOURCE_ID, dtype=np.int8),
        base_source=base_src_out.astype(np.int8),
        x=x_grid,
    )
    meta = {
        "kind": "tanaka_perturbed",
        "nx": int(nx), "length": length, "n_samples": int(n_kept),
        "source_id": PERTURBED_SOURCE_ID,
        "base_sources": list(TANAKA_SOURCES),
        "label": "order-6 pad-8 Craig-Sulem series, f64, tail-rejected",
        "tail_tol": args.tail_tol, "k_keep": args.k_keep,
        "r_range": [args.r_min, args.r_max], "seed": args.seed,
        "buckets": {name: n for name, (_, n) in buckets.items()},
    }
    Path(str(out_path).replace(".npz", ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
