"""Variant-A handcoded 3-term Craig-Sulem ansatz, closed-form least-squares fit.

Ansatz:
    gxi ≈ G0(D,h)·ξ + Σ_i a_i · G0(D,h)·(f_i(η) · G0(D,h)·ξ)

with f_i(η) ∈ {η, |D|^{1/2}η, η²} and G0(k,h) = |k|·tanh(h|k|).

3 learnable scalar params (a₀, a₁, a₂). Solved via 3×3 normal equations on a
random subset of a v5/v8 dataset. CPU-only — uses numpy.fft, no JAX.

Compares per-example relative-L2 error vs:
  - G0-only baseline (no learned terms)
  - v5 single-step val_loss (sobolev-1; the rel-L2 here is a related but not
    identical metric — see notes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/combined_dataset_v5.npz",
                   help="npz with eta, xi, gxi, depth, x (e.g. v5/v8)")
    p.add_argument("--n_samples", type=int, default=20000,
                   help="number of (eta,xi,gxi,h) examples to sample")
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--report", default="notes/figures/cs_dno_v5_operator_viz/handcoded_3term_ls.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = perf_counter()

    print(f"loading {args.dataset} (mmap)...", flush=True)
    d = np.load(args.dataset, mmap_mode="r")
    eta_all = d["eta"]
    xi_all = d["xi"]
    gxi_all = d["gxi"]
    depth_all = d["depth"]
    x = np.asarray(d["x"])
    N_total, nx = eta_all.shape
    L = float(x[-1] - x[0] + (x[1] - x[0]))  # period
    print(f"  N={N_total:,}, nx={nx}, L={L:.6f}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N_total, size=args.n_samples, replace=False)
    idx.sort()
    print(f"sampling {args.n_samples} examples...", flush=True)
    eta = np.asarray(eta_all[idx], dtype=np.float64)
    xi = np.asarray(xi_all[idx], dtype=np.float64)
    gxi_truth = np.asarray(gxi_all[idx], dtype=np.float64)
    h_per = np.asarray(depth_all[idx], dtype=np.float64).reshape(-1, 1)  # (N,1)
    print(f"  shapes: eta {eta.shape}, depth in [{h_per.min():.4f}, {h_per.max():.4f}]", flush=True)

    # Angular wavenumbers for L-periodic grid
    dx = L / nx
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi  # (nx,)
    kx_abs = np.abs(kx)  # (nx,)

    print("computing G0(k,h) per example...", flush=True)
    G0_kh = kx_abs[None, :] * np.tanh(h_per * kx_abs[None, :])  # (N, nx)

    print("FFTing xi, computing G0·ξ...", flush=True)
    xi_hat = np.fft.fft(xi, axis=-1)
    G0_xi = np.fft.ifft(G0_kh * xi_hat, axis=-1).real  # (N, nx)
    del xi_hat

    print("FFTing eta, computing |D|^{1/2}η...", flush=True)
    eta_hat = np.fft.fft(eta, axis=-1)
    half_eta = np.fft.ifft(np.sqrt(kx_abs)[None, :] * eta_hat, axis=-1).real
    del eta_hat

    print("building features f_i = G0·(eta_feat · G0·ξ)...", flush=True)
    # f0 = G0(D,h) · (η · G0·ξ)
    f0_inner = eta * G0_xi
    f0 = np.fft.ifft(G0_kh * np.fft.fft(f0_inner, axis=-1), axis=-1).real
    del f0_inner

    # f1 = G0(D,h) · (|D|^{1/2}η · G0·ξ)
    f1_inner = half_eta * G0_xi
    f1 = np.fft.ifft(G0_kh * np.fft.fft(f1_inner, axis=-1), axis=-1).real
    del f1_inner, half_eta

    # f2 = G0(D,h) · (η² · G0·ξ)
    f2_inner = (eta ** 2) * G0_xi
    f2 = np.fft.ifft(G0_kh * np.fft.fft(f2_inner, axis=-1), axis=-1).real
    del f2_inner

    residual = gxi_truth - G0_xi  # (N, nx)

    print("stacking design matrix and solving normal equations...", flush=True)
    X = np.stack([f0.reshape(-1), f1.reshape(-1), f2.reshape(-1)], axis=-1)  # (N*nx, 3)
    y = residual.reshape(-1)  # (N*nx,)
    XtX = X.T @ X  # (3,3)
    Xty = X.T @ y  # (3,)
    a = np.linalg.solve(XtX, Xty)
    print(f"\nleast-squares solution:")
    print(f"  a₀ (·η)         = {a[0]:+.6f}")
    print(f"  a₁ (·|D|^{{1/2}}η) = {a[1]:+.6f}")
    print(f"  a₂ (·η²)        = {a[2]:+.6f}")
    del X, y

    # Reconstruct prediction
    pred = G0_xi + (a[0] * f0 + a[1] * f1 + a[2] * f2)
    err = pred - gxi_truth

    # Relative L2 per example
    eps = 1e-12
    truth_norm = np.sqrt(np.sum(gxi_truth ** 2, axis=-1)) + eps
    rel_l2_3t = np.sqrt(np.sum(err ** 2, axis=-1)) / truth_norm
    rel_l2_g0 = np.sqrt(np.sum((G0_xi - gxi_truth) ** 2, axis=-1)) / truth_norm

    def _stats(name, arr):
        print(f"  {name:>22s}: median={np.median(arr):.5f}  mean={np.mean(arr):.5f}  p95={np.percentile(arr,95):.5f}")
        return {"median": float(np.median(arr)),
                "mean":   float(np.mean(arr)),
                "p95":    float(np.percentile(arr, 95))}

    print("\nrelative L2 per-example (gxi prediction error):")
    s_g0 = _stats("G0-only baseline", rel_l2_g0)
    s_3t = _stats("3-term ansatz",    rel_l2_3t)

    # Sobolev-k=1 weighted error (matches v5 training loss flavor)
    print("\nSobolev-1 weighted error (sum_k (1+k²)·|err_k|² / sum_k (1+k²)·|truth_k|²):")
    w = 1.0 + kx ** 2  # (nx,)
    truth_hat = np.fft.fft(gxi_truth, axis=-1)
    err_hat = np.fft.fft(err, axis=-1)
    err_g0_hat = np.fft.fft(G0_xi - gxi_truth, axis=-1)
    truth_pow_w = (w[None, :] * (truth_hat.real ** 2 + truth_hat.imag ** 2)).sum(axis=-1) + eps
    err_pow_w = (w[None, :] * (err_hat.real ** 2 + err_hat.imag ** 2)).sum(axis=-1)
    err_g0_pow_w = (w[None, :] * (err_g0_hat.real ** 2 + err_g0_hat.imag ** 2)).sum(axis=-1)
    sob_3t = err_pow_w / truth_pow_w
    sob_g0 = err_g0_pow_w / truth_pow_w
    s_g0_sob = _stats("G0-only sob1",   sob_g0)
    s_3t_sob = _stats("3-term sob1",    sob_3t)
    # v5 single-step val_loss is the *mean* of the per-example sob1 over the val set
    print(f"\n  v5 best_val_ckpt val_loss (sobolev-1 train metric) = 0.00101")
    print(f"  3-term ansatz   mean sob1 (this sample)            = {sob_3t.mean():.5f}")
    print(f"  ratio (3-term / v5)                                = {sob_3t.mean() / 0.00101:.2f}×")

    # Write report
    rpt = {
        "dataset": str(Path(args.dataset).resolve()),
        "n_samples": int(args.n_samples),
        "n_data_points": int(args.n_samples * nx),
        "nx": int(nx),
        "L": L,
        "seed": int(args.seed),
        "ansatz": "gxi = G0·xi + a0·G0·(eta·G0·xi) + a1·G0·(|D|^0.5·eta·G0·xi) + a2·G0·(eta²·G0·xi)",
        "a": {"a0_eta": float(a[0]), "a1_half_eta": float(a[1]), "a2_eta2": float(a[2])},
        "param_count": 3,
        "rel_l2": {"g0_only": s_g0, "3term": s_3t},
        "sobolev1": {"g0_only_mean": float(sob_g0.mean()),
                     "3term_mean": float(sob_3t.mean()),
                     "v5_ckpt_val_loss_ref": 0.00101,
                     "ratio_3term_over_v5": float(sob_3t.mean() / 0.00101)},
        "elapsed_seconds": float(perf_counter() - t0),
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rpt, indent=2))
    print(f"\nreport -> {out_path}")
    print(f"total wall = {rpt['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
