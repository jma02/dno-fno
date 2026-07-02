"""Generate a random-sea-state dataset for deep-water DNO training.

Each sample is a band-limited random superposition of Fourier modes drawn from
a Gaussian spectrum, with the surface potential xi derived from the linear
deep-water dispersion relation.  The DNO G(eta; h)xi is then evaluated exactly
(to truncation order M) via dno_series_eval.

Usage:
    uv run python playground/gen_random_sea.py --n_samples 100000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def _generate_profiles_numpy(
    n_samples: int,
    k_arr: np.ndarray,
    *,
    length: float,
    gravity: float,
    Hs_range: tuple[float, float],
    kp_range: tuple[float, float],
    bw_range: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all (eta, xi) profiles in pure numpy.  Returns (n, nx) arrays."""
    rng = np.random.default_rng(seed)
    n = len(k_arr)
    abs_k = np.abs(k_arr)

    # Draw random hyperparams for each sample
    Hs_all = rng.uniform(*Hs_range, size=n_samples)
    kp_all = rng.uniform(*kp_range, size=n_samples)
    bw_all = rng.uniform(*bw_range, size=n_samples)

    eta_all = np.empty((n_samples, n), dtype=np.float64)
    xi_all = np.empty((n_samples, n), dtype=np.float64)

    # Deep-water xi transfer: xi_hat = -i * sign(k) * (omega/|k|) * eta_hat
    omega_k = np.sqrt(gravity * abs_k)
    xi_ratio = np.where(k_arr != 0, omega_k / abs_k, 0.0)
    sign_k = np.sign(k_arr)

    for i in range(n_samples):
        sigma_k = kp_all[i] / bw_all[i]
        S = np.exp(-0.5 * ((abs_k - kp_all[i]) / sigma_k) ** 2)
        S[0] = 0.0
        amplitude = np.sqrt(S)
        phase = rng.uniform(0, 2 * np.pi, size=n)

        eta_hat = np.zeros(n, dtype=np.complex128)
        eta_hat[1 : n // 2] = amplitude[1 : n // 2] * np.exp(1j * phase[1 : n // 2])
        eta_hat[n // 2 + 1 :] = np.conj(eta_hat[1 : n // 2][::-1])
        eta = np.real(np.fft.ifft(eta_hat) * n)

        eta_std = np.std(eta)
        if eta_std > 1e-14:
            eta *= (Hs_all[i] / 4.0) / eta_std

        eta_all[i] = eta

        # Compute xi from scaled eta
        eta_fft = np.fft.fft(eta) / n
        xi_hat = -1j * sign_k * xi_ratio * eta_fft
        xi_all[i] = np.real(np.fft.ifft(xi_hat) * n)

    return eta_all, xi_all


def _batched_dno_eval(
    eta_all: np.ndarray,
    xi_all: np.ndarray,
    k: jnp.ndarray,
    depth: float,
    order: int,
    pad_factor: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Evaluate DNO in batches using the (..., nx) support in dno_series_eval."""
    n_samples = eta_all.shape[0]
    nx = eta_all.shape[1]
    gxi_all = np.empty((n_samples, nx), dtype=np.float32)

    # JIT-compile on first batch
    t0 = time.time()
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        eta_batch = jnp.asarray(eta_all[start:end])
        xi_batch = jnp.asarray(xi_all[start:end])
        gxi_batch = dno_series_eval(eta_batch, xi_batch, k, depth, order, pad_factor=pad_factor)
        gxi_all[start:end] = np.asarray(gxi_batch, dtype=np.float32)

        done = end
        elapsed = time.time() - t0
        rate = done / elapsed
        eta_left = (n_samples - done) / rate if rate > 0 else 0
        if done == end and (done <= batch_size or done % (batch_size * 10) == 0 or done == n_samples):
            print(f"  [{done:>7d}/{n_samples}]  {rate:.1f} samples/s  ETA {eta_left:.0f}s", flush=True)

    return gxi_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random sea state dataset.")
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1000.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--Hs_lo", type=float, default=0.1)
    parser.add_argument("--Hs_hi", type=float, default=0.6)
    parser.add_argument("--kp_lo", type=float, default=0.06)
    parser.add_argument("--kp_hi", type=float, default=0.50)
    parser.add_argument("--bw_lo", type=float, default=1.5)
    parser.add_argument("--bw_hi", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="playground/data/random_sea_deep.npz")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} random sea states (h={args.depth}, nx={args.nx})")
    print(f"  Hs in [{args.Hs_lo}, {args.Hs_hi}]")
    print(f"  kp in [{args.kp_lo}, {args.kp_hi}]")
    print(f"  bw in [{args.bw_lo}, {args.bw_hi}]")

    _, k = build_grid(args.nx, args.length)
    k_np = np.asarray(k)

    print("Generating profiles (numpy)...")
    t0 = time.time()
    eta_all, xi_all = _generate_profiles_numpy(
        args.n_samples, k_np,
        length=args.length, gravity=1.0,
        Hs_range=(args.Hs_lo, args.Hs_hi),
        kp_range=(args.kp_lo, args.kp_hi),
        bw_range=(args.bw_lo, args.bw_hi),
        seed=args.seed,
    )
    print(f"  Profiles done in {time.time() - t0:.1f}s")

    print("Evaluating DNO (batched)...")
    gxi_all = _batched_dno_eval(
        eta_all, xi_all, k, args.depth, args.dno_order, args.pad_factor,
        batch_size=args.batch_size,
    )

    # Convert to float32 for storage
    eta_f32 = eta_all.astype(np.float32)
    xi_f32 = xi_all.astype(np.float32)
    x_f32 = np.asarray(build_grid(args.nx, args.length)[0], dtype=np.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, eta=eta_f32, xi=xi_f32, gxi=gxi_all, x=x_f32)

    stats = {
        "n_samples": args.n_samples,
        "nx": args.nx,
        "length": args.length,
        "depth": args.depth,
        "feature_min": [float(np.min(eta_f32)), float(np.min(xi_f32))],
        "feature_max": [float(np.max(eta_f32)), float(np.max(xi_f32))],
        "feature_absmax": [float(np.max(np.abs(eta_f32))), float(np.max(np.abs(xi_f32)))],
        "target_min": float(np.min(gxi_all)),
        "target_max": float(np.max(gxi_all)),
        "target_absmax": float(np.max(np.abs(gxi_all))),
    }
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSaved {args.n_samples} samples -> {out}")
    print(f"  |eta|_max = {stats['feature_absmax'][0]:.4f}")
    print(f"  |xi|_max  = {stats['feature_absmax'][1]:.4f}")
    print(f"  |gxi|_max = {stats['target_absmax']:.4f}")


if __name__ == "__main__":
    main()
