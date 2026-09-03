#!/usr/bin/env python3
"""Discretization invariance probe for a trained CS-DNO checkpoint.

Loads a checkpoint produced by the canonical JAX trainer, picks test simulations
from a flat .npz dataset, and runs one-step inference at the native grid
resolution and at an arbitrary other resolution.  The test resolution may be
finer or coarser than the native 1024 grid.  For refined inputs the test output
is downsampled to the native grid; for coarser inputs the native output is
downsampled to the test grid.  Shared low-mode components are compared.

Optionally adds small random high-wavenumber modes to a refined input to see
whether the model remains stable and does not leak high-mode content into the
resolved low modes.

Runs on CPU by default so it does not contend with the running GPU workers.
"""

from __future__ import annotations

# Configure JAX to use CPU before any JAX import.
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _d in (
    REPO_ROOT,
    REPO_ROOT / "models" / "fno-jax",
    REPO_ROOT / "models" / "dno-net",
    REPO_ROOT / "train-jax-10m",
    REPO_ROOT / "solver" / "evals",
):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from model_rollout import (  # noqa: E402
    build_predict_gxi_batched,
    load_run,
)
from util import compute_log_depth  # noqa: E402


def change_resolution_rfft(f: np.ndarray, n_new: int) -> np.ndarray:
    """Spectrally resample a periodic 1-D real signal to n_new points.

    Works for upsampling (zero-padding high modes) and downsampling
    (truncating high modes) with the 'forward' FFT normalization.
    """
    n_old = f.shape[-1]
    n_old_freq = n_old // 2 + 1
    n_new_freq = n_new // 2 + 1
    f_hat = np.fft.rfft(f, norm="forward")
    f_hat_new = np.zeros(n_new_freq, dtype=f_hat.dtype)
    n_common = min(n_old_freq, n_new_freq)
    f_hat_new[:n_common] = f_hat[:n_common]
    return np.fft.irfft(f_hat_new, n=n_new, norm="forward")


def add_high_mode_noise(
    f: np.ndarray,
    n_native: int,
    relative_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add random high-wavenumber modes to a spectrally upsampled real signal.

    The standard deviation of the new complex Fourier coefficients is
    ``relative_std * rms(low_mode_coefficients)``.
    """
    n = f.shape[-1]
    n_native_freq = n_native // 2 + 1
    n_test_freq = n // 2 + 1
    if n_test_freq <= n_native_freq:
        return f

    f_hat = np.fft.rfft(f, norm="forward")
    low_rms = float(np.sqrt(np.mean(np.abs(f_hat[:n_native_freq]) ** 2)))
    high_std = relative_std * low_rms
    n_high = n_test_freq - n_native_freq

    f_hat_new = f_hat.copy()
    real_part = rng.normal(0.0, high_std, size=n_high).astype(np.float32)
    imag_part = rng.normal(0.0, high_std, size=n_high).astype(np.float32)
    f_hat_new[n_native_freq:] = real_part + 1j * imag_part
    return np.fft.irfft(f_hat_new, n=n, norm="forward")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error, robust to zero denominator."""
    denom = np.linalg.norm(b)
    if denom < 1e-12:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def rel_l2_mean_centered(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error after subtracting the mean from both signals."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    return rel_l2(a0, b0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("outputs/c27_h1_to_l2_full_20260717_212550"),
        help="Checkpoint run directory (must contain config.json and best_val_ckpt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/combined_dataset_v9.npz"),
        help="Flat .npz dataset with eta, xi, gxi, depth, x.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/discretization_invariance_probe"),
        help="Directory to write probe_results.json.",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=3,
        help="Number of test simulations to evaluate.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit simulation indices (overrides --n_simulations).",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=2048,
        help="Target resolution to test (e.g. 512 for coarse, 2048 for fine).",
    )
    parser.add_argument(
        "--high_mode_std",
        type=float,
        default=0.0,
        help="If > 0 and n_test > n_native, add random high-wavenumber modes with this relative std.",
    )
    parser.add_argument(
        "--domain_length",
        type=float,
        default=None,
        help="Override the model's domain_length for the test resolution (tests k-grid scaling).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    loaded = load_run(
        args.run_dir,
        checkpoint="best",
        domain_length_override=args.domain_length,
    )
    predict_batched = build_predict_gxi_batched(loaded)

    def predict(
        eta: np.ndarray,
        xi: np.ndarray,
        log_depth: float,
    ) -> np.ndarray:
        return np.asarray(
            predict_batched(
                jnp.asarray(eta, dtype=jnp.float32)[None, :],
                jnp.asarray(xi, dtype=jnp.float32)[None, :],
                jnp.asarray([log_depth], dtype=jnp.float32),
            )
        )[0]

    load_time = time.perf_counter() - t0
    print(f"Loaded checkpoint from {args.run_dir} in {load_time:.2f}s")
    print(f"Epoch: {loaded.epoch}, config model: {loaded.config.get('model')}")
    print(f"Model domain_length: {loaded.model.domain_length}")

    with np.load(args.dataset, mmap_mode="r") as z:
        eta = z["eta"]
        xi = z["xi"]
        gxi = z["gxi"]
        depth = z["depth"]
        x = z["x"]
        n_native = int(eta.shape[-1])
        print(f"Dataset: {args.dataset}")
        print(
            f"  simulations: {eta.shape[0]}, n_native: {n_native}, "
            f"domain: [{x[0]}, {x[-1]}]"
        )

    if args.indices is not None:
        test_indices = list(args.indices)
    else:
        rng = np.random.default_rng(42)
        test_indices = rng.choice(
            eta.shape[0], size=args.n_simulations, replace=False
        ).tolist()

    rng = np.random.default_rng(43)
    results: list[dict[str, object]] = []
    for idx in test_indices:
        if idx < 0 or idx >= eta.shape[0]:
            print(f"Skipping invalid index {idx}")
            continue

        eta_c = np.asarray(eta[idx], dtype=np.float32)
        xi_c = np.asarray(xi[idx], dtype=np.float32)
        gxi_c = np.asarray(gxi[idx], dtype=np.float32)
        depth_phys = float(depth[idx])
        log_h = float(compute_log_depth(np.asarray([depth_phys]))[0, 0])

        print(f"\nCase index {idx}: depth={depth_phys:.6f}, log_h={log_h:.6f}")

        # Native-resolution prediction.
        t0 = time.perf_counter()
        gxi_native = np.asarray(predict(eta_c, xi_c, log_h))
        t_native = time.perf_counter() - t0
        rel_l2_truth_mean = rel_l2_mean_centered(gxi_native, gxi_c)
        rel_l2_truth_raw = rel_l2(gxi_native, gxi_c)
        print(
            f"  native {n_native}: time={t_native:.3f}s, rel_l2_truth={rel_l2_truth_raw:.6f}, rel_l2_truth_mean_centered={rel_l2_truth_mean:.6f}"
        )

        # Test input at the new resolution.
        eta_test = change_resolution_rfft(eta_c, args.n_test)
        xi_test = change_resolution_rfft(xi_c, args.n_test)

        # Test-resolution prediction.
        t0 = time.perf_counter()
        gxi_test = np.asarray(predict(eta_test, xi_test, log_h))
        t_test = time.perf_counter() - t0
        print(f"  test   {args.n_test}: time={t_test:.3f}s")

        # Compare shared low-mode content.
        if args.n_test > n_native:
            compare_n = n_native
            gxi_test_compare = change_resolution_rfft(gxi_test, compare_n)
            gxi_native_compare = gxi_native
        elif args.n_test < n_native:
            compare_n = args.n_test
            gxi_test_compare = gxi_test
            gxi_native_compare = change_resolution_rfft(gxi_native, compare_n)
        else:
            compare_n = n_native
            gxi_test_compare = gxi_test
            gxi_native_compare = gxi_native

        rel_l2_cross = rel_l2(gxi_native_compare, gxi_test_compare)
        rel_l2_mean_cross = rel_l2_mean_centered(gxi_native_compare, gxi_test_compare)
        print(
            f"  {n_native} vs {args.n_test} rel_l2 (at {compare_n}): {rel_l2_cross:.6e}"
        )
        print(
            f"  {n_native} vs {args.n_test} rel_l2_mean_centered (at {compare_n}): {rel_l2_mean_cross:.6e}"
        )

        # Energy in modes above the native grid, only meaningful for n_test > n_native.
        high_mode_fraction: float = 0.0
        native_freq_count = n_native // 2 + 1
        if args.n_test > n_native:
            gxi_test_hat = np.fft.rfft(gxi_test, norm="forward")
            high_mode_energy = np.sum(np.abs(gxi_test_hat[native_freq_count:]) ** 2)
            total_mode_energy = np.sum(np.abs(gxi_test_hat) ** 2)
            high_mode_fraction = float(high_mode_energy / (total_mode_energy + 1e-30))
            print(
                f"  high-mode energy fraction above native k-grid: {high_mode_fraction:.6e}"
            )

        record: dict[str, object] = {
            "index": int(idx),
            "depth": depth_phys,
            "log_h": log_h,
            "n_native": n_native,
            "n_test": args.n_test,
            "domain_length": float(loaded.model.domain_length),
            "rel_l2_truth_raw": float(rel_l2_truth_raw),
            "rel_l2_truth_mean_centered": float(rel_l2_truth_mean),
            "rel_l2_cross": float(rel_l2_cross),
            "rel_l2_mean_cross": float(rel_l2_mean_cross),
            "high_mode_energy_fraction": high_mode_fraction,
            "time_native_s": float(t_native),
            "time_test_s": float(t_test),
        }

        # Optional high-mode injection test.
        if args.high_mode_std > 0 and args.n_test > n_native:
            eta_test_high = add_high_mode_noise(
                eta_test, n_native, args.high_mode_std, rng
            )
            xi_test_high = add_high_mode_noise(
                xi_test, n_native, args.high_mode_std, rng
            )
            t0 = time.perf_counter()
            gxi_test_high = np.asarray(predict(eta_test_high, xi_test_high, log_h))
            t_test_high = time.perf_counter() - t0
            gxi_test_high_down = change_resolution_rfft(gxi_test_high, n_native)
            rel_l2_with_high = rel_l2(gxi_test_compare, gxi_test_high_down)
            rel_l2_mean_with_high = rel_l2_mean_centered(
                gxi_test_compare, gxi_test_high_down
            )
            print(
                f"  high-mode std={args.high_mode_std}: time={t_test_high:.3f}s, rel_l2 vs no-injection: {rel_l2_with_high:.6e}, mean: {rel_l2_mean_with_high:.6e}"
            )

            gxi_test_high_hat = np.fft.rfft(gxi_test_high, norm="forward")
            high_mode_energy_high = np.sum(
                np.abs(gxi_test_high_hat[native_freq_count:]) ** 2
            )
            total_mode_energy_high = np.sum(np.abs(gxi_test_high_hat) ** 2)
            high_mode_fraction_high = float(
                high_mode_energy_high / (total_mode_energy_high + 1e-30)
            )
            print(
                f"  high-mode output energy fraction with injection: {high_mode_fraction_high:.6e}"
            )

            record["high_mode_std"] = float(args.high_mode_std)
            record["rel_l2_with_high_mode_injection"] = float(rel_l2_with_high)
            record["rel_l2_mean_with_high_mode_injection"] = float(
                rel_l2_mean_with_high
            )
            record["high_mode_energy_fraction_with_injection"] = high_mode_fraction_high

        results.append(record)

    out_path = args.output_dir / "probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    raise SystemExit(0)
