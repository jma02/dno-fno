"""Quantify how out-of-distribution the candidate test sets are.

Produces a summary table comparing key statistics across all training sources
and the two test files (dno_dataset.npz and stokes_bf_dataset.npz), after
applying the L=164 -> L=2pi DN-operator rescaling and the NX=512->1024
spectral upsample where needed.

Stats computed per source:
    eta_abs_max / eta_std      amplitude scale
    grad_eta_abs_max           steepness proxy (max |dx eta|)
    kp_peak                    dominant wavenumber from |FFT(eta)|
    kh                         kp_peak * depth
    xi_abs_max
    gxi_abs_max
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np


L_OLD = 164.0
L_NEW = 2.0 * math.pi
ALPHA = L_OLD / L_NEW
H_OLD_DNO_ASSUMED = 1.0


TRAIN_SOURCES: tuple[tuple[str, str], ...] = (
    ("random_sea_deep",   "data/random_sea_deep.npz"),
    ("random_sea_finite", "data/random_sea_finite.npz"),
    ("stokes_deep",       "data/stokes_deep.npz"),
    ("stokes_shallow",    "data/stokes_shallow.npz"),
    ("linear_shallow",    "data/linear_shallow.npz"),
    ("tanaka_g0",         "data/tanaka_2_g0.npz"),
    ("tanaka_g1",         "data/tanaka_2_g1.npz"),
    ("bf_g0",             "data/bf_2_g0.npz"),
    ("bf_g1",             "data/bf_2_g1.npz"),
)


def spectral_upsample(eta: np.ndarray, new_nx: int) -> np.ndarray:
    """Zero-pad eta in Fourier domain to expand spatial grid (band-limited exact)."""
    nx = eta.shape[-1]
    if new_nx == nx:
        return eta
    if new_nx < nx:
        raise ValueError(f"downsample not supported: nx={nx} -> {new_nx}")
    f = np.fft.rfft(eta, axis=-1)
    new_f_n = new_nx // 2 + 1
    pad_shape = list(f.shape); pad_shape[-1] = new_f_n - f.shape[-1]
    padded = np.concatenate([f, np.zeros(pad_shape, dtype=f.dtype)], axis=-1)
    return np.fft.irfft(padded, n=new_nx, axis=-1) * (new_nx / nx)


def per_sample_stats(eta: np.ndarray, xi: np.ndarray, gxi: np.ndarray, length: float) -> dict[str, np.ndarray]:
    """Vectorized stats over (n_samples, nx) arrays."""
    nx = eta.shape[-1]
    k = 2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx)  # (nx//2+1,)
    eta_hat = np.fft.rfft(eta, axis=-1)
    spec = np.abs(eta_hat[:, 1:])  # drop k=0
    kp_idx = np.argmax(spec, axis=-1) + 1
    kp = k[kp_idx]
    grad_eta = np.fft.irfft(1j * k[None, :] * eta_hat, n=nx, axis=-1)
    return {
        "eta_abs_max": np.abs(eta).max(axis=-1),
        "eta_std": eta.std(axis=-1),
        "grad_eta_abs_max": np.abs(grad_eta).max(axis=-1),
        "kp_peak": kp,
        "xi_abs_max": np.abs(xi).max(axis=-1),
        "gxi_abs_max": np.abs(gxi).max(axis=-1),
    }


def summarize(stats: dict[str, np.ndarray], depth: np.ndarray | float) -> dict[str, str]:
    def q(arr):
        a = np.asarray(arr).astype(np.float64)
        return f"{np.quantile(a, 0.05):.3g} / {np.quantile(a, 0.5):.3g} / {np.quantile(a, 0.95):.3g}"
    if np.isscalar(depth):
        kh = stats["kp_peak"] * float(depth)
        depth_str = f"{float(depth):.4g}"
    else:
        depth = np.asarray(depth)
        kh = stats["kp_peak"] * depth
        depth_str = q(depth)
    return {
        "eta_abs_max":      q(stats["eta_abs_max"]),
        "eta_std":          q(stats["eta_std"]),
        "grad_eta_abs_max": q(stats["grad_eta_abs_max"]),
        "kp_peak":          q(stats["kp_peak"]),
        "kh":               q(kh),
        "xi_abs_max":       q(stats["xi_abs_max"]),
        "gxi_abs_max":      q(stats["gxi_abs_max"]),
        "depth":            depth_str,
    }


def load_train_sample(path: Path, n_sample: int = 5000, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pull a random subset of (eta, xi, gxi, depth) from a per-batch zip."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        batch_ids = sorted({int(n[len("eta_batch_"):-len(".npy")])
                            for n in names if n.startswith("eta_batch_") and n.endswith(".npy")})
    rng = np.random.default_rng(seed)
    rng.shuffle(batch_ids)
    eta_parts, xi_parts, gxi_parts, depth_parts = [], [], [], []
    n_so_far = 0
    with np.load(path) as archive:
        for bid in batch_ids:
            tag = f"{bid:04d}"
            e = np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32)
            x_ = np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32)
            g = np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32)
            d_key = f"depth_batch_{tag}"
            d = np.asarray(archive[d_key], dtype=np.float32) if d_key in archive.files else np.full((e.shape[0],), np.nan, dtype=np.float32)
            eta_parts.append(e); xi_parts.append(x_); gxi_parts.append(g); depth_parts.append(d)
            n_so_far += e.shape[0]
            if n_so_far >= n_sample: break
    eta = np.concatenate(eta_parts, axis=0)[:n_sample]
    xi = np.concatenate(xi_parts, axis=0)[:n_sample]
    gxi = np.concatenate(gxi_parts, axis=0)[:n_sample]
    depth = np.concatenate(depth_parts, axis=0)[:n_sample]
    return {"eta": eta, "xi": xi, "gxi": gxi}, depth


def load_dno_test(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """Load and rescale the legacy dno_dataset.npz (L=164, h=1 assumed)."""
    h_new = H_OLD_DNO_ASSUMED / ALPHA
    out: dict[str, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(path) as zf:
        for label in ("soliton", "stokes", "linear"):
            with zf.open(f"{label}_eta.npy") as f:
                eta = np.load(f, allow_pickle=False).astype(np.float32) / ALPHA
            with zf.open(f"{label}_xi.npy") as f:
                xi = np.load(f, allow_pickle=False).astype(np.float32)
            with zf.open(f"{label}_Gxi.npy") as f:
                gxi = np.load(f, allow_pickle=False).astype(np.float32) * ALPHA
            out[label] = {"eta": eta, "xi": xi, "gxi": gxi}
    return out, h_new


def load_stokes_bf_test(path: Path, target_nx: int) -> dict[str, np.ndarray]:
    """Already on L=2pi, h=1000. Needs spectral upsample to target_nx."""
    with zipfile.ZipFile(path) as zf:
        with zf.open("stokes_eta.npy") as f:
            eta = np.load(f, allow_pickle=False).astype(np.float32)
        with zf.open("stokes_xi.npy") as f:
            xi = np.load(f, allow_pickle=False).astype(np.float32)
        with zf.open("stokes_Gxi.npy") as f:
            gxi = np.load(f, allow_pickle=False).astype(np.float32)
    if eta.shape[-1] != target_nx:
        eta = spectral_upsample(eta, target_nx).astype(np.float32)
        xi  = spectral_upsample(xi,  target_nx).astype(np.float32)
        gxi = spectral_upsample(gxi, target_nx).astype(np.float32)
    return {"eta": eta, "xi": xi, "gxi": gxi}


def main() -> None:
    rows: list[tuple[str, dict[str, str], int]] = []
    target_nx = 1024

    print("Sampling 5000 from each training source ...")
    for label, rel in TRAIN_SOURCES:
        path = Path(rel)
        if not path.exists():
            print(f"  {label}: missing, skipping")
            continue
        d, depth = load_train_sample(path, n_sample=5000)
        stats = per_sample_stats(d["eta"], d["xi"], d["gxi"], length=L_NEW)
        rows.append((label, summarize(stats, depth), d["eta"].shape[0]))
        print(f"  {label}: {d['eta'].shape[0]} samples loaded")

    # Test set 1: dno_dataset
    print("Loading dno_dataset (rescaled L=164 -> L=2pi, h_new ≈ 0.0383)...")
    dno, h_dno = load_dno_test(Path("data/dno_dataset.npz"))
    for label, d in dno.items():
        # Subsample to keep table compact
        n = min(d["eta"].shape[0], 5000)
        idx = np.random.default_rng(0).choice(d["eta"].shape[0], size=n, replace=False)
        stats = per_sample_stats(d["eta"][idx], d["xi"][idx], d["gxi"][idx], length=L_NEW)
        rows.append((f"TEST dno/{label} (rescaled)", summarize(stats, h_dno), n))

    # Test set 2: stokes_bf
    print("Loading stokes_bf_dataset (L=2pi, h=1000, NX=512 -> 1024)...")
    sbf = load_stokes_bf_test(Path("data/stokes_bf_dataset.npz"), target_nx=target_nx)
    n = min(sbf["eta"].shape[0], 5000)
    idx = np.random.default_rng(0).choice(sbf["eta"].shape[0], size=n, replace=False)
    stats = per_sample_stats(sbf["eta"][idx], sbf["xi"][idx], sbf["gxi"][idx], length=L_NEW)
    rows.append(("TEST stokes_bf", summarize(stats, 1000.0), n))

    # Print
    cols = ("eta_abs_max", "eta_std", "grad_eta_abs_max", "kp_peak", "kh",
            "xi_abs_max", "gxi_abs_max", "depth")
    header_w = 32
    print()
    print(f"{'source':<{header_w}} | {'n':>6} | " + " | ".join(f"{c:>22}" for c in cols))
    print("-" * (header_w + 8 + (24 + 3) * len(cols)))
    for label, summary, n in rows:
        print(f"{label:<{header_w}} | {n:>6} | " + " | ".join(f"{summary[c]:>22}" for c in cols))
    print()
    print("(values are 5%/50%/95% quantiles per column)")


if __name__ == "__main__":
    main()
