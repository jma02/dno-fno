"""Inspect the per-block φ(η) spectra for the C2 final CS-DNO.

φ is the pointwise per-branch activation inside each CraigSulemBlock.
If φ has low/mid-k Fourier content, then the product φ·ξ_branch will
scatter a pure high-k ξ into neighboring sidebands, which is exactly the
mechanism seen in the A2 transfer kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solver.evals import model_rollout as mr


def load_failing_state(run_dir: Path, regime: str, case_id: int, frame: int):
    npz_dir = run_dir / "eval_final_2reg_f64h_batched_cached_20260707_133545" / regime
    npz = np.load(npz_dir / f"{regime}_trajs.npz")
    case_idx = int(np.where(npz["case_ids"] == case_id)[0][0])
    h = float(npz["depths"][case_idx])
    time = float(npz["times"][frame])
    pred_eta = np.asarray(npz["pred_eta"][frame, case_idx, :], dtype=np.float64)
    truth_eta = np.asarray(npz["truth_eta"][frame, case_idx, :], dtype=np.float64)
    return pred_eta, truth_eta, h, time


def compute_raw_eta_features(eta: np.ndarray, h: float, L: float) -> dict[str, np.ndarray]:
    """Compute the raw η-features that the CS-DNO trunk uses (polynomials + spectral derivatives)."""
    N = eta.shape[-1]
    k = (2.0 * np.pi / L) * np.arange(N // 2 + 1)
    eta_hat = np.fft.rfft(eta)
    feats = {
        "eta": eta,
        "eta2": eta ** 2,
        "eta3": eta ** 3,
        "dx": np.fft.irfft(1j * k * eta_hat, n=N),
        "d2x": np.fft.irfft(-(k ** 2) * eta_hat, n=N),
        "half_deriv": np.fft.irfft(np.sqrt(np.abs(k)) * eta_hat, n=N),
        "hilbert": np.fft.irfft(1j * np.sign(k) * eta_hat, n=N),
    }
    # G0(h)·η
    g0 = k * np.tanh(h * k)
    feats["g0_eta"] = np.fft.irfft(g0 * eta_hat, n=N)
    feats["dx_g0_eta"] = np.fft.irfft(1j * k * g0 * eta_hat, n=N)
    return feats


def analyze_phi(run_dir: Path, eta: np.ndarray, h: float, time: float, suffix: str, out_dir: Path):
    loaded = mr.load_run(run_dir, checkpoint="final")
    nx = eta.shape[-1]
    L = float(loaded.config["domain_length"])
    log_depth = float(np.log(h))

    # phi depends only on η, so ξ=0 is fine (and makes block output zero).
    stacked = np.stack([eta, np.zeros_like(eta)], axis=-1).astype(np.float32)[None, :, :]
    depth_arg = np.full((1, 1), log_depth, dtype=np.float32)

    _, state = loaded.model.apply(
        {"params": loaded.params},
        jnp.asarray(stacked),
        jnp.asarray(depth_arg),
        capture_intermediates=True,
        mutable="intermediates",
    )
    intm = state["intermediates"]

    n_blocks = int(loaded.config["n_blocks"])
    n_branches = int(loaded.config["latent"])

    eta_features = np.asarray(intm["eta_feat_mix"]["__call__"][0][0])  # (N, C)
    raw_features = compute_raw_eta_features(eta, h, L)

    k = np.arange(nx // 2 + 1)
    dk = 2.0 * np.pi / L
    k_phys = k * dk

    block_results = []
    phi_specs = []  # list of (n_branches, n_freq) arrays
    for b in range(n_blocks):
        phi = np.asarray(intm[f"cs_block_{b}"]["phi_proj"]["__call__"][0][0])  # (N, n_branches)
        phi_hat = np.fft.rfft(phi, axis=0)  # (n_freq, n_branches)
        phi_mag = np.abs(phi_hat) / (nx / 2)  # physical amplitude
        phi_specs.append(phi_mag)

        # Mean spectrum over branches.
        mean_spec = np.mean(phi_mag, axis=1)
        max_spec = np.max(phi_mag, axis=1)
        std_spec = np.std(phi_mag, axis=1)

        # Energy in bands.
        bands = {
            "low": (k_phys < 32),
            "mid": (k_phys >= 32) & (k_phys < 96),
            "high": (k_phys >= 96) & (k_phys < 128),
            "very_high": (k_phys >= 128),
        }
        band_energy = {name: float(np.sum(mean_spec[mask] ** 2)) for name, mask in bands.items()}
        total_energy = float(np.sum(mean_spec ** 2))

        # Peak locations.
        peak_idx = np.argsort(mean_spec)[-10:][::-1]
        peaks = [{"k": int(k[i]), "amp": float(mean_spec[i]), "max_amp": float(max_spec[i])} for i in peak_idx]

        block_results.append({
            "block": b,
            "phi_std": float(phi.std()),
            "phi_mean": float(phi.mean()),
            "total_energy": total_energy,
            "band_energy": band_energy,
            "low_fraction": float(band_energy["low"] / (total_energy + 1e-30)),
            "top_peaks": peaks,
        })

    # eta_features spectrum.
    feat_hat = np.fft.rfft(eta_features, axis=0)  # (n_freq, C)
    feat_mag = np.abs(feat_hat) / (nx / 2)
    feat_mean_spec = np.mean(feat_mag, axis=1)
    feat_max_spec = np.max(feat_mag, axis=1)

    # Raw feature spectra.
    raw_specs = {}
    for name, f in raw_features.items():
        fhat = np.fft.rfft(f)
        raw_specs[name] = np.abs(fhat) / (nx / 2)

    summary = {
        "suffix": suffix,
        "case_id": 5,
        "frame": 40,
        "time": time,
        "depth": h,
        "n_blocks": n_blocks,
        "n_branches": n_branches,
        "block_results": block_results,
        "eta_features_mean_spectrum": [float(v) for v in feat_mean_spec],
        "eta_features_max_spectrum": [float(v) for v in feat_max_spec],
        "raw_feature_spectra": {name: [float(v) for v in spec] for name, spec in raw_specs.items()},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Per-block mean phi spectrum.
    ax = axes[0, 0]
    for b in range(n_blocks):
        ax.semilogy(k_phys, np.mean(phi_specs[b], axis=1), label=f"block {b}", lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_xlabel("k")
    ax.set_ylabel("mean |φ(k)| over branches")
    ax.set_title("Per-block φ(η) mean spectrum")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Per-block max phi spectrum.
    ax = axes[0, 1]
    for b in range(n_blocks):
        ax.semilogy(k_phys, np.max(phi_specs[b], axis=1), label=f"block {b}", lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_xlabel("k")
    ax.set_ylabel("max |φ(k)| over branches")
    ax.set_title("Per-block φ(η) max spectrum")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # eta_features spectrum.
    ax = axes[1, 0]
    ax.semilogy(k_phys, feat_mean_spec, label="mean over channels", lw=0.8)
    ax.semilogy(k_phys, feat_max_spec, label="max over channels", lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_xlabel("k")
    ax.set_ylabel("|η_feature(k)|")
    ax.set_title("Projected η-features spectrum")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Raw feature spectra.
    ax = axes[1, 1]
    for name, spec in raw_specs.items():
        ax.semilogy(k_phys, spec, label=name, lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_xlabel("k")
    ax.set_ylabel("|feature(k)|")
    ax.set_title("Raw η spatial features")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "phi_spectra.png", dpi=200)
    plt.close(fig)

    # Branch heatmaps for blocks 2 and 3 (the suspicious ones from A1).
    for b in [2, 3]:
        if b >= n_blocks:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        data = np.log10(np.maximum(phi_specs[b].T, 1e-12))  # (branches, n_freq)
        im = ax.imshow(data[:, :256], aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("k")
        ax.set_ylabel("branch")
        ax.set_title(f"block {b}: log10 |φ(k)| per branch")
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels([str(int(k_phys[i])) for i in [0, 64, 128, 192, 256]])
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"phi_heatmap_block{b}.png", dpi=200)
        plt.close(fig)

    print(f"Saved {suffix} phi analysis to {out_dir}")
    for b, res in enumerate(block_results):
        print(f"  block {b}: φ std={res['phi_std']:.3f}, low-fraction={res['low_fraction']:.3f}, top peak k={res['top_peaks'][0]['k']} amp={res['top_peaks'][0]['amp']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path,
                        default=Path("/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707"))
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    pred_eta, truth_eta, h, time = load_failing_state(run_dir, "tanaka_g0", 5, 40)

    if args.out_dir is None:
        out_dir = run_dir / "phi_spectra_case5_frame40"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix, eta in [("pred_eta", pred_eta), ("truth_eta", truth_eta)]:
        subdir = out_dir / suffix
        subdir.mkdir(parents=True, exist_ok=True)
        analyze_phi(run_dir, eta, h, time, suffix, subdir)


if __name__ == "__main__":
    main()
