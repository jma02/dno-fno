"""Per-regime error heatmaps: (wavenumber vs time) and (steepness vs time).

For each regime, render one PNG with two heatmaps side-by-side:
  - Left:  error-spectrum power |F[η_pred − η_truth]|² as a function of (k, t),
           IC-averaged.
  - Right: per-IC relative L2 η-error as a function of (t, IC), with ICs
           ordered along the y-axis by their t=0 max-steepness max|∂_x η|.

Usage:
    uv run python -m playground.plot_error_heatmaps \\
        --run_dir outputs/fno_w128b6_v3_hclip5_20260514_063811 \\
        --output_dir outputs/error_heatmaps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_traj(npz_path: Path) -> dict:
    with np.load(npz_path) as d:
        return {
            "times": np.asarray(d["times"]).astype(np.float64),
            "truth": np.asarray(d["truth_eta"]).astype(np.float64),
            "pred":  np.asarray(d["pred_eta"]).astype(np.float64),
        }


def err_spectrum(arr: dict, length: float) -> tuple[np.ndarray, np.ndarray]:
    t, p = arr["truth"], arr["pred"]
    bad = (~np.isfinite(p).all(axis=(0, 2))) | (~np.isfinite(t).all(axis=(0, 2)))
    t, p = t[:, ~bad, :], p[:, ~bad, :]
    err = p - t
    pwr = np.abs(np.fft.rfft(err, axis=-1)) ** 2     # (T, B, K)
    pwr_mean = np.nanmean(pwr, axis=1)               # (T, K)
    nk = pwr.shape[-1]
    k = (2.0 * np.pi / length) * np.arange(nk)
    return k, pwr_mean


def per_ic_rel_l2(arr: dict) -> np.ndarray:
    t, p = arr["truth"], arr["pred"]
    err = p - t
    num = np.sqrt(np.sum(err ** 2, axis=-1))
    den = np.sqrt(np.sum(t ** 2, axis=-1)) + 1e-12
    return num / den                                  # (T, B)


def ic_max_steepness(arr: dict, length: float) -> np.ndarray:
    eta0 = arr["truth"][0]                            # (B, nx)
    nx = eta0.shape[-1]
    eta0_x = np.gradient(eta0, length / nx, axis=-1)
    return np.max(np.abs(eta0_x), axis=-1)            # (B,)


def plot_regime(regime: str, npz_path: Path, length: float, out_path: Path) -> None:
    arr = load_traj(npz_path)
    times = arr["times"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax_k, ax_s = axes

    # ----- (k, t) error-spectrum heatmap -----
    k, pwr = err_spectrum(arr, length)
    log_pwr = np.log10(np.maximum(pwr, 1e-30))
    vmax = float(np.nanmax(log_pwr))
    vmin = vmax - 10.0
    im_k = ax_k.pcolormesh(
        times, k, log_pwr.T,
        shading="auto", cmap="magma", vmin=vmin, vmax=vmax,
    )
    ax_k.set_xlabel("time")
    ax_k.set_ylabel("wavenumber  k")
    ax_k.set_title(f"{regime}: error spectrum (mean over ICs)")
    cb_k = fig.colorbar(im_k, ax=ax_k)
    cb_k.set_label(r"$\log_{10}\,|\widehat{\eta_{\mathrm{pred}}-\eta_{\mathrm{truth}}}|^{2}$")

    # ----- (steepness, t) per-IC error heatmap -----
    rel = per_ic_rel_l2(arr)                          # (T, B)
    steep = ic_max_steepness(arr, length)             # (B,)
    finite_ic = np.isfinite(steep) & np.isfinite(rel).any(axis=0)
    steep_f = steep[finite_ic]
    rel_f = rel[:, finite_ic]
    order = np.argsort(steep_f)
    steep_sorted = steep_f[order]
    rel_sorted = rel_f[:, order]

    rel_plot = np.where(np.isfinite(rel_sorted), np.clip(rel_sorted, 0.0, 2.0), np.nan)
    im_s = ax_s.pcolormesh(
        times, steep_sorted, rel_plot.T,
        shading="auto", cmap="magma", vmin=0.0, vmax=1.0,
    )
    ax_s.set_xlabel("time")
    ax_s.set_ylabel(r"IC initial max-steepness $\;\max|\partial_x\eta_0|$")
    ax_s.set_title(f"{regime}: per-IC rel-L2 η-error")
    cb_s = fig.colorbar(im_s, ax=ax_s)
    cb_s.set_label("relative L2 error (clipped at 1.0)")

    n_finite = int(finite_ic.sum())
    n_total = int(arr["truth"].shape[1])
    if n_finite < n_total:
        fig.suptitle(f"{regime}    ({n_finite}/{n_total} ICs finite)", y=1.04, fontsize=10)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="Path to the trained-run directory containing eval_suite/")
    parser.add_argument(
        "--regimes",
        default="tanaka_g0,tanaka_g1,bf_g0,bf_g1,bf_modal,linear,"
                "stokes_deep,stokes_finite,random_sea_deep,random_sea_finite",
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--output_dir", default="outputs/error_heatmaps")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for regime in (r.strip() for r in args.regimes.split(",") if r.strip()):
        npz = run_dir / "eval_suite" / f"{regime}_trajs.npz"
        if not npz.exists():
            print(f"  skip {regime}: no trajs.npz")
            continue
        out = out_dir / f"{regime}_heatmaps.png"
        plot_regime(regime, npz, args.length, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
