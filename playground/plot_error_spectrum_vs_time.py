"""Plot |FFT_x[eta_pred - eta_truth]|^2 vs (wavenumber, time) for each regime.

Reads <run_dir>/eval_suite/<regime>_trajs.npz and produces a heatmap per regime
showing how the spectral content of the rollout error evolves in time.

Diagnoses:
- Uniform across all k -> error is broadband (consistent with random walk).
- Concentrated at high k -> spectral truncation / aliasing.
- Concentrated at low k -> large-scale coherent bias.
- Peaks at specific k -> resonance / mode-locking with a particular frequency.

CPU-only; just numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def power_spectrum_over_time(
    truth_eta: np.ndarray, pred_eta: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (k_values, time-mean error power E[t, k], truth power for reference)."""
    nx = truth_eta.shape[-1]
    err = pred_eta - truth_eta  # (T, B, nx)
    err_hat = np.fft.rfft(err, axis=-1)  # (T, B, nx//2+1)
    truth_hat = np.fft.rfft(truth_eta, axis=-1)
    err_power_per_ic = np.abs(err_hat) ** 2  # (T, B, K)
    truth_power_per_ic = np.abs(truth_hat) ** 2
    err_power = np.nanmean(err_power_per_ic, axis=1)  # mean over ICs, shape (T, K)
    truth_power = np.nanmean(truth_power_per_ic, axis=1)
    k_idx = np.arange(err_hat.shape[-1])
    k_phys = (2.0 * np.pi / length) * k_idx
    return k_phys, err_power, truth_power


def plot_regime(
    npz_path: Path,
    output_path: Path,
    length: float,
    regime_label: str,
    cap_k_max: float | None = None,
    n_time_curves: int = 8,
) -> dict[str, float]:
    with np.load(npz_path) as d:
        times = np.asarray(d["times"])
        truth_eta = np.asarray(d["truth_eta"])
        pred_eta = np.asarray(d["pred_eta"])

    # Drop ICs whose rollout went NaN/inf so they don't dominate the mean.
    bad_per_ic = ~np.isfinite(pred_eta).all(axis=(0, 2))
    if bad_per_ic.any():
        keep = ~bad_per_ic
        pred_eta = pred_eta[:, keep, :]
        truth_eta = truth_eta[:, keep, :]
        ic_kept = int(keep.sum())
    else:
        ic_kept = int(pred_eta.shape[1])

    k, err_power, truth_power = power_spectrum_over_time(truth_eta, pred_eta, length)
    if cap_k_max is not None:
        sel = k <= cap_k_max
        k = k[sel]
        err_power = err_power[:, sel]
        truth_power = truth_power[:, sel]

    n_t = times.shape[0]
    t_idx = np.linspace(0, n_t - 1, n_time_curves).round().astype(int)
    cmap = plt.get_cmap("viridis")
    t_min, t_max = float(times[0]), float(times[-1])

    # Decimate legend so 16-24 curves don't crowd it; keep 6-8 entries.
    legend_stride = max(1, len(t_idx) // 6)
    legend_t_idx = set(t_idx[::legend_stride].tolist()) | {int(t_idx[-1])}

    def alpha_for(ti: int) -> float:
        # fade early time (low signal) → opaque late time (the bias-accumulation curves)
        return 0.25 + 0.75 * (times[ti] - t_min) / max(t_max - t_min, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    eps = 1e-30

    # Left panel: |err(k)|^2 at multiple time snapshots
    for i, ti in enumerate(t_idx):
        color = cmap((times[ti] - t_min) / max(t_max - t_min, 1e-9))
        label = f"t={times[ti]:.1f}" if int(ti) in legend_t_idx else None
        axes[0].loglog(np.maximum(k, 1), np.maximum(err_power[ti], eps),
                       color=color, lw=1.2, alpha=alpha_for(int(ti)), label=label)
    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel(r"$|\widehat{\eta_{pred}-\eta_{truth}}|^{2}$ (mean over ICs)")
    axes[0].set_title(f"{regime_label}: rollout-error spectrum (N_IC={ic_kept})")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left", ncol=2)

    # Right panel: relative |err|^2 / |truth|^2 at the same snapshots
    rel = err_power / (truth_power + eps)
    for i, ti in enumerate(t_idx):
        color = cmap((times[ti] - t_min) / max(t_max - t_min, 1e-9))
        label = f"t={times[ti]:.1f}" if int(ti) in legend_t_idx else None
        axes[1].loglog(np.maximum(k, 1), np.maximum(rel[ti], eps),
                       color=color, lw=1.2, alpha=alpha_for(int(ti)), label=label)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="|err|=|truth|")
    axes[1].set_xlabel("wavenumber k")
    axes[1].set_ylabel(r"$|\widehat{err}|^{2}/|\widehat{truth}|^{2}$")
    axes[1].set_title(f"{regime_label}: relative error spectrum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right", ncol=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_min, vmax=t_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", pad=0.02, shrink=0.85)
    cbar.set_label("time")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    final_err = err_power[-1]
    final_truth = truth_power[-1]
    final_rel = final_err / (final_truth + eps)
    top_idx = int(np.argmax(final_err))
    summary = {
        "ic_kept": ic_kept,
        "final_t": float(times[-1]),
        "k_of_max_err_at_final_t": float(k[top_idx]),
        "max_err_power_final_t": float(final_err[top_idx]),
        "first_k_err_above_truth": float(
            k[final_rel > 1.0][0] if (final_rel > 1.0).any() else -1.0
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        default="outputs/fno_w128b6_v3_hclip5_20260514_063811",
        help="Run directory containing eval_suite/<regime>_trajs.npz",
    )
    parser.add_argument(
        "--regimes",
        default="tanaka_g0,tanaka_g1,bf_g0,bf_g1,bf_modal,linear,stokes_deep,stokes_finite,random_sea_deep,random_sea_finite",
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument(
        "--cap_k_max",
        type=float,
        default=None,
        help="Optional upper bound on wavenumber to plot (e.g. 200).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write plots; defaults to <run_dir>/eval_suite/error_spectrum.",
    )
    parser.add_argument(
        "--n_time_curves", type=int, default=16,
        help="Number of evenly-spaced time slices to overlay.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_dir = run_dir / "eval_suite"
    if not eval_dir.exists():
        raise FileNotFoundError(f"no eval_suite under {run_dir}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else eval_dir / "error_spectrum"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, object] = {}
    for regime in (r.strip() for r in args.regimes.split(",") if r.strip()):
        npz = eval_dir / f"{regime}_trajs.npz"
        if not npz.exists():
            print(f"skip {regime}: no trajs.npz")
            continue
        out_path = output_dir / f"{regime}_error_spectrum.png"
        try:
            summaries[regime] = plot_regime(
                npz, out_path, args.length, regime, args.cap_k_max,
                n_time_curves=args.n_time_curves,
            )
            print(f"  {regime} -> {out_path}")
        except Exception as e:
            summaries[regime] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {regime} ERROR: {e}")

    summary_path = output_dir / "error_spectrum_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
