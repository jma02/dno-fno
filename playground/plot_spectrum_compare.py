"""Coherent side-by-side spectrum plots: FNO vs DNO+PF on the same axes.

Produces one figure per regime showing |F[η_pred-η_truth]|² for both models
at three time slices (early, middle, late), with the truth spectrum at the
final time drawn as a faded reference. Less clutter than the per-model
overlay-many-times-slices plot.

Usage:
    uv run python -m playground.plot_spectrum_compare \\
        --run "FNO baseline=outputs/fno_w128b6_v3_hclip5_20260514_063811" \\
        --run "DNO+PF=outputs/dno_w128b6_l64_v3_pf1_20260603_064450" \\
        --output_dir outputs/spectrum_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_traj(npz_path: Path):
    with np.load(npz_path) as d:
        return {
            "times": np.asarray(d["times"]),
            "truth": np.asarray(d["truth_eta"]),
            "pred":  np.asarray(d["pred_eta"]),
        }


def err_power_meanic(arrs: dict, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (k, mean_over_ICs(|F[pred-truth]|²)) where bad ICs are dropped."""
    truth, pred = arrs["truth"], arrs["pred"]
    bad = (~np.isfinite(pred).all(axis=(0,2))) | (~np.isfinite(truth).all(axis=(0,2)))
    truth, pred = truth[:, ~bad, :], pred[:, ~bad, :]
    err = pred - truth
    pwr = np.abs(np.fft.rfft(err, axis=-1)) ** 2     # (T, B, K)
    pwr_mean = np.nanmean(pwr, axis=1)               # (T, K)
    nk = pwr.shape[-1]
    k = (2.0 * np.pi / length) * np.arange(nk)
    return k, pwr_mean


def truth_power_meanic(arrs: dict, length: float) -> tuple[np.ndarray, np.ndarray]:
    truth = arrs["truth"]
    bad = ~np.isfinite(truth).all(axis=(0,2))
    truth = truth[:, ~bad, :]
    pwr = np.abs(np.fft.rfft(truth, axis=-1)) ** 2
    return (
        (2.0 * np.pi / length) * np.arange(pwr.shape[-1]),
        np.nanmean(pwr, axis=1),
    )


def plot_regime(
    regime: str,
    runs: dict[str, Path],
    length: float,
    output_path: Path,
    cap_k_max: float | None,
):
    arrs = {label: load_traj(rd / "eval_suite" / f"{regime}_trajs.npz") for label, rd in runs.items()}
    times = next(iter(arrs.values()))["times"]
    n_t = times.shape[0]
    tmax = float(times[-1])

    # Three time slices: 10%, 50%, 100% of tmax
    slice_idx = [int(round((n_t - 1) * f)) for f in [0.10, 0.50, 1.00]]
    slice_labels = [f"t={times[i]:.1f}" for i in slice_idx]

    # Reference: truth spectrum at final time, IC-averaged
    truth_arr = next(iter(arrs.values()))
    k_t, truth_pwr = truth_power_meanic(truth_arr, length)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    eps = 1e-30

    # Colors per model (red = first listed, blue = second)
    model_colors = {list(runs.keys())[0]: "#d62728"}
    if len(runs) >= 2:
        model_colors[list(runs.keys())[1]] = "#1f77b4"

    # Line opacity per time slice (early light, late opaque)
    alphas = [0.35, 0.65, 1.00]
    linestyles = [":", "--", "-"]

    # Plot per (model, slice)
    for label, arr in arrs.items():
        k, err_pwr = err_power_meanic(arr, length)
        color = model_colors[label]
        for si, ti in enumerate(slice_idx):
            sel = (k <= cap_k_max) if cap_k_max is not None else slice(None)
            kk = k[sel] if cap_k_max is not None else k
            yy = err_pwr[ti][sel] if cap_k_max is not None else err_pwr[ti]
            ax.loglog(
                np.maximum(kk, 1), np.maximum(yy, eps),
                color=color, lw=1.6, alpha=alphas[si], linestyle=linestyles[si],
                label=f"{label}  {slice_labels[si]}",
            )

    # Truth-at-final-t reference (gray, dashed, half-opacity)
    sel = (k_t <= cap_k_max) if cap_k_max is not None else slice(None)
    kk_t = k_t[sel] if cap_k_max is not None else k_t
    yy_t = truth_pwr[-1][sel] if cap_k_max is not None else truth_pwr[-1]
    ax.loglog(
        np.maximum(kk_t, 1), np.maximum(yy_t, eps),
        color="#666", lw=1.4, ls="-", alpha=0.45, label=f"|truth|² at t={tmax:.1f}",
    )

    ax.set_xlabel("wavenumber k")
    ax.set_ylabel(r"$|\widehat{\eta_{pred}-\eta_{truth}}|^{2}$  (mean over ICs)")
    n_kept = []
    for label, arr in arrs.items():
        bad = (~np.isfinite(arr["pred"]).all(axis=(0,2))) | (~np.isfinite(arr["truth"]).all(axis=(0,2)))
        n_kept.append(f"{label}: {int((~bad).sum())}/{arr['pred'].shape[1]}")
    ax.set_title(f"{regime}: rollout error spectrum   ({' · '.join(n_kept)})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.9)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--run must be 'label=path', got {spec!r}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip()).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help='Repeatable. Format: "label=path_to_run_dir".')
    parser.add_argument(
        "--regimes",
        default="tanaka_g0,tanaka_g1,bf_g0,bf_g1,bf_modal,linear,stokes_deep,stokes_finite,random_sea_deep,random_sea_finite",
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--cap_k_max", type=float, default=None,
                        help="Upper bound on k to plot (default: no cap).")
    parser.add_argument("--output_dir", default="outputs/spectrum_compare")
    args = parser.parse_args()

    runs = dict(parse_run(s) for s in args.run)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime in (r.strip() for r in args.regimes.split(",") if r.strip()):
        missing = [lbl for lbl, rd in runs.items() if not (rd / "eval_suite" / f"{regime}_trajs.npz").exists()]
        if missing:
            print(f"  skip {regime}: missing trajs for {missing}")
            continue
        out_path = output_dir / f"{regime}_spectrum_compare.png"
        plot_regime(regime, runs, args.length, out_path, args.cap_k_max)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
