"""Per-regime plot: rollout error and solution mean wavenumber vs time.

Two stacked panels with a shared time axis:
  - Top:    relative L2 η-error (log scale).
  - Bottom: mean wavenumber of the truth ⟨k⟩(t)
            = Σ k|η̂(k,t)|² / Σ |η̂(k,t)|².

Three ICs per regime, binned by initial mean wavenumber:
  red = low-k, blue = mid-k, green = high-k.

Usage:
    uv run python -m playground.plot_error_vs_k_time \\
        --run_dir outputs/fno_w128b6_v3_hclip5_20260514_063811 \\
        --output_dir outputs/error_vs_k
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


def rel_l2(arr: dict) -> np.ndarray:
    t, p = arr["truth"], arr["pred"]
    err = p - t
    num = np.sqrt(np.sum(err ** 2, axis=-1))
    den = np.sqrt(np.sum(t ** 2, axis=-1)) + 1e-12
    return num / den                                     # (T, B)


def mean_k_truth(arr: dict, length: float) -> np.ndarray:
    eta = arr["truth"]
    pwr = np.abs(np.fft.rfft(eta, axis=-1)) ** 2
    nk = pwr.shape[-1]
    k = (2.0 * np.pi / length) * np.arange(nk)
    pwr[..., 0] = 0.0
    num = (pwr * k).sum(axis=-1)
    den = pwr.sum(axis=-1) + 1e-30
    return num / den                                     # (T, B)


def select_three_ics(arr: dict, length: float) -> tuple[np.ndarray, np.ndarray]:
    t, p = arr["truth"], arr["pred"]
    bad = (~np.isfinite(p).all(axis=(0, 2))) | (~np.isfinite(t).all(axis=(0, 2)))
    eta0 = t[0]
    pwr0 = np.abs(np.fft.rfft(eta0, axis=-1)) ** 2
    pwr0[:, 0] = 0.0
    nk = pwr0.shape[-1]
    k = (2.0 * np.pi / length) * np.arange(nk)
    k0 = (pwr0 * k).sum(axis=-1) / (pwr0.sum(axis=-1) + 1e-30)
    valid = np.where(~bad)[0]
    if valid.size < 3:
        return np.array([], dtype=int), np.array([])
    sv = valid[np.argsort(k0[valid])]
    chosen = np.array([sv[0], sv[sv.size // 2], sv[-1]])
    return chosen, k0[chosen]


def plot_regime(regime: str, npz_path: Path, length: float, out_path: Path) -> bool:
    arr = load_traj(npz_path)
    times = arr["times"]
    ics, k0s = select_three_ics(arr, length)
    if ics.size < 3:
        return False

    rel = rel_l2(arr)
    kmean = mean_k_truth(arr, length)

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    names = ["low-k", "mid-k", "high-k"]

    fig, (ax_e, ax_k) = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for ic, color, name, k0 in zip(ics, colors, names, k0s):
        ax_e.plot(times, np.maximum(rel[:, ic], 1e-6), color=color, lw=2.4,
                  label=f"{name}  (k₀={k0:.1f})")
        ax_k.plot(times, kmean[:, ic], color=color, lw=2.0)

    ax_e.set_ylabel("relative L2 η-error")
    ax_e.set_yscale("log")
    ax_e.set_title(regime, loc="left", fontsize=14, fontweight="bold")
    ax_e.grid(True, which="both", alpha=0.25)
    ax_e.legend(loc="lower right", fontsize=11, framealpha=0.95)

    ax_k.set_xlabel("time")
    ax_k.set_ylabel("mean wavenumber  ⟨k⟩")
    ax_k.grid(True, alpha=0.25)
    ax_k.set_ylim(bottom=0)

    for ax in (ax_e, ax_k):
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--regimes",
        default="tanaka_g0,tanaka_g1,bf_g0,bf_g1,bf_modal,linear,"
                "stokes_deep,stokes_finite,random_sea_deep,random_sea_finite",
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--output_dir", default="outputs/error_vs_k")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for regime in (r.strip() for r in args.regimes.split(",") if r.strip()):
        npz = run_dir / "eval_suite" / f"{regime}_trajs.npz"
        if not npz.exists():
            print(f"  skip {regime}: no trajs.npz")
            continue
        out = out_dir / f"{regime}_err_vs_k.png"
        ok = plot_regime(regime, npz, args.length, out)
        print(f"  {'wrote' if ok else 'skipped'} {out}")


if __name__ == "__main__":
    main()
