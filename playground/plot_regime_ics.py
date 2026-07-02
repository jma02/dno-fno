"""Per-regime IC plot: visualize the three ICs picked by `plot_error_vs_k_time.py`.

For each regime, loads `<run>/eval_suite/<regime>_trajs.npz`, applies the same
low/mid/high k0 selection (energy-weighted spectral centroid of η at t=0),
and writes one PNG showing:
  - η(x, t=0) overlay for the three ICs
  - ξ(x, t=0) overlay for the three ICs
  - log-power spectrum |η̂(k, 0)|² overlay with each centroid k₀ marked

The selected case_id, depth, and amplitude are annotated in the legend so a
reader can map the curve back to the data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#d62728", "#1f77b4", "#2ca02c")
NAMES = ("low-k", "mid-k", "high-k")


def select_three_ics(eta0: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Same selection as playground/plot_error_vs_k_time.py.

    eta0 has shape (B, N). Returns chosen indices and their k0.
    """
    pwr0 = np.abs(np.fft.rfft(eta0, axis=-1)) ** 2
    pwr0[:, 0] = 0.0
    nk = pwr0.shape[-1]
    k = (2.0 * np.pi / length) * np.arange(nk)
    k0 = (pwr0 * k).sum(axis=-1) / (pwr0.sum(axis=-1) + 1e-30)
    valid = np.where(np.isfinite(k0))[0]
    if valid.size < 3:
        return np.array([], dtype=int), np.array([])
    sv = valid[np.argsort(k0[valid])]
    chosen = np.array([sv[0], sv[sv.size // 2], sv[-1]])
    return chosen, k0[chosen]


def plot_regime_ics(regime: str, npz_path: Path, length: float, out_path: Path) -> bool:
    with np.load(npz_path) as d:
        truth_eta = np.asarray(d["truth_eta"], dtype=np.float64)
        truth_xi = np.asarray(d["truth_xi"], dtype=np.float64)
        depths = np.asarray(d["depths"], dtype=np.float64)
        case_ids = np.asarray(d["case_ids"])

    eta0 = truth_eta[0]  # (B, N)
    xi0 = truth_xi[0]
    nx = eta0.shape[-1]
    x = np.linspace(0.0, length, nx, endpoint=False)

    chosen, k0s = select_three_ics(eta0, length)
    if chosen.size < 3:
        return False

    pwr0_all = np.abs(np.fft.rfft(eta0, axis=-1)) ** 2
    nk = pwr0_all.shape[-1]
    k_axis = (2.0 * np.pi / length) * np.arange(nk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for ax, ic, color, name, k0 in zip([axes[0]] * 3, chosen, COLORS, NAMES, k0s):
        eta_max = float(np.max(np.abs(eta0[ic])))
        label = (f"{name}: case {int(case_ids[ic])}, "
                 f"h={float(depths[ic]):.3f}, "
                 f"|η|_max={eta_max:.3g}, k₀={k0:.2f}")
        ax.plot(x, eta0[ic], color=color, lw=1.6, label=label)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("η(x, t=0)")
    axes[0].set_title(f"{regime} — initial η", loc="left", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    for ic, color, _, _ in zip(chosen, COLORS, NAMES, k0s):
        axes[1].plot(x, xi0[ic], color=color, lw=1.6)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("ξ(x, t=0)")
    axes[1].set_title("initial ξ", loc="left", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.25)

    pwr_floor = max(1e-30, float(np.max(pwr0_all[chosen])) * 1e-10)
    for ic, color, _, k0 in zip(chosen, COLORS, NAMES, k0s):
        p = np.maximum(pwr0_all[ic], pwr_floor)
        axes[2].semilogy(k_axis, p, color=color, lw=1.4, alpha=0.85)
        axes[2].axvline(float(k0), color=color, ls=":", lw=1.0, alpha=0.7)
    k_show = min(80, nk - 1)
    axes[2].set_xlim(0, k_axis[k_show])
    axes[2].set_xlabel("wavenumber k")
    axes[2].set_ylabel("|η̂(k, 0)|²  (log)")
    axes[2].set_title("initial spectrum (centroid: dotted line)", loc="left",
                      fontsize=11, fontweight="bold")
    axes[2].grid(True, which="both", alpha=0.25)

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
        out = out_dir / f"{regime}_ics.png"
        ok = plot_regime_ics(regime, npz, args.length, out)
        print(f"  {'wrote' if ok else 'skipped'} {out}")


if __name__ == "__main__":
    main()
