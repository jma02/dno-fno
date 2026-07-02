"""Plot per-(k, t) spectral error |fft(pred - truth)|(k, t) for one or more cases
from a saved eval_suite trajectories npz.

Outputs a single figure organized as:
  rows 1–3: heatmap on (t, k) of log10 |error_hat_k| for eta, xi, gxi.
  rows 4–6: per-band ABSOLUTE log10 magnitudes — both |P_B truth|_2 (dashed) and
            |P_B err|_2 (solid) — for eta, xi, gxi. This separates "real high-k
            growth" (err >> truth) from "relative-metric inflation" (truth tiny,
            err modest in absolute terms).

Usage:
    python -m solver.evals.plot_spectral_error \\
        --trajs_npz outputs/.../eval_suite_f64h/tanaka_g0_trajs.npz \\
        --case_id 5 \\
        --out_png outputs/.../plots/spectral_error_cid5.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KBAND_EDGES: tuple[int, ...] = (0, 32, 64, 128, 256, 513)
LOG_FLOOR = 1e-14  # avoid log(0); below this is "machine zero"


def _band_l2(hat: np.ndarray) -> np.ndarray:
    """hat shape (T, kmax+1). Return per-band L2 magnitude (T, n_bands)."""
    nk = hat.shape[-1]
    edges = [min(e, nk) for e in KBAND_EDGES]
    bands = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bands.append(np.sqrt(np.sum(np.abs(hat[:, lo:hi]) ** 2, axis=-1)))
    return np.stack(bands, axis=-1)


def _heatmap(ax, t: np.ndarray, mag: np.ndarray, title: str, vmin: float, vmax: float, nan_t: float | None) -> None:
    """mag shape (T, K). Draw with t on x-axis, k on y-axis, log10 colour scale."""
    finite = np.isfinite(mag)
    safe = np.where(finite, np.maximum(mag, LOG_FLOOR), LOG_FLOOR)
    img = np.log10(safe)
    img = np.where(finite, img, np.nan)
    im = ax.imshow(
        img.T,
        origin="lower",
        aspect="auto",
        extent=(t[0], t[-1], 0, mag.shape[1] - 1),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_ylabel("k")
    ax.set_title(title)
    if nan_t is not None:
        ax.axvline(nan_t, color="red", lw=1.0, ls="--", alpha=0.8)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajs_npz", required=True)
    parser.add_argument("--case_id", type=int, required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--vmin", type=float, default=-12.0, help="log10 colour-scale floor")
    parser.add_argument("--vmax", type=float, default=2.0, help="log10 colour-scale ceiling")
    args = parser.parse_args()

    d = np.load(args.trajs_npz)
    case_ids = np.asarray(d["case_ids"])
    row_idx = np.nonzero(case_ids == args.case_id)[0]
    if row_idx.size == 0:
        raise SystemExit(f"case_id {args.case_id} not in {args.trajs_npz}")
    row = int(row_idx[0])
    depth = float(d["depths"][row])
    times = np.asarray(d["times"])
    nx = d["pred_eta"].shape[-1]

    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "eta": (np.asarray(d["pred_eta"][:, row, :]), np.asarray(d["truth_eta"][:, row, :])),
        "xi": (np.asarray(d["pred_xi"][:, row, :]), np.asarray(d["truth_xi"][:, row, :])),
        "gxi": (np.asarray(d["pred_gxi"][:, row, :]), np.asarray(d["truth_gxi"][:, row, :])),
    }

    # First NaN frame in any of the three fields → cascade onset
    nan_idx_candidates: list[int] = []
    for name, (p, _) in fields.items():
        bad = (~np.isfinite(p)).any(axis=-1)
        if bad.any():
            nan_idx_candidates.append(int(np.argmax(bad)))
    nan_t = float(times[min(nan_idx_candidates)]) if nan_idx_candidates else None

    fig, axes = plt.subplots(6, 1, figsize=(12, 17), constrained_layout=True)
    fig.suptitle(
        f"Spectral error  —  cid={args.case_id}, depth={depth:.3f}, nx={nx}"
        + (f", NaN at t={nan_t:.1f}" if nan_t is not None else ""),
        fontsize=12,
    )

    field_hats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    last_im = None
    for ax, name in zip(axes[:3], ("eta", "xi", "gxi")):
        pred, truth = fields[name]
        pred_hat = np.fft.rfft(pred, axis=-1)
        truth_hat = np.fft.rfft(truth, axis=-1)
        err_hat = pred_hat - truth_hat
        field_hats[name] = (pred_hat, truth_hat, err_hat)
        last_im = _heatmap(ax, times, np.abs(err_hat), f"|{name}_pred_hat − {name}_truth_hat|", args.vmin, args.vmax, nan_t)
    fig.colorbar(last_im, ax=axes[:3], label="log10 |error_hat_k|", shrink=0.85)

    # Per-band ABSOLUTE magnitudes for each field. Truth dashed, error solid.
    # If err curve sits well above truth curve, that band has REAL growth.
    # If both curves sit at machine zero ~1e-7, the band's relative-L2 looked
    # bad only because truth was tiny — metric inflation, not physics.
    cmap = plt.get_cmap("turbo")
    band_colors = [cmap(0.05 + 0.9 * i / max(1, len(KBAND_EDGES) - 2))
                   for i in range(len(KBAND_EDGES) - 1)]
    for ax, name in zip(axes[3:6], ("eta", "xi", "gxi")):
        _, truth_hat, err_hat = field_hats[name]
        truth_band = _band_l2(truth_hat)
        err_band = _band_l2(err_hat)
        for i in range(truth_band.shape[1]):
            label = f"k[{KBAND_EDGES[i]}–{KBAND_EDGES[i + 1]})"
            ax.semilogy(times, np.maximum(truth_band[:, i], LOG_FLOOR),
                        color=band_colors[i], lw=1.2, ls="--", alpha=0.7)
            ax.semilogy(times, np.maximum(err_band[:, i], LOG_FLOOR),
                        color=band_colors[i], lw=1.6, ls="-", label=label)
        if nan_t is not None:
            ax.axvline(nan_t, color="red", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel("t")
        ax.set_ylabel(f"|P_B {name}|_2")
        ax.set_title(f"{name} per-band absolute magnitude — solid: |err|, dashed: |truth|")
        ax.set_ylim(1e-8, 1e3)
        ax.legend(loc="lower right", ncol=5, fontsize=8)
        ax.grid(alpha=0.3, which="both")

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}", flush=True)

    # Quantitative readout: report final-frame |err|/|truth| ratio per band per field
    # so the artifact-vs-real question gets a number, not just a plot.
    final_finite_idx = int(np.argmax(np.cumsum(np.isfinite(field_hats["eta"][2]).all(axis=-1)))) if nan_t is None else None
    if final_finite_idx is None:
        # last finite frame across all three fields
        good = np.ones(times.shape, dtype=bool)
        for _, _, err_hat in field_hats.values():
            good &= np.isfinite(err_hat).all(axis=-1)
        final_finite_idx = int(np.nonzero(good)[0][-1]) if good.any() else 0
    print(f"final-finite frame t={float(times[final_finite_idx]):.2f}  per-band |err|/|truth|:", flush=True)
    for name in ("eta", "xi", "gxi"):
        _, truth_hat, err_hat = field_hats[name]
        tb = _band_l2(truth_hat)[final_finite_idx]
        eb = _band_l2(err_hat)[final_finite_idx]
        ratio_str = "  ".join(
            f"k[{KBAND_EDGES[i]}-{KBAND_EDGES[i + 1]}): {eb[i]:.2e}/{tb[i]:.2e}={eb[i]/(tb[i] + 1e-30):.2e}"
            for i in range(tb.shape[0])
        )
        print(f"  {name:>4}: {ratio_str}", flush=True)


if __name__ == "__main__":
    main()
