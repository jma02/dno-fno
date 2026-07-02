"""Fit the pre-spike relative-L2 error growth to three candidate models on a
single case from a saved eval_suite trajectories npz.

Models fitted on Y(t) = log E_rel(t):
  M1:  Y = a + b * t           ->  E_rel ~ exp(b*t)         (exponential)
  M2:  Y = a + p * log(t)      ->  E_rel ~ t^p              (power law)
  M3:  Y = a + b * sqrt(t)     ->  E_rel ~ exp(b*sqrt(t))   (stretched exp)

For each field (eta, xi, gxi) and a configurable pre-spike window, reports OLS
fit parameters and R^2, and overlays the three fits on the data.

Usage:
    python -m solver.evals.fit_error_growth \\
        --trajs_npz outputs/.../eval_suite_f64h/tanaka_g0_trajs.npz \\
        --case_id 5 \\
        --out_png outputs/.../plots/fit_error_growth_cid5.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FieldHat = tuple[np.ndarray, np.ndarray]
FitResult = dict[str, float]
LOG_FLOOR = 1e-30


def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-frame relative L2. pred/truth shape (T, nx)."""
    num = np.sqrt(np.sum((pred - truth) ** 2, axis=-1))
    den = np.sqrt(np.sum(truth ** 2, axis=-1))
    return num / np.maximum(den, LOG_FLOOR)


def _ols_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    """1D linear fit y = a + b*x. Returns intercept, slope, R^2."""
    n = x.size
    if n < 3:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": float(n)}
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    sxy = float(np.sum((x - xm) * (y - ym)))
    syy = float(np.sum((y - ym) ** 2))
    b = sxy / max(sxx, 1e-30)
    a = ym - b * xm
    r2 = (sxy * sxy) / max(sxx * syy, 1e-30)
    return {"a": float(a), "b": float(b), "r2": float(r2), "n": float(n)}


def _fit_all(t: np.ndarray, y_log: np.ndarray) -> dict[str, FitResult]:
    """Fit three candidate models to Y = log E_rel."""
    # Drop t=0 (log(0) and origin distortion); keep where Y is finite.
    keep = (t > 0) & np.isfinite(y_log)
    tt = t[keep]
    yy = y_log[keep]
    return {
        "linear_t":   _ols_fit(tt, yy),                # Y = a + b*t
        "log_t":      _ols_fit(np.log(tt), yy),         # Y = a + p*log(t)
        "sqrt_t":     _ols_fit(np.sqrt(tt), yy),        # Y = a + b*sqrt(t)
    }


def _eval_model(name: str, fit: FitResult, t: np.ndarray) -> np.ndarray:
    """Evaluate the fitted model at t (returns log E_rel)."""
    a, b = fit["a"], fit["b"]
    eps = 1e-12
    if name == "linear_t":
        return a + b * t
    if name == "log_t":
        return a + b * np.log(np.maximum(t, eps))
    if name == "sqrt_t":
        return a + b * np.sqrt(np.maximum(t, 0.0))
    raise ValueError(name)


def _pick_window(times: np.ndarray, rel: np.ndarray, nan_t: float | None,
                 t_lo: float, t_hi: float | None) -> tuple[float, float]:
    """Resolve (t_lo, t_hi). t_hi defaults to 0.9 * NaN time if available."""
    if t_hi is None:
        if nan_t is not None:
            t_hi = 0.9 * nan_t
        else:
            t_hi = float(times[-1])
    return float(t_lo), float(t_hi)


def _first_nan_t(arr: np.ndarray, times: np.ndarray) -> float | None:
    bad = (~np.isfinite(arr)).any(axis=-1)
    if not bad.any():
        return None
    return float(times[int(np.argmax(bad))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajs_npz", required=True)
    parser.add_argument("--case_id", type=int, required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--t_lo", type=float, default=2.0,
                        help="Start of fit window (exclude very early frames; avoids log(0)).")
    parser.add_argument("--t_hi", type=float, default=None,
                        help="End of fit window. Default = 0.9 * NaN_time if present, else t_end.")
    args = parser.parse_args()

    d = np.load(args.trajs_npz)
    case_ids = np.asarray(d["case_ids"])
    row_idx = np.nonzero(case_ids == args.case_id)[0]
    if row_idx.size == 0:
        raise SystemExit(f"case_id {args.case_id} not in {args.trajs_npz}")
    row = int(row_idx[0])
    depth = float(d["depths"][row])
    times = np.asarray(d["times"])

    fields = {
        "eta": (np.asarray(d["pred_eta"][:, row, :]), np.asarray(d["truth_eta"][:, row, :])),
        "xi":  (np.asarray(d["pred_xi"][:, row, :]),  np.asarray(d["truth_xi"][:, row, :])),
        "gxi": (np.asarray(d["pred_gxi"][:, row, :]), np.asarray(d["truth_gxi"][:, row, :])),
    }

    nan_t = _first_nan_t(fields["eta"][0], times)
    t_lo, t_hi = _pick_window(times, fields["eta"][0], nan_t, args.t_lo, args.t_hi)
    print(f"cid={args.case_id} depth={depth:.4f} nan_t={nan_t} -> fit window [{t_lo:.2f}, {t_hi:.2f}]",
          flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    fig.suptitle(
        f"Pre-spike rel-L2 error growth fits — cid={args.case_id}, depth={depth:.3f}"
        + (f", NaN @ t={nan_t:.1f}" if nan_t is not None else ""),
        fontsize=12,
    )

    summary: dict[str, dict[str, FitResult]] = {}
    for ax, name in zip(axes, ("eta", "xi", "gxi")):
        pred, truth = fields[name]
        rel = _rel_l2(pred, truth)
        y = np.log(np.maximum(rel, 1e-30))

        in_win = (times >= t_lo) & (times <= t_hi) & np.isfinite(y)
        if in_win.sum() < 5:
            print(f"  {name}: too few points in window, skipping fit", flush=True)
            continue
        fits = _fit_all(times[in_win], y[in_win])
        summary[name] = fits

        # Plot data on log y. Solid black = data; coloured lines = three fits.
        ax.semilogy(times, np.maximum(rel, 1e-30), color="0.15", lw=1.4, label="data")
        # Shade fit window
        ax.axvspan(t_lo, t_hi, color="0.85", alpha=0.5, zorder=0)
        # Evaluate each fit and overlay
        t_fit_plot = times[in_win]
        for label, key, color in (
            ("Y=a+bt  (exp)",     "linear_t", "tab:red"),
            ("Y=a+p·log t (pow)", "log_t",    "tab:blue"),
            ("Y=a+b·√t (str-exp)", "sqrt_t",  "tab:green"),
        ):
            y_pred = _eval_model(key, fits[key], t_fit_plot)
            r2 = fits[key]["r2"]
            ax.semilogy(t_fit_plot, np.exp(y_pred),
                        color=color, lw=1.5, ls="--",
                        label=f"{label}  R²={r2:.4f}")
            print(f"  {name:>4}  {key:>9}:  a={fits[key]['a']:+.4f}  "
                  f"b/p={fits[key]['b']:+.4f}  R²={r2:.4f}  n={int(fits[key]['n'])}", flush=True)

        if nan_t is not None:
            ax.axvline(nan_t, color="red", lw=1.0, ls=":", alpha=0.7)
        ax.set_xlabel("t")
        ax.set_ylabel(f"E_rel({name})")
        ax.set_title(f"{name} relative L2 — pre-spike fit window shaded")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(max(1e-6, rel[rel > 0].min() * 0.5), max(rel) * 5)

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nsaved {out_path}", flush=True)

    # Punch-line: which model wins per field
    print("\nBEST-FIT BY R^2:", flush=True)
    for name, fits in summary.items():
        best_key = max(fits.keys(), key=lambda k: fits[k]["r2"])
        best = fits[best_key]
        interp = {
            "linear_t": f"E_rel ~ exp({best['b']:+.4f} * t)  (exponential)",
            "log_t":    f"E_rel ~ t^{best['b']:.3f}          (power law)",
            "sqrt_t":   f"E_rel ~ exp({best['b']:+.4f} * √t) (stretched exp)",
        }[best_key]
        print(f"  {name:>4}: {best_key}  R²={best['r2']:.4f}   ->  {interp}", flush=True)


if __name__ == "__main__":
    main()
