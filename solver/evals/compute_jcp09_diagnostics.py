"""JCP09-canonical per-IC tanaka diagnostics on saved eval_suite trajs.npz.

Computes (per the cs_dno_jcp09_improvement_plan.md §4 pass criteria):
  - err_high(t) / truth_high(t)   at t = 2.4   (k > k_high_cut, default 30)
                                   pass < 0.1×
  - alpha_final                     dot(truth_final, pred_final) / ||truth|| ||pred||
                                   pass > 0.5  (anti-correlated < 0)

Usage:
    uv run python -m solver.evals.compute_jcp09_diagnostics \
        --trajs outputs/<run>/<eval_dir>/tanaka_g0_trajs.npz \
        [--t_diag 2.4] [--k_high_cut 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_diagnostics(
    trajs_path: Path, t_diag: float = 2.4, k_high_cut: int = 30,
) -> dict:
    data = np.load(trajs_path)
    times = data["times"].astype(np.float64)           # (n_t,)
    depths = data["depths"].astype(np.float64)         # (NB,)
    case_ids = data["case_ids"].astype(np.int64)       # (NB,)
    truth_eta = data["truth_eta"]                       # (n_t, NB, nx)
    pred_eta = data["pred_eta"]                         # (n_t, NB, nx)

    n_t, NB, nx = truth_eta.shape
    t_idx = int(np.argmin(np.abs(times - t_diag)))
    t_actual = float(times[t_idx])

    truth_t = truth_eta[t_idx]                          # (NB, nx)
    pred_t = pred_eta[t_idx]                            # (NB, nx)
    truth_final = truth_eta[-1]
    pred_final = pred_eta[-1]

    truth_hat = np.fft.rfft(truth_t, axis=-1)           # (NB, nx//2+1)
    pred_hat = np.fft.rfft(pred_t, axis=-1)
    err_hat = pred_hat - truth_hat
    n_freq = truth_hat.shape[-1]
    high_mask = np.arange(n_freq) >= k_high_cut

    truth_high_energy = np.sqrt(np.sum(np.abs(truth_hat[:, high_mask]) ** 2, axis=-1))
    err_high_energy = np.sqrt(np.sum(np.abs(err_hat[:, high_mask]) ** 2, axis=-1))
    truth_safe = np.where(truth_high_energy > 1e-30, truth_high_energy, 1e-30)
    ratio = err_high_energy / truth_safe                 # (NB,)

    pred_norm = np.linalg.norm(pred_final, axis=-1)
    truth_norm = np.linalg.norm(truth_final, axis=-1)
    dot = np.sum(pred_final * truth_final, axis=-1)
    denom = np.where(pred_norm * truth_norm > 1e-30, pred_norm * truth_norm, 1e-30)
    alpha_final = dot / denom                            # (NB,)
    finite_mask = np.isfinite(alpha_final)
    nan_per_ic = ~np.isfinite(pred_final).all(axis=-1)

    per_ic = []
    for j in range(NB):
        per_ic.append({
            "case_id": int(case_ids[j]),
            "h": float(depths[j]),
            "err_high_over_truth_high_at_t_diag": (
                float(ratio[j]) if np.isfinite(ratio[j]) else None
            ),
            "alpha_final": float(alpha_final[j]) if finite_mask[j] else None,
            "is_nan_final": bool(nan_per_ic[j]),
        })

    survivors = [p for p in per_ic if p["err_high_over_truth_high_at_t_diag"] is not None]
    median_ratio = (
        float(np.median([p["err_high_over_truth_high_at_t_diag"] for p in survivors]))
        if survivors else None
    )
    max_ratio = (
        float(np.max([p["err_high_over_truth_high_at_t_diag"] for p in survivors]))
        if survivors else None
    )

    return {
        "trajs": str(trajs_path),
        "t_diag_requested": float(t_diag),
        "t_diag_actual": t_actual,
        "k_high_cut": int(k_high_cut),
        "nx": int(nx),
        "NB": int(NB),
        "n_nan_final": int(np.sum(nan_per_ic)),
        "median_err_high_over_truth_high": median_ratio,
        "max_err_high_over_truth_high": max_ratio,
        "pass_threshold": 0.1,
        "per_ic": per_ic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajs", required=True, nargs="+",
                        help="One or more <regime>_trajs.npz files.")
    parser.add_argument("--t_diag", type=float, default=2.4)
    parser.add_argument("--k_high_cut", type=int, default=30)
    parser.add_argument("--out", default=None,
                        help="Output JSON path; default prints to stdout per file.")
    args = parser.parse_args()

    results = {}
    for tp in args.trajs:
        path = Path(tp).resolve()
        diag = compute_diagnostics(path, t_diag=args.t_diag, k_high_cut=args.k_high_cut)
        results[path.stem] = diag

        print(f"\n=== {path.parent.name}/{path.name} ===")
        print(f"  NB={diag['NB']} t_diag={diag['t_diag_actual']:.2f} k_cut={diag['k_high_cut']}")
        print(f"  NaN finals: {diag['n_nan_final']}/{diag['NB']}")
        if diag["median_err_high_over_truth_high"] is not None:
            ratio = diag["median_err_high_over_truth_high"]
            verdict = "PASS" if ratio < 0.1 else "FAIL"
            print(f"  err_high/truth_high  median={ratio:.4g}  max={diag['max_err_high_over_truth_high']:.4g}  [{verdict} <0.1]")
        for p in diag["per_ic"]:
            cid = p["case_id"]; h = p["h"]
            ratio_s = (
                f"{p['err_high_over_truth_high_at_t_diag']:.3g}"
                if p["err_high_over_truth_high_at_t_diag"] is not None else "  NaN"
            )
            alpha_s = (
                f"{p['alpha_final']:+.3f}"
                if p["alpha_final"] is not None else " NaN"
            )
            tag = " NaN" if p["is_nan_final"] else "    "
            print(f"    cid={cid:>9}  h={h:.4f}  err_high/truth_high={ratio_s}  alpha_final={alpha_s}{tag}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
