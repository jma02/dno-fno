"""Compact per-regime worst-IC table and overlap analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/home/johnma/dno-fno")
FORENSICS = REPO / "playground/diagnostics/rollout_forensics.json"


def main():
    with open(FORENSICS) as f:
        data = json.load(f)
    print("\n=== Per-regime worst-IC table (rel-L2 eta @ tmax; NaN = blow-up) ===\n")
    print(f"{'regime':22s} {'fno_med':>8s} {'dno_med':>8s} {'fno_worst':>10s} {'dno_worst':>10s} {'overlap':>8s}/5")
    overlap_counts = []
    for regime, R in data["regimes"].items():
        per_ic = R["per_ic"]
        fno_v = np.array([r["fno_rel_l2_tmax"] for r in per_ic])
        dno_v = np.array([r["dno_rel_l2_tmax"] for r in per_ic])
        fno_med = float(np.nanmedian(fno_v))
        dno_med = float(np.nanmedian(dno_v))
        fno_worst = float(np.nanmax(fno_v)) if np.isfinite(fno_v).any() else float("nan")
        dno_worst = float(np.nanmax(dno_v)) if np.isfinite(dno_v).any() else float("nan")
        # if any NaN, treat that as "worst"
        if (~np.isfinite(fno_v)).any():
            fno_worst_str = f"NaN({(~np.isfinite(fno_v)).sum()})"
        else:
            fno_worst_str = f"{fno_worst:.3f}"
        if (~np.isfinite(dno_v)).any():
            dno_worst_str = f"NaN({(~np.isfinite(dno_v)).sum()})"
        else:
            dno_worst_str = f"{dno_worst:.3f}"
        worst5_fno = set(R["worst5_fno_case_ids"])
        worst5_dno = set(R["worst5_dno_case_ids"])
        ov = len(worst5_fno & worst5_dno)
        overlap_counts.append(ov)
        print(f"{regime:22s} {fno_med:8.4f} {dno_med:8.4f} {fno_worst_str:>10s} {dno_worst_str:>10s}   {ov}/5")
    print(f"\nmean worst-5 overlap across regimes: {np.mean(overlap_counts):.2f}/5")

    print("\n=== Per-regime worst 5 ICs (case_id, fno_rel_l2, dno_rel_l2) ===")
    for regime, R in data["regimes"].items():
        print(f"\n{regime}: (worst by FNO)")
        per_ic = sorted(R["per_ic"], key=lambda r: -1.0 if not np.isfinite(r["fno_rel_l2_tmax"]) else -r["fno_rel_l2_tmax"])
        for r in per_ic[:5]:
            fno = f"NaN" if not np.isfinite(r["fno_rel_l2_tmax"]) else f"{r['fno_rel_l2_tmax']:.3f}"
            dno = f"NaN" if not np.isfinite(r["dno_rel_l2_tmax"]) else f"{r['dno_rel_l2_tmax']:.3f}"
            print(f"  case {r['case_id']:>9d}  fno={fno:>6s}  dno={dno:>6s}  stp={r['max_steepness']:.3f} kh={r['kh']:.3f}")


if __name__ == "__main__":
    main()
