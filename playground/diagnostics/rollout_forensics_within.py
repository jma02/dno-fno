"""Within-regime correlations: descriptor vs failure for each regime separately."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path("/home/johnma/dno-fno")
FORENSICS = REPO / "playground/diagnostics/rollout_forensics.json"

DESCRIPTORS = ["max_abs_eta", "max_steepness", "rms_eta", "k_peak", "bandwidth", "kh", "depth"]


def main():
    with open(FORENSICS) as f:
        data = json.load(f)
    print("\n=== Within-regime Spearman rho(descriptor, fno_tmax) | rho(., dno_tmax) ===\n")
    header = f"{'regime':22s} " + "  ".join(f"{d[:6]:>14s}" for d in DESCRIPTORS)
    print(header)
    within = {}
    for regime, R in data["regimes"].items():
        per_ic = R["per_ic"]
        if len(per_ic) < 4:
            continue
        fno = np.array([r["fno_rel_l2_tmax"] for r in per_ic], dtype=np.float64)
        dno = np.array([r["dno_rel_l2_tmax"] for r in per_ic], dtype=np.float64)
        line = f"{regime:22s} "
        within[regime] = {}
        for d in DESCRIPTORS:
            v = np.array([r[d] for r in per_ic], dtype=np.float64)
            mfno = np.isfinite(v) & np.isfinite(fno)
            mdno = np.isfinite(v) & np.isfinite(dno)
            if mfno.sum() < 4 or np.std(v[mfno]) < 1e-9 or np.std(fno[mfno]) < 1e-9:
                rf = float("nan")
            else:
                rf = float(spearmanr(v[mfno], fno[mfno]).statistic)
            if mdno.sum() < 4 or np.std(v[mdno]) < 1e-9 or np.std(dno[mdno]) < 1e-9:
                rd = float("nan")
            else:
                rd = float(spearmanr(v[mdno], dno[mdno]).statistic)
            line += f"  {rf:+.2f}/{rd:+.2f} "
            within[regime][d] = {"fno_spearman": rf, "dno_spearman": rd}
        print(line)
    data["within_regime_correlations"] = within
    with open(FORENSICS, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
