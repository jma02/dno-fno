"""Part 3: finer-grained error-spectrum analysis for the worst-3 finite ICs.

Split power spectrum into ~10 log-spaced bands and report where error mass sits.
Also report whether error spectrum is dominated by *resolved physical modes*
(roughly matched to truth peak band) vs noise floor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/home/johnma/dno-fno")
FNO_DIR = REPO / "outputs/fno_w128b6_v3_hclip5_20260514_063811/eval_suite"
DNO_DIR = REPO / "outputs/dno_w128b6_l64_v3_pf1_20260603_064450/eval_suite"
FORENSICS = REPO / "playground/diagnostics/rollout_forensics.json"

WORST3 = [
    ("linear", 59),
    ("bf_modal", 2000014),
    ("bf_g1", 1000001),
]


def band_breakdown(eta_truth: np.ndarray, eta_pred: np.ndarray, n_bins: int = 8) -> dict:
    nx = eta_truth.shape[0]
    half = nx // 2
    err = eta_pred - eta_truth
    err_pow = np.abs(np.fft.fft(err)[:half]) ** 2
    truth_pow = np.abs(np.fft.fft(eta_truth)[:half]) ** 2
    # log-spaced bin edges starting from k=1
    edges = np.unique(np.round(np.logspace(0, np.log10(half), n_bins + 1)).astype(int))
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1])
        ep = float(err_pow[lo:hi].sum())
        tp = float(truth_pow[lo:hi].sum())
        bins.append({"k_lo": lo, "k_hi": hi, "err_pow": ep, "truth_pow": tp})
    # normalize
    total_e = sum(b["err_pow"] for b in bins) + 1e-30
    total_t = sum(b["truth_pow"] for b in bins) + 1e-30
    for b in bins:
        b["err_frac"] = b["err_pow"] / total_e
        b["truth_frac"] = b["truth_pow"] / total_t
    # truth peak band
    peak = max(range(len(bins)), key=lambda i: bins[i]["truth_pow"])
    # error peak band
    err_peak = max(range(len(bins)), key=lambda i: bins[i]["err_pow"])
    return {
        "bins": bins,
        "truth_peak_band": peak,
        "err_peak_band": err_peak,
        "err_matches_truth_peak": peak == err_peak,
        "err_mass_above_truth_peak": float(sum(b["err_frac"] for i, b in enumerate(bins) if i > peak)),
    }


def main():
    with open(FORENSICS) as f:
        data = json.load(f)
    out = {}
    print("Finer-grained error-spectrum breakdown:")
    for regime, case_id in WORST3:
        fno_npz = np.load(FNO_DIR / f"{regime}_trajs.npz", mmap_mode="r")
        dno_npz = np.load(DNO_DIR / f"{regime}_trajs.npz", mmap_mode="r")
        case_ids = np.asarray(fno_npz["case_ids"])
        b = int(np.argmax(case_ids == case_id))
        truth_eta_T = np.asarray(fno_npz["truth_eta"][-1, b])
        fno_pred_T = np.asarray(fno_npz["pred_eta"][-1, b])
        dno_pred_T = np.asarray(dno_npz["pred_eta"][-1, b])
        # also at t/tmax=0.5
        T = fno_npz["times"].shape[0]
        mid = T // 2
        truth_eta_M = np.asarray(fno_npz["truth_eta"][mid, b])
        fno_pred_M = np.asarray(fno_npz["pred_eta"][mid, b])
        dno_pred_M = np.asarray(dno_npz["pred_eta"][mid, b])
        key = f"{regime}_case{case_id}"
        fno_T = band_breakdown(truth_eta_T, fno_pred_T)
        dno_T = band_breakdown(truth_eta_T, dno_pred_T)
        fno_M = band_breakdown(truth_eta_M, fno_pred_M)
        dno_M = band_breakdown(truth_eta_M, dno_pred_M)
        out[key] = {
            "fno_tmax": fno_T,
            "dno_tmax": dno_T,
            "fno_t50p": fno_M,
            "dno_t50p": dno_M,
        }
        print(f"\n{key} (tmax)")
        print("  band [k_lo,k_hi)  truth_frac    err_frac(FNO)  err_frac(DNO)")
        for i, bb in enumerate(fno_T["bins"]):
            db = dno_T["bins"][i]
            print(f"   {i}  [{bb['k_lo']:4d},{bb['k_hi']:4d})    {bb['truth_frac']:.3f}        {bb['err_frac']:.3f}          {db['err_frac']:.3f}")
        print(f"  truth_peak_band={fno_T['truth_peak_band']} fno_err_peak={fno_T['err_peak_band']} (match={fno_T['err_matches_truth_peak']})")
        print(f"                                          dno_err_peak={dno_T['err_peak_band']} (match={dno_T['err_matches_truth_peak']})")

    # also: for the "NaN" ICs, check eta blow-up point — when does NaN first appear
    print("\nNaN blow-up cases (FNO):")
    for regime in ("bf_g1", "bf_modal"):
        fno_npz = np.load(FNO_DIR / f"{regime}_trajs.npz", mmap_mode="r")
        case_ids = np.asarray(fno_npz["case_ids"])
        rel = np.asarray(fno_npz["rel_l2_eta"])
        pred = fno_npz["pred_eta"]
        T = rel.shape[0]
        for b in range(rel.shape[1]):
            r = rel[:, b]
            if (~np.isfinite(r[-1])):
                nan_idx = int(np.argmax(~np.isfinite(r)))
                # last finite eta
                if nan_idx > 0:
                    last_finite = nan_idx - 1
                    eta_max_lf = float(np.max(np.abs(pred[last_finite, b])))
                    eta_max_at_nan = float(np.nanmax(np.abs(pred[nan_idx, b])))
                else:
                    eta_max_lf = float("nan")
                    eta_max_at_nan = float("nan")
                print(f"  {regime} case {int(case_ids[b])}: nan at t-idx {nan_idx}/{T-1} (t={fno_npz['times'][nan_idx]:.2f}), eta_max_pre={eta_max_lf:.3f}")

    data["worst3_spectrum_bands"] = out
    with open(FORENSICS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nupdated {FORENSICS}")


if __name__ == "__main__":
    main()
