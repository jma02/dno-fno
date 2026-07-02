"""Part 2: error growth signature + error spectrum for the worst 3 ICs across regimes.

Reads playground/diagnostics/rollout_forensics.json from part 1, picks the top-3
worst ICs (by fno tmax rel-L2) globally, and characterizes:
  - growth shape (linear / exponential / stepped / blow-up) via R^2 fits
  - error spectrum at tmax (low/mid/high-k mass, cosine sim to truth spectrum)
  - same for DNO+PF on the same ICs

Also: compute per-regime mean rel-L2 difference (FNO - DNO_PF) at tmax to flag
where pushforward helps vs not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/home/johnma/dno-fno")
FNO_DIR = REPO / "outputs/fno_w128b6_v3_hclip5_20260514_063811/eval_suite"
DNO_DIR = REPO / "outputs/dno_w128b6_l64_v3_pf1_20260603_064450/eval_suite"
FORENSICS = REPO / "playground/diagnostics/rollout_forensics.json"


def classify_growth(rel_l2_t: np.ndarray, times: np.ndarray) -> dict:
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(rel_l2_t, dtype=np.float64)
    n = len(t)
    start = max(1, n // 50)
    ts = t[start:]
    ys = y[start:]
    finite = np.isfinite(ys) & (ys > 0)
    if finite.sum() < 5:
        return {"kind": "blown_up_early", "linear_r2": float("nan"), "exp_r2": float("nan")}
    nan_idx = -1
    if (~finite).any():
        nan_idx = int(np.argmax(~finite))
    ts_f = ts[finite]
    ys_f = ys[finite]
    A = np.vstack([ts_f, np.ones_like(ts_f)]).T
    coef_lin, *_ = np.linalg.lstsq(A, ys_f, rcond=None)
    yhat = A @ coef_lin
    ss_res = ((ys_f - yhat) ** 2).sum()
    ss_tot = ((ys_f - ys_f.mean()) ** 2).sum() + 1e-30
    r2_lin = float(1.0 - ss_res / ss_tot)
    logy = np.log(np.clip(ys_f, 1e-12, None))
    coef_exp, *_ = np.linalg.lstsq(A, logy, rcond=None)
    yhat_e = A @ coef_exp
    ss_res_e = ((logy - yhat_e) ** 2).sum()
    ss_tot_e = ((logy - logy.mean()) ** 2).sum() + 1e-30
    r2_exp = float(1.0 - ss_res_e / ss_tot_e)
    diffs = np.diff(ys_f)
    diffs_pos = diffs[diffs > 0]
    max_jump = float(diffs_pos.max() / (np.median(diffs_pos) + 1e-30)) if diffs_pos.size > 5 else float("nan")
    if nan_idx != -1:
        kind = "blew_up"
    elif np.isfinite(max_jump) and max_jump > 30.0 and r2_lin < 0.7:
        kind = "stepped"
    elif r2_exp > r2_lin + 0.05 and coef_exp[0] > 0.05:
        kind = "exponential"
    else:
        kind = "linear"
    return {
        "kind": kind,
        "linear_r2": r2_lin,
        "linear_slope": float(coef_lin[0]),
        "exp_r2": r2_exp,
        "exp_rate": float(coef_exp[0]),
        "max_jump_ratio": max_jump,
        "nan_t_idx": nan_idx,
        "final_rel_l2": float(ys[-1]) if np.isfinite(ys[-1]) else float("nan"),
    }


def err_spectrum(truth_eta_T: np.ndarray, pred_eta_T: np.ndarray) -> dict:
    nx = truth_eta_T.shape[0]
    half = nx // 2
    err = pred_eta_T - truth_eta_T
    err_hat = np.fft.fft(err)
    truth_hat = np.fft.fft(truth_eta_T)
    err_pow = np.abs(err_hat[:half]) ** 2
    truth_pow = np.abs(truth_hat[:half]) ** 2
    lo_split = half // 4
    md_split = half // 2
    lo = err_pow[:lo_split].sum()
    md = err_pow[lo_split:md_split].sum()
    hi = err_pow[md_split:].sum()
    total = lo + md + hi + 1e-30
    tlo = truth_pow[:lo_split].sum()
    tmd = truth_pow[lo_split:md_split].sum()
    thi = truth_pow[md_split:].sum()
    ttot = tlo + tmd + thi + 1e-30
    a = err_pow / total
    b = truth_pow / ttot
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    # dominant band
    bands = {"low": lo, "mid": md, "high": hi}
    dom = max(bands.items(), key=lambda kv: kv[1])[0]
    # peak-k of error
    pk = int(np.argmax(err_pow))
    return {
        "err_low_frac": float(lo / total),
        "err_mid_frac": float(md / total),
        "err_high_frac": float(hi / total),
        "truth_low_frac": float(tlo / ttot),
        "truth_mid_frac": float(tmd / ttot),
        "truth_high_frac": float(thi / ttot),
        "spectrum_cosine_sim": cos,
        "dominant_band": dom,
        "err_peak_k_idx": pk,
        "err_l2_total": float(np.sqrt(err_pow.sum())),
    }


def main():
    with open(FORENSICS) as f:
        data = json.load(f)

    # global worst by fno_rel_l2_tmax — separate NaN (blow-up) from finite
    flat = []
    for regime, R in data["regimes"].items():
        for r in R["per_ic"]:
            flat.append({
                "regime": regime,
                "case_id": r["case_id"],
                "fno_tmax": r["fno_rel_l2_tmax"],
                "dno_tmax": r["dno_rel_l2_tmax"],
                "max_steepness": r["max_steepness"],
                "max_abs_eta": r["max_abs_eta"],
                "kh": r["kh"],
            })
    flat_finite = [r for r in flat if np.isfinite(r["fno_tmax"])]
    flat_nan = [r for r in flat if not np.isfinite(r["fno_tmax"])]
    flat_fno = sorted(flat_finite, key=lambda r: -r["fno_tmax"])
    worst3_fno = flat_fno[:3]
    print(f"FNO NaN-blow-up ICs (count): {len(flat_nan)} / {len(flat)}")
    for r in flat_nan[:10]:
        print(f"  NaN: {r['regime']} case {r['case_id']} stp={r['max_steepness']:.3f}")
    print("Top 3 worst finite-error FNO ICs at tmax:")
    for r in worst3_fno:
        print(f"  {r['regime']} case {r['case_id']}: fno={r['fno_tmax']:.3f} dno={r['dno_tmax']:.3f} stp={r['max_steepness']:.3f}")

    # For each: load full trajectory and characterize
    growth_signature: dict = {}
    spectrum_signature: dict = {}
    for r in worst3_fno:
        regime = r["regime"]
        case_id = r["case_id"]
        # load
        fno_npz = np.load(FNO_DIR / f"{regime}_trajs.npz", mmap_mode="r")
        dno_npz = np.load(DNO_DIR / f"{regime}_trajs.npz", mmap_mode="r")
        case_ids = np.asarray(fno_npz["case_ids"])
        b = int(np.argmax(case_ids == case_id))
        times = np.asarray(fno_npz["times"])
        fno_rl_t = np.asarray(fno_npz["rel_l2_eta"][:, b])
        dno_rl_t = np.asarray(dno_npz["rel_l2_eta"][:, b])
        key = f"{regime}_case{case_id}"
        growth_signature[key] = {
            "regime": regime,
            "case_id": case_id,
            "fno": classify_growth(fno_rl_t, times),
            "dno": classify_growth(dno_rl_t, times),
        }
        # spectrum at tmax
        truth_eta_T = np.asarray(fno_npz["truth_eta"][-1, b])
        fno_pred_T = np.asarray(fno_npz["pred_eta"][-1, b])
        dno_pred_T = np.asarray(dno_npz["pred_eta"][-1, b])
        spectrum_signature[key] = {
            "fno": err_spectrum(truth_eta_T, fno_pred_T),
            "dno": err_spectrum(truth_eta_T, dno_pred_T),
        }

    # per-regime FNO vs DNO comparison (use nan-aware finite stats)
    regime_compare = {}
    for regime, R in data["regimes"].items():
        fno_vals = np.array([r["fno_rel_l2_tmax"] for r in R["per_ic"]], dtype=np.float64)
        dno_vals = np.array([r["dno_rel_l2_tmax"] for r in R["per_ic"]], dtype=np.float64)
        fno_f = fno_vals[np.isfinite(fno_vals)]
        dno_f = dno_vals[np.isfinite(dno_vals)]
        med_f = float(np.median(fno_f)) if fno_f.size else float("nan")
        med_d = float(np.median(dno_f)) if dno_f.size else float("nan")
        mean_f = float(np.mean(fno_f)) if fno_f.size else float("nan")
        mean_d = float(np.mean(dno_f)) if dno_f.size else float("nan")
        p95_f = float(np.percentile(fno_f, 95)) if fno_f.size else float("nan")
        p95_d = float(np.percentile(dno_f, 95)) if dno_f.size else float("nan")
        regime_compare[regime] = {
            "fno_median_tmax_finite": med_f,
            "dno_median_tmax_finite": med_d,
            "fno_mean_tmax_finite": mean_f,
            "dno_mean_tmax_finite": mean_d,
            "fno_p95_tmax_finite": p95_f,
            "dno_p95_tmax_finite": p95_d,
            "fno_nan_rate": R["fno_nan_rate"],
            "dno_nan_rate": R["dno_nan_rate"],
            "median_diff_pct": (med_f - med_d) / max(med_f, 1e-9) * 100.0 if med_f > 0 else float("nan"),
        }

    # merge into json
    data["worst3_global_fno"] = [{"regime": r["regime"], "case_id": r["case_id"], "fno_tmax": r["fno_tmax"], "dno_tmax": r["dno_tmax"]} for r in worst3_fno]
    data["growth_signatures"] = growth_signature
    data["spectrum_signatures"] = spectrum_signature
    data["regime_compare"] = regime_compare

    with open(FORENSICS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nupdated {FORENSICS}")

    # console output for the model report
    print("\n--- regime compare (fno vs dno) finite median tmax ---")
    for regime, c in regime_compare.items():
        delta = c["median_diff_pct"]
        print(f"  {regime:22s}: fno={c['fno_median_tmax_finite']:.4f}  dno={c['dno_median_tmax_finite']:.4f}  delta={delta:+7.1f}%  fno_nan={c['fno_nan_rate']:.2f} dno_nan={c['dno_nan_rate']:.2f}")

    print("\n--- growth signatures (worst-3 fno) ---")
    for k, v in growth_signature.items():
        print(f"  {k}")
        print(f"    fno kind={v['fno']['kind']} r2_lin={v['fno']['linear_r2']:.3f} r2_exp={v['fno']['exp_r2']:.3f} slope={v['fno']['linear_slope']:.4f} final={v['fno']['final_rel_l2']:.3f}")
        print(f"    dno kind={v['dno']['kind']} r2_lin={v['dno']['linear_r2']:.3f} r2_exp={v['dno']['exp_r2']:.3f} slope={v['dno']['linear_slope']:.4f} final={v['dno']['final_rel_l2']:.3f}")

    print("\n--- spectrum signatures (worst-3 at tmax) ---")
    for k, v in spectrum_signature.items():
        print(f"  {k}")
        print(f"    fno err_band: lo={v['fno']['err_low_frac']:.2f} mid={v['fno']['err_mid_frac']:.2f} hi={v['fno']['err_high_frac']:.2f} dom={v['fno']['dominant_band']} cos_truth={v['fno']['spectrum_cosine_sim']:.3f}")
        print(f"    dno err_band: lo={v['dno']['err_low_frac']:.2f} mid={v['dno']['err_mid_frac']:.2f} hi={v['dno']['err_high_frac']:.2f} dom={v['dno']['dominant_band']} cos_truth={v['dno']['spectrum_cosine_sim']:.3f}")

    # global correlations from part 1
    print("\n--- top descriptor correlations vs failure (Pearson r, Spearman rho) ---")
    corr = data["global"]["correlations"]
    rows = []
    for desc, m in corr.items():
        for model_key, c in m.items():
            rows.append((desc, model_key, abs(c["spearman_rho"]), c["pearson_r"], c["spearman_rho"], c["n"]))
    rows.sort(key=lambda r: -r[2])
    for desc, mk, _, pr, sp, n in rows[:16]:
        print(f"  {desc:18s} {mk:10s}  r={pr:+.3f}  rho={sp:+.3f}  n={n}")


if __name__ == "__main__":
    main()
