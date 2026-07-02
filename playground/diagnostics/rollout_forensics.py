"""Rollout forensics: per-IC failure analysis for FNO vs DNO+PF surrogates.

Computes:
  - Per-IC rel-L2(eta) at t/tmax ∈ {0.1, 0.5, 1.0} for both models
  - IC physical descriptors from truth_eta[0], truth_xi[0]
  - Pearson / Spearman correlations of descriptors vs failure
  - Error growth signature (linear vs exp vs stepped) for worst-3 ICs
  - Error spectrum at tmax for worst-3 ICs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path("/home/johnma/dno-fno")
FNO_DIR = REPO / "outputs/fno_w128b6_v3_hclip5_20260514_063811/eval_suite"
DNO_DIR = REPO / "outputs/dno_w128b6_l64_v3_pf1_20260603_064450/eval_suite"
REGIMES = [
    "tanaka_g0", "tanaka_g1", "bf_g0", "bf_g1", "bf_modal",
    "linear", "stokes_deep", "stokes_finite",
    "random_sea_deep", "random_sea_finite",
]

LENGTH = 2.0 * np.pi


def regime_depth(regime: str, depths_arr: np.ndarray) -> np.ndarray:
    """Returns per-IC depth array (already in the trajs file)."""
    return np.asarray(depths_arr, dtype=np.float64)


def physical_descriptors(eta0: np.ndarray, xi0: np.ndarray, depth: float) -> dict[str, float]:
    """Compute physical descriptors for a single IC (eta0 shape (nx,), xi0 shape (nx,))."""
    nx = eta0.shape[0]
    dx = LENGTH / nx
    # spatial derivatives via FFT
    k = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi  # angular wavenumbers
    eta_hat = np.fft.fft(eta0)
    deta_dx = np.real(np.fft.ifft(1j * k * eta_hat))

    # power spectrum (one-sided, abs of complex amplitudes)
    half = nx // 2
    eta_pow = (np.abs(eta_hat[:half]) ** 2)
    k_pos = np.abs(k[:half])
    # peak k (skip k=0 dc)
    if eta_pow[1:].sum() > 0:
        k_peak_idx = int(np.argmax(eta_pow[1:]) + 1)
        k_peak = float(k_pos[k_peak_idx])
    else:
        k_peak = 0.0
    # bandwidth: variance of k weighted by spectrum
    p = eta_pow.copy()
    if p.sum() > 0:
        p = p / p.sum()
        k_mean = float((k_pos * p).sum())
        k_var = float(((k_pos - k_mean) ** 2 * p).sum())
        bandwidth = float(np.sqrt(max(k_var, 0.0)))
    else:
        k_mean = 0.0
        bandwidth = 0.0
    kh = float(k_peak * depth)
    return {
        "max_abs_eta": float(np.max(np.abs(eta0))),
        "max_abs_xi": float(np.max(np.abs(xi0))),
        "max_steepness": float(np.max(np.abs(deta_dx))),
        "rms_eta": float(np.sqrt(np.mean(eta0 ** 2))),
        "k_peak": k_peak,
        "k_mean": k_mean,
        "bandwidth": bandwidth,
        "kh": kh,
        "depth": float(depth),
    }


def load_regime(regime: str, model_dir: Path) -> dict:
    path = model_dir / f"{regime}_trajs.npz"
    d = np.load(path, mmap_mode="r")
    return {
        "times": np.asarray(d["times"]),
        "depths": np.asarray(d["depths"]),
        "case_ids": np.asarray(d["case_ids"]),
        "truth_eta": d["truth_eta"],
        "truth_xi": d["truth_xi"],
        "pred_eta": d["pred_eta"],
        "rel_l2_eta": np.asarray(d["rel_l2_eta"]),  # (T, B)
    }


def rel_l2_at_fracs(rel_l2: np.ndarray, times: np.ndarray, fracs: list[float]) -> dict[float, np.ndarray]:
    """rel_l2 shape (T, B). Return {frac: (B,) array of values at nearest t/tmax}."""
    tmax = float(times[-1])
    out: dict[float, np.ndarray] = {}
    for f in fracs:
        t_target = f * tmax
        idx = int(np.argmin(np.abs(times - t_target)))
        # if frac=0.0 we'd want idx 0, but at idx 0 rel_l2 is undefined/0; use idx>=1
        if f == 0.0:
            idx = 0
        out[f] = np.asarray(rel_l2[idx], dtype=np.float64)
    return out


def classify_growth(rel_l2_t: np.ndarray, times: np.ndarray) -> dict:
    """Classify error growth as linear, exponential, or stepped.

    Skip t=0 (rel-L2 starts at 0). Fit log(rel_l2) ~ a*t + b for exponential
    and rel_l2 ~ a*t + b for linear. Compare R^2. Stepped detected by max
    derivative being much larger than mean derivative.
    Also flags NaN/Inf events.
    """
    t = np.asarray(times)
    y = np.asarray(rel_l2_t, dtype=np.float64)
    n = len(t)
    # skip first ~5 samples to avoid t=0 zero
    start = max(1, n // 50)
    ts = t[start:]
    ys = y[start:]
    finite = np.isfinite(ys) & (ys > 0)
    if finite.sum() < 5:
        return {"kind": "blown_up_early", "linear_r2": float("nan"), "exp_r2": float("nan"), "max_jump": float("nan")}
    # find first NaN/inf
    nan_idx = int(np.argmin(finite)) if (~finite).any() else -1
    ts_f = ts[finite]
    ys_f = ys[finite]
    # linear fit
    A = np.vstack([ts_f, np.ones_like(ts_f)]).T
    coef_lin, *_ = np.linalg.lstsq(A, ys_f, rcond=None)
    yhat_lin = A @ coef_lin
    ss_res_lin = ((ys_f - yhat_lin) ** 2).sum()
    ss_tot = ((ys_f - ys_f.mean()) ** 2).sum() + 1e-30
    r2_lin = float(1.0 - ss_res_lin / ss_tot)
    # exponential fit on log
    logy = np.log(ys_f + 1e-30)
    coef_exp, *_ = np.linalg.lstsq(A, logy, rcond=None)
    yhat_exp = A @ coef_exp
    ss_res_exp = ((logy - yhat_exp) ** 2).sum()
    ss_tot_log = ((logy - logy.mean()) ** 2).sum() + 1e-30
    r2_exp = float(1.0 - ss_res_exp / ss_tot_log)
    # stepped: jump ratio = max(diff) / median(diff) on finite chunk
    diffs = np.diff(ys_f)
    diffs_pos = diffs[diffs > 0]
    if diffs_pos.size > 5:
        max_jump = float(diffs_pos.max() / (np.median(diffs_pos) + 1e-30))
    else:
        max_jump = float("nan")
    # classify
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
        "nan_idx": nan_idx,
    }


def error_spectrum_signature(truth_eta_T: np.ndarray, pred_eta_T: np.ndarray) -> dict:
    """Compare error spectrum at one timestep vs truth spectrum.

    truth_eta_T, pred_eta_T: (nx,)
    Returns: dominant-mode region (low/mid/high), spectral overlap with truth
    """
    nx = truth_eta_T.shape[0]
    half = nx // 2
    err = pred_eta_T - truth_eta_T
    err_hat = np.fft.fft(err)
    truth_hat = np.fft.fft(truth_eta_T)
    err_pow = np.abs(err_hat[:half]) ** 2
    truth_pow = np.abs(truth_hat[:half]) ** 2
    # split into 3 bands: low (0–nx/8), mid (nx/8 – nx/4), high (nx/4 – nx/2)
    lo = err_pow[: half // 4].sum()
    md = err_pow[half // 4 : half // 2].sum()
    hi = err_pow[half // 2 :].sum()
    total = lo + md + hi + 1e-30
    # truth band fractions
    tlo = truth_pow[: half // 4].sum()
    tmd = truth_pow[half // 4 : half // 2].sum()
    thi = truth_pow[half // 2 :].sum()
    ttotal = tlo + tmd + thi + 1e-30
    # cosine similarity of (err_pow / total) vs (truth_pow / ttotal)
    a = err_pow / total
    b = truth_pow / ttotal
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "err_low_frac": float(lo / total),
        "err_mid_frac": float(md / total),
        "err_high_frac": float(hi / total),
        "truth_low_frac": float(tlo / ttotal),
        "truth_mid_frac": float(tmd / ttotal),
        "truth_high_frac": float(thi / ttotal),
        "spectrum_cosine_sim": cos,
    }


def main():
    out: dict = {"regimes": {}, "global": {}}
    desc_all: dict[str, list[float]] = {}
    err_all: dict[str, list[float]] = {"fno_tmax": [], "dno_tmax": []}

    for regime in REGIMES:
        fno_path = FNO_DIR / f"{regime}_trajs.npz"
        dno_path = DNO_DIR / f"{regime}_trajs.npz"
        if not fno_path.exists() or not dno_path.exists():
            print(f"skipping {regime}: file missing")
            continue
        print(f"processing {regime}")
        fno = load_regime(regime, FNO_DIR)
        dno = load_regime(regime, DNO_DIR)
        # sanity: same case_ids
        if not np.array_equal(fno["case_ids"], dno["case_ids"]):
            print(f"  WARN: case_id mismatch fno={fno['case_ids']} dno={dno['case_ids']}")

        case_ids = fno["case_ids"]
        depths = fno["depths"]
        times = fno["times"]
        B = len(case_ids)

        # rel-L2 at three horizons for both models
        fracs = [0.1, 0.5, 1.0]
        fno_rl = rel_l2_at_fracs(fno["rel_l2_eta"], times, fracs)
        dno_rl = rel_l2_at_fracs(dno["rel_l2_eta"], times, fracs)

        # IC descriptors from truth t=0
        truth_eta_0 = np.asarray(fno["truth_eta"][0])  # (B, nx)
        truth_xi_0 = np.asarray(fno["truth_xi"][0])
        per_ic = []
        for b in range(B):
            d = physical_descriptors(truth_eta_0[b], truth_xi_0[b], float(depths[b]))
            d["case_id"] = int(case_ids[b])
            d["depth"] = float(depths[b])
            d["fno_rel_l2_t10p"] = float(fno_rl[0.1][b])
            d["fno_rel_l2_t50p"] = float(fno_rl[0.5][b])
            d["fno_rel_l2_tmax"] = float(fno_rl[1.0][b])
            d["dno_rel_l2_t10p"] = float(dno_rl[0.1][b])
            d["dno_rel_l2_t50p"] = float(dno_rl[0.5][b])
            d["dno_rel_l2_tmax"] = float(dno_rl[1.0][b])
            per_ic.append(d)

        # rank worst by tmax (use max across both models for global view)
        # but also separate worst lists
        worst_fno = sorted(per_ic, key=lambda r: -r["fno_rel_l2_tmax"])[:5]
        worst_dno = sorted(per_ic, key=lambda r: -r["dno_rel_l2_tmax"])[:5]

        # IC-level overlap: worst-5 IDs in each
        worst_fno_ids = {r["case_id"] for r in worst_fno}
        worst_dno_ids = {r["case_id"] for r in worst_dno}
        overlap = sorted(worst_fno_ids & worst_dno_ids)

        out["regimes"][regime] = {
            "n_ics": B,
            "tmax": float(times[-1]),
            "fno_mean_tmax": float(np.mean(fno_rl[1.0])),
            "dno_mean_tmax": float(np.mean(dno_rl[1.0])),
            "fno_median_tmax": float(np.median(fno_rl[1.0])),
            "dno_median_tmax": float(np.median(dno_rl[1.0])),
            "fno_p95_tmax": float(np.percentile(fno_rl[1.0], 95)),
            "dno_p95_tmax": float(np.percentile(dno_rl[1.0], 95)),
            "fno_nan_rate": float(np.mean(~np.isfinite(fno_rl[1.0]))),
            "dno_nan_rate": float(np.mean(~np.isfinite(dno_rl[1.0]))),
            "per_ic": per_ic,
            "worst5_fno_case_ids": [r["case_id"] for r in worst_fno],
            "worst5_dno_case_ids": [r["case_id"] for r in worst_dno],
            "worst5_overlap_case_ids": overlap,
        }

        # collect for global correlations
        keys = ["max_abs_eta", "max_abs_xi", "max_steepness", "rms_eta",
                "k_peak", "bandwidth", "kh", "depth"]
        for k in keys:
            desc_all.setdefault(k, []).extend([r[k] for r in per_ic])
        err_all["fno_tmax"].extend([r["fno_rel_l2_tmax"] for r in per_ic])
        err_all["dno_tmax"].extend([r["dno_rel_l2_tmax"] for r in per_ic])

    # global correlations
    corr: dict = {}
    for descriptor, vals in desc_all.items():
        v = np.asarray(vals, dtype=np.float64)
        for model_key in ("fno_tmax", "dno_tmax"):
            err = np.asarray(err_all[model_key], dtype=np.float64)
            mask = np.isfinite(v) & np.isfinite(err)
            if mask.sum() < 4:
                continue
            try:
                pr = pearsonr(v[mask], err[mask])
                sr = spearmanr(v[mask], err[mask])
                corr.setdefault(descriptor, {})[model_key] = {
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                    "spearman_rho": float(sr.statistic),
                    "spearman_p": float(sr.pvalue),
                    "n": int(mask.sum()),
                }
            except Exception as e:
                print(f"  corr fail {descriptor} {model_key}: {e}")
    out["global"]["correlations"] = corr

    # write json
    out_path = REPO / "playground/diagnostics/rollout_forensics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
