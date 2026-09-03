"""Standalone overnight audit for the four failing Tanaka cases of v5.

Produces per-IC physical+spectral diagnostics and writes figures under
notes/figures/tanaka_audit/. Intentionally CPU-only; no JAX, no model load.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/johnma/dno-fno")
OUT_DIR = ROOT / "notes" / "figures" / "tanaka_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJS_DIR = (
    ROOT / "outputs" / "cs_dno_w512b8_l256_v5_2gpu_20260615_141717" / "eval_suite_f64h"
)

# Failing-IC manifest. tag -> {simulation_ids in trajs.npz}, file paths, dataset name.
FAILING = [
    {"tag": "g0", "cid": 5, "label": "tanaka_g0_cid5"},
    {"tag": "g0", "cid": 11, "label": "tanaka_g0_cid11"},
    {"tag": "g1", "cid": 1000006, "label": "tanaka_g1_cid1000006"},
    {"tag": "g1", "cid": 1000011, "label": "tanaka_g1_cid1000011"},
]


def load_trajs(tag: str) -> dict:
    f = TRAJS_DIR / f"tanaka_{tag}_trajs.npz"
    d = np.load(f, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    d.close()
    return out


def load_ic_from_dataset(tag: str, cid: int) -> dict:
    """Return eta(t=0), xi(t=0), times, depth, and spec for a given case."""
    f = ROOT / "data" / f"tanaka_2_adaptive_{tag}.npz"
    d = np.load(f, allow_pickle=True)
    # case 5 / 11 / 1000006 / 1000011 are all in batch 0000.
    batch = "0000"
    cids = d[f"simulation_id_batch_{batch}"]
    mask = cids == cid
    if not mask.any():
        # try other batches
        for b in range(20):
            cids_b = d[f"simulation_id_batch_{b:04d}"]
            if (cids_b == cid).any():
                batch = f"{b:04d}"
                cids = cids_b
                mask = cids == cid
                break
    rows = np.where(mask)[0]
    times = d[f"time_batch_{batch}"][rows]
    depths = d[f"depth_batch_{batch}"][rows]
    eta = d[f"eta_batch_{batch}"][rows]
    xi = d[f"xi_batch_{batch}"][rows]
    gxi = d[f"gxi_batch_{batch}"][rows]
    # Find t=0 row
    t0_i = int(np.argmin(np.abs(times)))
    # Specs
    specs = json.loads(bytes(d[f"specs_batch_{batch}.json"]).decode())
    # cid 5 in g0 -> spec index 5; cid 1000006 in g1 -> spec index 6 (= cid - 1000000)
    if tag == "g0":
        spec_idx = cid
    else:
        spec_idx = cid - 1_000_000
    case_spec = specs[spec_idx]
    out = {
        "eta_t0": eta[t0_i],
        "xi_t0": xi[t0_i],
        "gxi_t0": gxi[t0_i],
        "times_in_dataset": times,
        "depth": float(depths[t0_i]),
        "spec": case_spec,
    }
    # x grid
    out["x"] = d["x"]
    d.close()
    return out


def physical_params(eta_t0: np.ndarray, depth: float, x: np.ndarray) -> dict:
    """Compute amplitude metrics + an approximate half-width and ka."""
    amax = float(np.max(eta_t0))
    amin = float(np.min(eta_t0))
    a = float(np.max(np.abs(eta_t0)))  # peak-amplitude magnitude
    half_amp = 0.5 * a
    # Find the dominant crest: location of max eta_t0
    pk_i = int(np.argmax(np.abs(eta_t0)))
    # walk outward from pk_i to find where |eta| < half_amp
    n = eta_t0.size
    abs_eta = np.abs(eta_t0)

    # Walk in both directions on circular array
    def walk(start: int, step: int) -> int:
        i = start
        cnt = 0
        while cnt < n:
            j = (i + step) % n
            if abs_eta[j] < half_amp:
                return j
            i = j
            cnt += 1
        return -1

    left = walk(pk_i, -1)
    right = walk(pk_i, +1)
    dx = float(x[1] - x[0])
    if left >= 0 and right >= 0:
        # geodesic distance on circle
        d_right = ((right - pk_i) % n) * dx
        d_left = ((pk_i - left) % n) * dx
        half_width = d_right + d_left
    else:
        half_width = float("nan")
    # Characteristic k from half-width (one crest ~ wavelength = 2 * full width)
    if half_width > 0 and np.isfinite(half_width):
        # rough effective k = pi / half_width (sech^2 has half-width ~ acosh(sqrt(2))/k_sol ~ 0.88/k_sol)
        k_eff = float(np.pi / half_width)
    else:
        k_eff = float("nan")
    ka = k_eff * a if np.isfinite(k_eff) else float("nan")
    # Also: tanh(kh)/(kh) shallowness param
    if np.isfinite(k_eff):
        mu = k_eff * depth
        ursell_like = (a / depth) / (mu**2) if mu > 0 else float("nan")
    else:
        mu = ursell_like = float("nan")
    return {
        "a_pos": amax,
        "a_neg": amin,
        "a_abs": a,
        "a_over_h": a / depth,
        "depth": depth,
        "half_width": half_width,
        "k_eff": k_eff,
        "ka": ka,
        "kh": mu,
        "ursell": ursell_like,
    }


def tanaka_reference_shape(
    depth: float, target_a_over_h: float, x: np.ndarray
) -> tuple[float, float] | None:
    """Return a reference Tanaka-like sech^2 shape with the requested a/h on the same grid.

    We don't have a Tanaka solver in scope here — use a sech^2 KdV-like profile
    as a *qualitative* reference. KdV soliton: a sech^2(sqrt(3a/(4h^3)) * (x - x0)).
    """
    a = target_a_over_h * depth
    if a <= 0.0 or depth <= 0.0:
        return None
    kappa = float(np.sqrt(3.0 * a / (4.0 * depth**3)))
    # center at the same x as the dataset's IC peak
    return a, kappa  # caller will produce a shifted copy on the requested grid


def plot_ic_physical(case_info: list[dict]) -> None:
    """Per-IC figure: eta(x) at t=0 + KdV-soliton reference at same a/h."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, c in zip(axes.flat, case_info):
        eta = c["eta_t0"]
        x = c["x"]
        # KdV reference for comparison
        ref = tanaka_reference_shape(c["depth"], c["a_over_h"], x)
        if ref is not None:
            a_ref, kappa_ref = ref
            pk_i = int(np.argmax(np.abs(eta)))
            x0 = float(x[pk_i])
            sign = np.sign(eta[pk_i])
            ref_eta = sign * a_ref / np.cosh(kappa_ref * (x - x0)) ** 2
            ax.plot(x, ref_eta, "k--", lw=1.0, alpha=0.5, label="KdV ref a/h matched")
        ax.plot(x, eta, "C0-", lw=1.5, label=c["label"])
        ax.set_title(
            f"{c['label']}: h={c['depth']:.3f}, a/h={c['a_over_h']:.3f}, ka={c['ka']:.3f}",
            fontsize=10,
        )
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylabel("eta(t=0)")
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    fig.suptitle(
        "Failing-Tanaka ICs vs KdV-soliton reference (matched a/h)", fontsize=12
    )
    fig.tight_layout()
    out = OUT_DIR / "ic_profiles.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def plot_truth_energy_drift(case_info: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, c in zip(axes.flat, case_info):
        t = c["times"]
        edp = c["energy_drift_pred"]
        edt = c["energy_drift_truth"]
        # mask NaN
        ax.plot(t, edt, "C0-", lw=1.2, label="truth")
        ax.plot(t, edp, "C3--", lw=1.2, label="pred")
        ax.axvline(
            c["nan_t"],
            color="red",
            lw=0.8,
            ls=":",
            alpha=0.6,
            label=f"NaN/div @ t={c['nan_t']:.1f}",
        ) if c["nan_t"] is not None else None
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(f"{c['label']}: energy drift vs t", fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel("rel energy drift")
        ax.axhline(1e-3, color="k", lw=0.5, alpha=0.3, ls="--")
        ax.legend(fontsize=8)
    fig.suptitle("Energy drift: truth (CS-DNO M=6 spectral) is rock-solid", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "energy_drift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def spectral_attribution(c: dict) -> dict:
    """Compute hat_err(t,k), banded norms, and coherent/incoherent split."""
    pred = c["pred_eta"]  # (T, nx)
    truth = c["truth_eta"]  # (T, nx)
    times = c["times"]
    # truncate to first NaN in pred
    nan_t_mask = np.isnan(pred).any(axis=-1)
    if nan_t_mask.any():
        t_cut = int(np.argmax(nan_t_mask))
    else:
        t_cut = pred.shape[0]
    pred = pred[:t_cut]
    truth = truth[:t_cut]
    times = times[:t_cut]
    err = pred - truth
    # rfft over space
    hat_err = np.fft.rfft(err, axis=-1, norm="forward")
    hat_truth = np.fft.rfft(truth, axis=-1, norm="forward")
    # k bins (k = integer mode index here, since L=2pi means physical k = mode index)
    # k_low: k<10, k_carrier: 10<=k<30, k_high: k>=30
    k_idx = np.arange(hat_err.shape[-1])
    low = k_idx < 10
    carrier = (k_idx >= 10) & (k_idx < 30)
    high = k_idx >= 30

    def band_norm(hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # L2 norm reconstructed from spectrum: 2*sum |hat|^2 for k>0, plus |hat[0]|^2
        # (Parseval for rfft norm=forward: 2 * sum_{k>0} |hat[k]|^2 + |hat[0]|^2 = mean(x^2))
        contributing = mask & (k_idx > 0)
        sq_corrected = (np.abs(hat[:, contributing]) ** 2).sum(axis=-1) * 2.0
        if mask[0]:
            sq_corrected = sq_corrected + (np.abs(hat[:, 0]) ** 2)
        return np.sqrt(sq_corrected)

    err_low = band_norm(hat_err, low)
    err_carr = band_norm(hat_err, carrier)
    err_high = band_norm(hat_err, high)
    truth_low = band_norm(hat_truth, low)
    truth_carr = band_norm(hat_truth, carrier)
    truth_high = band_norm(hat_truth, high)
    # Coherent (projection) vs incoherent split (in physical space).
    # alpha(t) = <err, truth> / <truth, truth>; coherent_err(t,x) = alpha*truth; incoherent = err - coherent
    inner_et = np.einsum("tn,tn->t", err, truth)
    norm_t2 = np.einsum("tn,tn->t", truth, truth)
    alpha = np.where(norm_t2 > 0, inner_et / norm_t2, 0.0)
    coh = alpha[:, None] * truth
    inc = err - coh
    coh_l2 = np.sqrt((coh**2).mean(axis=-1))
    inc_l2 = np.sqrt((inc**2).mean(axis=-1))
    err_l2 = np.sqrt((err**2).mean(axis=-1))
    return {
        "times": times,
        "hat_err": hat_err,
        "hat_truth": hat_truth,
        "err_low": err_low,
        "err_carr": err_carr,
        "err_high": err_high,
        "truth_low": truth_low,
        "truth_carr": truth_carr,
        "truth_high": truth_high,
        "alpha": alpha,
        "coh_l2": coh_l2,
        "inc_l2": inc_l2,
        "err_l2": err_l2,
        "t_cut": t_cut,
    }


def plot_spectral_heatmap(c: dict, sp: dict) -> None:
    times = sp["times"]
    hat_err = sp["hat_err"]
    hat_truth = sp["hat_truth"]
    # We'll only show k up to 64 (well above carrier; nx=1024 -> kmax=512 but most energy below 32)
    k_show = 64
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    a0 = np.log10(np.maximum(np.abs(hat_err[:, :k_show]), 1e-16))
    a1 = np.log10(np.maximum(np.abs(hat_truth[:, :k_show]), 1e-16))
    vmin, vmax = -10.0, max(a0.max(), a1.max())
    im0 = axes[0].imshow(
        a0,
        aspect="auto",
        origin="lower",
        extent=[0, k_show, times.min(), times.max()],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"{c['label']}: log10|hat_err|(t, k)")
    axes[0].set_xlabel("k (mode index)")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(
        a1,
        aspect="auto",
        origin="lower",
        extent=[0, k_show, times.min(), times.max()],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"{c['label']}: log10|hat_truth|(t, k)")
    axes[1].set_xlabel("k (mode index)")
    plt.colorbar(im1, ax=axes[1])
    # carrier/high band markers
    for ax in axes:
        ax.axvline(10, color="white", lw=0.6, ls=":")
        ax.axvline(30, color="white", lw=0.6, ls=":")
    fig.tight_layout()
    out = OUT_DIR / f"spec_heatmap_{c['label']}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def plot_banded_norms(case_info: list[dict], specs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c, sp in zip(axes.flat, case_info, specs):
        t = sp["times"]
        ax.plot(t, sp["err_low"], "C0-", label="err k<10")
        ax.plot(t, sp["err_carr"], "C1-", label="err 10<=k<30")
        ax.plot(t, sp["err_high"], "C3-", label="err k>=30")
        ax.plot(t, sp["truth_low"], "C0--", lw=0.8, alpha=0.6, label="truth k<10")
        ax.plot(t, sp["truth_carr"], "C1--", lw=0.8, alpha=0.6, label="truth 10<=k<30")
        ax.plot(t, sp["truth_high"], "C3--", lw=0.8, alpha=0.6, label="truth k>=30")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("L2 amplitude per band")
        ax.set_title(c["label"], fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        if c["nan_t"] is not None:
            ax.axvline(c["nan_t"], color="red", lw=0.8, ls=":", alpha=0.7)
    fig.suptitle("Banded spectral error vs truth (per IC)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "banded_spectral.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def plot_coh_incoh(case_info: list[dict], specs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c, sp in zip(axes.flat, case_info, specs):
        t = sp["times"]
        ax.plot(t, sp["err_l2"], "k-", lw=1.2, label="|err|")
        ax.plot(t, sp["coh_l2"], "C0-", lw=1.0, label="coherent (alpha*truth)")
        ax.plot(t, sp["inc_l2"], "C3-", lw=1.0, label="incoherent (shape/phase)")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("L2 (per-pt RMS)")
        ax.set_title(f"{c['label']} (alpha_final={sp['alpha'][-1]:+.3f})", fontsize=10)
        ax.legend(fontsize=8)
        if c["nan_t"] is not None:
            ax.axvline(c["nan_t"], color="red", lw=0.8, ls=":", alpha=0.7)
    fig.suptitle("Coherent (amplitude) vs incoherent (phase/shape) error", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "coherent_incoherent.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main() -> dict:
    # Load trajs once per dataset
    trajs = {tag: load_trajs(tag) for tag in ["g0", "g1"]}

    case_info = []
    for ic in FAILING:
        tag = ic["tag"]
        cid = ic["cid"]
        t = trajs[tag]
        i = int(np.where(t["simulation_ids"] == cid)[0][0])
        times = t["times"]
        pred_eta = t["pred_eta"][:, i, :]
        truth_eta = t["truth_eta"][:, i, :]
        rl2 = t["rel_l2_eta"][:, i]
        edp = t["energy_drift_pred"][:, i]
        edt = t["energy_drift_truth"][:, i]
        depth = float(t["depths"][i])
        nan_t = None
        nan_mask = np.isnan(pred_eta).any(axis=-1)
        if nan_mask.any():
            nan_t = float(times[np.argmax(nan_mask)])
        # Sometimes there's no NaN but rl2 > 1 (cid 1000011); use that t
        if nan_t is None:
            finite_rl2 = np.where(np.isfinite(rl2), rl2, 0.0)
            if (finite_rl2 > 1.0).any():
                nan_t = float(times[np.argmax(finite_rl2 > 1.0)])
        ic_data = load_ic_from_dataset(tag, cid)
        # Use the *truth* eta(t=0) from the trajs for consistency with what the eval saw
        phys = physical_params(truth_eta[0], depth, ic_data["x"])
        case = {
            "label": ic["label"],
            "tag": tag,
            "cid": cid,
            "depth": depth,
            "times": times,
            "pred_eta": pred_eta,
            "truth_eta": truth_eta,
            "rl2": rl2,
            "energy_drift_pred": edp,
            "energy_drift_truth": edt,
            "nan_t": nan_t,
            "max_truth_drift": float(
                np.nanmax(np.abs(np.where(np.isfinite(edt), edt, 0.0)))
            ),
            "max_pred_drift": float(
                np.nanmax(np.abs(np.where(np.isfinite(edp), edp, 0.0)))
            ),
            "max_rl2": float(np.nanmax(np.where(np.isfinite(rl2), rl2, 0.0))),
            "eta_t0": truth_eta[0],
            "xi_t0": ic_data["xi_t0"],  # used only for ka info
            "x": ic_data["x"],
            "spec": ic_data["spec"],
            **phys,
        }
        case_info.append(case)

    # --- Physical plots ---
    plot_ic_physical(case_info)
    plot_truth_energy_drift(case_info)

    # --- Spectral attribution ---
    specs = [spectral_attribution(c) for c in case_info]
    plot_banded_norms(case_info, specs)
    plot_coh_incoh(case_info, specs)
    for c, sp in zip(case_info, specs):
        plot_spectral_heatmap(c, sp)

    # --- Tabular summary ---
    table_rows = []
    for c, sp in zip(case_info, specs):
        # Determine dominant band at end of usable trajectory
        end_i = sp["t_cut"] - 1
        bands: dict[str, float] = {
            "low": float(sp["err_low"][end_i]),
            "carrier": float(sp["err_carr"][end_i]),
            "high": float(sp["err_high"][end_i]),
        }
        dom = max(bands, key=bands.__getitem__)

        # Growth ratios per band (end / start)
        def growth(arr: np.ndarray) -> float:
            start_val = max(float(arr[0]), 1e-16)
            return float(arr[end_i]) / start_val

        gr = {
            "low": growth(sp["err_low"]),
            "carrier": growth(sp["err_carr"]),
            "high": growth(sp["err_high"]),
        }
        row = {
            "label": c["label"],
            "depth": c["depth"],
            "a_over_h": c["a_over_h"],
            "ka": c["ka"],
            "kh": c["kh"],
            "nan_t": c["nan_t"],
            "max_truth_drift": c["max_truth_drift"],
            "max_rl2": c["max_rl2"],
            "dominant_band_at_end": dom,
            "band_growth_ratios": gr,
            "alpha_final": float(sp["alpha"][end_i]),
            "coh_l2_final": float(sp["coh_l2"][end_i]),
            "inc_l2_final": float(sp["inc_l2"][end_i]),
            "spec": c["spec"],
            "amax_eta": c["a_pos"],
            "amin_eta": c["a_neg"],
        }
        table_rows.append(row)

    summary_path = OUT_DIR / "audit_summary.json"
    summary_path.write_text(json.dumps(table_rows, indent=2, default=float))
    print("wrote", summary_path)
    return {"case_info": case_info, "specs": specs, "table": table_rows}


if __name__ == "__main__":
    main()
