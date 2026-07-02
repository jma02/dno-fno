"""Audit training data + per-regime IC distributions vs. training.

Memory-safe: uses np.load(mmap_mode='r') and samples random indices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/johnma/dno-fno")
DATA = ROOT / "data/combined_dataset_v3.npz"
META = json.loads((ROOT / "data/combined_dataset_v3.meta.json").read_text())
LX = float(META["length"])  # 2 pi
NX = int(META["nx"])
DX = LX / NX
KX = np.fft.fftfreq(NX, d=DX) * 2.0 * np.pi  # physical wavenumbers

REGIMES = [
    "tanaka_g0", "tanaka_g1", "bf_g0", "bf_g1", "bf_modal",
    "linear", "stokes_deep", "stokes_finite",
    "random_sea_deep", "random_sea_finite",
]
SOURCE_LEGEND = {int(k): v for k, v in META["source_legend"].items()}
SOURCE_NAME_TO_ID = {v: int(k) for k, v in META["source_legend"].items()}
FNO_RUN = ROOT / "outputs/fno_w128b6_v3_hclip5_20260514_063811"

RNG = np.random.default_rng(20260604)
N_SAMPLE = 1000


def percentiles(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def histogram(x: np.ndarray, bins: int = 30) -> dict:
    hist, edges = np.histogram(x, bins=bins)
    return {"counts": hist.tolist(), "edges": edges.tolist()}


def eta_xi_stats(eta: np.ndarray, xi: np.ndarray) -> dict[str, np.ndarray]:
    # eta, xi shape (N, NX)
    eta_amp = np.max(np.abs(eta), axis=-1)
    xi_amp = np.max(np.abs(xi), axis=-1)
    # spatial derivative via spectral
    eta_hat = np.fft.fft(eta, axis=-1)
    deta_dx = np.real(np.fft.ifft(1j * KX[None, :] * eta_hat, axis=-1))
    steep = np.max(np.abs(deta_dx), axis=-1)
    # dominant wavenumber (positive half spectrum). Skip k=0.
    power = np.abs(eta_hat[:, : NX // 2]) ** 2
    power[:, 0] = 0.0
    kx_pos = KX[: NX // 2].copy()
    k_idx = np.argmax(power, axis=-1)
    k_dom = np.abs(kx_pos[k_idx])
    return {
        "eta_amp": eta_amp.astype(np.float32),
        "xi_amp": xi_amp.astype(np.float32),
        "steep": steep.astype(np.float32),
        "k_dom": k_dom.astype(np.float32),
    }


def summarize(arrs: dict[str, np.ndarray], *, with_hist: bool = False) -> dict:
    out: dict[str, dict] = {}
    for k, v in arrs.items():
        v = np.asarray(v).reshape(-1)
        out[k] = {"pct": percentiles(v)}
        if with_hist:
            out[k]["hist"] = histogram(v)
    return out


def ood_flag(value: float, ref: dict[str, float]) -> str:
    lo = ref["p05"]
    hi = ref["p95"]
    if value > ref["max"]:
        return "OUTSIDE_MAX"
    if value < ref["min"]:
        return "OUTSIDE_MIN"
    if value > hi:
        return "above_p95"
    if value < lo:
        return "below_p05"
    return "ok"


def audit_training() -> dict:
    archive = np.load(DATA, mmap_mode="r")
    eta = archive["eta"]
    xi = archive["xi"]
    gxi = archive["gxi"]
    depth = archive["depth"]
    times = archive["time"]
    source = archive["source"]
    n_total = eta.shape[0]
    print(f"training: n={n_total} nx={eta.shape[1]}")

    # global samples
    idx = np.sort(RNG.choice(n_total, size=N_SAMPLE, replace=False))
    eta_s = np.asarray(eta[idx])
    xi_s = np.asarray(xi[idx])
    gxi_s = np.asarray(gxi[idx])

    nan_eta = int(np.sum(~np.isfinite(eta_s)))
    nan_xi = int(np.sum(~np.isfinite(xi_s)))
    nan_gxi = int(np.sum(~np.isfinite(gxi_s)))

    stats = eta_xi_stats(eta_s, xi_s)
    stats["gxi_amp"] = np.max(np.abs(gxi_s), axis=-1).astype(np.float32)
    stats["depth_sample"] = np.asarray(depth[idx], dtype=np.float32)
    stats["time_sample"] = np.asarray(times[idx], dtype=np.float32)

    train_summary = summarize(stats, with_hist=True)
    train_summary["n_sampled"] = N_SAMPLE
    train_summary["n_total"] = n_total
    train_summary["nan_inf_counts"] = {
        "eta_voxels": nan_eta, "xi_voxels": nan_xi, "gxi_voxels": nan_gxi,
        "total_voxels": int(eta_s.size),
    }

    # per-source breakdown — sample inside each source
    src_idx_all = np.asarray(source[:])  # int8 vector, full but only 6M bytes
    per_source: dict[str, dict] = {}
    for sid, name in SOURCE_LEGEND.items():
        mask = np.where(src_idx_all == sid)[0]
        if mask.size == 0:
            continue
        take = RNG.choice(mask, size=min(N_SAMPLE, mask.size), replace=False)
        take = np.sort(take)
        e = np.asarray(eta[take])
        x = np.asarray(xi[take])
        d = np.asarray(depth[take], dtype=np.float32)
        t = np.asarray(times[take], dtype=np.float32)
        st = eta_xi_stats(e, x)
        st["depth"] = d
        st["time"] = t
        per_source[name] = summarize(st)
        per_source[name]["n_total"] = int(mask.size)
    train_summary["per_source"] = per_source

    # time-step / time gap analysis: per-source time histogram already in per_source
    return train_summary


def audit_rollouts(train_summary: dict) -> dict:
    out: dict[str, dict] = {}
    train_pct = {k: train_summary[k]["pct"] for k in ["eta_amp", "xi_amp", "steep", "k_dom"]}
    for regime in REGIMES:
        traj_path = FNO_RUN / "eval_suite" / f"{regime}_trajs.npz"
        if not traj_path.exists():
            out[regime] = {"error": "missing trajs file"}
            continue
        z = np.load(traj_path, mmap_mode="r")
        # initial conditions
        eta0 = np.asarray(z["truth_eta"][0])  # (n_ICs, nx)
        xi0 = np.asarray(z["truth_xi"][0])
        times = np.asarray(z["times"])
        depths = np.asarray(z["depths"]) if "depths" in z.files else None

        # finite checks across truth
        truth_eta_full = np.asarray(z["truth_eta"])
        truth_xi_full = np.asarray(z["truth_xi"])
        n_nan_truth = int(np.sum(~np.isfinite(truth_eta_full))) + int(np.sum(~np.isfinite(truth_xi_full)))
        # divergence in truth across time: max abs
        truth_eta_max_t = np.max(np.abs(truth_eta_full), axis=(1, 2))  # (T,)
        truth_xi_max_t = np.max(np.abs(truth_xi_full), axis=(1, 2))

        st = eta_xi_stats(eta0, xi0)
        flags = {}
        for key in ["eta_amp", "xi_amp", "steep", "k_dom"]:
            v = st[key]
            mn, md, mx = float(np.min(v)), float(np.median(v)), float(np.max(v))
            ref = train_pct[key]
            flags[key] = {
                "min": mn, "median": md, "max": mx, "p95": float(np.percentile(v, 95)),
                "train_p05": ref["p05"], "train_p95": ref["p95"],
                "verdict_median": ood_flag(md, ref),
                "verdict_max": ood_flag(mx, ref),
            }
        dt = float(times[1] - times[0]) if times.size > 1 else 0.0
        out[regime] = {
            "n_ICs": int(eta0.shape[0]),
            "T": int(times.size),
            "dt": dt,
            "depth": {
                "min": float(np.min(depths)) if depths is not None else None,
                "max": float(np.max(depths)) if depths is not None else None,
            },
            "ic_flags": flags,
            "truth_nan_inf": n_nan_truth,
            "truth_eta_max_t_first_last": [float(truth_eta_max_t[0]), float(truth_eta_max_t[-1])],
            "truth_xi_max_t_first_last": [float(truth_xi_max_t[0]), float(truth_xi_max_t[-1])],
        }
    return out


def main() -> None:
    print("auditing training...")
    train = audit_training()
    print("auditing rollouts...")
    rollouts = audit_rollouts(train)
    payload = {
        "meta": {
            "data_path": str(DATA),
            "n_total": train["n_total"],
            "n_sampled_global": train["n_sampled"],
            "n_sampled_per_source": N_SAMPLE,
            "nx": NX, "Lx": LX,
        },
        "training": {k: v for k, v in train.items() if k != "per_source"},
        "training_per_source": train["per_source"],
        "rollouts": rollouts,
    }
    out_path = ROOT / "playground/diagnostics/data_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out_path}")

    # Print compact human-readable summary
    print("\n=== TRAINING (global sample) ===")
    g = train
    for k in ["eta_amp", "xi_amp", "steep", "k_dom", "gxi_amp", "depth_sample"]:
        p = g[k]["pct"]
        print(f"  {k:12s}  min={p['min']:+.4g}  p05={p['p05']:+.4g}  p50={p['p50']:+.4g}  p95={p['p95']:+.4g}  p99={p['p99']:+.4g}  max={p['max']:+.4g}")
    print("  nan/inf:", g["nan_inf_counts"])

    print("\n=== ROLLOUT IC vs TRAINING ===")
    for regime, r in rollouts.items():
        if "error" in r:
            print(f"  {regime}: MISSING")
            continue
        fl = r["ic_flags"]
        ds = r["depth"]
        print(f"\n  [{regime}] n_ICs={r['n_ICs']} T={r['T']} dt={r['dt']:.4g} depth=[{ds['min']:.3g},{ds['max']:.3g}]")
        for key in ["eta_amp", "xi_amp", "steep", "k_dom"]:
            f = fl[key]
            print(f"    {key:8s} med={f['median']:+.4g} max={f['max']:+.4g} "
                  f"train_[p05,p95]=[{f['train_p05']:+.4g},{f['train_p95']:+.4g}] "
                  f"verdict_med={f['verdict_median']} verdict_max={f['verdict_max']}")


if __name__ == "__main__":
    main()
