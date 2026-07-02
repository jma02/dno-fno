"""Batched rollout test suite for FNO / SpectralDNO surrogates.

For each regime (linear, stokes_deep, stokes_finite, random_sea_finite,
random_sea_deep, tanaka_g0, tanaka_g1, bf_g0, bf_g1, bf_modal), pulls N ICs
from the corresponding gen-script data file (which contain validated t=0 ICs),
rolls each out under both the analytic-DNO solver (truth, batched float64) and
the trained surrogate (per-IC float32), and saves full trajectories + summary
metrics to ``<run_dir>/eval_suite/``.

Canonical rollout params come from the data file's meta.json (dt=0.08, f64,
filter=0.25, substeps=8, implicit_iterations=4, gl2_if). Linear/stokes are
analytic regimes with no time-rollout in their gen scripts; we pick dt=0.08
tmax=20 for the surrogate test and reproduce the truth at the same settings.

Typical use:
    uv run python -m solver.evals.eval_suite \\
        --run_dir outputs/fno_w128b6_v3_hclip5_20260514_063811 \\
        --n_ics 16 --gpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402
from solver.evals.model_rollout import (  # noqa: E402
    LoadedRun,
    build_predict_gxi_with_depth,
    load_run,
    rollout_surrogate,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


@dataclass
class IC:
    eta: np.ndarray
    xi: np.ndarray
    depth: float
    case_id: int
    meta: dict = field(default_factory=dict)


@dataclass
class RegimeConfig:
    name: str
    # Source spec: either ('batched_npz', path) for gen-script batched files
    # (bf/tanaka/random_sea), or ('test_npz', path, source_id) for the rescaled
    # legacy test set (linear/stokes_*).
    source: tuple
    # Rollout params (None means inherit from data meta where present)
    dt: float | None = None
    tmax: float | None = None
    n_ics: int = 16
    # Inner stepping: total inner dt = dt / substeps. Keep inner_dt at 0.01 to
    # match gen-script integration order. For long horizons we coarsen the
    # OUTER (save) dt and grow substeps proportionally so memory stays bounded.
    substeps: int = 8
    implicit_iters: int = 4
    filter_fraction: float = 0.25
    # "nonlinear" (default): full Zakharov + Craig-Sulem via time_integrator.
    # "linear_analytic": per-mode Fourier propagation under linearized Zakharov
    # (eta'=G_lin xi, xi'=-g eta) — the correct truth for single-mode linear ICs
    # whose nominal validity is linear-only.
    truth_kind: str = "nonlinear"


REGISTRY: dict[str, RegimeConfig] = {
    # Gen-script batched files. Short horizon (tmax=20) keeps dt=0.08 substeps=8.
    # Long horizon (tmax=200) coarsens save dt to 0.8 and grows substeps to 80 so
    # inner_dt stays 0.01 (matches gen-script integration order) but n_t drops 10x.
    "random_sea_finite": RegimeConfig(
        name="random_sea_finite",
        source=("batched_npz", DATA_DIR / "random_sea_finite.npz"),
    ),
    "random_sea_deep": RegimeConfig(
        name="random_sea_deep",
        source=("batched_npz", DATA_DIR / "random_sea_deep.npz"),
    ),
    "tanaka_g0": RegimeConfig(
        name="tanaka_g0",
        source=("batched_npz", DATA_DIR / "tanaka_2_adaptive_g0.npz"),
        dt=0.8, tmax=200.0, substeps=80,
    ),
    "tanaka_g1": RegimeConfig(
        name="tanaka_g1",
        source=("batched_npz", DATA_DIR / "tanaka_2_adaptive_g1.npz"),
        dt=0.8, tmax=200.0, substeps=80,
    ),
    "bf_g0": RegimeConfig(
        name="bf_g0",
        source=("batched_npz", DATA_DIR / "bf_2_adaptive_g0.npz"),
        dt=0.8, tmax=200.0, substeps=80,
    ),
    "bf_g1": RegimeConfig(
        name="bf_g1",
        source=("batched_npz", DATA_DIR / "bf_2_adaptive_g1.npz"),
        dt=0.8, tmax=200.0, substeps=80,
    ),
    "bf_modal": RegimeConfig(
        name="bf_modal",
        source=("batched_npz", DATA_DIR / "bf_2_adaptive_modal_shard_00.npz"),
        dt=0.8, tmax=200.0, substeps=80,
    ),
    # Legacy MATLAB test set, rescaled to L=2pi. No canonical rollout so we pick dt=0.08, tmax=20.
    # source_legend: 0=soliton, 1=stokes_deep, 2=stokes_finite, 3=linear
    "linear": RegimeConfig(
        name="linear",
        source=("test_npz", DATA_DIR / "test_dno_rescaled.npz", 3),
        dt=0.08, tmax=20.0,
    ),
    "stokes_deep": RegimeConfig(
        name="stokes_deep",
        source=("test_npz", DATA_DIR / "test_dno_rescaled.npz", 1),
        dt=0.08, tmax=20.0,
    ),
    "stokes_finite": RegimeConfig(
        name="stokes_finite",
        source=("test_npz", DATA_DIR / "test_dno_rescaled.npz", 2),
        dt=0.08, tmax=20.0,
    ),
}


def _load_batched_ics(path: Path, n_ics: int) -> tuple[list[IC], dict, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Pull t=0 ICs from a gen-script batched file (batch_0000, first N cases).

    Returns (ics, meta, saved_times_per_case, saved_eta_per_case).
    """
    with np.load(path, mmap_mode="r") as d:
        meta = json.loads(bytes(d["meta.json"]).decode())
        case_id = np.asarray(d["case_id_batch_0000"])
        depth_arr = np.asarray(d["depth_batch_0000"])
        time_arr = np.asarray(d["time_batch_0000"])
        eta_batch = np.asarray(d["eta_batch_0000"])
        xi_batch = np.asarray(d["xi_batch_0000"])

    case_ids = sorted(np.unique(case_id))[:n_ics]
    saved_times: dict[int, np.ndarray] = {}
    saved_eta: dict[int, np.ndarray] = {}
    ics: list[IC] = []
    for c in case_ids:
        rows = np.nonzero(case_id == c)[0]
        row0 = int(rows[0])
        ics.append(
            IC(
                eta=eta_batch[row0].astype(np.float64),
                xi=xi_batch[row0].astype(np.float64),
                depth=float(depth_arr[row0]),
                case_id=int(c),
                meta={"source_file": path.name},
            )
        )
        saved_times[int(c)] = np.asarray(time_arr[rows], dtype=np.float64)
        saved_eta[int(c)] = np.asarray(eta_batch[rows], dtype=np.float64)
    return ics, meta, saved_times, saved_eta


def _load_test_npz_ics(path: Path, source_id: int, n_ics: int) -> list[IC]:
    """Pull ICs from the rescaled-MATLAB test set filtered by source label."""
    with np.load(path, mmap_mode="r") as d:
        meta = json.loads(path.with_suffix(".meta.json").read_text()) if path.with_suffix(".meta.json").exists() else {}
        source = np.asarray(d["source"])
        depth = np.asarray(d["depth"])
        eta = np.asarray(d["eta"])
        xi = np.asarray(d["xi"])
    rows = np.nonzero(source == source_id)[0][:n_ics]
    if rows.size == 0:
        raise SystemExit(f"no source={source_id} ICs in {path}")
    return [
        IC(
            eta=eta[r].astype(np.float64),
            xi=xi[r].astype(np.float64),
            depth=float(depth[r]),
            case_id=int(r),
            meta={"source_file": path.name, "source_id": int(source_id)},
        )
        for r in rows
    ]


def build_ics_and_truth_targets(
    cfg: RegimeConfig,
) -> tuple[list[IC], float, float, dict[int, np.ndarray] | None, dict[int, np.ndarray] | None]:
    """Returns (ics, dt, tmax, saved_times, saved_eta_truth).

    For batched_npz: dt/tmax come from data meta; saved_* are the adaptive-subsampled
    trajectories (used as ground-truth reference). For test_npz: we override dt/tmax
    from RegimeConfig and there's no saved truth.
    """
    kind = cfg.source[0]
    if kind == "batched_npz":
        ics, meta, saved_t, saved_eta = _load_batched_ics(cfg.source[1], cfg.n_ics)
        # Allow override (lets us coarsen save dt while keeping inner_dt = dt/substeps).
        dt = float(cfg.dt) if cfg.dt is not None else float(meta["dt"])
        tmax = float(cfg.tmax) if cfg.tmax is not None else float(meta["tmax"])
        return ics, dt, tmax, saved_t, saved_eta
    if kind == "test_npz":
        ics = _load_test_npz_ics(cfg.source[1], int(cfg.source[2]), cfg.n_ics)
        if cfg.dt is None or cfg.tmax is None:
            raise SystemExit(f"test_npz regime '{cfg.name}' must specify dt and tmax")
        return ics, cfg.dt, cfg.tmax, None, None
    raise SystemExit(f"unknown source kind '{kind}' for regime {cfg.name}")


def truth_rollout_batched(
    ics: list[IC],
    times_f64: jnp.ndarray,
    nx: int,
    length: float,
    cfg: RegimeConfig,
) -> dict[str, np.ndarray]:
    """Batched f64 truth rollout across all ICs (one JIT, NB-broadcast)."""
    dtype = jnp.float64
    eta0 = np.stack([ic.eta for ic in ics], axis=0)
    xi0 = np.stack([ic.xi for ic in ics], axis=0)
    depths = np.asarray([ic.depth for ic in ics], dtype=np.float64)

    _, k_grid = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid, dtype=dtype)
    depth_2d = jnp.asarray(depths, dtype=dtype)[:, None]
    g0 = make_linear_dno_symbol(k_grid, depth_2d)
    sp = ti.SolverParams(
        nx=nx, length=length, depth=depth_2d, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=cfg.filter_fraction,
        k=k_grid, g0=g0,
    )
    state = ti.State(eta=jnp.asarray(eta0, dtype=dtype), xi=jnp.asarray(xi0, dtype=dtype))
    t0 = time.perf_counter()
    out = ti.batched_rollout(
        state, times_f64, sp, save_gxi=True,
        substeps_per_interval=cfg.substeps, method="gl2_if",
        implicit_iterations=cfg.implicit_iters, zero_mean_xi=True,
    )
    jax.block_until_ready(out["eta"])
    wall = time.perf_counter() - t0
    return {
        "eta": np.asarray(out["eta"]).astype(np.float32),    # (n_t, NB, nx)
        "xi": np.asarray(out["xi"]).astype(np.float32),
        "gxi": np.asarray(out["gxi"]).astype(np.float32),
        "wall_s": float(wall),
    }


def truth_rollout_linear_analytic(
    ics: list[IC],
    times: np.ndarray,
    nx: int,
    length: float,
    gravity: float = 1.0,
) -> dict[str, np.ndarray]:
    """Analytic per-mode Fourier propagation of the linearized Zakharov system.

    Solves d_t eta = G_lin xi, d_t xi = -g eta with G_lin(k) = k * tanh(h k).
    Per mode k: omega_k^2 = g * G_lin(k), and
        eta_k(t) = eta_k(0) cos(w t) + (G_lin/w) xi_k(0) sin(w t)
        xi_k(t)  = xi_k(0)  cos(w t) - (g/w)     eta_k(0) sin(w t)
    For k=0 (omega=0): eta_0(t) = eta_0(0), xi_0(t) = xi_0(0) - g eta_0(0) t.
    Returns float32 arrays shaped (n_t, NB, nx).
    """
    NB = len(ics)
    n_t = times.shape[0]
    eta0 = np.stack([ic.eta for ic in ics], axis=0).astype(np.float64)
    xi0 = np.stack([ic.xi for ic in ics], axis=0).astype(np.float64)
    depths = np.asarray([ic.depth for ic in ics], dtype=np.float64)[:, None]

    _, k_grid = build_grid(nx, length)
    k_rfft = np.fft.rfftfreq(nx, d=length / nx) * (2.0 * np.pi)
    k_rfft = k_rfft[None, :]
    G_lin = k_rfft * np.tanh(depths * k_rfft)
    omega = np.sqrt(gravity * G_lin)
    omega_safe = np.where(omega > 0, omega, 1.0)

    eta_hat0 = np.fft.rfft(eta0, axis=-1)
    xi_hat0 = np.fft.rfft(xi0, axis=-1)

    eta_out = np.empty((n_t, NB, nx), dtype=np.float32)
    xi_out = np.empty((n_t, NB, nx), dtype=np.float32)
    gxi_out = np.empty((n_t, NB, nx), dtype=np.float32)
    t0 = time.perf_counter()
    for ti_, t in enumerate(times):
        c, s = np.cos(omega * t), np.sin(omega * t)
        eta_hat = eta_hat0 * c + (G_lin / omega_safe) * xi_hat0 * s
        xi_hat = xi_hat0 * c - (gravity / omega_safe) * eta_hat0 * s
        # k=0: omega=0 so above formulas degenerate; xi_0 picks up -g*eta_0*t.
        xi_hat[:, 0] = xi_hat0[:, 0] - gravity * eta_hat0[:, 0] * t
        eta_hat[:, 0] = eta_hat0[:, 0]
        gxi_hat = G_lin * xi_hat
        eta_out[ti_] = np.fft.irfft(eta_hat, n=nx, axis=-1).astype(np.float32)
        xi_out[ti_] = np.fft.irfft(xi_hat, n=nx, axis=-1).astype(np.float32)
        gxi_out[ti_] = np.fft.irfft(gxi_hat, n=nx, axis=-1).astype(np.float32)
    wall = time.perf_counter() - t0
    return {"eta": eta_out, "xi": xi_out, "gxi": gxi_out, "wall_s": float(wall)}


def surrogate_rollout_per_ic(
    ics: list[IC],
    times_f32: jnp.ndarray,
    nx: int,
    length: float,
    cfg: RegimeConfig,
    loaded: LoadedRun,
    predict_gxi_with_depth: Callable,
    filter_gxi: bool = False,
    f64_harness: bool = False,
    filter_shape: str = "hard",
    houli_a: float = 36.0,
    houli_m: float = 36.0,
    cascade_gate_enabled: bool = False,
    cascade_k_cut: float = 32.0,
    cascade_r_threshold: float = 1e-3,
    cascade_sharpness: float = 10.0,
    cascade_houli_a: float = 0.69,
    cascade_houli_m: float = 4.0,
    cascade_k_eff: float = 128.0,
    cascade_filter_xi: bool = True,
) -> dict[str, np.ndarray]:
    """Per-IC surrogate rollout. One JIT covers all ICs (log_h is a jit arg).

    The model is always applied in f32. With f64_harness the integrator state,
    spectral ops, and linear flow run in f64, isolating model error from the
    harness's own f32 phase-drift floor (~0.11 rel-L2 at tanaka t=200).
    """
    n_t = int(times_f32.shape[0])
    NB = len(ics)
    pred_eta = np.empty((n_t, NB, nx), dtype=np.float32)
    pred_xi = np.empty((n_t, NB, nx), dtype=np.float32)
    pred_gxi = np.empty((n_t, NB, nx), dtype=np.float32)

    dtype = jnp.float64 if f64_harness else jnp.float32
    _, k_grid_np = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid_np, dtype=dtype)
    times = times_f32.astype(dtype)
    filt = cfg.filter_fraction
    substeps = cfg.substeps
    impl = cfg.implicit_iters

    @jax.jit
    def jitted(eta0: jnp.ndarray, xi0: jnp.ndarray, depth: jnp.ndarray, log_h: jnp.ndarray):
        g0 = make_linear_dno_symbol(k_grid, depth)
        sp = ti.SolverParams(
            nx=nx, length=length, depth=depth, gravity=1.0,
            dno_order=6, pad_factor=8, filter_fraction=filt, k=k_grid, g0=g0,
        )
        state0 = ti.State(eta=eta0, xi=xi0)

        def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
            gxi = predict_gxi_with_depth(
                eta.astype(jnp.float32), xi.astype(jnp.float32), log_h
            ).astype(dtype)
            if filter_gxi and filt < 1.0:
                gxi = ti.apply_filter(
                    gxi, k_grid, shape=filter_shape, filter_fraction=filt,
                    houli_a=houli_a, houli_m=houli_m,
                )
            return gxi

        out = rollout_surrogate(
            state0, times, sp, predict,
            substeps=substeps, zero_mean_xi=True, gl2_iterations=impl,
            filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m,
            cascade_gate_enabled=cascade_gate_enabled,
            cascade_k_cut=cascade_k_cut,
            cascade_r_threshold=cascade_r_threshold,
            cascade_sharpness=cascade_sharpness,
            cascade_houli_a=cascade_houli_a,
            cascade_houli_m=cascade_houli_m,
            cascade_k_eff=cascade_k_eff,
            cascade_filter_xi=cascade_filter_xi,
        )
        gxi_traj = jax.vmap(predict)(out["eta"], out["xi"])
        return out["eta"], out["xi"], gxi_traj

    total_wall = 0.0
    for j, ic in enumerate(ics):
        eta0 = jnp.asarray(ic.eta, dtype=dtype)
        xi0 = jnp.asarray(ic.xi, dtype=dtype)
        depth = jnp.asarray(ic.depth, dtype=dtype)
        log_h = jnp.asarray(np.log(max(ic.depth, 1e-12)), dtype=jnp.float32)
        t0 = time.perf_counter()
        eta_t, xi_t, gxi_t = jitted(eta0, xi0, depth, log_h)
        jax.block_until_ready(gxi_t)
        total_wall += time.perf_counter() - t0
        pred_eta[:, j, :] = np.asarray(eta_t)
        pred_xi[:, j, :] = np.asarray(xi_t)
        pred_gxi[:, j, :] = np.asarray(gxi_t)
    return {"eta": pred_eta, "xi": pred_xi, "gxi": pred_gxi, "wall_s": float(total_wall)}


def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """rel-L2 over axis=-1 (per (time, IC) pair)."""
    return np.linalg.norm(pred - truth, axis=-1) / (np.linalg.norm(truth, axis=-1) + 1e-12)


TRUTH_DRIFT_TOL = 1e-3  # healthy GL2 truth conserves energy to ~1e-7; >1e-3 means the truth itself blew up


def compute_metrics(
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    times: np.ndarray,
    length: float,
) -> dict:
    """All rel-L2 arrays have shape (n_t, NB). Aggregates are over truth-valid ICs only."""
    # shape (n_t, NB)
    re = _rel_l2(pred["eta"], truth["eta"])
    rx = _rel_l2(pred["xi"], truth["xi"])
    rg = _rel_l2(pred["gxi"], truth["gxi"])
    n_t, NB = re.shape

    nx = truth["eta"].shape[-1]
    dx = length / nx
    H_truth = 0.5 * np.sum(truth["xi"] * truth["gxi"] + truth["eta"] ** 2, axis=-1) * dx  # (n_t, NB)
    H_pred = 0.5 * np.sum(pred["xi"] * pred["gxi"] + pred["eta"] ** 2, axis=-1) * dx
    H0 = H_truth[:1, :]
    drift_pred = (H_pred - H0) / (np.abs(H0) + 1e-12)
    drift_truth = (H_truth - H0) / (np.abs(H0) + 1e-12)

    # An IC whose truth goes non-finite or violates its own energy conservation
    # cannot score the model; exclude it from every aggregate.
    truth_nonfinite = (
        ~np.isfinite(truth["eta"]).all(axis=(0, 2))
        | ~np.isfinite(truth["xi"]).all(axis=(0, 2))
        | ~np.isfinite(truth["gxi"]).all(axis=(0, 2))
    )
    drift_abs = np.abs(drift_truth)
    drift_abs[~np.isfinite(drift_abs)] = np.inf
    truth_invalid = truth_nonfinite | (drift_abs.max(axis=0) > TRUTH_DRIFT_TOL)
    valid = ~truth_invalid

    nan_per_ic = (np.isnan(re).any(axis=0) | np.isnan(rx).any(axis=0) | np.isnan(rg).any(axis=0))
    nan_rate = float(np.mean(nan_per_ic[valid]))
    diverged = (re[-1] > 1.0) | np.isnan(re[-1])
    div_rate = float(np.mean(diverged[valid]))

    out = {
        "n_t": int(n_t), "NB": int(NB),
        "n_truth_valid": int(valid.sum()),
        "truth_invalid_ics": [int(j) for j in np.where(truth_invalid)[0]],
        "truth_drift_tol": TRUTH_DRIFT_TOL,
        "nan_rate": nan_rate, "divergence_rate_final": div_rate,
    }
    for frac, lbl in ((0.1, "t10p"), (0.5, "t50p"), (1.0, "tfinal")):
        hi = int(min(n_t - 1, max(0, int(n_t * frac) - 1)))
        out[f"rel_l2_eta_mean_{lbl}"] = float(np.nanmean(re[hi, valid]))
        out[f"rel_l2_eta_median_{lbl}"] = float(np.nanmedian(re[hi, valid]))
        out[f"rel_l2_eta_p95_{lbl}"] = float(np.nanpercentile(re[hi, valid], 95))
        out[f"rel_l2_xi_mean_{lbl}"] = float(np.nanmean(rx[hi, valid]))
        out[f"rel_l2_gxi_mean_{lbl}"] = float(np.nanmean(rg[hi, valid]))
    out["energy_drift_pred_p95_at_tfinal"] = float(np.nanpercentile(np.abs(drift_pred[-1, valid]), 95))
    out["energy_drift_pred_median_at_tfinal"] = float(np.nanmedian(np.abs(drift_pred[-1, valid])))
    out["_arrays"] = {
        "rel_l2_eta": re, "rel_l2_xi": rx, "rel_l2_gxi": rg,
        "energy_drift_pred": drift_pred, "energy_drift_truth": drift_truth,
    }
    return out


def run_regime(
    regime: str,
    cfg: RegimeConfig,
    loaded: LoadedRun,
    predict_gxi: Callable,
    out_dir: Path,
    nx: int,
    length: float,
    filter_gxi: bool = False,
    f64_harness: bool = False,
    filter_shape: str = "hard",
    houli_a: float = 36.0,
    houli_m: float = 36.0,
    cascade_gate_enabled: bool = False,
    cascade_k_cut: float = 32.0,
    cascade_r_threshold: float = 1e-3,
    cascade_sharpness: float = 10.0,
    cascade_houli_a: float = 0.69,
    cascade_houli_m: float = 4.0,
    cascade_k_eff: float = 128.0,
    cascade_filter_xi: bool = True,
) -> dict:
    ics, dt, tmax, saved_t, saved_eta = build_ics_and_truth_targets(cfg)
    times_np = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=np.float64)
    n_t = int(times_np.shape[0])
    print(
        f"[{regime}] NB={len(ics)} dt={dt} tmax={tmax} n_t={n_t} "
        f"filter={cfg.filter_fraction} substeps={cfg.substeps} impl={cfg.implicit_iters}",
        flush=True,
    )

    if cfg.truth_kind == "linear_analytic":
        print(f"[{regime}] truth rollout (analytic linear Fourier propagation)...", flush=True)
        truth = truth_rollout_linear_analytic(ics, times_np, nx, length)
    else:
        print(f"[{regime}] truth rollout (batched f64)...", flush=True)
        truth = truth_rollout_batched(ics, jnp.asarray(times_np, dtype=jnp.float64), nx, length, cfg)
    print(f"[{regime}]   truth wall = {truth['wall_s']:.1f}s", flush=True)

    harness = "f64 harness / f32 model" if f64_harness else "f32"
    print(f"[{regime}] surrogate rollout (per-IC {harness})...", flush=True)
    times_in = jnp.asarray(times_np, dtype=jnp.float64 if f64_harness else jnp.float32)
    pred = surrogate_rollout_per_ic(
        ics, times_in, nx, length, cfg, loaded, predict_gxi,
        filter_gxi=filter_gxi, f64_harness=f64_harness,
        filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m,
        cascade_gate_enabled=cascade_gate_enabled,
        cascade_k_cut=cascade_k_cut,
        cascade_r_threshold=cascade_r_threshold,
        cascade_sharpness=cascade_sharpness,
        cascade_houli_a=cascade_houli_a,
        cascade_houli_m=cascade_houli_m,
        cascade_k_eff=cascade_k_eff,
        cascade_filter_xi=cascade_filter_xi,
    )
    print(f"[{regime}]   surrogate wall = {pred['wall_s']:.1f}s", flush=True)

    # Optional: sanity-check truth against saved adaptive samples (only batched_npz regimes)
    saved_err: list[dict] = []
    if saved_t is not None and saved_eta is not None:
        for j, ic in enumerate(ics):
            st = saved_t[ic.case_id]
            tgt = saved_eta[ic.case_id]
            idx = np.clip(np.round(st / dt).astype(np.int64), 0, n_t - 1)
            rolled = truth["eta"][idx, j, :].astype(np.float64)
            err = np.linalg.norm(rolled - tgt, axis=1) / (np.linalg.norm(tgt, axis=1) + 1e-12)
            saved_err.append({
                "case_id": ic.case_id, "depth": ic.depth,
                "saved_err_max": float(np.nanmax(err)), "saved_err_median": float(np.nanmedian(err)),
            })

    metrics = compute_metrics(truth, pred, times_np, length)

    np.savez_compressed(
        out_dir / f"{regime}_trajs.npz",
        times=times_np.astype(np.float32),
        depths=np.asarray([ic.depth for ic in ics], dtype=np.float32),
        case_ids=np.asarray([ic.case_id for ic in ics], dtype=np.int64),
        truth_eta=truth["eta"], truth_xi=truth["xi"], truth_gxi=truth["gxi"],
        pred_eta=pred["eta"], pred_xi=pred["xi"], pred_gxi=pred["gxi"],
        rel_l2_eta=metrics["_arrays"]["rel_l2_eta"],
        rel_l2_xi=metrics["_arrays"]["rel_l2_xi"],
        rel_l2_gxi=metrics["_arrays"]["rel_l2_gxi"],
        energy_drift_pred=metrics["_arrays"]["energy_drift_pred"],
        energy_drift_truth=metrics["_arrays"]["energy_drift_truth"],
    )

    summary = {k: v for k, v in metrics.items() if k != "_arrays"}
    summary.update({
        "regime": regime, "dt": dt, "tmax": tmax, "n_t": n_t, "length": length, "nx": nx,
        "n_ics": len(ics),
        "depths": [float(ic.depth) for ic in ics],
        "case_ids": [int(ic.case_id) for ic in ics],
        "epoch": loaded.epoch,
        "truth_wall_s": truth["wall_s"],
        "surrogate_wall_s": pred["wall_s"],
        "filter_fraction": cfg.filter_fraction,
        "filter_gxi": filter_gxi,
        "f64_harness": f64_harness,
        "substeps": cfg.substeps,
        "implicit_iters": cfg.implicit_iters,
        "cascade_gate_enabled": cascade_gate_enabled,
        "cascade_k_cut": cascade_k_cut,
        "cascade_r_threshold": cascade_r_threshold,
        "cascade_sharpness": cascade_sharpness,
        "cascade_houli_a": cascade_houli_a,
        "cascade_houli_m": cascade_houli_m,
        "cascade_k_eff": cascade_k_eff,
        "cascade_filter_xi": cascade_filter_xi,
        "saved_truth_reproduction": saved_err,
    })
    (out_dir / f"{regime}_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{regime}] done. NaN={summary['nan_rate']:.2f} div={summary['divergence_rate_final']:.2f} "
        f"η_med_tf={summary['rel_l2_eta_median_tfinal']:.4g} η_p95_tf={summary['rel_l2_eta_p95_tfinal']:.4g}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-regime rollout test suite at gen-script params.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--regimes", nargs="+", default=list(REGISTRY.keys()))
    parser.add_argument("--n_ics", type=int, default=16)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument(
        "--filter_gxi", action="store_true",
        help="Low-pass the surrogate's Gxi prediction at the regime's filter_fraction "
        "before it enters the RHS (matches the state filter already applied each substep).",
    )
    parser.add_argument(
        "--f64_harness", action="store_true",
        help="Integrate the surrogate rollout in f64 (model still applied in f32). "
        "Removes the harness's own f32 phase-drift floor so models can be ranked.",
    )
    parser.add_argument(
        "--f64_model", action="store_true",
        help="Apply the model in f64 at inference (params stay as stored; "
        "JAX promotes mixed-precision matmuls to f64). Diagnostic for f32 noise-floor cascade.",
    )
    parser.add_argument(
        "--filter_shape", choices=("hard", "houli"), default="hard",
        help="Spectral filter shape used by --filter_gxi: hard cutoff (default) or Hou-Li "
        "exponential exp(-a*(k/k_eff)^{2m}) (JCP09 eq 29). 'houli' avoids Gibbs at the cutoff.",
    )
    parser.add_argument("--houli_a", type=float, default=36.0,
                        help="Hou-Li 'a' parameter (JCP09 default 36).")
    parser.add_argument("--houli_m", type=float, default=36.0,
                        help="Hou-Li 'm' parameter (JCP09 default 36).")
    parser.add_argument(
        "--filter_fraction", type=float, default=None,
        help="Override the regime's default filter_fraction (state filter + gxi filter cutoff). "
        "k_eff = filter_fraction * k_max. Default keeps RegimeConfig value (0.25 for tanaka/bf).",
    )
    parser.add_argument(
        "--cascade_gate", action="store_true",
        help="Enable conditional Hou-Li smoothing on state when high-k energy ratio crosses threshold. "
        "Sigmoid-blended in log-r space so the gate is differentiable / safe to JIT.",
    )
    parser.add_argument("--cascade_k_cut", type=float, default=32.0,
                        help="|k| boundary for hi-band energy ratio (default 32).")
    parser.add_argument("--cascade_r_threshold", type=float, default=1e-3,
                        help="Energy-ratio threshold E_hi/E_lo where the gate is at 50%% (default 1e-3).")
    parser.add_argument("--cascade_sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness in log(r) space (default 10).")
    parser.add_argument("--cascade_houli_a", type=float, default=0.69,
                        help="Hou-Li a for the cascade gate's smoothing (default 0.69 ~ ln 2).")
    parser.add_argument("--cascade_houli_m", type=float, default=4.0,
                        help="Hou-Li m for the cascade gate's smoothing (default 4).")
    parser.add_argument("--cascade_k_eff", type=float, default=128.0,
                        help="Hou-Li k_eff for the cascade gate (default 128 ~ 1/4 of Nyquist).")
    parser.add_argument("--cascade_no_filter_xi", action="store_true",
                        help="Apply the cascade gate to eta only; leave xi untouched.")
    args = parser.parse_args()

    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)

    run_dir = Path(args.run_dir).resolve()
    default_name = "eval_suite"
    if args.filter_gxi:
        default_name += "_gxifilt"
        if args.filter_shape == "houli":
            default_name += "_houli"
    if args.filter_fraction is not None:
        default_name += f"_ff{args.filter_fraction:g}".replace(".", "p")
    if args.f64_harness:
        default_name += "_f64h"
    if args.f64_model:
        default_name += "_f64m"
    if args.cascade_gate:
        default_name += f"_cg_kc{args.cascade_k_cut:g}_rt{args.cascade_r_threshold:g}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    if args.f64_model:
        loaded.config["precision"] = "fp64"
    predict_gxi = build_predict_gxi_with_depth(loaded)

    summaries: dict[str, dict] = {}
    for regime in args.regimes:
        if regime not in REGISTRY:
            print(f"unknown regime '{regime}'; skipping", flush=True)
            continue
        cfg = REGISTRY[regime]
        cfg_filter_fraction = args.filter_fraction if args.filter_fraction is not None else cfg.filter_fraction
        cfg = RegimeConfig(
            name=cfg.name, source=cfg.source, dt=cfg.dt, tmax=cfg.tmax,
            n_ics=args.n_ics, substeps=cfg.substeps, implicit_iters=cfg.implicit_iters,
            filter_fraction=cfg_filter_fraction, truth_kind=cfg.truth_kind,
        )
        summaries[regime] = run_regime(
            regime, cfg, loaded, predict_gxi, out_dir, args.nx, args.length,
            filter_gxi=args.filter_gxi, f64_harness=args.f64_harness,
            filter_shape=args.filter_shape, houli_a=args.houli_a, houli_m=args.houli_m,
            cascade_gate_enabled=args.cascade_gate,
            cascade_k_cut=args.cascade_k_cut,
            cascade_r_threshold=args.cascade_r_threshold,
            cascade_sharpness=args.cascade_sharpness,
            cascade_houli_a=args.cascade_houli_a,
            cascade_houli_m=args.cascade_houli_m,
            cascade_k_eff=args.cascade_k_eff,
            cascade_filter_xi=not args.cascade_no_filter_xi,
        )

    (out_dir / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Aggregate -> {out_dir / 'all_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
