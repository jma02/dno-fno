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
    build_predict_gxi_batched,
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


def _try_load_cached_truth(
    regime: str, cache_dir: Path | None, expected_n_ics: int,
    expected_n_t: int, expected_nx: int, expected_case_ids: list[int],
) -> dict[str, np.ndarray] | None:
    """Load a truth dict from `{cache_dir}/{regime}_trajs.npz` if shape+case_ids match; else None."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{regime}_trajs.npz"
    if not path.exists():
        print(f"[{regime}] truth cache miss: {path} not found", flush=True)
        return None
    try:
        with np.load(path) as arch:
            eta = arch["truth_eta"]
            xi = arch["truth_xi"]
            gxi = arch["truth_gxi"]
            case_ids = arch["case_ids"].tolist()
    except (KeyError, ValueError) as e:
        print(f"[{regime}] truth cache miss: {e}", flush=True)
        return None
    if eta.shape != (expected_n_t, expected_n_ics, expected_nx):
        print(f"[{regime}] truth cache shape mismatch: got {eta.shape}, "
              f"expected ({expected_n_t}, {expected_n_ics}, {expected_nx})", flush=True)
        return None
    if list(case_ids) != list(expected_case_ids):
        print(f"[{regime}] truth cache case_ids mismatch", flush=True)
        return None
    print(f"[{regime}] truth cache HIT: {path}", flush=True)
    return {"eta": eta, "xi": xi, "gxi": gxi, "wall_s": 0.0}


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


def _apply_gxi_cascade_gate(
    gxi: jnp.ndarray,
    k_arr: jnp.ndarray,
    *,
    k_cut: float,
    r_threshold: float,
    high_abs_threshold: float,
    sharpness: float,
    houli_a: float,
    houli_m: float,
    k_eff: float,
) -> jnp.ndarray:
    """Conditionally damp model Gxi when its mid/high-k energy surges.

    Uses r = Σ|Gxi_hat(k>=k_cut)|² / Σ|Gxi_hat(k<k_cut)|² as the trigger and
    smoothly blends from identity to a Hou-Li multiplier. Supports both single
    trajectories ``(nx,)`` and batched ``(NB, nx)`` fields.
    """
    gxi_hat = jnp.fft.fft(gxi, axis=-1)
    k_abs = jnp.abs(k_arr)
    mask_hi = k_abs >= k_cut
    e_hi = jnp.sum(jnp.where(mask_hi, jnp.abs(gxi_hat) ** 2, 0.0), axis=-1)
    e_lo = jnp.sum(jnp.where(~mask_hi, jnp.abs(gxi_hat) ** 2, 0.0), axis=-1)
    ratio = e_hi / (e_lo + 1e-30)
    log_ratio = jnp.log(ratio + 1e-30)
    log_threshold = jnp.log(jnp.asarray(r_threshold, dtype=ratio.dtype) + 1e-30)
    ratio_weight = jax.nn.sigmoid(sharpness * (log_ratio - log_threshold))
    high_amp = jnp.sqrt(e_hi)
    high_abs = jnp.asarray(high_abs_threshold, dtype=high_amp.dtype)
    high_weight = jnp.where(
        high_abs > 0.0,
        jax.nn.sigmoid(sharpness * (jnp.log(high_amp + 1e-30) - jnp.log(high_abs))),
        jnp.ones_like(ratio_weight),
    )
    weight = ratio_weight * high_weight
    weight = weight[..., None] if weight.ndim else weight
    k_eff_arr = jnp.asarray(k_eff, dtype=k_abs.dtype)
    houli = jnp.exp(-houli_a * (k_abs / jnp.maximum(k_eff_arr, 1e-12)) ** (2.0 * houli_m))
    blend = (1.0 - weight) + weight * houli
    return jnp.real(jnp.fft.ifft(gxi_hat * blend, axis=-1)).astype(gxi.dtype)


def _apply_gxi_highband_limiter(
    gxi: jnp.ndarray,
    k_arr: jnp.ndarray,
    *,
    k_cut: float,
    r_max: float,
    high_abs_floor: float,
) -> jnp.ndarray:
    """Cap only the high-band part of model Gxi.

    The cap is ``sqrt(E_hi) <= max(high_abs_floor, sqrt(r_max) * sqrt(E_lo))``.
    This preserves all low modes exactly and rescales the high-band Fourier
    coefficients uniformly only when the model output exceeds the envelope.
    Supports both single ``(nx,)`` and batched ``(NB, nx)`` fields.
    """
    gxi_hat = jnp.fft.fft(gxi, axis=-1)
    k_abs = jnp.abs(k_arr)
    mask_hi = k_abs >= k_cut
    e_hi = jnp.sum(jnp.where(mask_hi, jnp.abs(gxi_hat) ** 2, 0.0), axis=-1)
    e_lo = jnp.sum(jnp.where(~mask_hi, jnp.abs(gxi_hat) ** 2, 0.0), axis=-1)
    hi_amp = jnp.sqrt(e_hi)
    lo_amp = jnp.sqrt(e_lo)
    allowed = jnp.maximum(
        jnp.asarray(high_abs_floor, dtype=hi_amp.dtype),
        jnp.sqrt(jnp.asarray(r_max, dtype=hi_amp.dtype)) * lo_amp,
    )
    scale = jnp.minimum(1.0, allowed / (hi_amp + 1e-30))
    scale = scale[..., None] if scale.ndim else scale
    limited_hat = jnp.where(mask_hi, gxi_hat * scale, gxi_hat)
    return jnp.real(jnp.fft.ifft(limited_hat, axis=-1)).astype(gxi.dtype)


def _eta_growth_weight(
    eta: jnp.ndarray,
    k_arr: jnp.ndarray,
    eta_amp0: jnp.ndarray,
    *,
    k_lo: float,
    k_hi: float,
    abs_floor: float,
    growth_factor: float,
    sharpness: float,
    nx: int,
) -> jnp.ndarray:
    eta_hat = jnp.fft.fft(eta, axis=-1)
    k_abs = jnp.abs(k_arr)
    mask = (k_abs >= k_lo) & (k_abs < k_hi)
    amp = jnp.sqrt(jnp.mean(jnp.where(mask, jnp.abs(eta_hat) ** 2, 0.0), axis=-1))
    amp = amp / jnp.asarray(nx, dtype=amp.dtype)
    trigger = jnp.maximum(
        jnp.asarray(abs_floor, dtype=amp.dtype),
        jnp.asarray(growth_factor, dtype=amp.dtype) * eta_amp0,
    )
    log_ratio = jnp.log(amp + 1e-30) - jnp.log(trigger + 1e-30)
    return jax.nn.sigmoid(jnp.asarray(sharpness, dtype=amp.dtype) * log_ratio)


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
    eta_growth_guard_enabled: bool = False,
    eta_growth_guard_k_lo: float = 64.0,
    eta_growth_guard_k_hi: float = 128.0,
    eta_growth_guard_abs_floor: float = 1e-4,
    eta_growth_guard_growth_factor: float = 100.0,
    eta_growth_guard_sharpness: float = 10.0,
    eta_growth_guard_houli_a: float = 0.69,
    eta_growth_guard_houli_m: float = 4.0,
    eta_growth_guard_k_eff: float = 64.0,
    eta_growth_guard_filter_xi: bool = True,
    eta_growth_gxi_limiter_enabled: bool = False,
    eta_growth_gxi_limiter_k_lo: float = 64.0,
    eta_growth_gxi_limiter_k_hi: float = 128.0,
    eta_growth_gxi_limiter_abs_floor: float = 1e-4,
    eta_growth_gxi_limiter_growth_factor: float = 100.0,
    eta_growth_gxi_limiter_sharpness: float = 10.0,
    gxi_cascade_gate_enabled: bool = False,
    gxi_cascade_k_cut: float = 32.0,
    gxi_cascade_r_threshold: float = 1e-3,
    gxi_cascade_high_abs_threshold: float = 0.0,
    gxi_cascade_sharpness: float = 10.0,
    gxi_cascade_houli_a: float = 0.69,
    gxi_cascade_houli_m: float = 4.0,
    gxi_cascade_k_eff: float = 128.0,
    gxi_highband_limiter_enabled: bool = False,
    gxi_highband_k_cut: float = 32.0,
    gxi_highband_r_max: float = 1e-2,
    gxi_highband_abs_floor: float = 5.0,
    gl2_residual_check: bool = False,
    gl2_residual_tol: float = 1e-2,
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
        eta_amp0 = _eta_growth_weight(
            eta0, k_grid, jnp.asarray(0.0, dtype=eta0.dtype),
            k_lo=eta_growth_gxi_limiter_k_lo,
            k_hi=eta_growth_gxi_limiter_k_hi,
            abs_floor=0.0,
            growth_factor=0.0,
            sharpness=0.0,
            nx=nx,
        )
        # With zero trigger and sharpness=0, the helper returns 0.5 rather than
        # an amplitude. Compute the actual initial amplitude directly.
        eta0_hat = jnp.fft.fft(eta0, axis=-1)
        mask0 = (jnp.abs(k_grid) >= eta_growth_gxi_limiter_k_lo) & (
            jnp.abs(k_grid) < eta_growth_gxi_limiter_k_hi
        )
        eta_amp0 = jnp.sqrt(jnp.mean(jnp.where(mask0, jnp.abs(eta0_hat) ** 2, 0.0), axis=-1))
        eta_amp0 = eta_amp0 / jnp.asarray(nx, dtype=eta_amp0.dtype)
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
            if gxi_cascade_gate_enabled:
                gxi = _apply_gxi_cascade_gate(
                    gxi, k_grid,
                    k_cut=gxi_cascade_k_cut,
                    r_threshold=gxi_cascade_r_threshold,
                    high_abs_threshold=gxi_cascade_high_abs_threshold,
                    sharpness=gxi_cascade_sharpness,
                    houli_a=gxi_cascade_houli_a,
                    houli_m=gxi_cascade_houli_m,
                    k_eff=gxi_cascade_k_eff,
                )
            if gxi_highband_limiter_enabled:
                gxi = _apply_gxi_highband_limiter(
                    gxi, k_grid,
                    k_cut=gxi_highband_k_cut,
                    r_max=gxi_highband_r_max,
                    high_abs_floor=gxi_highband_abs_floor,
                )
            if eta_growth_gxi_limiter_enabled:
                limited = _apply_gxi_highband_limiter(
                    gxi, k_grid,
                    k_cut=gxi_highband_k_cut,
                    r_max=gxi_highband_r_max,
                    high_abs_floor=gxi_highband_abs_floor,
                )
                w = _eta_growth_weight(
                    eta, k_grid, eta_amp0,
                    k_lo=eta_growth_gxi_limiter_k_lo,
                    k_hi=eta_growth_gxi_limiter_k_hi,
                    abs_floor=eta_growth_gxi_limiter_abs_floor,
                    growth_factor=eta_growth_gxi_limiter_growth_factor,
                    sharpness=eta_growth_gxi_limiter_sharpness,
                    nx=nx,
                )
                gxi = (1.0 - w) * gxi + w * limited
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
            eta_growth_guard_enabled=eta_growth_guard_enabled,
            eta_growth_guard_k_lo=eta_growth_guard_k_lo,
            eta_growth_guard_k_hi=eta_growth_guard_k_hi,
            eta_growth_guard_abs_floor=eta_growth_guard_abs_floor,
            eta_growth_guard_growth_factor=eta_growth_guard_growth_factor,
            eta_growth_guard_sharpness=eta_growth_guard_sharpness,
            eta_growth_guard_houli_a=eta_growth_guard_houli_a,
            eta_growth_guard_houli_m=eta_growth_guard_houli_m,
            eta_growth_guard_k_eff=eta_growth_guard_k_eff,
            eta_growth_guard_filter_xi=eta_growth_guard_filter_xi,
            gl2_residual_check=gl2_residual_check,
            gl2_residual_tol=gl2_residual_tol,
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


def surrogate_rollout_batched(
    ics: list[IC],
    times_f32: jnp.ndarray,
    nx: int,
    length: float,
    cfg: RegimeConfig,
    loaded: LoadedRun,
    predict_gxi_batched: Callable,
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
    eta_growth_guard_enabled: bool = False,
    eta_growth_guard_k_lo: float = 64.0,
    eta_growth_guard_k_hi: float = 128.0,
    eta_growth_guard_abs_floor: float = 1e-4,
    eta_growth_guard_growth_factor: float = 100.0,
    eta_growth_guard_sharpness: float = 10.0,
    eta_growth_guard_houli_a: float = 0.69,
    eta_growth_guard_houli_m: float = 4.0,
    eta_growth_guard_k_eff: float = 64.0,
    eta_growth_guard_filter_xi: bool = True,
    eta_growth_gxi_limiter_enabled: bool = False,
    eta_growth_gxi_limiter_k_lo: float = 64.0,
    eta_growth_gxi_limiter_k_hi: float = 128.0,
    eta_growth_gxi_limiter_abs_floor: float = 1e-4,
    eta_growth_gxi_limiter_growth_factor: float = 100.0,
    eta_growth_gxi_limiter_sharpness: float = 10.0,
    gxi_cascade_gate_enabled: bool = False,
    gxi_cascade_k_cut: float = 32.0,
    gxi_cascade_r_threshold: float = 1e-3,
    gxi_cascade_high_abs_threshold: float = 0.0,
    gxi_cascade_sharpness: float = 10.0,
    gxi_cascade_houli_a: float = 0.69,
    gxi_cascade_houli_m: float = 4.0,
    gxi_cascade_k_eff: float = 128.0,
    gxi_highband_limiter_enabled: bool = False,
    gxi_highband_k_cut: float = 32.0,
    gxi_highband_r_max: float = 1e-2,
    gxi_highband_abs_floor: float = 5.0,
    gl2_residual_check: bool = False,
    gl2_residual_tol: float = 1e-2,
) -> dict[str, np.ndarray]:
    """One-JIT surrogate rollout across all ICs.

    Same math as ``surrogate_rollout_per_ic`` but state carries a leading
    batch axis so kernels get NB× more work per launch. Fixes the GPU
    under-utilization (~150W of 300W) seen with the per-IC loop.
    """
    dtype = jnp.float64 if f64_harness else jnp.float32
    _, k_grid_np = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid_np, dtype=dtype)
    times = times_f32.astype(dtype)
    filt = cfg.filter_fraction
    substeps = cfg.substeps
    impl = cfg.implicit_iters

    eta0_np = np.stack([ic.eta for ic in ics], axis=0).astype(np.float64)
    xi0_np = np.stack([ic.xi for ic in ics], axis=0).astype(np.float64)
    depths_np = np.asarray([ic.depth for ic in ics], dtype=np.float64)
    log_depth = jnp.asarray(np.log(np.maximum(depths_np, 1e-12)), dtype=jnp.float32)
    depth_2d = jnp.asarray(depths_np, dtype=dtype)[:, None]
    g0 = make_linear_dno_symbol(k_grid, depth_2d)  # (NB, nx)
    sp = ti.SolverParams(
        nx=nx, length=length, depth=depth_2d, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=filt, k=k_grid, g0=g0,
    )

    def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        gxi = predict_gxi_batched(
            eta.astype(jnp.float32), xi.astype(jnp.float32), log_depth,
        ).astype(dtype)
        if gxi_cascade_gate_enabled:
            gxi = _apply_gxi_cascade_gate(
                gxi, k_grid,
                k_cut=gxi_cascade_k_cut,
                r_threshold=gxi_cascade_r_threshold,
                high_abs_threshold=gxi_cascade_high_abs_threshold,
                sharpness=gxi_cascade_sharpness,
                houli_a=gxi_cascade_houli_a,
                houli_m=gxi_cascade_houli_m,
                k_eff=gxi_cascade_k_eff,
            )
        if gxi_highband_limiter_enabled:
            gxi = _apply_gxi_highband_limiter(
                gxi, k_grid,
                k_cut=gxi_highband_k_cut,
                r_max=gxi_highband_r_max,
                high_abs_floor=gxi_highband_abs_floor,
            )
        if eta_growth_gxi_limiter_enabled:
            limited = _apply_gxi_highband_limiter(
                gxi, k_grid,
                k_cut=gxi_highband_k_cut,
                r_max=gxi_highband_r_max,
                high_abs_floor=gxi_highband_abs_floor,
            )
            w = _eta_growth_weight(
                eta, k_grid, eta_amp0,
                k_lo=eta_growth_gxi_limiter_k_lo,
                k_hi=eta_growth_gxi_limiter_k_hi,
                abs_floor=eta_growth_gxi_limiter_abs_floor,
                growth_factor=eta_growth_gxi_limiter_growth_factor,
                sharpness=eta_growth_gxi_limiter_sharpness,
                nx=nx,
            )
            w = w[..., None] if w.ndim else w
            gxi = (1.0 - w) * gxi + w * limited
        if filter_gxi and filt < 1.0:
            gxi = ti.apply_filter(
                gxi, k_grid, shape=filter_shape, filter_fraction=filt,
                houli_a=houli_a, houli_m=houli_m,
            )
        return gxi

    state0 = ti.State(
        eta=jnp.asarray(eta0_np, dtype=dtype),
        xi=jnp.asarray(xi0_np, dtype=dtype),
    )
    eta0_hat = jnp.fft.fft(state0.eta, axis=-1)
    mask0 = (jnp.abs(k_grid) >= eta_growth_gxi_limiter_k_lo) & (
        jnp.abs(k_grid) < eta_growth_gxi_limiter_k_hi
    )
    eta_amp0 = jnp.sqrt(jnp.mean(jnp.where(mask0, jnp.abs(eta0_hat) ** 2, 0.0), axis=-1))
    eta_amp0 = eta_amp0 / jnp.asarray(nx, dtype=eta_amp0.dtype)

    @jax.jit
    def jitted():
        return rollout_surrogate(
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
            eta_growth_guard_enabled=eta_growth_guard_enabled,
            eta_growth_guard_k_lo=eta_growth_guard_k_lo,
            eta_growth_guard_k_hi=eta_growth_guard_k_hi,
            eta_growth_guard_abs_floor=eta_growth_guard_abs_floor,
            eta_growth_guard_growth_factor=eta_growth_guard_growth_factor,
            eta_growth_guard_sharpness=eta_growth_guard_sharpness,
            eta_growth_guard_houli_a=eta_growth_guard_houli_a,
            eta_growth_guard_houli_m=eta_growth_guard_houli_m,
            eta_growth_guard_k_eff=eta_growth_guard_k_eff,
            eta_growth_guard_filter_xi=eta_growth_guard_filter_xi,
            gl2_residual_check=gl2_residual_check,
            gl2_residual_tol=gl2_residual_tol,
        )

    t0 = time.perf_counter()
    out = jitted()
    jax.block_until_ready(out["eta"])
    total_wall = time.perf_counter() - t0
    # rollout_surrogate returns (n_t, NB, nx) when state has leading NB axis
    return {
        "eta": np.asarray(out["eta"]).astype(np.float32),
        "xi": np.asarray(out["xi"]).astype(np.float32),
        "gxi": np.asarray(out["gxi"]).astype(np.float32),
        "wall_s": float(total_wall),
    }


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
    eta_growth_guard_enabled: bool = False,
    eta_growth_guard_k_lo: float = 64.0,
    eta_growth_guard_k_hi: float = 128.0,
    eta_growth_guard_abs_floor: float = 1e-4,
    eta_growth_guard_growth_factor: float = 100.0,
    eta_growth_guard_sharpness: float = 10.0,
    eta_growth_guard_houli_a: float = 0.69,
    eta_growth_guard_houli_m: float = 4.0,
    eta_growth_guard_k_eff: float = 64.0,
    eta_growth_guard_filter_xi: bool = True,
    eta_growth_gxi_limiter_enabled: bool = False,
    eta_growth_gxi_limiter_k_lo: float = 64.0,
    eta_growth_gxi_limiter_k_hi: float = 128.0,
    eta_growth_gxi_limiter_abs_floor: float = 1e-4,
    eta_growth_gxi_limiter_growth_factor: float = 100.0,
    eta_growth_gxi_limiter_sharpness: float = 10.0,
    gxi_cascade_gate_enabled: bool = False,
    gxi_cascade_k_cut: float = 32.0,
    gxi_cascade_r_threshold: float = 1e-3,
    gxi_cascade_high_abs_threshold: float = 0.0,
    gxi_cascade_sharpness: float = 10.0,
    gxi_cascade_houli_a: float = 0.69,
    gxi_cascade_houli_m: float = 4.0,
    gxi_cascade_k_eff: float = 128.0,
    gxi_highband_limiter_enabled: bool = False,
    gxi_highband_k_cut: float = 32.0,
    gxi_highband_r_max: float = 1e-2,
    gxi_highband_abs_floor: float = 5.0,
    truth_cache_dir: Path | None = None,
    batched_surrogate: bool = False,
    predict_gxi_batched: Callable | None = None,
    gl2_residual_check: bool = False,
    gl2_residual_tol: float = 1e-2,
) -> dict:
    ics, dt, tmax, saved_t, saved_eta = build_ics_and_truth_targets(cfg)
    times_np = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=np.float64)
    n_t = int(times_np.shape[0])
    print(
        f"[{regime}] NB={len(ics)} dt={dt} tmax={tmax} n_t={n_t} "
        f"filter={cfg.filter_fraction} substeps={cfg.substeps} impl={cfg.implicit_iters}",
        flush=True,
    )

    cached = _try_load_cached_truth(
        regime, truth_cache_dir, expected_n_ics=len(ics),
        expected_n_t=n_t, expected_nx=nx,
        expected_case_ids=[ic.case_id for ic in ics],
    )
    if cached is not None:
        truth = cached
    elif cfg.truth_kind == "linear_analytic":
        print(f"[{regime}] truth rollout (analytic linear Fourier propagation)...", flush=True)
        truth = truth_rollout_linear_analytic(ics, times_np, nx, length)
    else:
        print(f"[{regime}] truth rollout (batched f64)...", flush=True)
        truth = truth_rollout_batched(ics, jnp.asarray(times_np, dtype=jnp.float64), nx, length, cfg)
    print(f"[{regime}]   truth wall = {truth['wall_s']:.1f}s", flush=True)

    harness = "f64 harness / f32 model" if f64_harness else "f32"
    mode = "batched" if batched_surrogate else "per-IC"
    print(f"[{regime}] surrogate rollout ({mode} {harness})...", flush=True)
    times_in = jnp.asarray(times_np, dtype=jnp.float64 if f64_harness else jnp.float32)
    if batched_surrogate:
        if predict_gxi_batched is None:
            raise ValueError("batched_surrogate=True requires predict_gxi_batched")
        pred = surrogate_rollout_batched(
            ics, times_in, nx, length, cfg, loaded, predict_gxi_batched,
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
            eta_growth_guard_enabled=eta_growth_guard_enabled,
            eta_growth_guard_k_lo=eta_growth_guard_k_lo,
            eta_growth_guard_k_hi=eta_growth_guard_k_hi,
            eta_growth_guard_abs_floor=eta_growth_guard_abs_floor,
            eta_growth_guard_growth_factor=eta_growth_guard_growth_factor,
            eta_growth_guard_sharpness=eta_growth_guard_sharpness,
            eta_growth_guard_houli_a=eta_growth_guard_houli_a,
            eta_growth_guard_houli_m=eta_growth_guard_houli_m,
            eta_growth_guard_k_eff=eta_growth_guard_k_eff,
            eta_growth_guard_filter_xi=eta_growth_guard_filter_xi,
            eta_growth_gxi_limiter_enabled=eta_growth_gxi_limiter_enabled,
            eta_growth_gxi_limiter_k_lo=eta_growth_gxi_limiter_k_lo,
            eta_growth_gxi_limiter_k_hi=eta_growth_gxi_limiter_k_hi,
            eta_growth_gxi_limiter_abs_floor=eta_growth_gxi_limiter_abs_floor,
            eta_growth_gxi_limiter_growth_factor=eta_growth_gxi_limiter_growth_factor,
            eta_growth_gxi_limiter_sharpness=eta_growth_gxi_limiter_sharpness,
            gxi_cascade_gate_enabled=gxi_cascade_gate_enabled,
            gxi_cascade_k_cut=gxi_cascade_k_cut,
            gxi_cascade_r_threshold=gxi_cascade_r_threshold,
            gxi_cascade_high_abs_threshold=gxi_cascade_high_abs_threshold,
            gxi_cascade_sharpness=gxi_cascade_sharpness,
            gxi_cascade_houli_a=gxi_cascade_houli_a,
            gxi_cascade_houli_m=gxi_cascade_houli_m,
            gxi_cascade_k_eff=gxi_cascade_k_eff,
            gxi_highband_limiter_enabled=gxi_highband_limiter_enabled,
            gxi_highband_k_cut=gxi_highband_k_cut,
            gxi_highband_r_max=gxi_highband_r_max,
            gxi_highband_abs_floor=gxi_highband_abs_floor,
            gl2_residual_check=gl2_residual_check,
            gl2_residual_tol=gl2_residual_tol,
        )
        # batched returns (n_t, NB, nx) already; per-IC returns the same shape too.
    else:
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
            eta_growth_guard_enabled=eta_growth_guard_enabled,
            eta_growth_guard_k_lo=eta_growth_guard_k_lo,
            eta_growth_guard_k_hi=eta_growth_guard_k_hi,
            eta_growth_guard_abs_floor=eta_growth_guard_abs_floor,
            eta_growth_guard_growth_factor=eta_growth_guard_growth_factor,
            eta_growth_guard_sharpness=eta_growth_guard_sharpness,
            eta_growth_guard_houli_a=eta_growth_guard_houli_a,
            eta_growth_guard_houli_m=eta_growth_guard_houli_m,
            eta_growth_guard_k_eff=eta_growth_guard_k_eff,
            eta_growth_guard_filter_xi=eta_growth_guard_filter_xi,
            eta_growth_gxi_limiter_enabled=eta_growth_gxi_limiter_enabled,
            eta_growth_gxi_limiter_k_lo=eta_growth_gxi_limiter_k_lo,
            eta_growth_gxi_limiter_k_hi=eta_growth_gxi_limiter_k_hi,
            eta_growth_gxi_limiter_abs_floor=eta_growth_gxi_limiter_abs_floor,
            eta_growth_gxi_limiter_growth_factor=eta_growth_gxi_limiter_growth_factor,
            eta_growth_gxi_limiter_sharpness=eta_growth_gxi_limiter_sharpness,
            gxi_cascade_gate_enabled=gxi_cascade_gate_enabled,
            gxi_cascade_k_cut=gxi_cascade_k_cut,
            gxi_cascade_r_threshold=gxi_cascade_r_threshold,
            gxi_cascade_high_abs_threshold=gxi_cascade_high_abs_threshold,
            gxi_cascade_sharpness=gxi_cascade_sharpness,
            gxi_cascade_houli_a=gxi_cascade_houli_a,
            gxi_cascade_houli_m=gxi_cascade_houli_m,
            gxi_cascade_k_eff=gxi_cascade_k_eff,
            gxi_highband_limiter_enabled=gxi_highband_limiter_enabled,
            gxi_highband_k_cut=gxi_highband_k_cut,
            gxi_highband_r_max=gxi_highband_r_max,
            gxi_highband_abs_floor=gxi_highband_abs_floor,
            gl2_residual_check=gl2_residual_check,
            gl2_residual_tol=gl2_residual_tol,
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
        "eta_growth_guard_enabled": eta_growth_guard_enabled,
        "eta_growth_guard_k_lo": eta_growth_guard_k_lo,
        "eta_growth_guard_k_hi": eta_growth_guard_k_hi,
        "eta_growth_guard_abs_floor": eta_growth_guard_abs_floor,
        "eta_growth_guard_growth_factor": eta_growth_guard_growth_factor,
        "eta_growth_guard_sharpness": eta_growth_guard_sharpness,
        "eta_growth_guard_houli_a": eta_growth_guard_houli_a,
        "eta_growth_guard_houli_m": eta_growth_guard_houli_m,
        "eta_growth_guard_k_eff": eta_growth_guard_k_eff,
        "eta_growth_guard_filter_xi": eta_growth_guard_filter_xi,
        "eta_growth_gxi_limiter_enabled": eta_growth_gxi_limiter_enabled,
        "eta_growth_gxi_limiter_k_lo": eta_growth_gxi_limiter_k_lo,
        "eta_growth_gxi_limiter_k_hi": eta_growth_gxi_limiter_k_hi,
        "eta_growth_gxi_limiter_abs_floor": eta_growth_gxi_limiter_abs_floor,
        "eta_growth_gxi_limiter_growth_factor": eta_growth_gxi_limiter_growth_factor,
        "eta_growth_gxi_limiter_sharpness": eta_growth_gxi_limiter_sharpness,
        "gxi_cascade_gate_enabled": gxi_cascade_gate_enabled,
        "gxi_cascade_k_cut": gxi_cascade_k_cut,
        "gxi_cascade_r_threshold": gxi_cascade_r_threshold,
        "gxi_cascade_high_abs_threshold": gxi_cascade_high_abs_threshold,
        "gxi_cascade_sharpness": gxi_cascade_sharpness,
        "gxi_cascade_houli_a": gxi_cascade_houli_a,
        "gxi_cascade_houli_m": gxi_cascade_houli_m,
        "gxi_cascade_k_eff": gxi_cascade_k_eff,
        "gxi_highband_limiter_enabled": gxi_highband_limiter_enabled,
        "gxi_highband_k_cut": gxi_highband_k_cut,
        "gxi_highband_r_max": gxi_highband_r_max,
        "gxi_highband_abs_floor": gxi_highband_abs_floor,
        "cs_residual_highband_cap": bool(loaded.config.get("cs_residual_highband_cap", False)),
        "cs_block_k_cut": int(loaded.config.get("cs_block_k_cut", 0)),
        "cs_residual_highband_cap_k_cut": float(loaded.config.get("cs_residual_highband_cap_k_cut", 32.0)),
        "cs_residual_highband_cap_beta": float(loaded.config.get("cs_residual_highband_cap_beta", 0.10)),
        "cs_residual_highband_cap_floor": float(loaded.config.get("cs_residual_highband_cap_floor", 0.0)),
        "batched_surrogate": batched_surrogate,
        "gl2_residual_check": gl2_residual_check,
        "gl2_residual_tol": gl2_residual_tol,
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
    parser.add_argument(
        "--gxi_cascade_gate", action="store_true",
        help="Conditionally damp the surrogate's Gxi prediction when its own "
             "mid/high-k energy ratio crosses a threshold. This acts before "
             "Gxi enters the nonlinear RHS and is independent of --cascade_gate, "
             "which filters eta/xi state after a substep.",
    )
    parser.add_argument(
        "--eta_growth_guard", action="store_true",
        help="Conditionally damp eta/xi state high modes when eta k-band amplitude "
             "grows abnormally relative to that IC's initial amplitude. This is a "
             "temporal cascade detector, unlike static high-band ratio gates.",
    )
    parser.add_argument("--eta_growth_guard_k_lo", type=float, default=64.0,
                        help="Lower |k| edge of the eta growth detector band.")
    parser.add_argument("--eta_growth_guard_k_hi", type=float, default=128.0,
                        help="Upper |k| edge of the eta growth detector band.")
    parser.add_argument("--eta_growth_guard_abs_floor", type=float, default=1e-4,
                        help="Absolute eta band-amplitude floor before the guard can activate.")
    parser.add_argument("--eta_growth_guard_growth_factor", type=float, default=100.0,
                        help="Required growth over initial eta band amplitude before activation.")
    parser.add_argument("--eta_growth_guard_sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness in log(current/trigger) space.")
    parser.add_argument("--eta_growth_guard_houli_a", type=float, default=0.69,
                        help="Hou-Li a for the eta-growth guard's state damping.")
    parser.add_argument("--eta_growth_guard_houli_m", type=float, default=4.0,
                        help="Hou-Li m for the eta-growth guard's state damping.")
    parser.add_argument("--eta_growth_guard_k_eff", type=float, default=64.0,
                        help="Hou-Li k_eff for eta-growth guard damping.")
    parser.add_argument("--eta_growth_guard_no_filter_xi", action="store_true",
                        help="Apply eta-growth guard damping to eta only; leave xi untouched.")
    parser.add_argument(
        "--eta_growth_gxi_limiter", action="store_true",
        help="Use the temporal eta-growth detector to smoothly enable the Gxi "
             "high-band limiter. This preserves clean/broadband trajectories "
             "unless their eta high band grows abnormally relative to t=0.",
    )
    parser.add_argument("--eta_growth_gxi_limiter_k_lo", type=float, default=64.0,
                        help="Lower |k| edge of the eta detector band for conditional Gxi limiting.")
    parser.add_argument("--eta_growth_gxi_limiter_k_hi", type=float, default=128.0,
                        help="Upper |k| edge of the eta detector band for conditional Gxi limiting.")
    parser.add_argument("--eta_growth_gxi_limiter_abs_floor", type=float, default=1e-4,
                        help="Absolute eta band-amplitude floor before conditional Gxi limiting.")
    parser.add_argument("--eta_growth_gxi_limiter_growth_factor", type=float, default=100.0,
                        help="Required growth over initial eta band amplitude before conditional Gxi limiting.")
    parser.add_argument("--eta_growth_gxi_limiter_sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness for conditional Gxi limiting.")
    parser.add_argument("--gxi_cascade_k_cut", type=float, default=32.0,
                        help="|k| boundary for Gxi high-band energy ratio (default 32).")
    parser.add_argument("--gxi_cascade_r_threshold", type=float, default=1e-3,
                        help="Gxi E_hi/E_lo threshold where the gate is at 50%% (default 1e-3).")
    parser.add_argument("--gxi_cascade_high_abs_threshold", type=float, default=0.0,
                        help="Optional absolute sqrt(sum |Gxi_hat(k>=k_cut)|^2) threshold. "
                             "If >0, the Gxi gate activates only when both ratio and absolute "
                             "high-band amplitude are large. Default 0 disables this extra guard.")
    parser.add_argument("--gxi_cascade_sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness in log(r) space for the Gxi gate (default 10).")
    parser.add_argument("--gxi_cascade_houli_a", type=float, default=0.69,
                        help="Hou-Li a for conditional Gxi smoothing (default 0.69 ~ ln 2).")
    parser.add_argument("--gxi_cascade_houli_m", type=float, default=4.0,
                        help="Hou-Li m for conditional Gxi smoothing (default 4).")
    parser.add_argument("--gxi_cascade_k_eff", type=float, default=128.0,
                        help="Hou-Li k_eff for conditional Gxi smoothing (default 128).")
    parser.add_argument(
        "--gxi_highband_limiter", action="store_true",
        help="Cap only the high-band Fourier coefficients of surrogate Gxi before "
             "the nonlinear RHS. Preserves low modes exactly and rescales "
             "k>=cut only when sqrt(E_hi) exceeds max(abs_floor, sqrt(r_max)*sqrt(E_lo)).",
    )
    parser.add_argument("--gxi_highband_k_cut", type=float, default=32.0,
                        help="|k| boundary for the Gxi high-band limiter (default 32).")
    parser.add_argument("--gxi_highband_r_max", type=float, default=1e-2,
                        help="Maximum allowed high/low Gxi energy ratio above the abs floor.")
    parser.add_argument("--gxi_highband_abs_floor", type=float, default=5.0,
                        help="Absolute sqrt(sum |Gxi_hat(k>=k_cut)|^2) floor for the high-band cap.")
    parser.add_argument(
        "--cs_residual_highband_cap", action="store_true",
        help="Enable the model's parameter-free structural cap on only the learned "
             "CS-DNO residual high band. This preserves G_0 xi and changes no "
             "checkpoint tensor shapes.",
    )
    parser.add_argument("--cs_residual_highband_cap_k_cut", type=float, default=32.0,
                        help="|k| boundary for the learned-residual high-band cap.")
    parser.add_argument("--cs_residual_highband_cap_beta", type=float, default=0.10,
                        help="Residual high-band cap as beta*||G_0 xi||.")
    parser.add_argument("--cs_residual_highband_cap_floor", type=float, default=0.0,
                        help="Additive normalized-output floor for the residual high-band cap.")
    parser.add_argument("--cs_block_k_cut", type=int, default=None,
                        help="Eval-only override for the CS-DNO learned block transfer cutoff. "
                             "0 leaves all transfer kernels active; positive values block "
                             "learned transfer-kernel corrections for |k| >= cut.")
    parser.add_argument("--truth_cache", default=None,
                        help="Directory containing prior `{regime}_trajs.npz` files. If present and "
                             "shape+case_ids match, truth is loaded instead of recomputed. "
                             "Cheap way to avoid re-running the batched f64 truth rollout across "
                             "multiple checkpoints/eval configs on the same regime+n_ics.")
    parser.add_argument("--cascade_no_filter_xi", action="store_true",
                        help="Apply the cascade gate to eta only; leave xi untouched.")
    parser.add_argument(
        "--batched_surrogate", action="store_true",
        help="Run the surrogate rollout with all ICs stacked on a leading batch axis "
             "so kernels get NB× more work per launch (nx=1024 batch=1 is kernel-latency "
             "bound; batched draws full TDP). Math identical to the per-IC path.",
    )
    parser.add_argument(
        "--gl2_residual_check", action="store_true",
        help="Track GL2 Picard final-iter stage-update relative residual and mask "
             "only samples whose r_final >= tol with NaNs. This is diagnostic "
             "per-sample containment, not dt-halving recovery.",
    )
    parser.add_argument("--gl2_residual_tol", type=float, default=1e-2,
                        help="Divergence threshold. Sample is masked if Picard's "
                             "final-iter relative stage residual >= this. Default 1e-2 "
                             "sits ~6 orders above healthy (~1e-8) and ~1-2 orders "
                             "below non-contractive (~0.19).")
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
    if args.gxi_cascade_gate:
        default_name += f"_gxicg_kc{args.gxi_cascade_k_cut:g}_rt{args.gxi_cascade_r_threshold:g}"
    if args.gxi_highband_limiter:
        default_name += f"_gxihbl_kc{args.gxi_highband_k_cut:g}_r{args.gxi_highband_r_max:g}"
    if args.eta_growth_guard:
        default_name += (
            f"_etagg_k{args.eta_growth_guard_k_lo:g}-{args.eta_growth_guard_k_hi:g}"
            f"_g{args.eta_growth_guard_growth_factor:g}"
        ).replace(".", "p")
    if args.eta_growth_gxi_limiter:
        default_name += (
            f"_etagxihl_k{args.eta_growth_gxi_limiter_k_lo:g}-{args.eta_growth_gxi_limiter_k_hi:g}"
            f"_g{args.eta_growth_gxi_limiter_growth_factor:g}"
        ).replace(".", "p")
    if args.cs_residual_highband_cap:
        default_name += (
            f"_csrcap_kc{args.cs_residual_highband_cap_k_cut:g}"
            f"_b{args.cs_residual_highband_cap_beta:g}"
        )
    if args.cs_block_k_cut is not None:
        default_name += f"_csbkc{args.cs_block_k_cut}"
    if args.gl2_residual_check:
        default_name += f"_gl2rc_tol{args.gl2_residual_tol:g}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config_overrides = {}
    if args.cs_residual_highband_cap:
        model_config_overrides.update({
            "cs_residual_highband_cap": True,
            "cs_residual_highband_cap_k_cut": float(args.cs_residual_highband_cap_k_cut),
            "cs_residual_highband_cap_beta": float(args.cs_residual_highband_cap_beta),
            "cs_residual_highband_cap_floor": float(args.cs_residual_highband_cap_floor),
        })
    if args.cs_block_k_cut is not None:
        model_config_overrides["cs_block_k_cut"] = int(args.cs_block_k_cut)
    loaded = load_run(
        run_dir, checkpoint=args.checkpoint,
        config_overrides=model_config_overrides or None,
    )
    if args.f64_model:
        loaded.config["precision"] = "fp64"
    predict_gxi = build_predict_gxi_with_depth(loaded)
    predict_gxi_batched = build_predict_gxi_batched(loaded) if args.batched_surrogate else None

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
            eta_growth_guard_enabled=args.eta_growth_guard,
            eta_growth_guard_k_lo=args.eta_growth_guard_k_lo,
            eta_growth_guard_k_hi=args.eta_growth_guard_k_hi,
            eta_growth_guard_abs_floor=args.eta_growth_guard_abs_floor,
            eta_growth_guard_growth_factor=args.eta_growth_guard_growth_factor,
            eta_growth_guard_sharpness=args.eta_growth_guard_sharpness,
            eta_growth_guard_houli_a=args.eta_growth_guard_houli_a,
            eta_growth_guard_houli_m=args.eta_growth_guard_houli_m,
            eta_growth_guard_k_eff=args.eta_growth_guard_k_eff,
            eta_growth_guard_filter_xi=not args.eta_growth_guard_no_filter_xi,
            eta_growth_gxi_limiter_enabled=args.eta_growth_gxi_limiter,
            eta_growth_gxi_limiter_k_lo=args.eta_growth_gxi_limiter_k_lo,
            eta_growth_gxi_limiter_k_hi=args.eta_growth_gxi_limiter_k_hi,
            eta_growth_gxi_limiter_abs_floor=args.eta_growth_gxi_limiter_abs_floor,
            eta_growth_gxi_limiter_growth_factor=args.eta_growth_gxi_limiter_growth_factor,
            eta_growth_gxi_limiter_sharpness=args.eta_growth_gxi_limiter_sharpness,
            gxi_cascade_gate_enabled=args.gxi_cascade_gate,
            gxi_cascade_k_cut=args.gxi_cascade_k_cut,
            gxi_cascade_r_threshold=args.gxi_cascade_r_threshold,
            gxi_cascade_high_abs_threshold=args.gxi_cascade_high_abs_threshold,
            gxi_cascade_sharpness=args.gxi_cascade_sharpness,
            gxi_cascade_houli_a=args.gxi_cascade_houli_a,
            gxi_cascade_houli_m=args.gxi_cascade_houli_m,
            gxi_cascade_k_eff=args.gxi_cascade_k_eff,
            gxi_highband_limiter_enabled=args.gxi_highband_limiter,
            gxi_highband_k_cut=args.gxi_highband_k_cut,
            gxi_highband_r_max=args.gxi_highband_r_max,
            gxi_highband_abs_floor=args.gxi_highband_abs_floor,
            truth_cache_dir=Path(args.truth_cache).resolve() if args.truth_cache else None,
            batched_surrogate=args.batched_surrogate,
            predict_gxi_batched=predict_gxi_batched,
            gl2_residual_check=args.gl2_residual_check,
            gl2_residual_tol=args.gl2_residual_tol,
        )

    (out_dir / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Aggregate -> {out_dir / 'all_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
