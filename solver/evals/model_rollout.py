"""Shared rollout-evaluation infrastructure for trained FNO/SpectralDNO checkpoints.

Loads an orbax checkpoint produced by ``train-jax-10m/1d_dno_fno_jax.py``,
wraps the trained model as a `predict_gxi(eta, xi)` callable that's a drop-in
replacement for ``dno_series_eval`` inside the time integrator, and provides a
`rollout_surrogate` driver mirroring `solver.solvers.time_integrator.rollout`.

Per-regime entrypoints (rollout_stokes/linear/tanaka/bf) call these helpers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in (
    REPO_ROOT,
    REPO_ROOT / "models" / "fno-jax",
    REPO_ROOT / "models" / "dno-net",
    REPO_ROOT / "train-jax-10m",
):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from dno_net_v2 import CraigSulemDNO
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import dno_series_eval, myfft, myifft


class LoadedRun(NamedTuple):
    model: object
    params: dict
    config: dict
    stats: dict
    norm_mode: str
    epoch: int


def load_run(run_dir: str | Path, *, checkpoint: str = "best") -> LoadedRun:
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_dir = run_dir / ("best_val_ckpt" if checkpoint == "best" else "final_ckpt")
    metadata = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    stats = metadata["stats"]
    norm_mode = config.get("norm", "minmax")

    # Checkpoints trained with mixed-precision (cs_fft_fp64=True) require x64
    # to be enabled at inference time too; otherwise the .astype(jnp.float64)
    # casts at the FFT boundaries silently downgrade to fp32 and rollout
    # diverges from training behavior.
    if bool(config.get("cs_fft_fp64", False)) or config.get("precision") == "fp64":
        jax.config.update("jax_enable_x64", True)

    restored = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])

    if config.get("model", "fno") == "spectral_dno":
        model = SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(stats["feature_absmax"]).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", stats["target_absmax"])),
        )
    elif config.get("model", "fno") == "cs_dno":
        model = CraigSulemDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            n_polys=int(config.get("cs_n_polys", 3)),
            use_first_deriv=bool(config.get("cs_use_first_deriv", True)),
            use_second_deriv=bool(config.get("cs_use_second_deriv", True)),
            use_half_deriv=bool(config.get("cs_use_half_deriv", True)),
            use_hilbert=bool(config.get("cs_use_hilbert", True)),
            use_g0_eta=bool(config.get("cs_use_g0_eta", False)),
            use_g0_eta_dx=bool(config.get("cs_use_g0_eta_dx", False)),
            mult_hidden=int(config.get("cs_mult_hidden", 32)),
            use_g1_baseline=bool(config.get("cs_use_g1_baseline", False)),
            g1_k_cut=int(config.get("cs_g1_k_cut", 0)),
            fft_fp64=bool(config.get("cs_fft_fp64", False)),
            tie_xi_out_mult=bool(config.get("cs_tie_xi_out_mult", False)),
            phi_bias_free=bool(config.get("cs_phi_bias_free", False)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(stats["feature_absmax"]).reshape(-1)[1])),
            eta_scale=float(config.get("eta_scale", np.asarray(stats["feature_absmax"]).reshape(-1)[0])),
            target_scale=float(config.get("target_scale", stats["target_absmax"])),
        )
    else:
        model = FNO1d(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            domain_length=float(stats.get("domain_length", 2.0 * np.pi)),
            xi_scale=float(np.asarray(stats["feature_absmax"]).reshape(-1)[1]),
            target_scale=float(stats["target_absmax"]),
            eta_features=bool(config.get("fno_eta_features", False)),
        )

    return LoadedRun(model, params, config, stats, norm_mode, int(metadata["epoch"]))


def build_predict_gxi_with_depth(loaded: LoadedRun) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a jit'd ``(eta, xi, log_depth) -> gxi`` callable.

    Like ``build_predict_gxi`` but depth is a runtime arg, so one jit covers many ICs
    at different depths without re-compilation. ``eta``/``xi`` are 1D ``(N,)``; ``log_depth``
    is a scalar ``(1,)`` or 0-d array.
    """
    model_dtype = jnp.float64 if loaded.config.get("precision") == "fp64" else jnp.float32

    if loaded.norm_mode == "scale":
        feat_absmax = jnp.asarray(loaded.stats["feature_absmax"], dtype=model_dtype).reshape(1, 1, 2)
        feat_absmax = jnp.where(feat_absmax > 0, feat_absmax, 1.0)
        target_scale = float(loaded.stats["target_absmax"]) or 1.0

        @jax.jit
        def predict(eta: jnp.ndarray, xi: jnp.ndarray, log_depth: jnp.ndarray) -> jnp.ndarray:
            stacked = jnp.stack((eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1)[None, :, :]
            inp = stacked / feat_absmax
            depth_arg = jnp.asarray(log_depth, dtype=model_dtype).reshape(1, 1)
            out = loaded.model.apply({"params": loaded.params}, inp, depth_arg)
            gxi = (out[0, :, 0] * target_scale).astype(eta.dtype)
            return gxi - gxi.mean()

    else:
        feat_min = jnp.asarray(loaded.stats["feature_min"], dtype=model_dtype).reshape(1, 1, 2)
        feat_max = jnp.asarray(loaded.stats["feature_max"], dtype=model_dtype).reshape(1, 1, 2)
        feat_range = feat_max - feat_min + 1e-8
        tgt_min = float(loaded.stats["target_min"])
        tgt_max = float(loaded.stats["target_max"])
        tgt_range = tgt_max - tgt_min + 1e-8

        @jax.jit
        def predict(eta: jnp.ndarray, xi: jnp.ndarray, log_depth: jnp.ndarray) -> jnp.ndarray:
            stacked = jnp.stack((eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1)[None, :, :]
            inp = ((stacked - feat_min) / feat_range) * 2.0 - 1.0
            depth_arg = jnp.asarray(log_depth, dtype=model_dtype).reshape(1, 1)
            out = loaded.model.apply({"params": loaded.params}, inp, depth_arg)
            denorm = ((out[0, :, 0] + 1.0) * 0.5) * tgt_range + tgt_min
            gxi = denorm.astype(eta.dtype)
            return gxi - gxi.mean()

    return predict


def build_predict_gxi(loaded: LoadedRun, depth: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a jit'd ``(eta, xi) -> gxi`` callable for a single sample (1, N).

    Depth is the physical depth (h); the model receives ``log(h)``.
    """
    model_dtype = jnp.float64 if loaded.config.get("precision") == "fp64" else jnp.float32
    log_depth = jnp.full((1, 1), float(np.log(max(float(depth), 1e-12))), dtype=model_dtype)

    if loaded.norm_mode == "scale":
        feat_absmax = jnp.asarray(loaded.stats["feature_absmax"], dtype=model_dtype).reshape(1, 1, 2)
        feat_absmax = jnp.where(feat_absmax > 0, feat_absmax, 1.0)
        target_scale = float(loaded.stats["target_absmax"]) or 1.0

        @jax.jit
        def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
            stacked = jnp.stack((eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1)[None, :, :]
            inp = stacked / feat_absmax
            out = loaded.model.apply({"params": loaded.params}, inp, log_depth)
            gxi = (out[0, :, 0] * target_scale).astype(eta.dtype)
            return gxi - gxi.mean()

    else:
        feat_min = jnp.asarray(loaded.stats["feature_min"], dtype=model_dtype).reshape(1, 1, 2)
        feat_max = jnp.asarray(loaded.stats["feature_max"], dtype=model_dtype).reshape(1, 1, 2)
        feat_range = feat_max - feat_min + 1e-8
        tgt_min = float(loaded.stats["target_min"])
        tgt_max = float(loaded.stats["target_max"])
        tgt_range = tgt_max - tgt_min + 1e-8

        @jax.jit
        def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
            stacked = jnp.stack((eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1)[None, :, :]
            inp = ((stacked - feat_min) / feat_range) * 2.0 - 1.0
            out = loaded.model.apply({"params": loaded.params}, inp, log_depth)
            denorm = ((out[0, :, 0] + 1.0) * 0.5) * tgt_range + tgt_min
            gxi = denorm.astype(eta.dtype)
            return gxi - gxi.mean()

    return predict


# ---------------------------------------------------------------------------
# Surrogate-driven time integration (mirrors solver.solvers.time_integrator)
# ---------------------------------------------------------------------------

def _rhs_nonlinear_surrogate(
    state: ti.State,
    params: ti.SolverParams,
    predict_gxi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    filter_shape: str = "hard",
    houli_a: float = 36.0,
    houli_m: float = 36.0,
) -> ti.State:
    eta_x = ti.spectral_dx(state.eta, params.k)
    xi_x = ti.spectral_dx(state.xi, params.k)
    gxi = predict_gxi(state.eta, state.xi)
    linear_gxi = ti.linear_dno_action(state.xi, params.g0)
    eta_t = gxi - linear_gxi
    numerator = gxi + eta_x * xi_x
    xi_t = -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)
    if params.filter_fraction < 1.0:
        eta_t = ti.apply_filter(eta_t, params.k, shape=filter_shape,
                                filter_fraction=params.filter_fraction,
                                houli_a=houli_a, houli_m=houli_m)
        xi_t = ti.apply_filter(xi_t, params.k, shape=filter_shape,
                               filter_fraction=params.filter_fraction,
                               houli_a=houli_a, houli_m=houli_m)
    return ti.State(eta=eta_t, xi=xi_t)


def _rhs_nonlinear_if_surrogate(
    v_hat: ti.SpectralState,
    t: float | jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    filter_shape: str = "hard",
    houli_a: float = 36.0,
    houli_m: float = 36.0,
) -> ti.SpectralState:
    physical_hat = ti.apply_linear_flow_hat(v_hat, t, params)
    physical_state = ti.State(eta=myifft(physical_hat.eta_hat), xi=myifft(physical_hat.xi_hat))
    nl = _rhs_nonlinear_surrogate(physical_state, params, predict_gxi,
                                  filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m)
    nl_hat = ti.SpectralState(eta_hat=myfft(nl.eta, params.nx), xi_hat=myfft(nl.xi, params.nx))
    return ti.apply_linear_flow_hat(nl_hat, -t, params)


def _apply_cascade_gate(
    state: ti.State,
    k_arr: jnp.ndarray,
    nx: int,
    *,
    k_cut: float,
    r_threshold: float,
    sharpness: float,
    houli_a: float,
    houli_m: float,
    k_eff: float,
    filter_xi: bool,
) -> ti.State:
    """Conditional Hou-Li gated by mid-high-k energy ratio in η.

    Compute r = Σ|η̂(k≥k_cut)|² / Σ|η̂(k<k_cut)|². Smooth-blend a Hou-Li
    multiplier c(k) = exp(-a(|k|/k_eff)^{2m}) into the state with weight
    f = sigmoid(sharpness·(log r − log r_threshold)). At r ≪ threshold,
    f → 0 and the gate is identity; the carrier-quiet phase is untouched.
    Activates only during cascade ignition.
    """
    eta_hat = myfft(state.eta, nx)
    k_abs = jnp.abs(k_arr)
    mask_hi = k_abs >= k_cut
    e_hi = jnp.sum(jnp.where(mask_hi, jnp.abs(eta_hat) ** 2, 0.0))
    e_lo = jnp.sum(jnp.where(~mask_hi, jnp.abs(eta_hat) ** 2, 0.0))
    r = e_hi / (e_lo + 1e-30)
    log_r = jnp.log(r + 1e-30)
    log_thresh = jnp.log(jnp.asarray(r_threshold, dtype=r.dtype) + 1e-30)
    f = jax.nn.sigmoid(sharpness * (log_r - log_thresh))
    c_houli = jnp.exp(-houli_a * (k_abs / k_eff) ** (2 * houli_m))
    blend = (1.0 - f) + f * c_houli
    new_eta = myifft(eta_hat * blend)
    if filter_xi:
        xi_hat = myfft(state.xi, nx)
        new_xi = myifft(xi_hat * blend)
    else:
        new_xi = state.xi
    return ti.State(eta=new_eta, xi=new_xi)


def _gl2_if_step_surrogate(
    state: ti.State,
    t: float,
    dt: float,
    params: ti.SolverParams,
    predict_gxi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    iterations: int = 4,
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
) -> ti.State:
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = ti.SpectralState(eta_hat=myfft(state.eta, params.nx), xi_hat=myfft(state.xi, params.nx))
    v0 = ti.apply_linear_flow_hat(state_hat, -t, params)

    _add = lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y, a, b)
    _scale = lambda a, s: jax.tree_util.tree_map(lambda x: s * x, a)

    def body_fn(_: int, stages: tuple[ti.SpectralState, ti.SpectralState]) -> tuple[ti.SpectralState, ti.SpectralState]:
        s1, s2 = stages
        f1 = _rhs_nonlinear_if_surrogate(s1, t + c1 * dt, params, predict_gxi,
                                         filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m)
        f2 = _rhs_nonlinear_if_surrogate(s2, t + c2 * dt, params, predict_gxi,
                                         filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m)
        cand1 = _add(v0, _scale(_add(_scale(f1, a11), _scale(f2, a12)), dt))
        cand2 = _add(v0, _scale(_add(_scale(f1, a21), _scale(f2, a22)), dt))
        return cand1, cand2

    s1, s2 = jax.lax.fori_loop(0, iterations, body_fn, (v0, v0))
    f1 = _rhs_nonlinear_if_surrogate(s1, t + c1 * dt, params, predict_gxi,
                                     filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m)
    f2 = _rhs_nonlinear_if_surrogate(s2, t + c2 * dt, params, predict_gxi,
                                     filter_shape=filter_shape, houli_a=houli_a, houli_m=houli_m)
    v1 = _add(v0, _scale(_add(f1, f2), 0.5 * dt))

    next_hat = ti.apply_linear_flow_hat(v1, t + dt, params)
    next_state = ti.State(eta=myifft(next_hat.eta_hat), xi=myifft(next_hat.xi_hat))
    if params.filter_fraction < 1.0:
        next_state = ti.State(
            eta=ti.apply_filter(next_state.eta, params.k, shape=filter_shape,
                                filter_fraction=params.filter_fraction,
                                houli_a=houli_a, houli_m=houli_m),
            xi=ti.apply_filter(next_state.xi, params.k, shape=filter_shape,
                               filter_fraction=params.filter_fraction,
                               houli_a=houli_a, houli_m=houli_m),
        )
    if cascade_gate_enabled:
        next_state = _apply_cascade_gate(
            next_state, params.k, params.nx,
            k_cut=cascade_k_cut, r_threshold=cascade_r_threshold,
            sharpness=cascade_sharpness, houli_a=cascade_houli_a,
            houli_m=cascade_houli_m, k_eff=cascade_k_eff,
            filter_xi=cascade_filter_xi,
        )
    return next_state


def rollout_surrogate(
    initial: ti.State,
    times: jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    substeps: int = 8,
    zero_mean_xi: bool = True,
    gl2_iterations: int = 4,
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
) -> dict[str, jnp.ndarray]:
    state = ti.State(eta=jnp.asarray(initial.eta), xi=jnp.asarray(initial.xi))
    if zero_mean_xi:
        state = ti.project_zero_mean_xi(state)

    gxi0 = predict_gxi(state.eta, state.xi)
    if times.shape[0] == 1:
        return {"times": times, "eta": state.eta[None, :], "xi": state.xi[None, :], "gxi": gxi0[None, :]}

    dts = times[1:] - times[:-1]

    def step_fn(carry: ti.State, data: tuple[jnp.ndarray, jnp.ndarray]):
        cur_t, dt_int = data
        dt_sub = dt_int / substeps

        def body_fn(sub_idx: int, sub: ti.State) -> ti.State:
            sub_t = cur_t + dt_sub * sub_idx
            nxt = _gl2_if_step_surrogate(sub, sub_t, dt_sub, params, predict_gxi,
                                         iterations=gl2_iterations,
                                         filter_shape=filter_shape,
                                         houli_a=houli_a, houli_m=houli_m,
                                         cascade_gate_enabled=cascade_gate_enabled,
                                         cascade_k_cut=cascade_k_cut,
                                         cascade_r_threshold=cascade_r_threshold,
                                         cascade_sharpness=cascade_sharpness,
                                         cascade_houli_a=cascade_houli_a,
                                         cascade_houli_m=cascade_houli_m,
                                         cascade_k_eff=cascade_k_eff,
                                         cascade_filter_xi=cascade_filter_xi)
            if zero_mean_xi:
                nxt = ti.project_zero_mean_xi(nxt)
            return nxt

        next_state = jax.lax.fori_loop(0, substeps, body_fn, carry)
        gxi_saved = predict_gxi(next_state.eta, next_state.xi)
        return next_state, (next_state.eta, next_state.xi, gxi_saved)

    _, (saved_eta, saved_xi, saved_gxi) = jax.lax.scan(step_fn, state, (times[:-1], dts))
    eta = jnp.concatenate((state.eta[None, :], saved_eta), axis=0)
    xi = jnp.concatenate((state.xi[None, :], saved_xi), axis=0)
    gxi = jnp.concatenate((gxi0[None, :], saved_gxi), axis=0)
    return {"times": times, "eta": eta, "xi": xi, "gxi": gxi}


# ---------------------------------------------------------------------------
# Comparison / plotting
# ---------------------------------------------------------------------------

def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=-1) / (np.linalg.norm(truth, axis=-1) + 1e-12)


def compare_rollouts(
    pred_eta: np.ndarray,
    pred_xi: np.ndarray,
    truth_eta: np.ndarray,
    truth_xi: np.ndarray,
    truth_gxi: np.ndarray | None = None,
    pred_gxi: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    eta_err = _rel_l2(pred_eta, truth_eta)
    xi_err = _rel_l2(pred_xi, truth_xi)
    out: dict[str, np.ndarray | float] = {
        "eta_rel_l2": eta_err,
        "xi_rel_l2": xi_err,
        "eta_final_rel_l2": float(eta_err[-1]),
        "xi_final_rel_l2": float(xi_err[-1]),
        "eta_mean_rel_l2": float(eta_err.mean()),
        "xi_mean_rel_l2": float(xi_err.mean()),
    }
    if truth_gxi is not None and pred_gxi is not None:
        gxi_err = _rel_l2(pred_gxi, truth_gxi)
        out["gxi_rel_l2"] = gxi_err
        out["gxi_final_rel_l2"] = float(gxi_err[-1])
        out["gxi_mean_rel_l2"] = float(gxi_err.mean())
    return out


def save_comparison_plot(
    out_path: Path,
    *,
    title: str,
    x: np.ndarray,
    t: np.ndarray,
    truth_eta: np.ndarray,
    pred_eta: np.ndarray,
    truth_xi: np.ndarray,
    pred_xi: np.ndarray,
    metrics: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_t = truth_eta.shape[0]
    snap_idx = (0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1)

    fig, axes = plt.subplots(3, len(snap_idx) + 1, figsize=(3.2 * (len(snap_idx) + 1), 7.5))

    axes[0, 0].semilogy(t, np.asarray(metrics["eta_rel_l2"]) + 1e-16, label="eta")
    axes[0, 0].semilogy(t, np.asarray(metrics["xi_rel_l2"]) + 1e-16, label="xi")
    if "gxi_rel_l2" in metrics:
        axes[0, 0].semilogy(t, np.asarray(metrics["gxi_rel_l2"]) + 1e-16, label="gxi")
    axes[0, 0].set_title("relative L2 vs time")
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("rel L2"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[1, 0].axis("off"); axes[2, 0].axis("off")

    for i, idx in enumerate(snap_idx, start=1):
        axes[0, i].plot(x, truth_eta[idx], "k", lw=1.0, label="truth")
        axes[0, i].plot(x, pred_eta[idx], "r--", lw=0.9, label="surrogate")
        axes[0, i].set_title(f"eta @ t={t[idx]:.2f}")
        axes[0, i].grid(alpha=0.3)
        if i == 1:
            axes[0, i].legend(fontsize=8)

        axes[1, i].plot(x, truth_xi[idx], "k", lw=1.0)
        axes[1, i].plot(x, pred_xi[idx], "r--", lw=0.9)
        axes[1, i].set_title(f"xi @ t={t[idx]:.2f}")
        axes[1, i].grid(alpha=0.3)

        diff = pred_eta[idx] - truth_eta[idx]
        axes[2, i].plot(x, diff, "tab:purple", lw=0.9)
        axes[2, i].set_title(f"eta_pred - eta_truth")
        axes[2, i].grid(alpha=0.3)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def predict_collisions_from_ic(
    x: np.ndarray,
    eta0: np.ndarray,
    gxi0: np.ndarray,
    t_max: float,
    *,
    peak_thresh: float = 0.25,
    flank_min: int = 4,
    flank_max: int = 30,
) -> list[float]:
    """Estimate soliton collision times in (0, t_max] from a t=0 snapshot.

    Locates peaks in |eta0|, estimates each peak's signed celerity from
    c = -G(eta)xi / d_x(eta) evaluated on its flanks, and enumerates all
    pairwise meet-times in the periodic domain.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return []
    x = np.asarray(x).astype(np.float64)
    eta0 = np.asarray(eta0).astype(np.float64)
    gxi0 = np.asarray(gxi0).astype(np.float64)
    nx = x.size
    dx = float(x[1] - x[0])
    L = dx * nx
    abs_max = float(np.max(np.abs(eta0)))
    if abs_max <= 0:
        return []
    eta_x = np.gradient(eta0, dx)
    peaks, _ = find_peaks(np.abs(eta0), height=peak_thresh * abs_max)
    if peaks.size < 2:
        return []
    centers: list[float] = []
    speeds: list[float] = []
    for p in peaks:
        offsets = list(range(-flank_max, -flank_min + 1)) + list(range(flank_min, flank_max + 1))
        cs = [
            -gxi0[(p + off) % nx] / eta_x[(p + off) % nx]
            for off in offsets if abs(eta_x[(p + off) % nx]) > 1e-6
        ]
        if not cs:
            continue
        centers.append(float(x[p]))
        speeds.append(float(np.median(cs)))
    out: list[float] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            ds = speeds[i] - speeds[j]
            if abs(ds) < 1e-8:
                continue
            period = L / abs(ds)
            t0 = ((centers[j] - centers[i]) / ds) % period
            if t0 < 1e-3:
                t0 += period
            t = t0
            while t <= t_max + 1e-6:
                out.append(float(t))
                t += period
    return sorted(out)


def save_paper_error_plot(
    out_path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    truth_eta: np.ndarray,
    pred_eta: np.ndarray,
    truth_xi: np.ndarray,
    pred_xi: np.ndarray,
    truth_gxi: np.ndarray,
    pred_gxi: np.ndarray,
    title: str | None = None,
    dpi: int = 220,
    annotate_collisions: bool = False,
) -> Path:
    """Paper-ready 3-panel space-time normalized-error heatmap.

    Each panel shows |pred(x,t) - truth(x,t)| / max_x |truth(x,t)| for eta, xi,
    and G(eta) xi. Per-frame normalization keeps the error meaningful even when
    the truth amplitude varies over the trajectory. A single shared colorbar
    spans all three panels so colors are directly comparable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (r"$\eta$",           np.asarray(truth_eta), np.asarray(pred_eta)),
        (r"$\xi$",            np.asarray(truth_xi),  np.asarray(pred_xi)),
        (r"$G(\eta)\,\xi$",   np.asarray(truth_gxi), np.asarray(pred_gxi)),
    ]
    errors: list[np.ndarray] = []
    for _, truth, pred in panels:
        denom = np.maximum(np.max(np.abs(truth), axis=-1, keepdims=True), 1e-12)
        errors.append(np.abs(pred - truth) / denom)
    stacked = np.concatenate([e.reshape(-1) for e in errors])
    positive = stacked[stacked > 0]
    if positive.size > 0:
        vmin_shared = max(float(np.quantile(positive, 0.05)), 1e-6)
        vmax_shared = max(float(np.quantile(positive, 0.995)), 10.0 * vmin_shared)
    else:
        vmin_shared, vmax_shared = 1e-6, 1e-1

    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True, constrained_layout=True)
    extent = [float(x.min()), float(x.max()), float(t.min()), float(t.max())]
    norm = LogNorm(vmin=vmin_shared, vmax=vmax_shared)
    im = None
    for ax, (label, _, _), err in zip(axes, panels, errors):
        err_floored = np.maximum(err, vmin_shared)
        im = ax.imshow(err_floored, origin="lower", aspect="auto", extent=extent,
                       cmap="magma", norm=norm)
        ax.set_xlabel(r"$x$")
        ax.set_title(label)

    if annotate_collisions:
        t_arr = np.asarray(t)
        coll_times = predict_collisions_from_ic(
            np.asarray(x), np.asarray(truth_eta)[0], np.asarray(truth_gxi)[0],
            t_max=float(t_arr.max()),
        )
        for ax in axes:
            for ct in coll_times:
                ax.axhline(ct, color="white", lw=1.2, ls="--", alpha=0.95, zorder=5)
        if coll_times:
            axes[-1].text(
                1.02, 0.98, "predicted\ncollisions",
                transform=axes[-1].transAxes, fontsize=8,
                color="white", va="top", ha="left",
                bbox=dict(facecolor="black", edgecolor="white", lw=0.5, alpha=0.6, pad=2),
            )
    axes[0].set_ylabel(r"$t$")
    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02, location="right")
    cbar.set_label(r"$|\widehat{u}-u|\,/\,\max_x|u|$")
    if title:
        fig.suptitle(title, fontsize=11)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_rollout_npz(
    out_path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    truth_eta: np.ndarray,
    pred_eta: np.ndarray,
    truth_xi: np.ndarray,
    pred_xi: np.ndarray,
    truth_gxi: np.ndarray,
    pred_gxi: np.ndarray,
    depth: float,
) -> Path:
    """Persist raw rollout arrays so plots can be regenerated without rerunning."""
    np.savez_compressed(
        out_path,
        x=np.asarray(x).astype(np.float32),
        t=np.asarray(t).astype(np.float32),
        truth_eta=np.asarray(truth_eta).astype(np.float32),
        pred_eta=np.asarray(pred_eta).astype(np.float32),
        truth_xi=np.asarray(truth_xi).astype(np.float32),
        pred_xi=np.asarray(pred_xi).astype(np.float32),
        truth_gxi=np.asarray(truth_gxi).astype(np.float32),
        pred_gxi=np.asarray(pred_gxi).astype(np.float32),
        depth=np.float32(depth),
    )
    return out_path


def save_rollout_gif(
    out_path: Path,
    *,
    title: str,
    x: np.ndarray,
    t: np.ndarray,
    truth_eta: np.ndarray,
    pred_eta: np.ndarray,
    truth_xi: np.ndarray,
    pred_xi: np.ndarray,
    truth_gxi: np.ndarray,
    pred_gxi: np.ndarray,
    fps: int = 30,
    n_frames: int = 200,
    dpi: int = 120,
) -> Path:
    # Defer import so the rollout scripts don't pay matplotlib startup cost when --gif is off.
    from solver.evals.render_rollout_movie import render_rollout_gif

    payload = {
        "x": np.asarray(x),
        "t": np.asarray(t),
        "truth_eta": np.asarray(truth_eta),
        "pred_eta": np.asarray(pred_eta),
        "truth_xi": np.asarray(truth_xi),
        "pred_xi": np.asarray(pred_xi),
        "truth_gxi": np.asarray(truth_gxi),
        "pred_gxi": np.asarray(pred_gxi),
    }
    return render_rollout_gif(payload, out_path, title=title, fps=fps, n_frames=n_frames, dpi=dpi)


def truth_gxi_from_state(
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    k: jnp.ndarray,
    depth: float,
    *,
    dno_order: int = 6,
    pad_factor: int = 8,
) -> jnp.ndarray:
    """Compute G(eta) xi via the high-order DNO series for ground-truth comparison."""
    return jax.vmap(
        lambda e, xi_row: dno_series_eval(e, xi_row, k, depth, dno_order, pad_factor=pad_factor)
    )(eta, xi)
