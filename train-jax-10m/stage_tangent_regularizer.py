"""GL2 stage-tangent regularizer.

Implements the training-time regularizer described in
`notes/gl2_stage_tangent_regularization_plan.md`. Pure functions only; the
trainer wires this into `loss_for_params` via a `lax.cond` gate.

Local-time-origin construction (t_n = 0). Because the linear flow is an
isometry in the energy norm, this is equivalent to the production
absolute-time construction up to floating-point noise — the parity test
in `tests/test_stage_tangent_regularizer.py` verifies this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from solver.solvers.dno_series_jax import dno_series_eval, myfft, myifft
from solver.solvers.time_integrator import (
    SpectralState,
    State,
    _hat_to_state,
    _state_to_hat,
    _tree_add,
    _tree_scale,
    apply_lowpass,
    dealiased_zakharov_xi_rhs,
    make_linear_dno_symbol,
    spectral_dx,
)


class GL2Coeffs(NamedTuple):
    c1: jnp.ndarray
    c2: jnp.ndarray
    a11: jnp.ndarray
    a12: jnp.ndarray
    a21: jnp.ndarray
    a22: jnp.ndarray


def gl2_coefficients(dtype: jnp.dtype = jnp.float64) -> GL2Coeffs:
    """Two-stage Gauss-Legendre Butcher tableau — must match production values in
    `solver/solvers/time_integrator.py:309-315`."""
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=dtype))
    c1 = jnp.asarray(0.5, dtype=dtype) - sqrt3 / jnp.asarray(6.0, dtype=dtype)
    c2 = jnp.asarray(0.5, dtype=dtype) + sqrt3 / jnp.asarray(6.0, dtype=dtype)
    a11 = jnp.asarray(0.25, dtype=dtype)
    a12 = jnp.asarray(0.25, dtype=dtype) - sqrt3 / jnp.asarray(6.0, dtype=dtype)
    a21 = jnp.asarray(0.25, dtype=dtype) + sqrt3 / jnp.asarray(6.0, dtype=dtype)
    a22 = jnp.asarray(0.25, dtype=dtype)
    return GL2Coeffs(c1=c1, c2=c2, a11=a11, a12=a12, a21=a21, a22=a22)


def soft_projector_band(
    k: jnp.ndarray, k_lo: float, k_hi: float,
    taper_lo: float = 8.0, taper_hi: float = 16.0,
) -> jnp.ndarray:
    """Smooth band-pass Fourier mask, real-valued, shape matches `k`.

    Rolls on cosine-half-cycle from `k_lo - taper_lo` to `k_lo`, plateaus at 1
    through `k_hi`, rolls off from `k_hi` to `k_hi + taper_hi`. Symmetric in
    positive/negative wavenumber."""
    ak = jnp.abs(k)
    rise_start = jnp.asarray(k_lo - taper_lo, dtype=ak.dtype)
    rise_end = jnp.asarray(k_lo, dtype=ak.dtype)
    fall_start = jnp.asarray(k_hi, dtype=ak.dtype)
    fall_end = jnp.asarray(k_hi + taper_hi, dtype=ak.dtype)
    taper_lo_safe = jnp.maximum(jnp.asarray(taper_lo, dtype=ak.dtype), 1e-9)
    taper_hi_safe = jnp.maximum(jnp.asarray(taper_hi, dtype=ak.dtype), 1e-9)

    rise = 0.5 * (1.0 - jnp.cos(jnp.pi * (ak - rise_start) / taper_lo_safe))
    fall = 0.5 * (1.0 + jnp.cos(jnp.pi * (ak - fall_start) / taper_hi_safe))
    mask = jnp.where(ak <= rise_start, 0.0, jnp.where(ak < rise_end, rise, 1.0))
    mask = jnp.where(
        ak >= fall_end, 0.0,
        jnp.where(ak > fall_start, fall, mask),
    )
    return mask


def soft_projector_low(k: jnp.ndarray, k_hi: float, taper: float = 8.0) -> jnp.ndarray:
    """Smooth low-pass Fourier mask: flat 1 below `k_hi - taper`, cosine roll-off to `k_hi`."""
    ak = jnp.abs(k)
    fall_start = jnp.asarray(k_hi - taper, dtype=ak.dtype)
    fall_end = jnp.asarray(k_hi, dtype=ak.dtype)
    taper_safe = jnp.maximum(jnp.asarray(taper, dtype=ak.dtype), 1e-9)
    fall = 0.5 * (1.0 + jnp.cos(jnp.pi * (ak - fall_start) / taper_safe))
    mask = jnp.where(ak <= fall_start, 1.0, jnp.where(ak < fall_end, fall, 0.0))
    return mask


def apply_projector_physical(field: jnp.ndarray, mask_k: jnp.ndarray) -> jnp.ndarray:
    """Apply a real-valued Fourier mask to a physical-space field (real output).

    `field` has shape (..., nx); `mask_k` is broadcastable to it."""
    nx = field.shape[-1]
    return myifft(mask_k.astype(myfft(field, nx).dtype) * myfft(field, nx))


def energy_norm_sq(
    deta: jnp.ndarray, dxi: jnp.ndarray, g0: jnp.ndarray,
    gravity: float = 1.0, dx: float = 1.0,
) -> jnp.ndarray:
    """Linear-Hamiltonian energy norm: g*||deta||^2 + <dxi, G0 dxi>, integrated in x.

    Uses Parseval on the xi term with our convention `myfft`. Returns a
    (batch,) array when `deta`/`dxi` are (batch, nx). `g0` broadcasts along
    the batch axis (either (nx,) or (batch, nx))."""
    nx = deta.shape[-1]
    e_eta = gravity * jnp.sum(deta * deta, axis=-1) * dx
    dxi_hat = myfft(dxi, nx)
    e_xi = (dx / nx) * jnp.sum(g0.astype(dxi_hat.real.dtype) * (dxi_hat.conj() * dxi_hat).real, axis=-1)
    return e_eta + e_xi


def _make_probe_raw(
    rng: jax.Array, kind: str, projector_mask: jnp.ndarray,
    shape: tuple[int, ...], dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random field, projector-band it, zero out the un-probed channel."""
    key_e, key_x = jax.random.split(rng)
    z_eta = jax.random.normal(key_e, shape, dtype=dtype)
    z_xi = jax.random.normal(key_x, shape, dtype=dtype)
    deta = apply_projector_physical(z_eta, projector_mask)
    dxi = apply_projector_physical(z_xi, projector_mask)
    if kind == "eta":
        dxi = jnp.zeros_like(dxi)
    elif kind == "xi":
        deta = jnp.zeros_like(deta)
    elif kind != "joint":
        raise ValueError(f"probe kind must be 'eta', 'xi', or 'joint'; got {kind!r}")
    return deta, dxi


def construct_probe(
    rng: jax.Array, kind: str, projector_mask: jnp.ndarray,
    g0: jnp.ndarray, gravity: float, dx: float,
    shape: tuple[int, ...], dtype: jnp.dtype, energy_floor: float = 1e-30,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (deta, dxi) probe normalized to unit energy norm per sample."""
    deta, dxi = _make_probe_raw(rng, kind, projector_mask, shape, dtype)
    e = energy_norm_sq(deta, dxi, g0, gravity=gravity, dx=dx)
    scale = 1.0 / jnp.sqrt(e + energy_floor)
    return deta * scale[..., None], dxi * scale[..., None]


def picard_step(
    F: Callable[[SpectralState, jnp.ndarray], SpectralState],
    stages: tuple[SpectralState, SpectralState],
    v0: SpectralState,
    dt: jnp.ndarray,
    coeffs: GL2Coeffs,
) -> tuple[SpectralState, SpectralState]:
    """One Picard iteration of the two-stage GL2 stage system in the v-frame.

    T(V) = (v0, v0) + dt * (A ⊗ I) @ (F(V1, c1*dt), F(V2, c2*dt))

    Mirrors the body_fn inside `gauss_legendre_2_if_step` at
    `solver/solvers/time_integrator.py:320-338`. Callers thread `t_local=0`
    (local origin) or `t_local=t_n` (absolute) into F via a closure."""
    stage1, stage2 = stages
    f1 = F(stage1, coeffs.c1 * dt)
    f2 = F(stage2, coeffs.c2 * dt)
    cand1 = _tree_add(
        v0,
        _tree_scale(_tree_add(_tree_scale(f1, coeffs.a11), _tree_scale(f2, coeffs.a12)), dt),
    )
    cand2 = _tree_add(
        v0,
        _tree_scale(_tree_add(_tree_scale(f1, coeffs.a21), _tree_scale(f2, coeffs.a22)), dt),
    )
    return cand1, cand2


@dataclass(frozen=True)
class ReferenceIFParams:
    """Bundle of scalar/array knobs for the reference (order-N Craig-Sulem) IF-RHS.

    Kept separate from `SolverParams` so it can carry a per-sample `depth`/`g0`
    without redefining production types."""
    k: jnp.ndarray
    g0: jnp.ndarray
    depth: jnp.ndarray
    gravity: float
    dno_order: int
    pad_factor: int
    filter_fraction: float
    nx: int


def _apply_linear_flow_batched(
    state_hat: SpectralState, tau: jnp.ndarray,
    k: jnp.ndarray, g0: jnp.ndarray, gravity: float,
) -> SpectralState:
    """Batched linear-flow propagator. Mirrors
    `solver/solvers/time_integrator.py:183-190` but takes `g0` explicitly so
    it can carry a batch axis (per-sample depth)."""
    omega = jnp.sqrt(gravity * g0)
    coswt = jnp.cos(omega * tau)
    sin_over_omega = jnp.where(omega > 0, jnp.sin(omega * tau) / jnp.maximum(omega, 1e-30), 0.0)
    eta_hat = coswt * state_hat.eta_hat + sin_over_omega * g0 * state_hat.xi_hat
    xi_hat = coswt * state_hat.xi_hat - sin_over_omega * gravity * state_hat.eta_hat
    return SpectralState(eta_hat=eta_hat, xi_hat=xi_hat)


def _rhs_nonlinear_from_gxi(
    state: State, gxi: jnp.ndarray,
    k: jnp.ndarray, g0: jnp.ndarray,
    filter_fraction: float,
) -> State:
    """Zakharov nonlinear RHS given a physical `gxi`. Mirrors
    `solver/solvers/time_integrator.py:211-231`. Note the filter is applied
    *inside* the RHS in production; we preserve that convention."""
    eta_x = spectral_dx(state.eta, k)
    xi_x = spectral_dx(state.xi, k)
    linear_gxi = myifft(g0.astype(myfft(state.xi, state.xi.shape[-1]).dtype)
                        * myfft(state.xi, state.xi.shape[-1]))
    eta_t = gxi - linear_gxi
    xi_t = dealiased_zakharov_xi_rhs(eta_x, xi_x, gxi)
    if filter_fraction < 1.0:
        eta_t = apply_lowpass(eta_t, k, filter_fraction)
        xi_t = apply_lowpass(xi_t, k, filter_fraction)
    return State(eta=eta_t, xi=xi_t)


def make_reference_F(rp: ReferenceIFParams) -> Callable[[SpectralState, jnp.ndarray], SpectralState]:
    """Build F_ref(v_hat, t_local) using the order-N Craig-Sulem series as the DNO.

    Vmap over the batch axis so `depth` (batched) is honored per-sample.
    `t_local` is a scalar; batching lives inside the state arrays."""
    def dno_per_sample(eta_i: jnp.ndarray, xi_i: jnp.ndarray, depth_i: jnp.ndarray) -> jnp.ndarray:
        return dno_series_eval(eta_i, xi_i, rp.k, depth_i, rp.dno_order, pad_factor=rp.pad_factor)
    dno_batched = jax.vmap(dno_per_sample, in_axes=(0, 0, 0))

    def F(v_hat: SpectralState, t_local: jnp.ndarray) -> SpectralState:
        phys_hat = _apply_linear_flow_batched(v_hat, t_local, rp.k, rp.g0, rp.gravity)
        phys_state = _hat_to_state(phys_hat)
        gxi = dno_batched(phys_state.eta, phys_state.xi, rp.depth)
        nl_state = _rhs_nonlinear_from_gxi(phys_state, gxi, rp.k, rp.g0, rp.filter_fraction)
        nl_hat = _state_to_hat(nl_state, rp.nx)
        return _apply_linear_flow_batched(nl_hat, -t_local, rp.k, rp.g0, rp.gravity)
    return F


def make_model_F(
    apply_fn: Callable,
    model_params,
    batch_depth: jnp.ndarray,
    norm_inputs_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    denorm_targets_fn: Callable[[jnp.ndarray], jnp.ndarray],
    filter_predictions_fn: Callable[[jnp.ndarray], jnp.ndarray],
    rp: ReferenceIFParams,
) -> Callable[[SpectralState, jnp.ndarray], SpectralState]:
    """Build F_model(v_hat, t_local) using the learned surrogate for gxi.

    `apply_fn`/`model_params`/`batch_depth`/normalizers are captured as
    closure vars; F participates in autodiff w.r.t. model_params through
    `apply_fn`."""
    def F(v_hat: SpectralState, t_local: jnp.ndarray) -> SpectralState:
        phys_hat = _apply_linear_flow_batched(v_hat, t_local, rp.k, rp.g0, rp.gravity)
        phys_state = _hat_to_state(phys_hat)
        inputs = norm_inputs_fn(phys_state.eta, phys_state.xi)
        preds = apply_fn({"params": model_params}, inputs, batch_depth)
        preds = filter_predictions_fn(preds)
        gxi = denorm_targets_fn(preds)[..., 0]
        nl_state = _rhs_nonlinear_from_gxi(phys_state, gxi, rp.k, rp.g0, rp.filter_fraction)
        nl_hat = _state_to_hat(nl_state, rp.nx)
        return _apply_linear_flow_batched(nl_hat, -t_local, rp.k, rp.g0, rp.gravity)
    return F


def finite_secant(
    T_of_V: Callable[[tuple[SpectralState, SpectralState]], tuple[SpectralState, SpectralState]],
    V_bar: tuple[SpectralState, SpectralState],
    probe: tuple[SpectralState, SpectralState],
    eps: jnp.ndarray,
) -> tuple[SpectralState, SpectralState]:
    """Forward finite secant [T(V_bar + eps*probe) - T(V_bar)] / eps."""
    V_plus = jax.tree_util.tree_map(lambda vb, p: vb + eps * p, V_bar, probe)
    T_plus = T_of_V(V_plus)
    T_bar = T_of_V(V_bar)
    return jax.tree_util.tree_map(lambda tp, tb: (tp - tb) / eps, T_plus, T_bar)


def _project_and_hat_to_phys(
    w_hat: SpectralState, projector_mask: jnp.ndarray,
) -> State:
    """Extract physical (eta, xi) from a spectral tangent, applying the Fourier mask.

    Because the projector is applied in Fourier space, we can equivalently
    multiply the hat directly then ifft."""
    dtype = w_hat.eta_hat.dtype
    mask = projector_mask.astype(dtype)
    return State(eta=myifft(mask * w_hat.eta_hat), xi=myifft(mask * w_hat.xi_hat))


def stage_match_loss(
    w_theta: tuple[SpectralState, SpectralState],
    w_ref: tuple[SpectralState, SpectralState],
    projector_B_mask: jnp.ndarray,
    g0: jnp.ndarray,
    gravity: float,
    dx: float,
    response_floor: float,
) -> jnp.ndarray:
    """L_match = mean_b [ Σ_s (1/2) * ||P_B (w_theta - w_ref)||_E^2 / (||P_B w_ref||_E^2 + floor) ]."""
    def per_stage(wt: SpectralState, wr: SpectralState) -> jnp.ndarray:
        wt_p = _project_and_hat_to_phys(wt, projector_B_mask)
        wr_p = _project_and_hat_to_phys(wr, projector_B_mask)
        num = energy_norm_sq(wt_p.eta - wr_p.eta, wt_p.xi - wr_p.xi, g0, gravity, dx)
        den = energy_norm_sq(wr_p.eta, wr_p.xi, g0, gravity, dx) + response_floor
        return num / den
    l1 = per_stage(w_theta[0], w_ref[0])
    l2 = per_stage(w_theta[1], w_ref[1])
    return jnp.mean(0.5 * l1 + 0.5 * l2)


def stage_gain_loss(
    w_theta: tuple[SpectralState, SpectralState],
    w_ref: tuple[SpectralState, SpectralState],
    probe: tuple[SpectralState, SpectralState],
    projector_B_mask: jnp.ndarray,
    g0: jnp.ndarray,
    gravity: float,
    dx: float,
    gain_margin_rel: float,
    gain_margin_abs: float,
    eps_norm: float,
) -> jnp.ndarray:
    """Hinge on model gain exceeding reference gain by a margin.

    r = ||P_B w||_E / (||q||_E + eps_norm)
    loss = mean(relu(r_theta - r_ref - (rel * r_ref + abs))^2)."""
    def per_stage(wt: SpectralState, wr: SpectralState, q: SpectralState) -> jnp.ndarray:
        wt_p = _project_and_hat_to_phys(wt, projector_B_mask)
        wr_p = _project_and_hat_to_phys(wr, projector_B_mask)
        q_p = State(eta=myifft(q.eta_hat), xi=myifft(q.xi_hat))
        r_theta = jnp.sqrt(energy_norm_sq(wt_p.eta, wt_p.xi, g0, gravity, dx))
        r_ref = jnp.sqrt(energy_norm_sq(wr_p.eta, wr_p.xi, g0, gravity, dx))
        q_norm = jnp.sqrt(energy_norm_sq(q_p.eta, q_p.xi, g0, gravity, dx))
        r_theta = r_theta / (q_norm + eps_norm)
        r_ref = r_ref / (q_norm + eps_norm)
        margin = gain_margin_rel * r_ref + gain_margin_abs
        return jax.nn.relu(r_theta - r_ref - margin) ** 2
    g1 = per_stage(w_theta[0], w_ref[0], probe[0])
    g2 = per_stage(w_theta[1], w_ref[1], probe[1])
    return jnp.mean(0.5 * g1 + 0.5 * g2)


# ---------------------------------------------------------------------------
# Full regularizer driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRegConfig:
    interval: int
    microbatch: int
    warmup_steps: int
    k_lo: float
    k_hi: float
    k_low_hi: float
    taper_lo: float
    taper_hi: float
    taper_low: float
    eps_min: float
    eps_max: float
    gain_margin_rel: float
    gain_margin_abs: float
    response_floor: float
    eps_norm: float
    reference_order: int
    reference_pad: int
    reference_picard_predictors: int
    dt: float
    gravity: float
    filter_fraction: float


def _promote_state(eta: jnp.ndarray, xi: jnp.ndarray, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    return eta.astype(dtype), xi.astype(dtype)


def sample_microbatch(
    rng: jax.Array, eta: jnp.ndarray, xi: jnp.ndarray, depth_per_sample: jnp.ndarray,
    local_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Deterministic random subset of the per-device batch, no replacement."""
    n = eta.shape[0]
    perm = jax.random.permutation(rng, n)
    idx = perm[:local_size]
    return eta[idx], xi[idx], depth_per_sample[idx]


def _log_uniform(rng: jax.Array, lo: float, hi: float, shape, dtype) -> jnp.ndarray:
    log_lo = jnp.log(jnp.asarray(lo, dtype=dtype))
    log_hi = jnp.log(jnp.asarray(hi, dtype=dtype))
    u = jax.random.uniform(rng, shape, dtype=dtype, minval=log_lo, maxval=log_hi)
    return jnp.exp(u)


def compute_stage_reg(
    rng: jax.Array,
    apply_fn: Callable,
    model_params,
    eta_phys: jnp.ndarray,
    xi_phys: jnp.ndarray,
    depth_phys: jnp.ndarray,
    batch_depth_local: jnp.ndarray,
    norm_inputs_fn: Callable,
    denorm_targets_fn: Callable,
    filter_predictions_fn: Callable,
    k: jnp.ndarray,
    dx: float,
    cfg: StageRegConfig,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Compute (L_match, L_gain, diagnostics) for one microbatch.

    Args:
      eta_phys/xi_phys: (microbatch, nx) physical fields.
      depth_phys: (microbatch,) per-sample physical depth.
      batch_depth_local: (microbatch, k) the model's depth-input tensor for the same
        microbatch (already log-clipped to match the model's normalization).
      norm_inputs_fn/denorm_targets_fn/filter_predictions_fn: trainer closures.
      k: (nx,) wavenumber grid.
      cfg: static config.
    """
    B, nx = eta_phys.shape
    g0 = jax.vmap(lambda h: make_linear_dno_symbol(k.astype(dtype), h))(depth_phys.astype(dtype))
    rp = ReferenceIFParams(
        k=k.astype(dtype),
        g0=g0,
        depth=depth_phys.astype(dtype),
        gravity=cfg.gravity,
        dno_order=cfg.reference_order,
        pad_factor=cfg.reference_pad,
        filter_fraction=cfg.filter_fraction,
        nx=nx,
    )
    F_ref = make_reference_F(rp)
    F_model = make_model_F(
        apply_fn=apply_fn, model_params=model_params,
        batch_depth=batch_depth_local,
        norm_inputs_fn=norm_inputs_fn,
        denorm_targets_fn=denorm_targets_fn,
        filter_predictions_fn=filter_predictions_fn,
        rp=rp,
    )

    coeffs = gl2_coefficients(dtype=dtype)
    dt_arr = jnp.asarray(cfg.dt, dtype=dtype)

    v_n_hat = SpectralState(eta_hat=myfft(eta_phys, nx), xi_hat=myfft(xi_phys, nx))
    V0 = (v_n_hat, v_n_hat)
    # Reference-Picard predictor: run picard_step reference_picard_predictors times.
    V_bar = V0
    for _ in range(int(cfg.reference_picard_predictors)):
        V_bar = picard_step(F_ref, V_bar, v_n_hat, dt_arr, coeffs)
    V_bar = jax.lax.stop_gradient(V_bar)

    # Build probes: two probe types per fire, cycling eta/xi/joint by RNG-split.
    # Sample a scalar eps in LogUniform(eps_min, eps_max), applied to all samples.
    key_eps, key_mid, key_low, key_kind = jax.random.split(rng, 4)
    eps = _log_uniform(key_eps, cfg.eps_min, cfg.eps_max, (), dtype)

    kind_id = jax.random.randint(key_kind, (), 0, 3)
    # Mid-to-mid probe: P_B on both eta and xi.
    proj_B_mask = soft_projector_band(
        k.astype(dtype), cfg.k_lo, cfg.k_hi, cfg.taper_lo, cfg.taper_hi,
    )
    proj_L_mask = soft_projector_low(k.astype(dtype), cfg.k_low_hi, cfg.taper_low)

    def _make(kind_str: str, key: jax.Array, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return construct_probe(
            key, kind_str, mask, g0, cfg.gravity, dx, (B, nx), dtype,
        )
    # jax.lax.switch selects a probe kind by kind_id. Same kind applied to both stages.
    key_mid_e, key_mid_x, key_mid_j = jax.random.split(key_mid, 3)
    key_low_e, key_low_x, key_low_j = jax.random.split(key_low, 3)
    mid_deta, mid_dxi = jax.lax.switch(
        kind_id,
        (
            lambda: _make("eta", key_mid_e, proj_B_mask),
            lambda: _make("xi", key_mid_x, proj_B_mask),
            lambda: _make("joint", key_mid_j, proj_B_mask),
        ),
    )
    low_deta, low_dxi = jax.lax.switch(
        kind_id,
        (
            lambda: _make("eta", key_low_e, proj_L_mask),
            lambda: _make("xi", key_low_x, proj_L_mask),
            lambda: _make("joint", key_low_j, proj_L_mask),
        ),
    )

    # Compose probes into spectral pytree matching V shape (same probe for both stages).
    def to_spectral_probe(deta: jnp.ndarray, dxi: jnp.ndarray) -> SpectralState:
        return SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe_mid = to_spectral_probe(mid_deta, mid_dxi)
    probe_low = to_spectral_probe(low_deta, low_dxi)
    probe_mid_V = (probe_mid, probe_mid)
    probe_low_V = (probe_low, probe_low)

    # Picard maps as callables T(V) = one_step(F, V, v_n, dt, coeffs).
    def T_ref(V):
        return picard_step(F_ref, V, v_n_hat, dt_arr, coeffs)

    def T_model(V):
        return picard_step(F_model, V, v_n_hat, dt_arr, coeffs)

    # Response secants (mid probe).
    w_ref_mid = jax.lax.stop_gradient(finite_secant(T_ref, V_bar, probe_mid_V, eps))
    w_theta_mid = finite_secant(T_model, V_bar, probe_mid_V, eps)

    # Response secants (low probe).
    w_ref_low = jax.lax.stop_gradient(finite_secant(T_ref, V_bar, probe_low_V, eps))
    w_theta_low = finite_secant(T_model, V_bar, probe_low_V, eps)

    l_match_mid = stage_match_loss(
        w_theta_mid, w_ref_mid, proj_B_mask, g0, cfg.gravity, dx, cfg.response_floor,
    )
    l_match_low = stage_match_loss(
        w_theta_low, w_ref_low, proj_B_mask, g0, cfg.gravity, dx, cfg.response_floor,
    )
    l_gain_mid = stage_gain_loss(
        w_theta_mid, w_ref_mid, probe_mid_V, proj_B_mask, g0, cfg.gravity, dx,
        cfg.gain_margin_rel, cfg.gain_margin_abs, cfg.eps_norm,
    )
    l_gain_low = stage_gain_loss(
        w_theta_low, w_ref_low, probe_low_V, proj_B_mask, g0, cfg.gravity, dx,
        cfg.gain_margin_rel, cfg.gain_margin_abs, cfg.eps_norm,
    )

    l_match = 0.5 * (l_match_mid + l_match_low)
    l_gain = 0.5 * (l_gain_mid + l_gain_low)

    diagnostics = {
        "stage_reg_match_mid": l_match_mid,
        "stage_reg_match_low": l_match_low,
        "stage_reg_gain_mid": l_gain_mid,
        "stage_reg_gain_low": l_gain_low,
        "stage_reg_eps": eps,
        "stage_reg_kind_id": kind_id,
    }
    return l_match, l_gain, diagnostics
