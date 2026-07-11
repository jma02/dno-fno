"""CPU tests for `stage_tangent_regularizer.py`.

Per plan §8. No pytest dep — standalone `assert`-based script. Run with

    JAX_ENABLE_X64=True JAX_PLATFORMS=cpu uv run python train-jax-10m/tests_stage_tangent_regularizer.py

Prints one line per test with PASS/FAIL and exits nonzero on any failure.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train-jax-10m"))
sys.path.insert(0, str(REPO / "models" / "dno-net"))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from solver.solvers.dno_series_jax import build_grid, myfft
from solver.solvers.time_integrator import (
    SolverParams, SpectralState, State, apply_linear_flow_hat,
    make_linear_dno_symbol, _state_to_hat, _hat_to_state,
)
from stage_tangent_regularizer import (
    ReferenceIFParams, StageRegConfig,
    apply_projector_physical, compute_stage_reg, construct_probe, energy_norm_sq,
    finite_secant, gl2_coefficients, make_reference_F, picard_step,
    soft_projector_band, soft_projector_low, stage_gain_loss, stage_match_loss,
)


PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def _report(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    (PASS if ok else FAIL).append((name, detail) if not ok else name)


def _run(name: str, fn):
    try:
        fn()
        _report(name, True)
    except Exception as e:  # noqa: BLE001 — test harness
        _report(name, False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Math tests
# ---------------------------------------------------------------------------


def test_gl2_coeffs_match_production():
    """Coefficients must equal the exact values inlined in gauss_legendre_2_if_step."""
    c = gl2_coefficients(dtype=jnp.float64)
    sqrt3 = float(np.sqrt(3.0))
    assert abs(float(c.c1) - (0.5 - sqrt3 / 6.0)) < 1e-14
    assert abs(float(c.c2) - (0.5 + sqrt3 / 6.0)) < 1e-14
    assert float(c.a11) == 0.25
    assert abs(float(c.a12) - (0.25 - sqrt3 / 6.0)) < 1e-14
    assert abs(float(c.a21) - (0.25 + sqrt3 / 6.0)) < 1e-14
    assert float(c.a22) == 0.25


def _small_state(nx: int = 32, batch: int = 2, depth: float = 0.5, seed: int = 0):
    x, k = build_grid(nx, 2.0 * np.pi)
    key = jax.random.PRNGKey(seed)
    key_e, key_x = jax.random.split(key)
    eta = 0.02 * jax.random.normal(key_e, (batch, nx), dtype=jnp.float64)
    xi = 0.05 * jax.random.normal(key_x, (batch, nx), dtype=jnp.float64)
    depths = jnp.full((batch,), depth, dtype=jnp.float64)
    return x, k.astype(jnp.float64), eta, xi, depths


def test_picard_map_matches_production_body_fn():
    """One Picard iteration in the training helper == one iteration of
    production gauss_legendre_2_if_step body, evaluated identically."""
    nx, batch, dt = 32, 1, 0.01
    x, k, eta, xi, depths = _small_state(nx=nx, batch=batch)
    depth = float(depths[0])
    g0 = make_linear_dno_symbol(k, depth)

    # Production body: replicate one Picard update of gauss_legendre_2_if_step body_fn
    # for absolute time t=0 (local origin).
    params = SolverParams(
        nx=nx, length=2.0 * np.pi, depth=depth, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
        k=k, g0=g0,
    )
    state = State(eta=eta[0], xi=xi[0])
    state_hat = _state_to_hat(state, nx)
    v0 = apply_linear_flow_hat(state_hat, 0.0, params)
    from solver.solvers.time_integrator import rhs_nonlinear_if, _tree_add, _tree_scale
    coeffs = gl2_coefficients(dtype=jnp.float64)
    f1 = rhs_nonlinear_if(v0, float(coeffs.c1) * dt, params)
    f2 = rhs_nonlinear_if(v0, float(coeffs.c2) * dt, params)
    cand1_prod = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, coeffs.a11), _tree_scale(f2, coeffs.a12)), dt))
    cand2_prod = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, coeffs.a21), _tree_scale(f2, coeffs.a22)), dt))

    # Training helper (batch axis of 1 for parity).
    rp = ReferenceIFParams(
        k=k, g0=jnp.broadcast_to(g0, (batch, nx)), depth=depths,
        gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0, nx=nx,
    )
    F_ref = make_reference_F(rp)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    stages = (v_n_hat, v_n_hat)
    cand1_train, cand2_train = picard_step(F_ref, stages, v_n_hat, jnp.asarray(dt), coeffs)

    err_eta_1 = float(jnp.max(jnp.abs(cand1_train.eta_hat[0] - cand1_prod.eta_hat)))
    err_eta_2 = float(jnp.max(jnp.abs(cand2_train.eta_hat[0] - cand2_prod.eta_hat)))
    err_xi_1 = float(jnp.max(jnp.abs(cand1_train.xi_hat[0] - cand1_prod.xi_hat)))
    err_xi_2 = float(jnp.max(jnp.abs(cand2_train.xi_hat[0] - cand2_prod.xi_hat)))
    tol = 1e-10
    assert err_eta_1 < tol, f"stage1 eta_hat diff {err_eta_1}"
    assert err_eta_2 < tol, f"stage2 eta_hat diff {err_eta_2}"
    assert err_xi_1 < tol, f"stage1 xi_hat diff {err_xi_1}"
    assert err_xi_2 < tol, f"stage2 xi_hat diff {err_xi_2}"


def test_linear_flow_preserves_energy_norm():
    """||apply_linear_flow(v, tau)||_E == ||v||_E for all tau (linear Hamiltonian isometry).

    Energy is compared in spectral space to avoid the myfft-Nyquist-zero convention
    losing energy through a physical roundtrip (myfft zeros the Nyquist mode, so a
    physical→spectral→physical roundtrip of a random-signal that has nonzero Nyquist
    is not identity)."""
    nx, batch = 64, 3
    dx = float(2.0 * np.pi / nx)
    _, k, eta, xi, depths = _small_state(nx=nx, batch=batch, depth=0.3, seed=1)
    g0 = jax.vmap(lambda h: make_linear_dno_symbol(k, h))(depths)
    v_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))

    def spectral_energy(vh: SpectralState) -> jnp.ndarray:
        return (dx / nx) * jnp.sum(
            1.0 * (vh.eta_hat.conj() * vh.eta_hat).real
            + g0 * (vh.xi_hat.conj() * vh.xi_hat).real,
            axis=-1,
        )

    e0 = spectral_energy(v_hat)
    from stage_tangent_regularizer import _apply_linear_flow_batched
    for tau in [0.001, 0.05, 0.5, 1.7, -0.4]:
        v_tau = _apply_linear_flow_batched(v_hat, jnp.asarray(tau), k, g0, gravity=1.0)
        e_tau = spectral_energy(v_tau)
        rel = float(jnp.max(jnp.abs(e_tau - e0) / (jnp.abs(e0) + 1e-30)))
        assert rel < 1e-12, f"tau={tau} rel energy drift {rel:.3e}"


def test_local_vs_absolute_time_agree():
    """T_local(V; t_n=0, dt) == e^{-L t_n} * T_absolute(e^{+L t_n} V; t_n, dt) * e^{+L t_n}.

    Since we only care about physical states (i.e. applied linear flow back to
    real time), local and absolute constructions must produce the same physical
    u_hat_new modulo float error. Equivalently, the energy norm of the
    difference must be tiny."""
    nx, batch, dt = 64, 2, 0.01
    _, k, eta, xi, depths = _small_state(nx=nx, batch=batch, seed=2)
    depth = float(depths[0])
    g0 = make_linear_dno_symbol(k, depth)
    coeffs = gl2_coefficients(dtype=jnp.float64)

    def run_one_picard(t_n: float):
        params = SolverParams(
            nx=nx, length=2.0 * np.pi, depth=depth, gravity=1.0,
            dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
            k=k, g0=g0,
        )
        state = State(eta=eta[0], xi=xi[0])
        state_hat = _state_to_hat(state, nx)
        v0 = apply_linear_flow_hat(state_hat, -t_n, params)
        from solver.solvers.time_integrator import rhs_nonlinear_if, _tree_add, _tree_scale
        f1 = rhs_nonlinear_if(v0, t_n + float(coeffs.c1) * dt, params)
        f2 = rhs_nonlinear_if(v0, t_n + float(coeffs.c2) * dt, params)
        cand1 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, coeffs.a11), _tree_scale(f2, coeffs.a12)), dt))
        cand2 = _tree_add(v0, _tree_scale(_tree_add(_tree_scale(f1, coeffs.a21), _tree_scale(f2, coeffs.a22)), dt))
        # Map both stages back to physical (u_hat) at their true times: stage_i lives at t_n + c_i * dt.
        u1_hat = apply_linear_flow_hat(cand1, t_n + float(coeffs.c1) * dt, params)
        u2_hat = apply_linear_flow_hat(cand2, t_n + float(coeffs.c2) * dt, params)
        return u1_hat, u2_hat

    u_local = run_one_picard(0.0)
    u_abs = run_one_picard(1.7)
    for i, (a, b) in enumerate(zip(u_local, u_abs)):
        diff_eta = float(jnp.max(jnp.abs(a.eta_hat - b.eta_hat)))
        diff_xi = float(jnp.max(jnp.abs(a.xi_hat - b.xi_hat)))
        assert diff_eta < 1e-9 and diff_xi < 1e-9, (
            f"stage{i+1}: physical u_hat differs between local (t=0) and absolute (t=1.7) "
            f"eta {diff_eta:.3e} xi {diff_xi:.3e}"
        )


def test_projector_support_and_reality():
    """Band projector zeroes modes outside its taper region; low projector is 1 near k=0;
    both preserve real-valued fields.

    Uses nx=256 so |k| goes up to 128 and the plan's default band [32, 112] w/
    taper 8/16 fits inside without hitting Nyquist."""
    nx = 256
    _, k = build_grid(nx, 2.0 * np.pi)
    k = k.astype(jnp.float64)
    mask_B = soft_projector_band(k, k_lo=32.0, k_hi=112.0, taper_lo=8.0, taper_hi=16.0)
    mask_L = soft_projector_low(k, k_hi=32.0, taper=8.0)

    # Support checks
    ak = jnp.abs(k)
    below_B = ak < 32.0 - 8.0
    plateau_B = (ak >= 32.0) & (ak <= 112.0)
    above_B = ak >= 112.0 + 16.0
    assert int(jnp.sum(above_B)) > 0, "test grid must include modes above P_B"
    assert float(jnp.max(mask_B[below_B])) < 1e-12
    assert float(jnp.min(mask_B[plateau_B])) > 1.0 - 1e-12
    assert float(jnp.max(mask_B[above_B])) < 1e-12

    plateau_L = ak <= 32.0 - 8.0
    beyond_L = ak >= 32.0
    assert float(jnp.min(mask_L[plateau_L])) > 1.0 - 1e-12
    assert float(jnp.max(mask_L[beyond_L])) < 1e-12

    # Reality preservation
    x = 0.5 * jax.random.normal(jax.random.PRNGKey(3), (2, nx), dtype=jnp.float64)
    y = apply_projector_physical(x, mask_B)
    assert float(jnp.max(jnp.abs(jnp.imag(y)))) < 1e-12


def test_probe_unit_energy_norm():
    """construct_probe(eta|xi|joint) produces per-sample unit energy norm."""
    nx, batch = 128, 4
    _, k = build_grid(nx, 2.0 * np.pi)
    k = k.astype(jnp.float64)
    depths = jnp.array([0.05, 0.2, 0.5, 1.0], dtype=jnp.float64)
    g0 = jax.vmap(lambda h: make_linear_dno_symbol(k, h))(depths)
    mask = soft_projector_band(k, 32.0, 112.0, 8.0, 16.0)
    dx = float(2.0 * np.pi / nx)
    for kind in ("eta", "xi", "joint"):
        deta, dxi = construct_probe(
            jax.random.PRNGKey(4), kind, mask, g0, 1.0, dx,
            (batch, nx), jnp.float64,
        )
        e = energy_norm_sq(deta, dxi, g0, gravity=1.0, dx=dx)
        rel = float(jnp.max(jnp.abs(e - 1.0)))
        assert rel < 1e-9, f"probe kind={kind} rel norm err {rel:.3e}"


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


def _reference_only_setup(nx: int = 64, batch: int = 2, dt: float = 0.01, depth: float = 0.3):
    _, k, eta, xi, depths = _small_state(nx=nx, batch=batch, depth=depth, seed=5)
    g0 = jax.vmap(lambda h: make_linear_dno_symbol(k, h))(depths)
    rp = ReferenceIFParams(
        k=k, g0=g0, depth=depths, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0, nx=nx,
    )
    return k, eta, xi, depths, g0, rp


def test_model_equals_reference_gives_zero_loss():
    """If F_model == F_ref, both losses collapse to (near-zero within float noise)."""
    nx, batch, dt = 64, 2, 0.01
    k, eta, xi, depths, g0, rp = _reference_only_setup(nx, batch, dt)
    F = make_reference_F(rp)
    coeffs = gl2_coefficients(jnp.float64)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    V_bar = picard_step(F, (v_n_hat, v_n_hat), v_n_hat, jnp.asarray(dt), coeffs)
    dx = float(2.0 * np.pi / nx)
    mask_B = soft_projector_band(k, 32.0, 112.0, 8.0, 16.0)
    deta, dxi = construct_probe(jax.random.PRNGKey(6), "joint", mask_B, g0, 1.0, dx, (batch, nx), jnp.float64)
    q_hat = SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe = (q_hat, q_hat)
    T = lambda V: picard_step(F, V, v_n_hat, jnp.asarray(dt), coeffs)
    w_ref = finite_secant(T, V_bar, probe, jnp.asarray(1e-4))
    w_theta = w_ref  # identical
    lm = float(stage_match_loss(w_theta, w_ref, mask_B, g0, 1.0, dx, response_floor=1e-8))
    lg = float(stage_gain_loss(w_theta, w_ref, probe, mask_B, g0, 1.0, dx, 0.05, 1e-3, 1e-8))
    assert lm < 1e-14, f"L_match should be ~0 for identical, got {lm}"
    assert lg == 0.0, f"L_gain should be 0 for identical (relu at margin), got {lg}"


def test_amplified_mid_band_response_increases_losses():
    """A model whose response is 3× the reference in the P_B band produces both a
    positive L_match and a positive L_gain.

    Uses band [8, 24] tapers (2, 4) scaled to fit inside nx=64 (max |k| = 32) so
    the projector has genuine plateau support."""
    nx, batch, dt = 64, 2, 0.01
    k, eta, xi, depths, g0, rp = _reference_only_setup(nx, batch, dt)
    F = make_reference_F(rp)
    coeffs = gl2_coefficients(jnp.float64)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    V_bar = picard_step(F, (v_n_hat, v_n_hat), v_n_hat, jnp.asarray(dt), coeffs)
    dx = float(2.0 * np.pi / nx)
    mask_B = soft_projector_band(k, 8.0, 24.0, 2.0, 4.0)
    deta, dxi = construct_probe(jax.random.PRNGKey(7), "joint", mask_B, g0, 1.0, dx, (batch, nx), jnp.float64)
    q_hat = SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe = (q_hat, q_hat)
    T = lambda V: picard_step(F, V, v_n_hat, jnp.asarray(dt), coeffs)
    w_ref = finite_secant(T, V_bar, probe, jnp.asarray(1e-4))
    # Synthetic w_theta = 3 * w_ref inside P_B (multiply spectral coeffs by mask factor).
    factor = jnp.where(mask_B > 0.5, 3.0, 1.0).astype(w_ref[0].eta_hat.dtype)
    def scale(ws):
        return SpectralState(eta_hat=factor * ws.eta_hat, xi_hat=factor * ws.xi_hat)
    w_theta = (scale(w_ref[0]), scale(w_ref[1]))
    lm = float(stage_match_loss(w_theta, w_ref, mask_B, g0, 1.0, dx, 1e-8))
    lg = float(stage_gain_loss(w_theta, w_ref, probe, mask_B, g0, 1.0, dx, 0.05, 1e-3, 1e-8))
    assert lm > 1e-3, f"L_match should be positive, got {lm}"
    assert lg > 1e-6, f"L_gain should be positive, got {lg}"


def test_low_to_mid_probe_detects_synthetic_mid_response():
    """Probe in P_L (low modes) — but score P_B (mid) — surfaces a synthetic
    mid-band response emanating from a low-mode perturbation."""
    nx, batch, dt = 128, 1, 0.01
    k, eta, xi, depths, g0, rp = _reference_only_setup(nx, batch, dt)
    F = make_reference_F(rp)
    coeffs = gl2_coefficients(jnp.float64)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    V_bar = picard_step(F, (v_n_hat, v_n_hat), v_n_hat, jnp.asarray(dt), coeffs)
    dx = float(2.0 * np.pi / nx)
    mask_B = soft_projector_band(k, 32.0, 112.0, 8.0, 16.0)
    mask_L = soft_projector_low(k, 32.0, 8.0)
    deta, dxi = construct_probe(jax.random.PRNGKey(8), "joint", mask_L, g0, 1.0, dx, (batch, nx), jnp.float64)
    q_hat = SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe = (q_hat, q_hat)
    T = lambda V: picard_step(F, V, v_n_hat, jnp.asarray(dt), coeffs)
    w_ref = finite_secant(T, V_bar, probe, jnp.asarray(1e-4))
    # Inject synthetic mid-band content into w_theta = w_ref + 0.1 * (mask_B * white noise)
    seed = jax.random.PRNGKey(9)
    e_noise = 0.1 * jax.random.normal(seed, (batch, nx), dtype=jnp.float64)
    x_noise = 0.1 * jax.random.normal(jax.random.fold_in(seed, 1), (batch, nx), dtype=jnp.float64)
    inject_eta = apply_projector_physical(e_noise, mask_B)
    inject_xi = apply_projector_physical(x_noise, mask_B)
    inj_hat = SpectralState(eta_hat=myfft(inject_eta, nx), xi_hat=myfft(inject_xi, nx))
    w_theta = (SpectralState(
        eta_hat=w_ref[0].eta_hat + inj_hat.eta_hat,
        xi_hat=w_ref[0].xi_hat + inj_hat.xi_hat,
    ), SpectralState(
        eta_hat=w_ref[1].eta_hat + inj_hat.eta_hat,
        xi_hat=w_ref[1].xi_hat + inj_hat.xi_hat,
    ))
    lm = float(stage_match_loss(w_theta, w_ref, mask_B, g0, 1.0, dx, 1e-8))
    assert lm > 1.0, f"low-to-mid injection should trigger L_match, got {lm}"


def test_finite_secant_agrees_with_jvp():
    """For a smooth F, finite secant → jax.jvp as eps → 0."""
    nx, batch, dt = 64, 1, 0.01
    k, eta, xi, depths, g0, rp = _reference_only_setup(nx, batch, dt)
    F = make_reference_F(rp)
    coeffs = gl2_coefficients(jnp.float64)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    V_bar = picard_step(F, (v_n_hat, v_n_hat), v_n_hat, jnp.asarray(dt), coeffs)
    dx = float(2.0 * np.pi / nx)
    mask_B = soft_projector_band(k, 32.0, 112.0, 8.0, 16.0)
    deta, dxi = construct_probe(jax.random.PRNGKey(10), "joint", mask_B, g0, 1.0, dx, (batch, nx), jnp.float64)
    q_hat = SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe = (q_hat, q_hat)
    T = lambda V: picard_step(F, V, v_n_hat, jnp.asarray(dt), coeffs)
    w_sec = finite_secant(T, V_bar, probe, jnp.asarray(1e-6))
    _, w_jvp = jax.jvp(T, (V_bar,), (probe,))
    for i in range(2):
        max_err = float(
            jnp.max(jnp.abs(w_sec[i].eta_hat - w_jvp[i].eta_hat))
            + jnp.max(jnp.abs(w_sec[i].xi_hat - w_jvp[i].xi_hat))
        )
        assert max_err < 5e-3, f"secant vs jvp stage{i+1}: {max_err:.3e}"


def test_gradients_are_finite_and_nonzero_on_unstable_model():
    """A parameterized model whose response is (1 + theta * mask_B) * reference has a
    finite, nonzero gradient of L_match w.r.t. theta. Band scaled to fit nx=64."""
    nx, batch, dt = 64, 2, 0.01
    k, eta, xi, depths, g0, rp = _reference_only_setup(nx, batch, dt)
    F = make_reference_F(rp)
    coeffs = gl2_coefficients(jnp.float64)
    v_n_hat = SpectralState(eta_hat=myfft(eta, nx), xi_hat=myfft(xi, nx))
    V_bar = jax.lax.stop_gradient(picard_step(F, (v_n_hat, v_n_hat), v_n_hat, jnp.asarray(dt), coeffs))
    dx = float(2.0 * np.pi / nx)
    mask_B = soft_projector_band(k, 8.0, 24.0, 2.0, 4.0)
    deta, dxi = construct_probe(jax.random.PRNGKey(11), "joint", mask_B, g0, 1.0, dx, (batch, nx), jnp.float64)
    q_hat = SpectralState(eta_hat=myfft(deta, nx), xi_hat=myfft(dxi, nx))
    probe = (q_hat, q_hat)

    T_ref_fn = lambda V: picard_step(F, V, v_n_hat, jnp.asarray(dt), coeffs)
    w_ref = jax.lax.stop_gradient(finite_secant(T_ref_fn, V_bar, probe, jnp.asarray(1e-4)))

    # Model = reference * (1 + theta * mask_B). Loss depends on theta.
    def loss_theta(theta):
        factor = (1.0 + theta * mask_B).astype(w_ref[0].eta_hat.dtype)
        def scale(ws):
            return SpectralState(eta_hat=factor * ws.eta_hat, xi_hat=factor * ws.xi_hat)
        w_theta = (scale(w_ref[0]), scale(w_ref[1]))
        return stage_match_loss(w_theta, w_ref, mask_B, g0, 1.0, dx, 1e-8)
    g = float(jax.grad(loss_theta)(jnp.asarray(0.5, dtype=jnp.float64)))
    assert np.isfinite(g), f"grad not finite: {g}"
    assert abs(g) > 1e-4, f"grad too small: {g}"


# ---------------------------------------------------------------------------
# Integration smoke: full compute_stage_reg on a stub model
# ---------------------------------------------------------------------------


def _stub_linear_model(k_np: np.ndarray, target_scale: float = 1.0):
    """Return (apply_fn, params, norm_inputs, denorm_targets, filter_predictions)
    where the model implements gxi = G_0(h) * xi via k*tanh(h*k) on the fly.
    This is the reference's flat-surface limit and lets us drive a full
    compute_stage_reg without a trained network."""
    k = jnp.asarray(k_np, dtype=jnp.float64)

    def apply_fn(vars_dict, batch_inputs, batch_depth):
        eta = batch_inputs[..., 0]
        xi = batch_inputs[..., 1]
        del eta  # unused; linear baseline is xi-only
        h = jnp.exp(batch_depth[:, 0])  # physical depth
        g0 = jax.vmap(lambda hi: make_linear_dno_symbol(k, hi))(h)
        xi_hat = jnp.fft.fft(xi, axis=-1)
        # myfft convention zeroes Nyquist — do the same for parity with reference.
        xi_hat = xi_hat.at[..., xi.shape[-1] // 2].set(0)
        gxi = jnp.real(jnp.fft.ifft(g0 * xi_hat, axis=-1))
        return gxi[..., None]

    params = {"dummy": jnp.zeros((), dtype=jnp.float64)}
    def norm_inputs(eta, xi):
        return jnp.stack([eta, xi], axis=-1)
    def denorm_targets(preds):
        return preds
    def filter_predictions(preds):
        return preds
    return apply_fn, params, norm_inputs, denorm_targets, filter_predictions


def test_compute_stage_reg_runs_and_produces_finite_loss():
    """Full driver runs on a stub 'model' matching the reference's flat-surface limit."""
    nx, batch = 64, 4
    x, k_np = build_grid(nx, 2.0 * np.pi)
    _, _, eta, xi, depths = _small_state(nx=nx, batch=batch, depth=0.2, seed=12)
    apply_fn, params, norm_inputs, denorm_targets, filter_preds = _stub_linear_model(np.asarray(k_np))
    # Model's batch_depth = log(h) as (B, 1)
    batch_depth = jnp.log(depths)[:, None]
    cfg = StageRegConfig(
        interval=32, microbatch=batch, warmup_steps=100,
        k_lo=8.0, k_hi=24.0, k_low_hi=6.0,
        taper_lo=2.0, taper_hi=4.0, taper_low=2.0,
        eps_min=1e-5, eps_max=1e-3,
        gain_margin_rel=0.05, gain_margin_abs=1e-3,
        response_floor=1e-8, eps_norm=1e-8,
        reference_order=6, reference_pad=8, reference_picard_predictors=1,
        dt=0.01, gravity=1.0, filter_fraction=2.0 / 3.0,
    )
    l_match, l_gain, diag = compute_stage_reg(
        rng=jax.random.PRNGKey(13),
        apply_fn=apply_fn, model_params=params,
        eta_phys=eta, xi_phys=xi, depth_phys=depths,
        batch_depth_local=batch_depth,
        norm_inputs_fn=norm_inputs, denorm_targets_fn=denorm_targets,
        filter_predictions_fn=filter_preds,
        k=jnp.asarray(k_np, dtype=jnp.float64),
        dx=float(2.0 * np.pi / nx),
        cfg=cfg, dtype=jnp.float64,
    )
    lm, lg = float(l_match), float(l_gain)
    assert np.isfinite(lm) and np.isfinite(lg), f"non-finite: lm={lm} lg={lg}"
    # The stub model has no eta channel: its gxi = G_0(h)*xi, while the reference
    # includes G_1, G_2, ... The mid-band mismatch should be *positive* but bounded.
    # We only assert finiteness + a nonzero L_match; L_gain may or may not fire.
    assert lm > 0.0, f"L_match should be nonzero for a nontrivial mismatch, got {lm}"
    for _, v in diag.items():
        assert np.isfinite(float(v)), "diagnostic not finite"


def test_compute_stage_reg_is_differentiable_wrt_params():
    """Loss must yield a nonzero gradient of L_match w.r.t. model params — the
    entry-point that the trainer actually uses."""
    nx, batch = 64, 4
    _, k_np = build_grid(nx, 2.0 * np.pi)
    _, _, eta, xi, depths = _small_state(nx=nx, batch=batch, depth=0.2, seed=14)

    # Parameterize the stub so the loss actually depends on a "weight".
    k = jnp.asarray(k_np, dtype=jnp.float64)

    def apply_fn(vars_dict, batch_inputs, batch_depth):
        params = vars_dict["params"]
        xi = batch_inputs[..., 1]
        h = jnp.exp(batch_depth[:, 0])
        g0 = jax.vmap(lambda hi: make_linear_dno_symbol(k, hi))(h)
        xi_hat = jnp.fft.fft(xi, axis=-1)
        xi_hat = xi_hat.at[..., xi.shape[-1] // 2].set(0)
        # Multiply by (1 + w * mask_B) to inject controllable mid-band gain.
        mask_B = soft_projector_band(k, 8.0, 24.0, 2.0, 4.0)
        gxi = jnp.real(jnp.fft.ifft(g0 * (1.0 + params["w"] * mask_B) * xi_hat, axis=-1))
        return gxi[..., None]

    params = {"w": jnp.asarray(0.5, dtype=jnp.float64)}
    def norm_inputs(eta, xi):
        return jnp.stack([eta, xi], axis=-1)
    denorm_targets = lambda preds: preds
    filter_preds = lambda preds: preds
    batch_depth = jnp.log(depths)[:, None]
    cfg = StageRegConfig(
        interval=32, microbatch=batch, warmup_steps=100,
        k_lo=8.0, k_hi=24.0, k_low_hi=6.0,
        taper_lo=2.0, taper_hi=4.0, taper_low=2.0,
        eps_min=1e-5, eps_max=1e-3,
        gain_margin_rel=0.05, gain_margin_abs=1e-3,
        response_floor=1e-8, eps_norm=1e-8,
        reference_order=6, reference_pad=8, reference_picard_predictors=1,
        dt=0.01, gravity=1.0, filter_fraction=2.0 / 3.0,
    )

    def loss(params_dict):
        lm, lg, _ = compute_stage_reg(
            rng=jax.random.PRNGKey(15),
            apply_fn=apply_fn, model_params=params_dict,
            eta_phys=eta, xi_phys=xi, depth_phys=depths,
            batch_depth_local=batch_depth,
            norm_inputs_fn=norm_inputs, denorm_targets_fn=denorm_targets,
            filter_predictions_fn=filter_preds,
            k=k, dx=float(2.0 * np.pi / nx),
            cfg=cfg, dtype=jnp.float64,
        )
        return lm + lg

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert np.isfinite(float(grads["w"]))
    assert abs(float(grads["w"])) > 1e-6, f"grad w too small: {float(grads['w'])}"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def test_compute_stage_reg_composes_with_cs_dno():
    """Full end-to-end integration: instantiate the actual CraigSulemDNO the trainer
    uses, evaluate compute_stage_reg with the plan's default band, and take grad w.r.t.
    the model's Flax params. Confirms the whole pipeline compiles and gradients flow.

    Uses physical smooth-wavetrain ICs (low harmonic content) rather than random
    Gaussian fields: the order-6 Craig-Sulem series is only stable on smooth low-mode
    inputs. Random Gaussian noise (even 2/3-filtered) drives the reference response
    into a 1e40+ range where the model's finite response is negligible in comparison,
    saturating L_match at 1.0 and producing spuriously zero gradients — a test-setup
    artifact, not a module bug (see the passing stub-model tests above)."""
    sys.path.insert(0, str(REPO / "models" / "dno-net"))
    from dno_net_v2 import CraigSulemDNO

    nx, batch = 256, 4  # nx=256 so k range covers plan's default band [32, 128]
    domain_length = 2.0 * np.pi
    _, k_np = build_grid(nx, domain_length)
    k = jnp.asarray(k_np, dtype=jnp.float64)
    depths = jnp.array([0.3, 0.4, 0.5, 0.6], dtype=jnp.float64)
    xg = jnp.linspace(0.0, domain_length, nx, endpoint=False)
    eta = jnp.stack([0.02 * jnp.cos(km * xg) for km in [2, 3, 4, 5]])
    xi = jnp.stack([0.05 * jnp.sin(km * xg) for km in [2, 3, 4, 5]])
    batch_depth = jnp.log(depths)[:, None]

    model = CraigSulemDNO(
        modes=32, width=32, n_blocks=1, latent=32, n_polys=3,
        mult_hidden=16, domain_length=domain_length,
        xi_scale=1.0, eta_scale=1.0, target_scale=1.0,
    )
    params = model.init(
        jax.random.PRNGKey(19),
        jnp.zeros((1, nx, 2), dtype=jnp.float64),
        jnp.zeros((1, 1), dtype=jnp.float64),
    )["params"]

    def norm_inputs(eta_, xi_):
        return jnp.stack([eta_, xi_], axis=-1)

    def denorm_targets(preds):
        return preds

    def filter_preds(preds):
        return preds

    cfg = StageRegConfig(
        interval=32, microbatch=batch, warmup_steps=500,
        k_lo=32.0, k_hi=128.0, k_low_hi=32.0,
        taper_lo=8.0, taper_hi=16.0, taper_low=8.0,
        eps_min=1e-6, eps_max=1e-3,
        gain_margin_rel=0.05, gain_margin_abs=1e-3,
        response_floor=1e-8, eps_norm=1e-12,
        reference_order=6, reference_pad=8, reference_picard_predictors=1,
        dt=0.01, gravity=1.0, filter_fraction=2.0 / 3.0,
    )

    def loss_of_params(p):
        lm, lg, _ = compute_stage_reg(
            rng=jax.random.PRNGKey(20),
            apply_fn=model.apply, model_params=p,
            eta_phys=eta, xi_phys=xi, depth_phys=depths,
            batch_depth_local=batch_depth,
            norm_inputs_fn=norm_inputs, denorm_targets_fn=denorm_targets,
            filter_predictions_fn=filter_preds,
            k=k, dx=float(domain_length / nx),
            cfg=cfg, dtype=jnp.float64,
        )
        return lm + 0.1 * lg

    val, grads = jax.value_and_grad(loss_of_params)(params)
    assert np.isfinite(float(val)), f"loss not finite: {val}"
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norm = float(sum(jnp.sum(g ** 2) for g in grad_leaves))
    assert np.isfinite(grad_norm), "grad has non-finite entries"
    assert grad_norm > 0.0, "grad is identically zero — model params disconnected from loss"


TESTS = [
    ("math/gl2_coeffs_match_production", test_gl2_coeffs_match_production),
    ("math/picard_map_matches_production", test_picard_map_matches_production_body_fn),
    ("math/linear_flow_preserves_energy_norm", test_linear_flow_preserves_energy_norm),
    ("math/local_vs_absolute_time_agree", test_local_vs_absolute_time_agree),
    ("math/projector_support_and_reality", test_projector_support_and_reality),
    ("math/probe_unit_energy_norm", test_probe_unit_energy_norm),
    ("loss/model_eq_ref_gives_zero", test_model_equals_reference_gives_zero_loss),
    ("loss/amplified_midband_triggers", test_amplified_mid_band_response_increases_losses),
    ("loss/low_to_mid_detected", test_low_to_mid_probe_detects_synthetic_mid_response),
    ("loss/secant_matches_jvp", test_finite_secant_agrees_with_jvp),
    ("loss/grads_finite_nonzero", test_gradients_are_finite_and_nonzero_on_unstable_model),
    ("integration/compute_stage_reg_runs", test_compute_stage_reg_runs_and_produces_finite_loss),
    ("integration/compute_stage_reg_diff", test_compute_stage_reg_is_differentiable_wrt_params),
    ("integration/composes_with_cs_dno", test_compute_stage_reg_composes_with_cs_dno),
]


def main() -> int:
    for name, fn in TESTS:
        _run(name, fn)
    print()
    print(f"summary: {len(PASS)} pass, {len(FAIL)} fail (of {len(TESTS)} total)")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL {n}: {d}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
