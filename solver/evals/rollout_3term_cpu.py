"""Train + rollout the 3-term handcoded Craig-Sulem ansatz on CPU.

Smoke test: does the 147-param ansatz integrate stably or blow up to NaN?
Uses reduced integrator settings so it fits in ~10-15 min on CPU; numbers
won't directly match eval_suite_f64h but qualitative divergence vs bounded-
ness is what we want.

Steps:
1. Train 3-term ansatz on v5 sample (~3 min, 2000 SGD steps, 147 params).
2. Per-IC rollout 4 tanaka_g0 ICs with reduced substeps + gl2_iter.
3. Compare to the cached truth tape (apples-vs-oranges: their truth used
   substeps=80, gl2_iter=4; we use 16, 2 — so this answers the boundedness
   question, not "does it match eval_suite numbers").

Run: uv run python solver/evals/rollout_3term_cpu.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"  # unconditional — script name promises CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hide GPUs from JAX entirely

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

from solver.solvers import time_integrator as ti
from solver.evals.model_rollout import rollout_surrogate


# ---------------------------------------------------------------------------
# 3-term ansatz
# ---------------------------------------------------------------------------

class TermMultiplier(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, k_features: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(k_features)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class ThreeTermAnsatz(nn.Module):
    mlp_hidden: int = 8

    def setup(self) -> None:
        self.M_eta  = TermMultiplier(self.mlp_hidden)
        self.M_half = TermMultiplier(self.mlp_hidden)
        self.M_eta2 = TermMultiplier(self.mlp_hidden)

    def __call__(self, eta: jnp.ndarray, xi: jnp.ndarray, h: jnp.ndarray,
                 kx_abs: jnp.ndarray) -> jnp.ndarray:
        h_col = h[:, None] if h.ndim == 1 else h
        G0_kh = kx_abs[None, :] * jnp.tanh(h_col * kx_abs[None, :])
        xi_hat = jnp.fft.fft(xi, axis=-1)
        G0_xi = jnp.fft.ifft(G0_kh * xi_hat, axis=-1).real
        eta_hat = jnp.fft.fft(eta, axis=-1)
        half_eta = jnp.fft.ifft(jnp.sqrt(kx_abs)[None, :] * eta_hat, axis=-1).real

        k_max = jnp.max(kx_abs) + 1e-12
        k_norm = (kx_abs / k_max)[None, :]
        hk = h_col * kx_abs[None, :]
        tanh_hk = jnp.tanh(hk)
        k_feat = jnp.stack([
            jnp.broadcast_to(k_norm, eta.shape),
            jnp.broadcast_to(h_col,  eta.shape),
            tanh_hk,
            k_norm * tanh_hk,
        ], axis=-1)

        def apply_outer(field, mlp):
            mult = mlp(k_feat)
            return jnp.fft.ifft(mult * jnp.fft.fft(field, axis=-1), axis=-1).real

        pred = G0_xi
        pred = pred + apply_outer(eta * G0_xi,        self.M_eta)
        pred = pred + apply_outer(half_eta * G0_xi,   self.M_half)
        pred = pred + apply_outer((eta ** 2) * G0_xi, self.M_eta2)
        return pred


def sobolev1_mean(err: jnp.ndarray, truth: jnp.ndarray, kx_full: jnp.ndarray) -> jnp.ndarray:
    w = (1.0 + kx_full ** 2)[None, :]
    err_hat   = jnp.fft.fft(err,   axis=-1)
    truth_hat = jnp.fft.fft(truth, axis=-1)
    err_pow   = w * (err_hat.real   ** 2 + err_hat.imag   ** 2)
    truth_pow = w * (truth_hat.real ** 2 + truth_hat.imag ** 2)
    return (err_pow.sum(axis=-1) / (truth_pow.sum(axis=-1) + 1e-12)).mean()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_ansatz(dataset: str, n_train: int, batch_size: int, steps: int,
                 lr: float, mlp_hidden: int, seed: int, nx_expected: int
                 ) -> tuple[dict, ThreeTermAnsatz, np.ndarray]:
    print(f"loading {dataset} (mmap)...", flush=True)
    d = np.load(dataset, mmap_mode="r")
    eta_all = d["eta"]; xi_all = d["xi"]; gxi_all = d["gxi"]; depth_all = d["depth"]
    x = np.asarray(d["x"])
    N_total, nx = eta_all.shape
    assert nx == nx_expected, f"nx mismatch: {nx} != {nx_expected}"
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    dx = L / nx
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    kx_abs = np.abs(kx).astype(np.float32)
    kx_full = kx.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(N_total, size=n_train, replace=False); idx.sort()
    print(f"sampling {n_train} examples...", flush=True)
    eta = np.asarray(eta_all[idx], dtype=np.float32)
    xi  = np.asarray(xi_all[idx],  dtype=np.float32)
    gxi = np.asarray(gxi_all[idx], dtype=np.float32)
    h   = np.asarray(depth_all[idx], dtype=np.float32)

    eta_j = jnp.asarray(eta); xi_j = jnp.asarray(xi)
    gxi_j = jnp.asarray(gxi); h_j = jnp.asarray(h)
    kx_abs_j = jnp.asarray(kx_abs); kx_full_j = jnp.asarray(kx_full)

    model = ThreeTermAnsatz(mlp_hidden=mlp_hidden)
    rng_key = jax.random.PRNGKey(seed)
    params = model.init(rng_key, eta_j[:1], xi_j[:1], h_j[:1], kx_abs_j)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"param count: {n_params}", flush=True)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, eta_b, xi_b, gxi_b, h_b):
        pred = model.apply(params, eta_b, xi_b, h_b, kx_abs_j)
        return sobolev1_mean(pred - gxi_b, gxi_b, kx_full_j)

    @jax.jit
    def train_step(params, opt_state, eta_b, xi_b, gxi_b, h_b):
        loss, grads = jax.value_and_grad(loss_fn)(params, eta_b, xi_b, gxi_b, h_b)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print(f"\ntraining {steps} steps, batch={batch_size}, lr={lr}...", flush=True)
    for step in range(steps):
        b_idx = rng.integers(0, n_train, size=batch_size)
        b_idx_j = jnp.asarray(b_idx)
        params, opt_state, loss = train_step(
            params, opt_state,
            eta_j[b_idx_j], xi_j[b_idx_j],
            gxi_j[b_idx_j], h_j[b_idx_j])
        if step == 0 or (step + 1) % 200 == 0:
            print(f"  step {step+1:>5d}/{steps}  train sob1 = {float(loss):.6f}", flush=True)
    return params, model, kx_abs


# ---------------------------------------------------------------------------
# Per-IC rollout
# ---------------------------------------------------------------------------

def rollout_one_regime(params, model, kx_abs: np.ndarray, trajs_path: str,
                       n_ics: int, substeps: int, gl2_iter: int,
                       regime_name: str,
                       dno_order: int = 6, pad_factor: int = 8,
                       filter_fraction: float = 0.25,
                       length: float = 2.0 * np.pi) -> dict:
    print(f"\n=== regime {regime_name} ({trajs_path.split('/')[-1]}) ===", flush=True)
    trajs = np.load(trajs_path)
    times = np.asarray(trajs["times"])
    NB = trajs["truth_eta"].shape[1]
    n_ics = min(n_ics, NB)
    depths = np.asarray(trajs["depths"])[:n_ics]
    truth_eta = np.asarray(trajs["truth_eta"])[:, :n_ics, :]
    truth_xi  = np.asarray(trajs["truth_xi"])[:,  :n_ics, :]
    nx = truth_eta.shape[-1]
    print(f"  NB={n_ics} tmax={float(times[-1]):.1f} h_range=[{depths.min():.4f},{depths.max():.4f}]", flush=True)

    kx_abs_j = jnp.asarray(kx_abs, dtype=jnp.float32)
    times_j = jnp.asarray(times, dtype=jnp.float32)
    pred_eta_all: list[np.ndarray] = []
    pred_xi_all: list[np.ndarray] = []
    walls: list[float] = []

    # Build sp once with a placeholder depth; only `depth` and `g0` change per IC, but we
    # rebuild sp per IC anyway since `g0 = make_linear_dno_symbol(k, depth)` is cheap.
    # JIT one rollout function: takes (eta0, xi0, depth_scalar) so the trace is reused.
    from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol

    @jax.jit
    def jitted_rollout(eta0: jnp.ndarray, xi0: jnp.ndarray, depth_scalar: jnp.ndarray):
        _, k = build_grid(nx, length)
        g0 = make_linear_dno_symbol(k, depth_scalar)
        sp = ti.SolverParams(
            nx=nx, length=length, depth=depth_scalar, gravity=1.0,
            dno_order=dno_order, pad_factor=pad_factor,
            filter_fraction=filter_fraction, k=k, g0=g0,
        )
        h_arr = depth_scalar.reshape(1)

        def predict(eta_1: jnp.ndarray, xi_1: jnp.ndarray) -> jnp.ndarray:
            return model.apply(params,
                               eta_1[None, :], xi_1[None, :],
                               h_arr, kx_abs_j)[0]

        out = rollout_surrogate(
            ti.State(eta=eta0, xi=xi0),
            times_j, sp, predict,
            substeps=substeps, zero_mean_xi=True, gl2_iterations=gl2_iter,
        )
        return out["eta"], out["xi"]

    for i in range(n_ics):
        h_i = float(depths[i])
        depth_arr = jnp.asarray(h_i, dtype=jnp.float32)

        eta0 = jnp.asarray(truth_eta[0, i], dtype=jnp.float32)
        xi0  = jnp.asarray(truth_xi[0,  i], dtype=jnp.float32)

        t0 = perf_counter()
        eta_pred, xi_pred = jitted_rollout(eta0, xi0, depth_arr)
        eta_pred.block_until_ready()
        wall = perf_counter() - t0
        walls.append(wall)
        pe = np.asarray(eta_pred)
        px = np.asarray(xi_pred)
        pred_eta_all.append(pe)
        pred_xi_all.append(px)
        nan_frac = float(np.isnan(pe).any(axis=-1).mean())
        eta_fin_norm = float(np.linalg.norm(pe[-1]))
        truth_fin_norm = float(np.linalg.norm(truth_eta[-1, i]))
        rl2_fin = float(np.linalg.norm(pe[-1] - truth_eta[-1, i]) /
                        (truth_fin_norm + 1e-12))
        print(f"  IC {i:2d} h={h_i:.4f}: wall={wall:.1f}s, NaN={nan_frac:.3f}, "
              f"||eta_fin||={eta_fin_norm:.3g} (truth={truth_fin_norm:.3g}), "
              f"rel_l2_fin={rl2_fin:.4f}", flush=True)

    pred_eta = np.stack(pred_eta_all, axis=1)
    pred_xi  = np.stack(pred_xi_all,  axis=1)

    eta_err = np.linalg.norm(pred_eta - truth_eta, axis=-1)
    eta_norm = np.linalg.norm(truth_eta, axis=-1) + 1e-12
    rel_l2 = eta_err / eta_norm  # (n_t, B)
    # Treat NaN-bearing time steps as worst (rel_l2=inf) for nan_rate accounting;
    # the median/p* statistics below ignore NaNs gracefully via nanmedian.
    nan_mask_per_step = np.isnan(pred_eta).any(axis=-1)  # (n_t, B)
    nan_rate_final = float(nan_mask_per_step[-1].mean())
    nan_rate_ever  = float(nan_mask_per_step.any(axis=0).mean())

    rl2_fin = rel_l2[-1]
    rl2_fin_no_nan = rl2_fin[~np.isnan(rl2_fin) & ~np.isinf(rl2_fin)]
    if rl2_fin_no_nan.size > 0:
        med = float(np.median(rl2_fin_no_nan))
        p95 = float(np.percentile(rl2_fin_no_nan, 95))
        mx  = float(np.max(rl2_fin_no_nan))
    else:
        med = p95 = mx = float("nan")

    print(f"  >> regime {regime_name}: NaN-rate(final/ever) = {nan_rate_final:.2f}/{nan_rate_ever:.2f}  "
          f"rel_l2_eta_final median={med:.4f} p95={p95:.4f} max={mx:.4f}", flush=True)

    return {
        "regime": regime_name,
        "times": np.asarray(times),
        "depths": depths,
        "rel_l2_eta_final": rl2_fin,
        "nan_rate_final": nan_rate_final,
        "nan_rate_ever": nan_rate_ever,
        "rel_l2_eta_final_median": med,
        "rel_l2_eta_final_p95": p95,
        "rel_l2_eta_final_max": mx,
        "walls": walls,
    }


DEFAULT_REGIMES = [
    "tanaka_g0", "tanaka_g1",
    "bf_g0", "bf_g1", "bf_modal",
    "linear",
    "stokes_deep", "stokes_finite",
    "random_sea_deep", "random_sea_finite",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/combined_dataset_v5.npz")
    p.add_argument("--trajs_dir",
                   default="outputs/fno_w128b6_eta_features_v7_trim_20260618_120313/eval_suite_f64h")
    p.add_argument("--regimes", nargs="+", default=DEFAULT_REGIMES,
                   help="regime names matching {trajs_dir}/{name}_trajs.npz")
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--mlp_hidden", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--n_ics", type=int, default=16)
    p.add_argument("--substeps", type=int, default=16)
    p.add_argument("--gl2_iter", type=int, default=2)
    p.add_argument("--report",
                   default="notes/figures/cs_dno_v5_operator_viz/rollout_3term_cpu.json")
    args = p.parse_args()

    t0 = perf_counter()
    print(f"jax devices: {jax.devices()}", flush=True)
    nx = 1024
    params, model, kx_abs = train_ansatz(
        args.dataset, args.n_train, args.batch_size,
        args.steps, args.lr, args.mlp_hidden, args.seed, nx,
    )
    t_train = perf_counter() - t0
    print(f"\ntrain wall = {t_train:.1f}s", flush=True)

    trajs_dir = Path(args.trajs_dir)
    summaries: dict[str, dict] = {}
    for regime in args.regimes:
        trajs_path = trajs_dir / f"{regime}_trajs.npz"
        if not trajs_path.exists():
            print(f"\n!! skipping {regime}: missing {trajs_path}", flush=True)
            continue
        r = rollout_one_regime(
            params, model, kx_abs, str(trajs_path),
            n_ics=args.n_ics, substeps=args.substeps, gl2_iter=args.gl2_iter,
            regime_name=regime,
        )
        summaries[regime] = {
            "n_ics": int(len(r["rel_l2_eta_final"])),
            "tmax": float(r["times"][-1]),
            "h_range": [float(r["depths"].min()), float(r["depths"].max())],
            "nan_rate_final": r["nan_rate_final"],
            "nan_rate_ever":  r["nan_rate_ever"],
            "rel_l2_eta_final_median": r["rel_l2_eta_final_median"],
            "rel_l2_eta_final_p95":    r["rel_l2_eta_final_p95"],
            "rel_l2_eta_final_max":    r["rel_l2_eta_final_max"],
            "rel_l2_eta_final_per_ic": [float(v) for v in r["rel_l2_eta_final"]],
            "depths": [float(v) for v in r["depths"]],
            "wall_seconds": float(sum(r["walls"])),
        }

    t_total = perf_counter() - t0
    print(f"\ntotal wall = {t_total:.1f}s", flush=True)

    print("\n=== summary across regimes ===")
    print(f"{'regime':22s} {'NB':>3s} {'tmax':>6s} {'NaN_fin':>8s} {'NaN_ever':>9s} "
          f"{'med':>7s} {'p95':>7s} {'max':>7s}")
    for regime, s in summaries.items():
        print(f"{regime:22s} {s['n_ics']:3d} {s['tmax']:6.1f} {s['nan_rate_final']:8.2f} "
              f"{s['nan_rate_ever']:9.2f} {s['rel_l2_eta_final_median']:7.4f} "
              f"{s['rel_l2_eta_final_p95']:7.4f} {s['rel_l2_eta_final_max']:7.4f}")

    rpt = {
        "train_seconds": float(t_train),
        "total_seconds": float(t_total),
        "n_ics_per_regime": args.n_ics,
        "substeps": args.substeps,
        "gl2_iter": args.gl2_iter,
        "regimes": summaries,
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rpt, indent=2))
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
