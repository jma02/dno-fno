"""Variant-B handcoded 3-term Craig-Sulem ansatz with a learnable per-term k-multiplier.

Ansatz:
    gxi ≈ G0(D,h)·ξ + Σ_i  ifft( M_i(k,h) · fft( f_i(η) · G0·ξ ) )

with f_i(η) ∈ {η, |D|^{1/2}η, η²} and one tiny depth-aware multiplier MLP M_i
per term (input: (k_norm, h, tanh(hk), k·tanh(hk)) → scalar).

Mirrors cs_dno's `DepthAwareMultiplier` at cs_mult_hidden=8 with the OUTER
multiplier per term learnable and the INNER G0 hardcoded — the goal is to
test whether a single small k-dependent scalar per term is enough to close
the gap variant A (3 fixed scalars) left at 10× v5.

Param count: 3 MLPs × (4×8 + 8 + 8×1 + 1) = 147.

CPU-only via JAX_PLATFORMS=cpu. SGD with Adam.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/combined_dataset_v5.npz")
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--mlp_hidden", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--report", default="notes/figures/cs_dno_v5_operator_viz/handcoded_3term_mlp.json")
    return p.parse_args()


class TermMultiplier(nn.Module):
    """Tiny per-term depth-aware multiplier — Dense(h)→gelu→Dense(1)."""
    hidden: int

    @nn.compact
    def __call__(self, k_features: jnp.ndarray) -> jnp.ndarray:
        # k_features: (..., 4)  → scalar (...,)
        x = nn.Dense(self.hidden)(k_features)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class ThreeTermAnsatz(nn.Module):
    mlp_hidden: int = 8

    def setup(self) -> None:
        self.M_eta    = TermMultiplier(self.mlp_hidden)
        self.M_half   = TermMultiplier(self.mlp_hidden)
        self.M_eta2   = TermMultiplier(self.mlp_hidden)

    def __call__(self, eta: jnp.ndarray, xi: jnp.ndarray, h: jnp.ndarray,
                 kx_abs: jnp.ndarray) -> jnp.ndarray:
        """eta, xi: (B, nx); h: (B,); kx_abs: (nx,). Returns gxi pred (B, nx)."""
        nx = eta.shape[-1]
        h_col = h[:, None]                                     # (B,1)
        G0_kh = kx_abs[None, :] * jnp.tanh(h_col * kx_abs[None, :])  # (B,nx)
        xi_hat = jnp.fft.fft(xi, axis=-1)
        G0_xi = jnp.fft.ifft(G0_kh * xi_hat, axis=-1).real    # (B,nx)
        eta_hat = jnp.fft.fft(eta, axis=-1)
        half_eta = jnp.fft.ifft(jnp.sqrt(kx_abs)[None, :] * eta_hat, axis=-1).real

        # Build per-(B, nx) k-feature tensor for the multipliers
        # k_features: (B, nx, 4) = [k/k_max, h, tanh(h·k), (k/k_max)·tanh(h·k)]
        k_max = jnp.max(kx_abs) + 1e-12
        k_norm = (kx_abs / k_max)[None, :]                    # (1, nx)
        hk     = h_col * kx_abs[None, :]                       # (B, nx)
        tanh_hk = jnp.tanh(hk)
        k_feat = jnp.stack([
            jnp.broadcast_to(k_norm, eta.shape),
            jnp.broadcast_to(h_col,  eta.shape),
            tanh_hk,
            k_norm * tanh_hk,
        ], axis=-1)                                           # (B, nx, 4)

        def apply_outer(field, mlp):
            mult = mlp(k_feat)                                # (B, nx)
            field_hat = jnp.fft.fft(field, axis=-1)
            return jnp.fft.ifft(mult * field_hat, axis=-1).real

        pred = G0_xi.copy()
        pred = pred + apply_outer(eta * G0_xi,      self.M_eta)
        pred = pred + apply_outer(half_eta * G0_xi, self.M_half)
        pred = pred + apply_outer((eta ** 2) * G0_xi, self.M_eta2)
        return pred, G0_xi


def sobolev1_mean(err: jnp.ndarray, truth: jnp.ndarray, kx_full: jnp.ndarray) -> jnp.ndarray:
    """sum_k (1+k²) |err_k|² / sum_k (1+k²) |truth_k|², averaged over batch."""
    w = (1.0 + kx_full ** 2)[None, :]
    err_hat   = jnp.fft.fft(err,   axis=-1)
    truth_hat = jnp.fft.fft(truth, axis=-1)
    err_pow   = w * (err_hat.real   ** 2 + err_hat.imag   ** 2)
    truth_pow = w * (truth_hat.real ** 2 + truth_hat.imag ** 2)
    per = err_pow.sum(axis=-1) / (truth_pow.sum(axis=-1) + 1e-12)
    return per.mean()


def rel_l2_per_example(err: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-12
    return jnp.sqrt(jnp.sum(err ** 2, axis=-1)) / (jnp.sqrt(jnp.sum(truth ** 2, axis=-1)) + eps)


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    print(f"jax devices: {jax.devices()}", flush=True)

    print(f"loading {args.dataset} (mmap)...", flush=True)
    d = np.load(args.dataset, mmap_mode="r")
    eta_all = d["eta"]
    xi_all = d["xi"]
    gxi_all = d["gxi"]
    depth_all = d["depth"]
    x = np.asarray(d["x"])
    N_total, nx = eta_all.shape
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    dx = L / nx
    print(f"  N={N_total:,}, nx={nx}, L={L:.6f}", flush=True)

    rng = np.random.default_rng(args.seed)
    n_all = args.n_train + args.n_val
    idx = rng.choice(N_total, size=n_all, replace=False)
    idx.sort()
    print(f"sampling {n_all} examples ({args.n_train} train + {args.n_val} val)...", flush=True)
    eta = np.asarray(eta_all[idx], dtype=np.float32)
    xi  = np.asarray(xi_all[idx],  dtype=np.float32)
    gxi = np.asarray(gxi_all[idx], dtype=np.float32)
    h   = np.asarray(depth_all[idx], dtype=np.float32)

    # Shuffle and split
    perm = rng.permutation(n_all)
    eta, xi, gxi, h = eta[perm], xi[perm], gxi[perm], h[perm]
    eta_train, xi_train, gxi_train, h_train = eta[:args.n_train], xi[:args.n_train], gxi[:args.n_train], h[:args.n_train]
    eta_val,   xi_val,   gxi_val,   h_val   = eta[args.n_train:], xi[args.n_train:], gxi[args.n_train:], h[args.n_train:]

    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    kx_abs = np.abs(kx).astype(np.float32)
    kx_full = kx.astype(np.float32)

    eta_train_j = jnp.asarray(eta_train); xi_train_j = jnp.asarray(xi_train)
    gxi_train_j = jnp.asarray(gxi_train); h_train_j = jnp.asarray(h_train)
    eta_val_j = jnp.asarray(eta_val); xi_val_j = jnp.asarray(xi_val)
    gxi_val_j = jnp.asarray(gxi_val); h_val_j = jnp.asarray(h_val)
    kx_abs_j = jnp.asarray(kx_abs); kx_full_j = jnp.asarray(kx_full)

    model = ThreeTermAnsatz(mlp_hidden=args.mlp_hidden)
    rng_key = jax.random.PRNGKey(args.seed)
    init_key, dropkey = jax.random.split(rng_key)
    params = model.init(init_key,
                        eta_train_j[:1], xi_train_j[:1], h_train_j[:1], kx_abs_j)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"param count: {n_params}", flush=True)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, eta_b, xi_b, gxi_b, h_b):
        pred, _ = model.apply(params, eta_b, xi_b, h_b, kx_abs_j)
        return sobolev1_mean(pred - gxi_b, gxi_b, kx_full_j)

    @jax.jit
    def train_step(params, opt_state, eta_b, xi_b, gxi_b, h_b):
        loss, grads = jax.value_and_grad(loss_fn)(params, eta_b, xi_b, gxi_b, h_b)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def val_metrics(params, eta_b, xi_b, gxi_b, h_b):
        pred, G0_xi = model.apply(params, eta_b, xi_b, h_b, kx_abs_j)
        sob_3t = sobolev1_mean(pred - gxi_b, gxi_b, kx_full_j)
        sob_g0 = sobolev1_mean(G0_xi - gxi_b, gxi_b, kx_full_j)
        rl2_3t = rel_l2_per_example(pred - gxi_b, gxi_b)
        rl2_g0 = rel_l2_per_example(G0_xi - gxi_b, gxi_b)
        return sob_3t, sob_g0, rl2_3t, rl2_g0

    print(f"\ntraining {args.steps} steps, batch={args.batch_size}, lr={args.lr}...", flush=True)
    losses = []
    n_train = args.n_train
    for step in range(args.steps):
        b_idx = rng.integers(0, n_train, size=args.batch_size)
        b_idx_j = jnp.asarray(b_idx)
        params, opt_state, loss = train_step(
            params, opt_state,
            eta_train_j[b_idx_j], xi_train_j[b_idx_j],
            gxi_train_j[b_idx_j], h_train_j[b_idx_j])
        losses.append(float(loss))
        if step == 0 or (step + 1) % 100 == 0:
            print(f"  step {step+1:>5d}/{args.steps}  train sob1 = {loss:.6f}", flush=True)

    # Validation (batched)
    print("\nvalidating...", flush=True)
    val_bs = 256
    sob_3t_list, sob_g0_list, rl2_3t_list, rl2_g0_list = [], [], [], []
    for i in range(0, args.n_val, val_bs):
        sl = slice(i, min(i + val_bs, args.n_val))
        s3, sg, r3, rg = val_metrics(params,
            eta_val_j[sl], xi_val_j[sl], gxi_val_j[sl], h_val_j[sl])
        sob_3t_list.append(float(s3) * (sl.stop - sl.start))
        sob_g0_list.append(float(sg) * (sl.stop - sl.start))
        rl2_3t_list.append(np.asarray(r3))
        rl2_g0_list.append(np.asarray(rg))
    sob_3t_mean = sum(sob_3t_list) / args.n_val
    sob_g0_mean = sum(sob_g0_list) / args.n_val
    rl2_3t = np.concatenate(rl2_3t_list)
    rl2_g0 = np.concatenate(rl2_g0_list)

    print("\n=== validation results ===")
    print(f"  param count               = {n_params}")
    print(f"  Sobolev-1 mean (G0 only)  = {sob_g0_mean:.5f}")
    print(f"  Sobolev-1 mean (3-term)   = {sob_3t_mean:.5f}")
    print(f"  v5 ckpt val_loss ref      = 0.00101")
    print(f"  ratio (3-term / v5)       = {sob_3t_mean / 0.00101:.2f}×")
    print(f"  variant-A reference ratio = 10.39×")
    print()
    print(f"  rel-L2 G0-only  median={np.median(rl2_g0):.5f}  mean={np.mean(rl2_g0):.5f}  p95={np.percentile(rl2_g0,95):.5f}")
    print(f"  rel-L2 3-term   median={np.median(rl2_3t):.5f}  mean={np.mean(rl2_3t):.5f}  p95={np.percentile(rl2_3t,95):.5f}")

    rpt = {
        "n_params": int(n_params),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "mlp_hidden": args.mlp_hidden,
        "sobolev1": {
            "g0_only_mean": float(sob_g0_mean),
            "3term_mean":   float(sob_3t_mean),
            "v5_ckpt_ref":  0.00101,
            "ratio_3term_over_v5": float(sob_3t_mean / 0.00101),
            "variant_a_ratio_ref": 10.39,
        },
        "rel_l2": {
            "g0_only": {"median": float(np.median(rl2_g0)), "mean": float(np.mean(rl2_g0)), "p95": float(np.percentile(rl2_g0,95))},
            "3term":   {"median": float(np.median(rl2_3t)), "mean": float(np.mean(rl2_3t)), "p95": float(np.percentile(rl2_3t,95))},
        },
        "training_loss_final_100_steps_mean": float(np.mean(losses[-100:])),
        "elapsed_seconds": float(perf_counter() - t0),
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rpt, indent=2))
    print(f"\nreport -> {out_path}")
    print(f"total wall = {rpt['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
