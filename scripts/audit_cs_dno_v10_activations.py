"""Part-2/3 companion to viz_cs_dno_v5_operators.py:

Runs the v10 (`cs_dno_w512b8_l256_v9`) checkpoint forward on selected
initial conditions from the tanaka_g0 2x2 eval and:

  * measures the per-block output magnitude ‖block_b(x)‖_L2 / ‖ξ‖_L2
  * measures the G_0 residual ‖pred_gxi − G_0(η,h)ξ‖ / ‖G_0(η,h)ξ‖
    where G_0(η,h)ξ is the *depth-conditioned linear DNO applied to ξ*
    (independent of η — it's the flat-bottom operator at this depth).
  * lists the top-3 activating branches per block, for failing vs healthy ICs
  * repeats at the last-finite time step of failing ICs to see how the
    neural residual grows relative to G_0 right before NaN.
  * effective-rank (# of branches with peak weight > 1e-3) per block for M_xi and M_out
  * top-16 branches M_xi curve plot per block (h ≈ 0.3)

Outputs to notes/figures/cs_dno_v10_operator_viz/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in (
    REPO_ROOT,
    REPO_ROOT / "models" / "dno-net",
    REPO_ROOT / "models" / "fno-jax",
    REPO_ROOT / "train-jax-10m",
    REPO_ROOT / "solver",
):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from solver.evals.model_rollout import load_run, build_predict_gxi_batched
from dno_net_v2 import CraigSulemDNO  # noqa: E402


RUN_DIR = REPO_ROOT / "outputs" / "cs_dno_w512b8_l256_v9_h100x2_20260703_035745"
TRAJ_PATH = RUN_DIR / "eval_2x2v3_off_20260706_070633" / "tanaka_g0_trajs.npz"
OUT_DIR = REPO_ROOT / "notes" / "figures" / "cs_dno_v10_operator_viz"


# ----------------------------------------------------------------------------
# Static per-block multiplier utilities (mirror viz_cs_dno_v5_operators.py).
# ----------------------------------------------------------------------------

def eval_multiplier(mlp_params, k_grid, h_values):
    K = k_grid.shape[0]
    H = h_values.shape[0]
    k_b = jnp.broadcast_to(k_grid[None, :], (H, K))
    h_b = jnp.broadcast_to(h_values[:, None], (H, K))
    kh = h_b * k_b
    feats = jnp.stack([k_b, h_b, jnp.tanh(kh), k_b * jnp.tanh(kh)], axis=-1)
    hidden_w = mlp_params["hidden"]["kernel"]
    hidden_b = mlp_params["hidden"]["bias"]
    out_w = mlp_params["out"]["kernel"]
    out_b = mlp_params["out"]["bias"]
    z = jnp.tanh(feats @ hidden_w + hidden_b)
    out = z @ out_w + out_b
    return np.asarray(out)   # (H, K, C)


def effective_rank(m_all: np.ndarray, thresh_abs: float = 1e-3) -> dict:
    """Multiple views of "effective rank" for a block's multiplier tensor.

    m_all has shape (H, K, n_branches). We report:
      * n_peak_gt_1e-3   — absolute peak > 1e-3
      * n_peak_gt_1e-2_of_max — peak > 1% of the largest peak in this block
      * n_peak_gt_1e-1_of_max — peak > 10% of the largest peak in this block
      * top10_peak_share — fraction of total L2 energy carried by top-10 branches
    """
    peak_per_branch = np.max(np.abs(m_all), axis=(0, 1))  # (C,)
    max_peak = float(np.max(peak_per_branch))
    l2_per_branch = np.sqrt(np.mean(m_all ** 2, axis=(0, 1)))
    l2_total = np.sum(l2_per_branch ** 2) + 1e-24
    top10 = np.sort(l2_per_branch)[-10:]
    return {
        "peak_gt_1e-3": int(np.sum(peak_per_branch > thresh_abs)),
        "peak_gt_1e-2_of_max": int(np.sum(peak_per_branch > 1e-2 * max_peak)),
        "peak_gt_1e-1_of_max": int(np.sum(peak_per_branch > 1e-1 * max_peak)),
        "top10_l2_share": float(np.sum(top10 ** 2) / l2_total),
        "max_peak": max_peak,
    }


def plot_top16_branches(
    out_path: Path,
    m_curves_per_block: list[np.ndarray],  # (n_branches, K) at h ≈ 0.3
    k_grid: np.ndarray,
    title: str,
) -> None:
    n_blocks = len(m_curves_per_block)
    n_cols = 2
    n_rows = (n_blocks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharex=True)
    axes_flat = axes.ravel()
    for b in range(n_blocks):
        ax = axes_flat[b]
        curves = m_curves_per_block[b]  # (n_branches, K)
        peaks = np.max(np.abs(curves), axis=1)
        top16 = np.argsort(-peaks)[:16]
        cmap = plt.cm.tab20(np.linspace(0, 1, 16))
        for rank, j in enumerate(top16):
            ax.plot(k_grid, curves[j], color=cmap[rank], lw=0.8, alpha=0.8)
        # Overlay the G_0 reference shape (rescaled to fit).
        h_mid = 0.3
        g0 = np.abs(k_grid) * np.tanh(h_mid * np.abs(k_grid))
        peak_learned = float(np.max(np.abs(curves[top16])))
        peak_ref = float(np.max(g0))
        if peak_ref > 0:
            g0_scaled = g0 * (peak_learned * 0.9 / peak_ref)
        else:
            g0_scaled = g0
        ax.plot(k_grid, g0_scaled, color="k", ls=":", lw=1.6,
                label="ref G_0 (rescaled)" if b == 0 else None)
        ax.plot(k_grid, -g0_scaled, color="k", ls=":", lw=1.6)
        ax.set_title(f"block {b}", fontsize=10)
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
    for j in range(n_blocks, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Per-block forward-activation via a light re-implementation of the block.
# ----------------------------------------------------------------------------

def eta_features_forward(eta_norm: jnp.ndarray, params: dict, config: dict) -> jnp.ndarray:
    """Reproduce CraigSulemDNO._eta_spatial_features + trunk -> eta_features.

    Returns (B, N, width/2) in the model's working precision.

    We do NOT re-use loaded.model.apply because we want the *intermediate*
    per-block outputs which the top-level model does not expose.
    """
    batch_size, grid_size = eta_norm.shape
    n_freq = grid_size // 2 + 1
    domain_length = float(config["domain_length"])
    k_arr = (2.0 * jnp.pi / domain_length) * jnp.arange(n_freq)

    feats = [eta_norm]
    if int(config["cs_n_polys"]) >= 2:
        feats.append(eta_norm ** 2)
    if int(config["cs_n_polys"]) >= 3:
        feats.append(eta_norm ** 3)

    eta_hat = jnp.fft.rfft(eta_norm, axis=-1)
    if bool(config["cs_use_first_deriv"]):
        feats.append(jnp.fft.irfft(1j * k_arr[None, :] * eta_hat, n=grid_size, axis=-1))
    if bool(config["cs_use_second_deriv"]):
        feats.append(jnp.fft.irfft(-(k_arr ** 2)[None, :] * eta_hat, n=grid_size, axis=-1))
    if bool(config["cs_use_half_deriv"]):
        feats.append(jnp.fft.irfft(jnp.sqrt(k_arr)[None, :] * eta_hat, n=grid_size, axis=-1))
    if bool(config["cs_use_hilbert"]):
        sgn = jnp.sign(k_arr).astype(eta_hat.dtype)
        feats.append(jnp.fft.irfft(-1j * sgn[None, :] * eta_hat, n=grid_size, axis=-1))

    raw = jnp.stack(feats, axis=-1)   # (B, N, n_raw)
    W_proj = params["eta_feat_proj"]["kernel"]
    b_proj = params["eta_feat_proj"]["bias"]
    W_mix = params["eta_feat_mix"]["kernel"]
    b_mix = params["eta_feat_mix"]["bias"]
    h = jax.nn.gelu(raw @ W_proj + b_proj)
    h = jax.nn.gelu(h @ W_mix + b_mix)
    return h  # (B, N, width/2)


def block_forward(
    eta_features: jnp.ndarray,   # (B, N, width/2)
    xi_phys: jnp.ndarray,        # (B, N)
    depth: jnp.ndarray,          # (B, 1) log h
    block_params: dict,
    domain_length: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reproduce CraigSulemBlock.__call__ and expose per-branch activations.

    Returns
    -------
    block_out : (B, N)             — the block's contribution (physical, unnormalized)
    prods     : (B, N, n_branches) — φ_j(η) · (M_xi_j ξ) branches (spatial)
    m_out_arr : (B, K, n_branches) — depth-conditioned M_out multiplier evaluated at input depth
    branch_contribs : (B, n_branches) — per-branch |contribution|_L2 (spatial)
    """
    batch_size, grid_size, _ = eta_features.shape
    n_freq = grid_size // 2 + 1

    # phi_proj: (n_features_in, n_branches)
    W_phi = block_params["phi_proj"]["kernel"]
    b_phi = block_params["phi_proj"]["bias"]
    phi = eta_features @ W_phi + b_phi                          # (B, N, n_branches)

    # Multipliers.
    k_arr = (2.0 * jnp.pi / domain_length) * jnp.arange(n_freq)
    h_clip = jnp.minimum(depth, jnp.log(5.0))
    h_phys = jnp.exp(h_clip)                                    # (B, 1)
    k_b = jnp.broadcast_to(k_arr[None, :], (batch_size, n_freq))
    h_b = jnp.broadcast_to(h_phys, (batch_size, n_freq))
    kh = h_b * k_b
    feats = jnp.stack([k_b, h_b, jnp.tanh(kh), k_b * jnp.tanh(kh)], axis=-1)

    def _apply_mult(mlp_params):
        z = jnp.tanh(feats @ mlp_params["hidden"]["kernel"] + mlp_params["hidden"]["bias"])
        return z @ mlp_params["out"]["kernel"] + mlp_params["out"]["bias"]  # (B, K, C)

    m_xi_arr = _apply_mult(block_params["m_xi"])   # (B, K, n_branches)
    m_out_arr = _apply_mult(block_params["m_out"])

    # ξ branching in Fourier.
    xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")     # (B, K)
    xi_branched_hat = m_xi_arr * xi_hat[..., None]              # (B, K, n_branches)
    xi_branched = jnp.fft.irfft(xi_branched_hat, n=grid_size, axis=1, norm="forward")

    prods = phi * xi_branched                                   # (B, N, n_branches)

    # Per-branch physical contribution (before summing): M_out_j applied per branch,
    # then physical inverse FFT. This lets us pick the top-activating branches.
    prods_hat = jnp.fft.rfft(prods, axis=1, norm="forward")     # (B, K, n_branches)
    per_branch_hat = m_out_arr * prods_hat                       # (B, K, n_branches)
    per_branch_phys = jnp.fft.irfft(per_branch_hat, n=grid_size, axis=1, norm="forward")
    branch_contribs = jnp.sqrt(jnp.mean(per_branch_phys ** 2, axis=1))  # (B, n_branches)

    out_hat = jnp.sum(per_branch_hat, axis=-1)                  # (B, K)
    block_out = jnp.fft.irfft(out_hat, n=grid_size, axis=-1, norm="forward")
    return block_out, prods, m_out_arr, branch_contribs


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def apply_g0(xi: np.ndarray, depth: float, domain_length: float) -> np.ndarray:
    N = xi.shape[-1]
    k_full = 2.0 * np.pi / domain_length * np.concatenate([
        np.arange(0, N // 2 + 1),
        np.arange(1 - N // 2, 0),
    ])
    g0 = np.abs(k_full) * np.tanh(depth * np.abs(k_full))
    xi_hat = np.fft.fft(xi, axis=-1)
    return np.real(np.fft.ifft(g0 * xi_hat, axis=-1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load run.
    loaded = load_run(RUN_DIR, checkpoint="best")
    config = loaded.config
    params = loaded.params
    print(f"loaded v10 run: model={config['model']} n_blocks={config['n_blocks']} "
          f"latent={config['latent']} mult_hidden={config['cs_mult_hidden']}")

    # ------------------------------------------------------------------
    # Part 1 addendum: effective rank + top-16 branches per block.
    # ------------------------------------------------------------------
    n_blocks = int(config["n_blocks"])
    modes = int(config["modes"])
    domain_length = float(config["domain_length"])
    k_grid = (2.0 * np.pi / domain_length) * np.arange(modes)
    h_values = np.asarray([0.05, 0.1, 0.3, 1.0, 5.0], dtype=np.float64)
    k_grid_j = jnp.asarray(k_grid, dtype=jnp.float32)
    h_values_j = jnp.asarray(h_values, dtype=jnp.float32)

    eff_rank_xi_per_block = []
    eff_rank_out_per_block = []
    top16_curves_xi = []
    # Also measure the composed phi_proj column L2 to count "load-bearing" branches
    # in a way that's directly comparable to v5's ~50/256 finding.
    W_proj_np = np.asarray(params["eta_feat_proj"]["kernel"])
    W_mix_np = np.asarray(params["eta_feat_mix"]["kernel"])
    phi_col_l2_per_block = []
    for b in range(n_blocks):
        block_params = params[f"cs_block_{b}"]
        m_xi_all = eval_multiplier(block_params["m_xi"], k_grid_j, h_values_j)
        m_out_all = eval_multiplier(block_params["m_out"], k_grid_j, h_values_j)
        eff_rank_xi_per_block.append(effective_rank(m_xi_all))
        eff_rank_out_per_block.append(effective_rank(m_out_all))
        h_mid = int(np.argmin(np.abs(h_values - 0.3)))
        curves = m_xi_all[h_mid].T
        top16_curves_xi.append(curves)
        # composed phi: (n_features, latent). Column L2 is what v5 report used.
        W_phi = np.asarray(block_params["phi_proj"]["kernel"])
        composed = W_proj_np @ W_mix_np @ W_phi   # (n_feats, latent)
        col_l2 = np.linalg.norm(composed, axis=0)
        phi_col_l2_per_block.append(col_l2)

    plot_top16_branches(
        OUT_DIR / "m_xi_top16_by_block.png",
        top16_curves_xi, k_grid,
        title="cs_dno v10 — top-16 M_xi branches per block at h=0.3 (black dotted = G_0 shape)",
    )
    print("\n===== Effective rank per block =====")
    print(f"{'block':>5} | {'M_xi peak_>10%_max':>18} | {'M_out peak_>10%_max':>18} | "
          f"{'phi_col L2 >10% max':>21} | {'top10 L2 share (M_xi)':>22}")
    phi_col_significant = []
    for b in range(n_blocks):
        col_l2 = phi_col_l2_per_block[b]
        n_phi = int(np.sum(col_l2 > 0.1 * float(np.max(col_l2))))
        phi_col_significant.append(n_phi)
        rxi = eff_rank_xi_per_block[b]
        rout = eff_rank_out_per_block[b]
        print(f"{b:>5} | {rxi['peak_gt_1e-1_of_max']:>18} | "
              f"{rout['peak_gt_1e-1_of_max']:>18} | {n_phi:>21} | "
              f"{rxi['top10_l2_share']:>22.3f}")

    # ------------------------------------------------------------------
    # Part 2/3: forward on selected ICs.
    # ------------------------------------------------------------------
    traj = np.load(TRAJ_PATH)
    depths_np = np.asarray(traj["depths"])
    times_np = np.asarray(traj["times"])
    pred_eta = np.asarray(traj["pred_eta"])   # (T, I, N)
    pred_xi = np.asarray(traj["pred_xi"])
    pred_gxi = np.asarray(traj["pred_gxi"])

    healthy_idx = [0, 4, 6, 16]   # 16 is a fresh finite one (h=0.066)
    nan_idx = [5, 11, 22, 23]

    # Sanity-check that nothing among these is NaN at t=0.
    for i in healthy_idx + nan_idx:
        assert not np.isnan(pred_eta[0, i]).any(), f"IC {i} has NaN at t=0"

    # Build the batched predictor for a scale-normalized model.
    feat_absmax = np.asarray(loaded.stats["feature_absmax"], dtype=np.float32)
    target_scale = float(loaded.stats["target_absmax"])

    # Also pre-compute h_clip logs to feed model.
    def run_batch(eta_batch: np.ndarray, xi_batch: np.ndarray, h_batch: np.ndarray):
        """Return (block_outs_np, gxi_full_np, branch_contribs_per_block).

        eta_batch, xi_batch: (B, N) physical.
        h_batch: (B,) physical depths.
        """
        eta_norm_np = eta_batch / feat_absmax[0]
        xi_norm_np = xi_batch / feat_absmax[1]
        xi_phys_np = xi_norm_np * feat_absmax[1]   # == xi_batch, kept for clarity

        eta_norm = jnp.asarray(eta_norm_np, dtype=jnp.float32)
        xi_norm = jnp.asarray(xi_norm_np, dtype=jnp.float32)
        xi_phys = jnp.asarray(xi_phys_np, dtype=jnp.float32)
        depth_log = jnp.log(jnp.maximum(jnp.asarray(h_batch, dtype=jnp.float32), 1e-12))[:, None]

        # Build eta trunk features.
        eta_features = eta_features_forward(eta_norm, params, config)

        # Iterate blocks; accumulate.
        residual = jnp.zeros_like(xi_phys)
        block_outs = []
        branch_contribs_per_block = []
        for b in range(n_blocks):
            block_params_b = params[f"cs_block_{b}"]
            block_out, _prods, _m_out, branch_contribs = block_forward(
                eta_features, xi_phys, depth_log, block_params_b, domain_length,
            )
            residual = residual + block_out
            block_outs.append(np.asarray(block_out))
            branch_contribs_per_block.append(np.asarray(branch_contribs))

        residual_np = np.asarray(residual) / target_scale

        # Linear baseline: G_0(h) ξ (in normalized OUTPUT space -> convert to physical).
        N = xi_phys.shape[-1]
        n_freq = N // 2 + 1
        k_r = (2.0 * np.pi / domain_length) * np.arange(n_freq)
        xi_hat = np.fft.rfft(np.asarray(xi_phys), axis=-1)
        h_phys = np.asarray(jnp.exp(jnp.minimum(depth_log, jnp.log(5.0))))
        h_col = h_phys.reshape(-1, 1)
        g0 = np.abs(k_r)[None, :] * np.tanh(h_col * np.abs(k_r)[None, :])
        g0_xi = np.fft.irfft(g0 * xi_hat, n=N, axis=-1)

        # Full pred_gxi = G_0(ξ) + residual_scaled  (matching the model)
        # residual returned above already divided by target_scale but ADDED to normalized output.
        # The model builds baseline in NORMALIZED output space; final pred_gxi_physical =
        # normalized_output * target_scale. Let me instead reconstruct exactly like the model:
        #    baseline_norm = G_0(xi_phys) / target_scale
        #    residual_norm = residual_raw / target_scale
        #    pred_gxi_phys = (baseline_norm + residual_norm) * target_scale
        #                  = G_0(xi_phys) + residual_raw
        pred_gxi_phys = g0_xi + np.asarray(residual)
        # Zero-mean subtract (as batched predictor does)
        pred_gxi_phys = pred_gxi_phys - pred_gxi_phys.mean(axis=-1, keepdims=True)
        return {
            "block_outs": np.stack(block_outs, axis=0),  # (n_blocks, B, N)
            "branch_contribs": np.stack(branch_contribs_per_block, axis=0),  # (n_blocks, B, C)
            "residual_phys": np.asarray(residual),        # (B, N) physical (already summed)
            "g0_xi_phys": g0_xi,                          # (B, N)
            "pred_gxi_phys": pred_gxi_phys,               # (B, N)
        }

    # -------- t=0 pass on all 8 ICs (healthy + failing) --------
    ic_all = healthy_idx + nan_idx
    eta0 = pred_eta[0, ic_all, :]
    xi0 = pred_xi[0, ic_all, :]
    h_all = depths_np[ic_all]

    out0 = run_batch(eta0, xi0, h_all)

    xi_l2 = np.linalg.norm(xi0, axis=-1)
    g0_l2 = np.linalg.norm(out0["g0_xi_phys"], axis=-1)
    residual_l2 = np.linalg.norm(out0["residual_phys"], axis=-1)
    block_norms = np.linalg.norm(out0["block_outs"], axis=-1)   # (n_blocks, B)
    gxi_pred_l2 = np.linalg.norm(out0["pred_gxi_phys"], axis=-1)
    gxi_true_pred_l2 = np.linalg.norm(pred_gxi[0, ic_all, :], axis=-1)   # sanity check

    # Sanity check that our reconstructed gxi matches pred_gxi[0] from the saved traj.
    gxi_match_err = np.linalg.norm(out0["pred_gxi_phys"] - pred_gxi[0, ic_all, :], axis=-1) / (
        gxi_pred_l2 + 1e-12
    )
    print(f"pred_gxi reconstruction error (should be ~0): {gxi_match_err}")

    # Per-IC report.
    rows_t0 = []
    for j, i in enumerate(ic_all):
        top3 = []
        for b in range(n_blocks):
            bc = out0["branch_contribs"][b, j]
            order = np.argsort(-bc)[:3]
            top3.append([(int(idx), float(bc[idx])) for idx in order])
        rows_t0.append({
            "ic": int(i),
            "h": float(h_all[j]),
            "status": "healthy" if i in healthy_idx else "failing",
            "xi_l2": float(xi_l2[j]),
            "g0_xi_l2": float(g0_l2[j]),
            "residual_l2": float(residual_l2[j]),
            "gxi_pred_l2": float(gxi_pred_l2[j]),
            "residual_over_g0": float(residual_l2[j] / (g0_l2[j] + 1e-12)),
            "block_norm_over_xi_l2": [
                float(block_norms[b, j] / (xi_l2[j] + 1e-12)) for b in range(n_blocks)
            ],
            "top3_branches_per_block": top3,
        })

    # -------- t = t_last_finite for failing ICs --------
    rows_pre_nan = []
    for i in nan_idx:
        is_nan = np.isnan(pred_eta[:, i]).any(axis=-1)   # (T,)
        t_nan = int(np.argmax(is_nan))
        t_ref = max(t_nan - 1, 0)
        eta_ref = pred_eta[t_ref, i, :][None, :]
        xi_ref = pred_xi[t_ref, i, :][None, :]
        h_ref = depths_np[i:i + 1]
        out_ref = run_batch(eta_ref, xi_ref, h_ref)

        xi_l2_r = float(np.linalg.norm(xi_ref))
        g0_l2_r = float(np.linalg.norm(out_ref["g0_xi_phys"]))
        residual_l2_r = float(np.linalg.norm(out_ref["residual_phys"]))
        block_norms_r = np.linalg.norm(out_ref["block_outs"], axis=-1)  # (n_blocks, 1)
        top3_r = []
        for b in range(n_blocks):
            bc = out_ref["branch_contribs"][b, 0]
            order = np.argsort(-bc)[:3]
            top3_r.append([(int(idx), float(bc[idx])) for idx in order])
        rows_pre_nan.append({
            "ic": int(i),
            "h": float(h_ref[0]),
            "t_idx": t_ref,
            "t": float(times_np[t_ref]),
            "xi_l2": xi_l2_r,
            "g0_xi_l2": g0_l2_r,
            "residual_l2": residual_l2_r,
            "residual_over_g0": residual_l2_r / (g0_l2_r + 1e-12),
            "block_norm_over_xi_l2": [
                float(block_norms_r[b, 0] / (xi_l2_r + 1e-12)) for b in range(n_blocks)
            ],
            "top3_branches_per_block": top3_r,
        })

    # Overlap of top-10 activated branches, failing vs healthy, per block.
    top10_per_ic_block = {}
    for j, i in enumerate(ic_all):
        top10_per_ic_block[int(i)] = []
        for b in range(n_blocks):
            bc = out0["branch_contribs"][b, j]
            top10_per_ic_block[int(i)].append(np.argsort(-bc)[:10].tolist())
    overlap = []
    for b in range(n_blocks):
        heal_union = set()
        fail_union = set()
        for i in healthy_idx:
            heal_union |= set(top10_per_ic_block[i][b])
        for i in nan_idx:
            fail_union |= set(top10_per_ic_block[i][b])
        jac = len(heal_union & fail_union) / max(1, len(heal_union | fail_union))
        overlap.append({
            "block": b,
            "n_healthy_union": len(heal_union),
            "n_failing_union": len(fail_union),
            "intersection": len(heal_union & fail_union),
            "jaccard": float(jac),
        })

    print("\n===== Overlap of top-10 activating branches (healthy vs failing, t=0) =====")
    for row in overlap:
        print(f"block {row['block']}: |healthy∪| = {row['n_healthy_union']}, "
              f"|failing∪| = {row['n_failing_union']}, "
              f"|∩| = {row['intersection']} (jaccard = {row['jaccard']:.2f})")

    diag = {
        "run_dir": str(RUN_DIR),
        "effective_rank_M_xi_per_block": eff_rank_xi_per_block,
        "effective_rank_M_out_per_block": eff_rank_out_per_block,
        "phi_col_significant_count_10pct_of_max": phi_col_significant,
        "top_branch_overlap": overlap,
        "t0_report": rows_t0,
        "pre_nan_report": rows_pre_nan,
    }
    with open(OUT_DIR / "activation_diagnostics.json", "w", encoding="utf-8") as h:
        json.dump(diag, h, indent=2)

    # -------- Print concise summary --------
    print("\n===== Part 2 (t=0): residual / G_0 ratio and per-block magnitudes =====")
    for r in rows_t0:
        print(f"IC {r['ic']:2d} h={r['h']:.3f} [{r['status']:7s}] "
              f"res/G0 = {r['residual_over_g0']:.3f}  "
              f"|block_b|/|ξ| = "
              + " ".join(f"{v:.2f}" for v in r['block_norm_over_xi_l2']))
    print("\n===== Part 3 (t_last_finite): failing ICs =====")
    for r in rows_pre_nan:
        print(f"IC {r['ic']:2d} h={r['h']:.3f} t={r['t']:6.2f}  "
              f"res/G0 = {r['residual_over_g0']:.3f}  "
              f"|block_b|/|ξ| = "
              + " ".join(f"{v:.2f}" for v in r['block_norm_over_xi_l2']))

    print("\nTop-3 branches per block at t=0 (verify overlap between failing and healthy):")
    header = "IC/block " + " ".join(f"b{b}" for b in range(n_blocks))
    print(header)
    for r in rows_t0:
        cells = []
        for b in range(n_blocks):
            top3 = r['top3_branches_per_block'][b]
            cells.append("[" + ",".join(str(idx) for idx, _ in top3) + "]")
        print(f"IC {r['ic']:2d} ({r['status'][:4]}): " + " ".join(cells))


if __name__ == "__main__":
    main()
