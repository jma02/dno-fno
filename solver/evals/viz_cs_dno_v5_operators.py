"""Visualize the learned Craig-Sulem operator symbols inside a cs_dno checkpoint.

For each block we:
  1. Evaluate the depth-aware Fourier multipliers M_xi(k, h), M_out(k, h)
     on a (k, h) grid by running the multiplier MLP forward (no model.apply).
  2. Extract the phi_proj Dense kernel mapping the 7 eta-features to the
     latent=256 branches; summarize per-feature L1 weight contribution.

References overlaid on the multiplier plots are the known CS symbols
(|k|, the linear DNO G0(k,h) = |k| tanh(h|k|), and the identity).

CPU-only by construction: JAX_PLATFORMS is forced to "cpu" before jax import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parents[2]
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dno_net_v2 import CraigSulemDNO  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "cs_dno_w512b8_l256_v5_2gpu_20260615_141717"
DEFAULT_OUT_DIR = REPO_ROOT / "notes" / "figures" / "cs_dno_v5_operator_viz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR),
                        help="Path to a cs_dno training run directory (must contain config.json and best_val_ckpt/).")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory to write figures into.")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_from_config(config: dict) -> CraigSulemDNO:
    if bool(config.get("cs_fft_fp64", False)) or bool(
        config.get("cs_g1_fft_fp64", False)
    ):
        jax.config.update("jax_enable_x64", True)
    return CraigSulemDNO(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config["n_blocks"]),
        latent=int(config["latent"]),
        n_polys=int(config["cs_n_polys"]),
        use_first_deriv=bool(config["cs_use_first_deriv"]),
        use_second_deriv=bool(config["cs_use_second_deriv"]),
        use_half_deriv=bool(config["cs_use_half_deriv"]),
        use_hilbert=bool(config["cs_use_hilbert"]),
        mult_hidden=int(config["cs_mult_hidden"]),
        use_g1_baseline=bool(config.get("cs_use_g1_baseline", False)),
        g1_k_cut=int(config.get("cs_g1_k_cut", 128)),
        fft_fp64=bool(config.get("cs_fft_fp64", False)),
        g1_fft_fp64=bool(config.get("cs_g1_fft_fp64", False)),
        tie_xi_out_mult=bool(config.get("cs_tie_xi_out_mult", False)),
        phi_bias_free=bool(config.get("cs_phi_bias_free", False)),
        residual_eta_order=int(config.get("cs_residual_eta_order", 1)),
        depth_scaled_residual=bool(config.get("cs_depth_scaled_residual", False)),
        block_k_cut=int(config.get("cs_block_k_cut", 0)),
        residual_highband_cap=bool(config.get("cs_residual_highband_cap", False)),
        residual_highband_cap_k_cut=float(config.get("cs_residual_highband_cap_k_cut", 32.0)),
        residual_highband_cap_beta=float(config.get("cs_residual_highband_cap_beta", 0.10)),
        residual_highband_cap_floor=float(config.get("cs_residual_highband_cap_floor", 0.0)),
        output_highband_cap=bool(config.get("cs_output_highband_cap", False)),
        output_highband_cap_k_cut=float(config.get("cs_output_highband_cap_k_cut", 32.0)),
        output_highband_cap_r_max=float(config.get("cs_output_highband_cap_r_max", 1e-2)),
        output_highband_cap_abs_floor=float(config.get("cs_output_highband_cap_abs_floor", 5.0)),
        domain_length=float(config["domain_length"]),
        xi_scale=float(config["xi_scale"]),
        eta_scale=float(config["eta_scale"]),
        target_scale=float(config["target_scale"]),
        h_clip_max=5.0,
    )


def load_params(run_dir: Path, checkpoint_name: str):
    checkpoint_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return checkpoint_dir, params


def eta_feature_names(config: dict) -> list[str]:
    """Return the ordered list of eta-feature labels exactly matching dno_net_v2."""
    names: list[str] = []
    n_polys = int(config["cs_n_polys"])
    for p in range(1, n_polys + 1):
        if p == 1:
            names.append("eta")
        elif p == 2:
            names.append("eta^2")
        elif p == 3:
            names.append("eta^3")
        else:
            names.append(f"eta^{p}")
    if bool(config["cs_use_first_deriv"]):
        names.append("d_x eta")
    if bool(config["cs_use_second_deriv"]):
        names.append("d_xx eta")
    if bool(config["cs_use_half_deriv"]):
        names.append("|D|^1/2 eta")
    if bool(config["cs_use_hilbert"]):
        names.append("H eta")
    return names


def eval_multiplier(
    mlp_params: dict,
    k_grid: jnp.ndarray,  # (K,) angular wavenumbers k_phys = 2*pi*n/L
    h_values: jnp.ndarray,  # (H,) physical depths
    branch_idx: int | None = None,
) -> np.ndarray:
    """Replicate DepthAwareMultiplier.__call__ on a (k, h) grid.

    mlp_params: dict {"hidden": {"kernel","bias"}, "out": {"kernel","bias"}}
    Returns array of shape (H, K, C) where C = number of branches. If
    branch_idx is given returns (H, K) — the mean over branches.
    """
    K = k_grid.shape[0]
    H = h_values.shape[0]
    # (H, K)
    k_b = jnp.broadcast_to(k_grid[None, :], (H, K))
    h_b = jnp.broadcast_to(h_values[:, None], (H, K))
    kh = h_b * k_b
    feats = jnp.stack([k_b, h_b, jnp.tanh(kh), k_b * jnp.tanh(kh)], axis=-1)  # (H, K, 4)

    hidden_w = mlp_params["hidden"]["kernel"]
    hidden_b = mlp_params["hidden"]["bias"]
    out_w = mlp_params["out"]["kernel"]
    out_b = mlp_params["out"]["bias"]

    z = jnp.tanh(feats @ hidden_w + hidden_b)  # (H, K, hidden)
    out = z @ out_w + out_b                    # (H, K, C)
    out_np = np.asarray(out)
    if branch_idx is not None:
        return out_np[..., branch_idx]
    return out_np


def reference_symbols(k_grid: np.ndarray, h_values: np.ndarray) -> dict[str, np.ndarray]:
    """Compute classical CS reference symbols evaluated on the same (k, h) grid."""
    K = k_grid.shape[0]
    H = h_values.shape[0]
    k_b = np.broadcast_to(k_grid[None, :], (H, K))
    h_b = np.broadcast_to(h_values[:, None], (H, K))
    return {
        "|k|": np.abs(k_b),
        "G0=|k|tanh(h|k|)": np.abs(k_b) * np.tanh(h_b * np.abs(k_b)),
        "identity (1)": np.ones_like(k_b),
        "i*k (d_x sym)": k_b,  # |i*k| magnitude
    }


def plot_multipliers_by_block(
    out_path: Path,
    title: str,
    mult_curves_per_block: list[np.ndarray],  # length n_blocks, each (H, K)
    k_grid: np.ndarray,
    h_values: np.ndarray,
    refs: dict[str, np.ndarray],
) -> None:
    n_blocks = len(mult_curves_per_block)
    n_cols = 2
    n_rows = (n_blocks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharex=True)
    axes_flat = axes.ravel()
    h_palette = plt.cm.viridis(np.linspace(0.05, 0.95, len(h_values)))
    # Determine a global y-limit for the learned curves so subplots are comparable
    # but references (which are O(|k|)) don't dominate.
    all_learned = np.concatenate([c.ravel() for c in mult_curves_per_block])
    y_top = max(1e-3, float(np.percentile(all_learned, 99.5)) * 1.5)

    for b in range(n_blocks):
        ax = axes_flat[b]
        curves = mult_curves_per_block[b]  # (H, K), branch-RMS magnitude
        for hi, h in enumerate(h_values):
            ax.plot(k_grid, curves[hi], color=h_palette[hi], lw=1.1,
                    label=f"learned h={h:.2g}" if b == 0 else None)
        # Reference symbols at mid h. We rescale them so the *shape* fits in the
        # axis (the comparison of interest is curvature/sign, not absolute scale —
        # absolute scale just reveals the rank-1 sum's per-branch shrinkage).
        ref_styles = {
            "|k|": ("0.35", "--", 1.6),
            "G0=|k|tanh(h|k|)": ("0.10", ":", 2.0),
            "identity (1)": ("0.55", "-.", 1.4),
        }
        mid = int(np.argmin(np.abs(h_values - 0.3)))
        for ref_name, ref_arr in refs.items():
            if ref_name not in ref_styles:
                continue
            color, ls, lw = ref_styles[ref_name]
            ref_curve = ref_arr[mid].copy()
            # Rescale references so peak == y_top * 0.9 — they're shown for shape
            # comparison only. Identity is constant 1, just clip to y_top.
            peak = float(np.max(ref_curve))
            if peak > 0:
                ref_curve = ref_curve * (y_top * 0.9 / peak)
            ax.plot(k_grid, ref_curve, color=color, ls=ls, lw=lw,
                    label=f"ref: {ref_name} (rescaled)" if b == 0 else None)
        ax.set_title(f"block {b}", fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylim(0, y_top)
        ax.grid(True, alpha=0.3)
    for j in range(n_blocks, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_phi_proj_feature_bars(
    out_path: Path,
    phi_kernels_per_block: list[np.ndarray],  # (n_features, latent) each
    feature_names: list[str],
) -> None:
    n_blocks = len(phi_kernels_per_block)
    n_cols = 2
    n_rows = (n_blocks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.4 * n_rows), sharey=False)
    axes_flat = axes.ravel()
    x_idx = np.arange(len(feature_names))
    for b in range(n_blocks):
        ax = axes_flat[b]
        ker = phi_kernels_per_block[b]
        contrib = np.mean(np.abs(ker), axis=1)  # (n_features,)
        ax.bar(x_idx, contrib, color="tab:blue")
        ax.set_title(f"block {b}", fontsize=10)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(n_blocks, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("phi_proj: mean |coefficient| per feature (averaged over latent=256 branches)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_phi_proj_heatmap_block0(
    out_path: Path,
    phi_kernel: np.ndarray,  # (n_features, latent)
    feature_names: list[str],
) -> None:
    # Sort latent columns by L2 norm descending so structure is visible.
    col_norms = np.linalg.norm(phi_kernel, axis=0)
    order = np.argsort(-col_norms)
    sorted_ker = phi_kernel[:, order]
    fig, ax = plt.subplots(figsize=(12, 3.0))
    vmax = float(np.max(np.abs(sorted_ker)))
    if vmax == 0.0:
        vmax = 1e-8
    im = ax.imshow(sorted_ker, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("latent branch (sorted by column L2 norm, desc)")
    ax.set_title("cs_block_0.phi_proj.kernel — eta-feature × latent branch")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summarize_block_geometry(
    m_xi: np.ndarray,  # (H, K, n_branches)
    m_out: np.ndarray,  # (H, K, n_branches)
    k_grid: np.ndarray,
    h_values: np.ndarray,
) -> dict[str, float]:
    """Return scalar diagnostics for one block.

    We use branch-RMS magnitude and compute the **shape correlation** of
    the (k -> rms_M(k, h_mid)) curve against the canonical CS symbols. Shape
    correlation is the relevant question here — the absolute scale is largely
    set by the rank-1 sum's implicit shrinkage and is not directly comparable.
    """
    xi_mag = np.sqrt(np.mean(m_xi ** 2, axis=-1))   # (H, K)
    out_mag = np.sqrt(np.mean(m_out ** 2, axis=-1)) # (H, K)
    h_mid = int(np.argmin(np.abs(h_values - 0.3)))
    k_pos = k_grid > 0
    abs_k = np.abs(k_grid)
    g0_mid = abs_k * np.tanh(h_values[h_mid] * abs_k)

    def shape_corr(a: np.ndarray, b: np.ndarray) -> float:
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        denom = np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12
        return float(np.dot(a0, b0) / denom)

    xi_curve = xi_mag[h_mid][k_pos]
    out_curve = out_mag[h_mid][k_pos]
    g0_curve = g0_mid[k_pos]
    k_curve = abs_k[k_pos]
    ones_curve = np.ones_like(k_curve)
    return {
        "rms_xi_mid_h_mean": float(np.mean(xi_mag[h_mid])),
        "rms_out_mid_h_mean": float(np.mean(out_mag[h_mid])),
        "xi_shape_corr_|k|": shape_corr(xi_curve, k_curve),
        "xi_shape_corr_G0": shape_corr(xi_curve, g0_curve),
        "xi_shape_corr_identity": shape_corr(xi_curve, ones_curve),
        "out_shape_corr_|k|": shape_corr(out_curve, k_curve),
        "out_shape_corr_G0": shape_corr(out_curve, g0_curve),
        "out_shape_corr_identity": shape_corr(out_curve, ones_curve),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"backend devices: {jax.devices()}")
    print(f"run_dir: {run_dir}")
    print(f"out_dir: {out_dir}")

    config = load_config(run_dir)
    print(f"model={config.get('model')} n_blocks={config['n_blocks']} "
          f"latent={config['latent']} modes={config['modes']} "
          f"mult_hidden={config['cs_mult_hidden']}")

    checkpoint_path, params = load_params(run_dir, args.checkpoint)
    print(f"loaded params from {checkpoint_path}")

    # Top-level param keys (sanity).
    top_keys = sorted(params.keys())
    print(f"top-level param keys: {top_keys}")

    feature_names = eta_feature_names(config)
    print(f"eta features ({len(feature_names)}): {feature_names}")

    # (k, h) grid. k_phys = 2*pi*n/L; with L=2pi this gives k_phys = n.
    n_freq = int(config["modes"])
    domain_length = float(config["domain_length"])
    k_indices = np.arange(n_freq)
    k_grid = (2.0 * np.pi / domain_length) * k_indices  # shape (modes,)
    h_values = np.asarray([0.05, 0.1, 0.3, 1.0, 5.0], dtype=np.float64)
    print(f"k grid: [{k_grid[0]:.3g}, ..., {k_grid[-1]:.3g}] ({k_grid.size} points)")
    print(f"h grid: {h_values.tolist()}")

    k_grid_j = jnp.asarray(k_grid, dtype=jnp.float32)
    h_values_j = jnp.asarray(h_values, dtype=jnp.float32)

    n_blocks = int(config["n_blocks"])
    m_xi_curves: list[np.ndarray] = []   # per-block (H, K) branch-averaged |M_xi|
    m_out_curves: list[np.ndarray] = []
    phi_kernels: list[np.ndarray] = []   # per-block (n_features, latent)
    per_block_diag: list[dict[str, float]] = []

    for b in range(n_blocks):
        block_key = f"cs_block_{b}"
        block_params = params[block_key]
        # Sanity: list sub-keys for the first block only.
        if b == 0:
            print(f"{block_key} sub-keys: {sorted(block_params.keys())}")
            print(f"  m_xi sub-keys: {sorted(block_params['m_xi'].keys())}")
            print(f"  m_xi.hidden.kernel shape: {tuple(block_params['m_xi']['hidden']['kernel'].shape)}")
            print(f"  m_xi.out.kernel    shape: {tuple(block_params['m_xi']['out']['kernel'].shape)}")
            print(f"  phi_proj.kernel    shape: {tuple(block_params['phi_proj']['kernel'].shape)}")

        m_xi_all = eval_multiplier(block_params["m_xi"], k_grid_j, h_values_j)
        m_out_all = eval_multiplier(block_params["m_out"], k_grid_j, h_values_j)
        # RMS over branches: preserves per-branch scale (mean(|.|) over branches
        # would average toward zero when branches partly cancel; RMS is the L2
        # norm of the per-branch multiplier vector divided by sqrt(n_branches)).
        m_xi_mag = np.sqrt(np.mean(m_xi_all ** 2, axis=-1))
        m_out_mag = np.sqrt(np.mean(m_out_all ** 2, axis=-1))
        m_xi_curves.append(m_xi_mag)
        m_out_curves.append(m_out_mag)

        phi_kernel = np.asarray(block_params["phi_proj"]["kernel"])
        # phi_proj input is the eta_feat_mix output (width/2 channels), not the raw
        # 7-feature stack. To get a per-feature contribution we have to compose:
        #   raw_eta_feats (n_feats) -> eta_feat_proj -> gelu -> eta_feat_mix -> gelu -> phi
        # We approximate per-feature contribution by the L1 of the linear chain
        # (ignoring the gelu, which is monotone and roughly preserves sign).
        # We will assemble this once below using the top-level eta_feat layers.
        phi_kernels.append(phi_kernel)

        diag = summarize_block_geometry(m_xi_all, m_out_all, k_grid, h_values)
        per_block_diag.append(diag)
        print(f"block {b}: rms<M_xi>(h~.3)={diag['rms_xi_mid_h_mean']:.3g} "
              f"rms<M_out>(h~.3)={diag['rms_out_mid_h_mean']:.3g} | "
              f"M_xi shape-corr  |k|={diag['xi_shape_corr_|k|']:+.2f} "
              f"G0={diag['xi_shape_corr_G0']:+.2f} "
              f"id={diag['xi_shape_corr_identity']:+.2f}")

    # Compose the linear part of the eta-feature trunk so per-feature attribution
    # at the phi_proj input is well-defined. The chain is:
    #   raw_feats @ W_proj + b_proj  -> gelu -> @ W_mix + b_mix -> gelu -> phi
    # Ignoring gelu (treating it as identity), the effective linear kernel from
    # raw_eta_feats -> phi_proj_out is  W_proj @ W_mix @ W_phi.
    W_proj = np.asarray(params["eta_feat_proj"]["kernel"])  # (n_feats, width/2)
    W_mix = np.asarray(params["eta_feat_mix"]["kernel"])    # (width/2, width/2)
    print(f"eta_feat_proj.kernel: {W_proj.shape}, eta_feat_mix.kernel: {W_mix.shape}")
    composed_per_block = [W_proj @ W_mix @ k for k in phi_kernels]  # each (n_feats, latent)

    # Plots.
    refs = reference_symbols(k_grid, h_values)
    version_tag = "v10" if "v9_h100x2" in str(run_dir) else "v5"
    plot_multipliers_by_block(
        out_dir / "m_xi_by_block.png",
        title=f"cs_dno {version_tag} — M_xi(k, h) magnitude per block (branch-averaged)",
        mult_curves_per_block=m_xi_curves,
        k_grid=k_grid,
        h_values=h_values,
        refs=refs,
    )
    plot_multipliers_by_block(
        out_dir / "m_out_by_block.png",
        title=f"cs_dno {version_tag} — M_out(k, h) magnitude per block (branch-averaged)",
        mult_curves_per_block=m_out_curves,
        k_grid=k_grid,
        h_values=h_values,
        refs=refs,
    )
    plot_phi_proj_feature_bars(
        out_dir / "phi_proj_features.png",
        phi_kernels_per_block=composed_per_block,
        feature_names=feature_names,
    )
    plot_phi_proj_heatmap_block0(
        out_dir / "phi_proj_heatmap.png",
        phi_kernel=composed_per_block[0],
        feature_names=feature_names,
    )

    # Save numerical diagnostics for the report.
    diag_json = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "h_values": h_values.tolist(),
        "k_values": k_grid.tolist(),
        "feature_names": feature_names,
        "per_block": per_block_diag,
        "per_block_feature_l1": [
            {name: float(np.mean(np.abs(comp[i]))) for i, name in enumerate(feature_names)}
            for comp in composed_per_block
        ],
    }
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diag_json, handle, indent=2)

    print("\n--- per-block feature contributions (mean |W_proj @ W_mix @ W_phi| over latent) ---")
    for b, comp in enumerate(composed_per_block):
        per_feat = np.mean(np.abs(comp), axis=1)  # (n_feats,)
        ranked = sorted(zip(feature_names, per_feat), key=lambda t: -t[1])
        top3 = ", ".join(f"{n}={v:.3g}" for n, v in ranked[:3])
        print(f"block {b}: top features -> {top3}")

    print(f"\nWrote: m_xi_by_block.png, m_out_by_block.png, phi_proj_features.png, "
          f"phi_proj_heatmap.png, diagnostics.json to {out_dir}")


if __name__ == "__main__":
    main()
