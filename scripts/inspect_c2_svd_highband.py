"""Compute the real-Fourier SVD of the high-high block (k=96..127) of the
C2 final CS-DNO for a fixed failing-case eta, and compare to the order-6
truth DNO. This is the SVD part of diagnostic A2."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solver.evals import model_rollout as mr
from solver.solvers import dno_series_jax as ds


def load_failing_state(run_dir: Path, regime: str, case_id: int, frame: int):
    npz_dir = run_dir / "eval_final_2reg_f64h_batched_cached_20260707_133545" / regime
    npz = np.load(npz_dir / f"{regime}_trajs.npz")
    case_idx = int(np.where(npz["case_ids"] == case_id)[0][0])
    h = float(npz["depths"][case_idx])
    time = float(npz["times"][frame])
    pred_eta = np.asarray(npz["pred_eta"][frame, case_idx, :], dtype=np.float64)
    truth_eta = np.asarray(npz["truth_eta"][frame, case_idx, :], dtype=np.float64)
    return pred_eta, truth_eta, h, time


def build_real_operator_matrix(
    apply_operator,
    x: np.ndarray,
    k_modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build real-Fourier operator matrix L with rows (cos_out, sin_out) and
    cols (cos_in, sin_in) for k in k_modes.

    Returns (L, cos_basis, sin_basis).
    """
    N = x.shape[-1]
    L = x[-1] + x[1] - x[0]  # domain length
    # Orthonormal real Fourier basis on the periodic grid.
    cos_basis = {}
    sin_basis = {}
    norm = np.sqrt(N / 2.0)
    for k in k_modes:
        cos_basis[k] = np.cos(2.0 * np.pi * k * x / L) / norm
        sin_basis[k] = np.sin(2.0 * np.pi * k * x / L) / norm

    cols = []
    for k_in in k_modes:
        for label, basis in [("cos", cos_basis), ("sin", sin_basis)]:
            gxi = apply_operator(basis[k_in])
            row = []
            for k_out in k_modes:
                row.append(np.sum(cos_basis[k_out] * gxi))
                row.append(np.sum(sin_basis[k_out] * gxi))
            cols.append(row)
    return np.array(cols).T, cos_basis, sin_basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path,
                        default=Path("/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707"))
    parser.add_argument("--case_id", type=int, default=5)
    parser.add_argument("--frame", type=int, default=40)
    parser.add_argument("--use_truth_eta", action="store_true")
    parser.add_argument("--k_lo", type=int, default=96)
    parser.add_argument("--k_hi", type=int, default=128)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    loaded = mr.load_run(run_dir, checkpoint="final")
    predict_fn = mr.build_predict_gxi_batched(loaded)

    pred_eta, truth_eta, h, time = load_failing_state(run_dir, "tanaka_g0", args.case_id, args.frame)
    eta = truth_eta if args.use_truth_eta else pred_eta

    x, k_full = ds.build_grid(eta.shape[-1], float(loaded.config["domain_length"]))
    x = np.asarray(x)
    log_depth = float(np.log(h))
    k_modes = np.arange(args.k_lo, args.k_hi)

    def model_op(xi: np.ndarray) -> np.ndarray:
        return np.asarray(predict_fn(
            jnp.asarray(eta, dtype=jnp.float32)[None, :],
            jnp.asarray(xi, dtype=jnp.float32)[None, :],
            jnp.full((1,), log_depth, dtype=jnp.float32),
        )[0, :]).astype(np.float64)

    truth_jit = jax.jit(lambda xi: ds.dno_series_eval(
        jnp.asarray(eta, dtype=jnp.float64),
        jnp.asarray(xi, dtype=jnp.float64),
        k_full,
        h,
        order=6,
        pad_factor=8,
    ))

    def truth_op(xi: np.ndarray) -> np.ndarray:
        return np.asarray(truth_jit(xi))

    suffix = "truth_eta" if args.use_truth_eta else "pred_eta"
    if args.out_dir is None:
        out_dir = run_dir / f"svd_highband_{suffix}_case{args.case_id}_frame{args.frame}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing real-Fourier SVD for high-high block k={args.k_lo}..{args.k_hi-1}, case {args.case_id}, {suffix}, t={time:.2f} ...")

    L_model, _, _ = build_real_operator_matrix(model_op, x, k_modes)
    L_truth, _, _ = build_real_operator_matrix(truth_op, x, k_modes)

    s_model = np.linalg.svd(L_model, compute_uv=False)
    s_truth = np.linalg.svd(L_truth, compute_uv=False)

    summary = {
        "case_id": args.case_id,
        "frame": args.frame,
        "time": time,
        "depth": h,
        "eta_source": suffix,
        "k_lo": args.k_lo,
        "k_hi": args.k_hi,
        "matrix_shape": list(L_model.shape),
        "max_sv_model": float(s_model[0]),
        "max_sv_truth": float(s_truth[0]),
        "max_sv_ratio": float(s_model[0] / (s_truth[0] + 1e-30)),
        "top_sv_model": [float(v) for v in s_model[:20]],
        "top_sv_truth": [float(v) for v in s_truth[:20]],
    }

    # Per-mode diagonal (G0-like) from truth.
    g0_diag = np.array([k * np.tanh(h * k) for k in k_modes])
    summary["g0_diag"] = [float(v) for v in g0_diag]

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "operators.npz",
        L_model=L_model,
        L_truth=L_truth,
        k_modes=k_modes,
        g0_diag=g0_diag,
    )

    print(f"Saved outputs to {out_dir}")
    print(f"Max singular value: model={s_model[0]:.3f}, truth={s_truth[0]:.3f}, ratio={s_model[0]/(s_truth[0]+1e-30):.3f}")
    print(f"Top 5 model SVs: {s_model[:5]}")
    print(f"Top 5 truth SVs: {s_truth[:5]}")


if __name__ == "__main__":
    main()
