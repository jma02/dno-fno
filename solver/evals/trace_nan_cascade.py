"""Step-by-step diagnostic of a known-NaN tanaka rollout.

Runs the GL2-IF surrogate stepper one substep at a time on a single failing IC,
printing per-substep band-resolved spectra of (eta, xi, gxi, eta_t, xi_t) so we
can see WHERE in the operator chain the cascade ignites.

Usage:
    uv run python -m solver.evals.trace_nan_cascade \\
        --run_dir outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621 \\
        --npz data/tanaka_2_adaptive_g0.npz \\
        --case_id 5 \\
        --max_substeps 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import (
    _gl2_if_step_surrogate,
    _rhs_nonlinear_surrogate,
    build_predict_gxi,
    load_run,
)
from solver.solvers import time_integrator as ti


KBANDS = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 513)]


def band_energy(field_hat: np.ndarray, k_lo: int, k_hi: int) -> float:
    """Sqrt sum |X̂_k|^2 over k in [k_lo, k_hi). Field is rfft output (n//2+1)."""
    band = field_hat[k_lo: min(k_hi, field_hat.shape[-1])]
    return float(np.sqrt(np.sum(np.abs(band) ** 2)))


def field_stats(name: str, field: np.ndarray) -> str:
    max_abs = float(np.max(np.abs(field)))
    has_nan = bool(np.any(np.isnan(field)))
    has_inf = bool(np.any(np.isinf(field)))
    fh = np.fft.rfft(field)
    bands = " ".join(
        f"k[{lo:>3}-{hi:>3}]={band_energy(fh, lo, hi):>8.2e}"
        for lo, hi in KBANDS
    )
    flag = ""
    if has_nan:
        flag = "  [NaN!]"
    elif has_inf:
        flag = "  [Inf!]"
    return f"{name:>6}: max={max_abs:>9.3e}  {bands}{flag}"


def find_first_bad_case(
    npz_path: Path, case_id: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    with np.load(npz_path, mmap_mode="r") as d:
        meta = json.loads(bytes(d["meta.json"]).decode())
        ids = np.asarray(d["case_id_batch_0000"])
        depths = np.asarray(d["depth_batch_0000"])
        eta = np.asarray(d["eta_batch_0000"])
        xi = np.asarray(d["xi_batch_0000"])
    rows = np.nonzero(ids == case_id)[0]
    if rows.size == 0:
        raise SystemExit(f"case_id {case_id} not found in {npz_path}")
    row0 = int(rows[0])
    return (
        eta[row0].astype(np.float64),
        xi[row0].astype(np.float64),
        float(depths[row0]),
        np.asarray([meta["dt"], meta["tmax"]], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--case_id", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=80)
    parser.add_argument("--impl", type=int, default=4)
    parser.add_argument("--filter_fraction", type=float, default=0.25)
    parser.add_argument("--max_substeps", type=int, default=400,
                        help="Total inner substeps to walk before stopping (NaN halts earlier).")
    parser.add_argument("--every", type=int, default=5,
                        help="Print every Nth substep (always print at NaN / Inf / first 5).")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    args = parser.parse_args()

    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    eta0, xi0, depth, meta_dtmax = find_first_bad_case(Path(args.npz), args.case_id)
    nx = int(eta0.shape[-1])
    outer_dt = float(meta_dtmax[0])
    inner_dt = outer_dt / float(args.substeps)
    print(f"# run_dir   : {args.run_dir}")
    print(f"# epoch     : {loaded.epoch}  (cs_tie_xi_out_mult={loaded.config.get('cs_tie_xi_out_mult')}, "
          f"cs_mult_hidden={loaded.config.get('cs_mult_hidden')})")
    print(f"# npz       : {args.npz}  case_id={args.case_id}  depth={depth:.4f}")
    print(f"# grid      : nx={nx}  outer_dt={outer_dt:.4f}  inner_dt={inner_dt:.4f}  "
          f"substeps={args.substeps}  impl={args.impl}  filter_fraction={args.filter_fraction}")

    params = ti.make_solver_params(
        nx=nx, length=2.0 * np.pi, depth=depth, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=args.filter_fraction,
    )
    params = ti.cast_solver_params_dtype(params, jnp.float64)
    predict = build_predict_gxi(loaded, depth)

    state = ti.State(
        eta=jnp.asarray(eta0, dtype=jnp.float64),
        xi=jnp.asarray(xi0 - xi0.mean(), dtype=jnp.float64),
    )
    print(f"# IC stats:")
    print("  " + field_stats("eta0", np.asarray(state.eta)))
    print("  " + field_stats("xi0", np.asarray(state.xi)))
    g0_xi0 = ti.linear_dno_action(state.xi, params.g0)
    print("  " + field_stats("g0xi0", np.asarray(g0_xi0)))
    print()

    t = 0.0
    for step in range(args.max_substeps):
        # Inspect the RHS at the CURRENT state before stepping — gives the predict_gxi
        # spectrum the integrator is about to feed into the substep.
        rhs = _rhs_nonlinear_surrogate(state, params, predict)
        gxi = predict(state.eta, state.xi)
        # Pieces of xi_t to disentangle "model output explodes" vs "nonlinear xi_t formula explodes".
        eta_x = ti.spectral_dx(state.eta, params.k)
        xi_x = ti.spectral_dx(state.xi, params.k)
        numerator = gxi + eta_x * xi_x
        denom = 1.0 + eta_x ** 2

        prefix = f"[s={step:>4} t={t:>6.3f}]"
        eta_nan = bool(jnp.any(jnp.isnan(state.eta)) or jnp.any(jnp.isinf(state.eta)))
        is_print = (step < 5) or (step % args.every == 0) or eta_nan

        if is_print:
            print(prefix)
            print("  " + field_stats("eta", np.asarray(state.eta)))
            print("  " + field_stats("xi", np.asarray(state.xi)))
            print("  " + field_stats("gxi", np.asarray(gxi)))
            print("  " + field_stats("eta_t", np.asarray(rhs.eta)))
            print("  " + field_stats("xi_t", np.asarray(rhs.xi)))
            print("  " + field_stats("etax_xix", np.asarray(eta_x * xi_x)))
            print("  " + field_stats("num=gxi+etax_xix", np.asarray(numerator)))
            print("  " + field_stats("1+etax^2", np.asarray(denom)))
            print()

        if eta_nan:
            print(f"!! halt: η/ξ NaN at substep {step} (t={t:.4f})")
            break

        state = _gl2_if_step_surrogate(
            state, t, inner_dt, params, predict, iterations=args.impl,
        )
        t += inner_dt
    else:
        print(f"# completed {args.max_substeps} substeps without NaN (t={t:.4f})")


if __name__ == "__main__":
    main()
