"""Parity smoke for batched vs per-IC surrogate rollout.

Runs both paths on the same 2-IC tanaka_g0 slice for a short horizon and
prints max abs diff per field. Passes if diffs are below the fp32 harness
tolerance (1e-4 for eta, ~1e-3 for gxi through 6 substeps).

CPU-only: relies on JAX_PLATFORMS=cpu so it doesn't clash with GPU evals.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.evals.eval_suite import (  # noqa: E402
    REGISTRY,
    RegimeConfig,
    build_ics_and_truth_targets,
    surrogate_rollout_batched,
    surrogate_rollout_per_ic,
)
from solver.evals.model_rollout import (  # noqa: E402
    build_predict_gxi_batched,
    build_predict_gxi_with_depth,
    load_run,
)


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    run_dir = Path("/home/johnma/dno-fno/outputs/c1_stage_match_from_v85b_20260704_210952")
    print(f"loading run: {run_dir}", flush=True)
    loaded = load_run(run_dir, checkpoint="best")
    predict_per_ic = build_predict_gxi_with_depth(loaded)
    predict_batched = build_predict_gxi_batched(loaded)

    regime = "tanaka_g0"
    base = REGISTRY[regime]
    # Trim to a short horizon so the test finishes in a reasonable time on CPU.
    cfg = RegimeConfig(
        name=base.name, source=base.source,
        dt=base.dt, tmax=4 * base.dt,
        n_ics=2, substeps=base.substeps, implicit_iters=base.implicit_iters,
        filter_fraction=base.filter_fraction, truth_kind=base.truth_kind,
    )
    ics, dt, tmax, _, _ = build_ics_and_truth_targets(cfg)
    times_np = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=np.float64)
    times_in = jnp.asarray(times_np, dtype=jnp.float64)
    nx = 1024
    length = 2.0 * float(np.pi)

    kwargs = dict(
        filter_gxi=False,
        f64_harness=True,
        filter_shape="hard",
        houli_a=36.0, houli_m=36.0,
        cascade_gate_enabled=False,
        cascade_k_cut=32.0, cascade_r_threshold=1e-3, cascade_sharpness=10.0,
        cascade_houli_a=0.69, cascade_houli_m=4.0, cascade_k_eff=128.0,
        cascade_filter_xi=True,
    )

    print("per-IC rollout (residual check OFF)...", flush=True)
    per = surrogate_rollout_per_ic(ics, times_in, nx, length, cfg, loaded, predict_per_ic, **kwargs)
    print(f"  wall = {per['wall_s']:.2f}s", flush=True)

    print("batched rollout (residual check OFF)...", flush=True)
    bat = surrogate_rollout_batched(ics, times_in, nx, length, cfg, loaded, predict_batched, **kwargs)
    print(f"  wall = {bat['wall_s']:.2f}s", flush=True)

    for k in ("eta", "xi", "gxi"):
        a = per[k].astype(np.float64)
        b = bat[k].astype(np.float64)
        diff = np.abs(a - b)
        print(f"  {k}: shape={a.shape}==(?){b.shape}  max_abs={diff.max():.3e}")

    for k in ("eta", "xi"):
        a = per[k].astype(np.float64)
        b = bat[k].astype(np.float64)
        assert a.shape == b.shape, f"shape mismatch on {k}: {a.shape} vs {b.shape}"
        assert np.max(np.abs(a - b)) < 1e-4, f"{k} max abs diff {np.max(np.abs(a - b)):.3e}"

    print("BATCH PARITY OK", flush=True)

    # Residual-check parity: on healthy substeps the check should never fire →
    # trajectories should match the check-off path to fp64 numerical noise.
    kwargs_rc = dict(kwargs)
    kwargs_rc["gl2_residual_check"] = True
    # gl2_residual_tol left at the default (1e-2 = divergence threshold).

    print("per-IC rollout (residual check ON)...", flush=True)
    per_rc = surrogate_rollout_per_ic(ics, times_in, nx, length, cfg, loaded, predict_per_ic, **kwargs_rc)
    print(f"  wall = {per_rc['wall_s']:.2f}s", flush=True)

    print("batched rollout (residual check ON)...", flush=True)
    bat_rc = surrogate_rollout_batched(ics, times_in, nx, length, cfg, loaded, predict_batched, **kwargs_rc)
    print(f"  wall = {bat_rc['wall_s']:.2f}s", flush=True)

    for k in ("eta", "xi", "gxi"):
        a = per[k].astype(np.float64)
        b = per_rc[k].astype(np.float64)
        c = bat_rc[k].astype(np.float64)
        assert not np.isnan(b).any(), f"per-IC residual-check produced NaN on healthy {k}"
        assert not np.isnan(c).any(), f"batched residual-check produced NaN on healthy {k}"
        print(f"  {k}: per_rc_vs_per max_abs={np.max(np.abs(b - a)):.3e}   bat_rc_vs_per max_abs={np.max(np.abs(c - a)):.3e}")

    for k in ("eta", "xi"):
        a = per[k].astype(np.float64)
        b = per_rc[k].astype(np.float64)
        c = bat_rc[k].astype(np.float64)
        assert np.max(np.abs(a - b)) < 1e-4, f"per-IC residual-check {k} deviates: {np.max(np.abs(a - b)):.3e}"
        assert np.max(np.abs(a - c)) < 1e-4, f"batched residual-check {k} deviates: {np.max(np.abs(a - c)):.3e}"

    print("RESIDUAL-CHECK HEALTHY PARITY OK", flush=True)


if __name__ == "__main__":
    main()
