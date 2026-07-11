"""Block-7 ablation eval for v10 on tanaka_g0.

Zeros cs_block_7/phi_proj (kernel + bias) in the v10 params so block 7
contributes exactly zero to the model output, then runs the standard
f64-harness batched surrogate rollout on tanaka_g0 at n_ics=32.

Prediction (from weight-audit subagent 2026-07-06):
- Baseline v10 patch-off tanaka_g0: 4/32 NaN, div=5/32
- Block 7 dominates pre-NaN on IC 22 (2.84× |ξ|) and IC 23 (1.95× |ξ|)
- If block 7 is a runaway amplifier and not load-bearing, IC 22 + IC 23 cure.
  Prediction: NaN drops to 2/32.

Output writes to a temp dir under scratchpad + prints a summary.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.evals.eval_suite import (   # noqa: E402
    REGISTRY, RegimeConfig, build_ics_and_truth_targets, run_regime,
)
from solver.evals.model_rollout import (   # noqa: E402
    load_run, build_predict_gxi_batched, build_predict_gxi_with_depth,
)


def zero_block7(params: dict) -> dict:
    """Return a new params tree with cs_block_7/phi_proj set to zero.

    We zero phi_proj (both kernel and bias) so `phi = 0` everywhere in block 7;
    since the block computes `Σ_i m_out_i · rfft(phi_i * xi_branched_i)`, that
    forces the block output to be identically zero regardless of state or depth.
    Leaving m_xi / m_out intact keeps the params tree shape identical to the
    original ckpt so the rest of the code (loader, apply_fn) is unaffected.
    """
    from copy import deepcopy
    p = deepcopy(params)
    b7 = p["cs_block_7"]
    for k in list(b7["phi_proj"].keys()):
        b7["phi_proj"][k] = jnp.zeros_like(b7["phi_proj"][k])
    return p


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    run_dir = Path("/home/johnma/dno-fno/outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745")
    print(f"loading v10: {run_dir}", flush=True)
    loaded = load_run(run_dir, checkpoint="best")

    # Sanity: block 7 phi_proj is nonzero before ablation.
    b7_pre = loaded.params["cs_block_7"]["phi_proj"]
    pre_l2 = float(jnp.sqrt(jnp.sum(b7_pre["kernel"]**2 + b7_pre["bias"]**2)))
    print(f"  cs_block_7/phi_proj L2 before ablation: {pre_l2:.3e}", flush=True)

    ablated_params = zero_block7(loaded.params)
    b7_post = ablated_params["cs_block_7"]["phi_proj"]
    post_l2 = float(jnp.sqrt(jnp.sum(b7_post["kernel"]**2 + b7_post["bias"]**2)))
    print(f"  cs_block_7/phi_proj L2 after ablation:  {post_l2:.3e}", flush=True)
    assert post_l2 == 0.0, "phi_proj should be exactly zero after ablation"

    # Reconstruct a LoadedRun-shaped object with ablated params.
    from solver.evals.model_rollout import LoadedRun
    ablated = LoadedRun(
        model=loaded.model, params=ablated_params, config=loaded.config,
        stats=loaded.stats, norm_mode=loaded.norm_mode, epoch=loaded.epoch,
    )
    predict_batched = build_predict_gxi_batched(ablated)
    predict_per_ic = build_predict_gxi_with_depth(ablated)

    regime = "tanaka_g0"
    base = REGISTRY[regime]
    cfg = RegimeConfig(
        name=base.name, source=base.source,
        dt=base.dt, tmax=base.tmax,
        n_ics=32, substeps=base.substeps, implicit_iters=base.implicit_iters,
        filter_fraction=base.filter_fraction, truth_kind=base.truth_kind,
    )
    out_dir = run_dir / "ablate_block7_tanaka_g0_20260706_144500"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output_dir: {out_dir}", flush=True)

    length = 2.0 * float(np.pi)
    nx = 1024
    baseline_eval_dir = run_dir / "eval_2x2v3_off_20260706_070633"
    t0 = time.time()
    summary = run_regime(
        regime, cfg, ablated, predict_per_ic, out_dir,
        nx=nx, length=length,
        filter_gxi=False, f64_harness=True,
        filter_shape="hard", houli_a=36.0, houli_m=36.0,
        cascade_gate_enabled=False,
        cascade_k_cut=32.0, cascade_r_threshold=1e-3, cascade_sharpness=10.0,
        cascade_houli_a=0.69, cascade_houli_m=4.0, cascade_k_eff=128.0,
        cascade_filter_xi=True,
        truth_cache_dir=baseline_eval_dir,
        batched_surrogate=True,
        predict_gxi_batched=predict_batched,
        gl2_residual_check=False,
        gl2_residual_tol=1e-2,
    )
    wall = time.time() - t0
    trajs = {}

    # Compare to baseline
    baseline_summ = json.loads(
        (run_dir / "eval_2x2v3_off_20260706_070633/tanaka_g0_summary.json").read_text()
    )
    print()
    print("===== BLOCK 7 ABLATION RESULT (tanaka_g0, n_ics=32) =====")
    print(f"  wall: {wall:.1f}s")
    print()
    print(f"                 | baseline (patch-off) | block7 ablated |")
    print(f"  NaN rate       | {baseline_summ['nan_rate']:.3f}              | {summary['nan_rate']:.3f}          |")
    print(f"  divergence     | {baseline_summ['divergence_rate_final']:.3f}              | {summary['divergence_rate_final']:.3f}          |")
    print(f"  eta med tfinal | {baseline_summ.get('rel_l2_eta_median_tfinal', float('nan'))}  | {summary.get('rel_l2_eta_median_tfinal', float('nan'))}       |")
    print(f"  eta p95 tfinal | {baseline_summ.get('rel_l2_eta_p95_tfinal', float('nan'))}  | {summary.get('rel_l2_eta_p95_tfinal', float('nan'))}       |")

    # Save summary (trajs npz is already written by run_regime into out_dir).
    (out_dir / "tanaka_g0_summary.json").write_text(json.dumps(summary, indent=2))

    # Which ICs NaN'd? Read trajs npz that run_regime saves.
    trajs_path = out_dir / f"tanaka_g0_trajs.npz"
    if trajs_path.exists():
        pe = np.load(trajs_path)["pred_eta"]
        nans_per = np.isnan(pe).reshape(pe.shape[0], pe.shape[1], -1).any(axis=(0, 2))
        nan_ics = np.where(nans_per)[0].tolist()
        print()
        print(f"  Baseline NaN ICs: [5, 11, 22, 23]  (depths: 0.276, 0.234, 0.272, 0.209)")
        print(f"  Ablated  NaN ICs: {nan_ics}")


if __name__ == "__main__":
    main()
