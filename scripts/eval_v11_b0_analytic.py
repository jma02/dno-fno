"""v11-B0 eval: analytic G_0 + G_1 with 0 learned params on tanaka_g0.

v11-B0 = CraigSulemDNO(n_blocks=0, use_g1_baseline=True, g1_k_cut=64). No block
loop, no learned residual — the model returns the closed-form Craig-Sulem series
truncated at order 1 (G_0 + G_1). Uses v10's dataset scales so xi/eta/target
normalizations match; loads NO checkpoint (params tree is empty).

Runs the standard f64_harness batched surrogate on tanaka_g0 at n_ics=32 via the
same eval_suite pipeline used by every other model, so numbers are directly
comparable to v10 baseline (0.01232 med, 4/32 NaN) and C1 (0.01506 med, 3/32).
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.evals.eval_suite import REGISTRY, RegimeConfig, run_regime   # noqa: E402
from solver.evals.model_rollout import (   # noqa: E402
    LoadedRun, build_predict_gxi_batched, build_predict_gxi_with_depth,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "dno-net"))
from dno_net_v2 import CraigSulemDNO   # noqa: E402


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    v10_run_dir = Path("/home/johnma/dno-fno/outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745")
    v10_config = json.loads((v10_run_dir / "config.json").read_text())
    v10_meta = json.loads((v10_run_dir / "best_val_ckpt" / "metadata.json").read_text())
    stats = v10_meta["stats"]

    xi_scale = float(v10_config.get("xi_scale", np.asarray(stats["feature_absmax"]).reshape(-1)[1]))
    eta_scale = float(v10_config.get("eta_scale", np.asarray(stats["feature_absmax"]).reshape(-1)[0]))
    target_scale = float(v10_config.get("target_scale", stats["target_absmax"]))
    domain_length = float(v10_config.get("domain_length", stats.get("domain_length", 2.0 * np.pi)))

    print(f"v11-B0: n_blocks=0, use_g1_baseline=True, g1_k_cut=64")
    print(f"  xi_scale={xi_scale:.4f}  eta_scale={eta_scale:.4f}  target_scale={target_scale:.4f}")
    print(f"  domain_length={domain_length:.6f}")

    model = CraigSulemDNO(
        modes=64, width=32, n_blocks=0, latent=8,
        n_polys=1, use_first_deriv=False, use_second_deriv=False,
        use_half_deriv=False, use_hilbert=False,
        use_g0_eta=False, use_g0_eta_dx=False,
        mult_hidden=8,
        use_g1_baseline=True, g1_k_cut=64,
        fft_fp64=False,
        tie_xi_out_mult=False, phi_bias_free=False,
        domain_length=domain_length,
        xi_scale=xi_scale, eta_scale=eta_scale, target_scale=target_scale,
        h_clip_max=5.0,
    )

    # Init params (empty tree with n_blocks=0). Shapes chosen to match eval-time
    # inputs: (B, N, 2) inputs and (B, 1) depth.
    key = jax.random.PRNGKey(0)
    dummy_inputs = jnp.zeros((1, 1024, 2), dtype=jnp.float32)
    dummy_depth = jnp.zeros((1, 1), dtype=jnp.float32)
    params = model.init(key, dummy_inputs, dummy_depth)["params"]

    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"  trainable params: {n_params}")
    for name, p in params.items():
        print(f"    {name}: shape={p.shape if hasattr(p, 'shape') else p}")

    loaded = LoadedRun(
        model=model, params=params,
        config={**v10_config,
                "n_blocks": 0, "cs_use_g1_baseline": True, "cs_g1_k_cut": 64,
                "cs_n_polys": 1, "cs_use_first_deriv": False,
                "cs_use_second_deriv": False, "cs_use_half_deriv": False,
                "cs_use_hilbert": False, "cs_fft_fp64": False,
                "precision": "fp32", "model": "cs_dno", "norm": "scale"},
        stats=stats, norm_mode="scale", epoch=0,
    )

    predict_per_ic = build_predict_gxi_with_depth(loaded)
    predict_batched = build_predict_gxi_batched(loaded)

    regime = "tanaka_g0"
    base = REGISTRY[regime]
    cfg = RegimeConfig(
        name=base.name, source=base.source, dt=base.dt, tmax=base.tmax,
        n_ics=32, substeps=base.substeps, implicit_iters=base.implicit_iters,
        filter_fraction=base.filter_fraction, truth_kind=base.truth_kind,
    )

    out_dir = Path("/home/johnma/dno-fno/outputs/v11_b0_analytic_20260706_150500")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache = v10_run_dir / "eval_2x2v3_off_20260706_070633"
    print(f"  output_dir: {out_dir}")
    print(f"  truth cache: {baseline_cache}")

    t0 = time.time()
    summary = run_regime(
        regime, cfg, loaded, predict_per_ic, out_dir,
        nx=1024, length=domain_length,
        filter_gxi=False, f64_harness=True,
        filter_shape="hard", houli_a=36.0, houli_m=36.0,
        cascade_gate_enabled=False,
        truth_cache_dir=baseline_cache,
        batched_surrogate=True,
        predict_gxi_batched=predict_batched,
    )
    wall = time.time() - t0

    print()
    print("===== v11-B0 (analytic G_0+G_1, 0 params) — tanaka_g0 n_ics=32 =====")
    print(f"  wall: {wall:.1f}s")
    print()
    baseline_summ = json.loads((baseline_cache / "tanaka_g0_summary.json").read_text())
    c1_summ = json.loads(
        (v10_run_dir.parent / "c1_stage_match_from_v85b_20260704_210952"
         / "eval_2x2v3_off_20260706_070633" / "tanaka_g0_summary.json").read_text()
    )
    print(f"                 | v10 baseline | C1 finetune | v11-B0 (0 params) |")
    print(f"  NaN rate       | {baseline_summ['nan_rate']:.3f}        | {c1_summ['nan_rate']:.3f}      | {summary['nan_rate']:.3f}            |")
    print(f"  divergence     | {baseline_summ['divergence_rate_final']:.3f}        | {c1_summ['divergence_rate_final']:.3f}      | {summary['divergence_rate_final']:.3f}            |")
    print(f"  eta med tfinal | {baseline_summ.get('rel_l2_eta_median_tfinal', float('nan')):.5f}    | {c1_summ.get('rel_l2_eta_median_tfinal', float('nan')):.5f}   | {summary.get('rel_l2_eta_median_tfinal', float('nan')):.5f}          |")
    print(f"  eta p95 tfinal | {baseline_summ.get('rel_l2_eta_p95_tfinal', float('nan')):.5f}    | {c1_summ.get('rel_l2_eta_p95_tfinal', float('nan')):.5f}   | {summary.get('rel_l2_eta_p95_tfinal', float('nan')):.5f}          |")

    trajs_path = out_dir / "tanaka_g0_trajs.npz"
    if trajs_path.exists():
        pe = np.load(trajs_path)["pred_eta"]
        nans_per = np.isnan(pe).reshape(pe.shape[0], pe.shape[1], -1).any(axis=(0, 2))
        nan_ics = np.where(nans_per)[0].tolist()
        print()
        print(f"  v10 baseline NaN ICs: [5, 11, 22, 23]")
        print(f"  v11-B0 NaN ICs:       {nan_ics}")


if __name__ == "__main__":
    main()
