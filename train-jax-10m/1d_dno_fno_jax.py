from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from flax.training import train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from dno_net_v2 import CraigSulemDNO
from losses import build_loss, count_params
from util import (
    FlatParams,
    NormStats,
    build_split_indices,
    device_prefetch,
    evaluate,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    make_normalizers,
    require_jax_devices,
    save_final_representative_plot,
    save_loss_history_plot,
)
from eval_on_dno_dataset import evaluate_run_on_dno_dataset
from solver.solvers.dno_series_jax import build_grid, dno_series_eval
from solver.solvers.time_integrator import apply_filter


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    state: train_state.TrainState,
    train_loss: float,
    val_loss: float,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
    stats: dict[str, object],
    async_manager: checkpoints.AsyncManager,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    host_state = jax.device_get(state)
    to_np = lambda v: np.asarray(v)
    payload = {
        "params": jax.tree_util.tree_map(to_np, host_state.params),
        "opt_state": jax.tree_util.tree_map(to_np, host_state.opt_state),
        "step": int(host_state.step),
    }

    metadata = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stats": stats,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    checkpoints.save_checkpoint(
        ckpt_dir=output_dir,
        target=payload,
        step=epoch,
        prefix="ckpt_",
        keep=1,
        overwrite=True,
        async_manager=async_manager,
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D JAX FNO on the in-memory Tanaka dataset.")
    parser.add_argument("--model", choices=("fno", "spectral_dno", "cs_dno"), default="fno")
    parser.add_argument("--fno_eta_features", action="store_true", default=False,
                        help="Concat cs_dno-style spectral η features (η², η³, ∂η, ∂²η, ½∂η, ℋη) "
                             "to FNO input. Tests whether the feature stack alone (without the "
                             "multilinear Φ_i × M_i structure) recovers cs_dno's win.")
    parser.add_argument("--cs_n_polys", type=int, default=3,
                        help="Highest power of η used in the CS-DNO spatial features. "
                             "n_polys=3 includes η, η², η³.")
    parser.add_argument("--cs_use_first_deriv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cs_use_second_deriv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cs_use_half_deriv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cs_use_hilbert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cs_use_g0_eta", action=argparse.BooleanOptionalAction, default=False,
                        help="JCP09 Tier 2D: add G_0(h)·η to the η-feature stack (depth-aware).")
    parser.add_argument("--cs_use_g0_eta_dx", action=argparse.BooleanOptionalAction, default=False,
                        help="JCP09 Tier 2D: add ∂_x G_0(h)·η to the η-feature stack.")
    parser.add_argument("--cs_mult_hidden", type=int, default=32,
                        help="Hidden size of the depth-aware Fourier-multiplier MLP.")
    parser.add_argument("--cs_use_g1_baseline", action="store_true", default=False,
                        help="Add the closed-form first-order Craig-Sulem term "
                             "G_1(η,h)ξ = -G_0(η G_0 ξ) - ∂_x(η ∂_x ξ) on top of the G_0 "
                             "baseline. Forces the small-amplitude limit to be correct by "
                             "construction; the learned blocks then model G_2+.")
    parser.add_argument("--cs_g1_k_cut", type=int, default=128,
                        help="Zero the analytic G_1 term's output modes at k >= this cutoff. "
                             "The f32 FFT chain's k²-amplified rounding noise exceeds the "
                             "genuine G_1 signal above k≈128 and the H¹ loss weights that "
                             "band by k. 0 disables.")
    parser.add_argument("--cs_fft_fp64", action="store_true", default=False,
                        help="Run every FFT chain in cs_dno (G_0 baseline, ∂x, η spatial features, "
                             "G_1 baseline if on, and each CraigSulemBlock's rfft → mult → irfft "
                             "→ mult → rfft → mult → irfft pipeline) in fp64/complex128. Learned "
                             "multiplier weights, Dense layers, and optimizer state stay fp32 — "
                             "only the spectral path is widened. Cheaper than --precision fp64 "
                             "(no 2× memory blow-up) and targets the noise source directly. "
                             "Enables jax_enable_x64.")
    parser.add_argument("--cs_phi_bias_free", action="store_true", default=False,
                        help="Remove bias terms from the η-feature trunk and per-block φ "
                             "projections so all blocks vanish identically at η=0 and the "
                             "prediction equals the G_0 baseline exactly in the flat-surface "
                             "limit, for all trained weights.")
    parser.add_argument("--cs_tie_xi_out_mult", action="store_true", default=False,
                        help="Inside each CraigSulemBlock, tie M_xi = M_out so the block is "
                             "self-adjoint in ξ ↔ ψ. Matches the true G(η)'s symmetry; "
                             "roughly halves the multiplier-MLP parameter count.")
    parser.add_argument("--norm", choices=("minmax", "scale"), default="minmax")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32",
                        help="Numerical precision for params/activations/optimizer. fp64 enables "
                             "jax_enable_x64 and casts all internal arrays. Roughly 5-10× slower "
                             "on H100 (and an order of magnitude on A100) — only use if you have "
                             "evidence the f32 ceiling is the bottleneck.")
    parser.add_argument("--dataset", default="combined_dataset.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of train/val indices to keep (after the 80/10/10 "
                        "split). 1.0 = full dataset. Useful for quick A/B comparisons.")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--sobolev_k", type=int, default=1)
    parser.add_argument("--filter_gxi_fraction", type=float, default=1.0,
                        help="Train-time low-pass filter on the model's gxi prediction "
                        "(JCP09 Tier 1A). 1.0 disables; 0.25 matches the dataset's "
                        "filter_fraction=0.25 label preprocessing. Applied symmetrically "
                        "to predictions, pushforward gxi rollouts, and Hamiltonian gxi.")
    parser.add_argument("--filter_shape", choices=("hard", "houli"), default="hard",
                        help="Spectral filter shape for --filter_gxi_fraction (JCP09 Tier 1B). "
                        "'houli' = exp(-a*(k/k_eff)^{2m}) smooth roll-off avoids Gibbs at cutoff.")
    parser.add_argument("--houli_a", type=float, default=36.0,
                        help="Hou-Li 'a' parameter (JCP09 default 36).")
    parser.add_argument("--houli_m", type=float, default=36.0,
                        help="Hou-Li 'm' parameter (JCP09 default 36).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--total_epochs", type=int, default=None,
                        help="LR-schedule budget in epochs (decay_steps = total_epochs * "
                             "steps_per_epoch). Defaults to --epochs. Use when resuming a "
                             "partial run to keep the cosine shape matched to the original "
                             "budget — e.g. resume from ep 13 of a 40-epoch run with "
                             "--epochs 27 --total_epochs 40.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--skip_dno_eval", action="store_true")
    parser.add_argument("--dno_eval_dataset", default="test_dno_rescaled.npz")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--resume_from", default=None,
                        help="Path to a checkpoint dir (e.g. outputs/<run>/best_val_ckpt) "
                             "to warm-start from. Restores params, Adam (mu, nu, count) and the "
                             "cosine-schedule step counter so LR resumes mid-schedule. "
                             "The state.step counter (used by hamiltonian warmup) is reset to 0 "
                             "unless --keep_schedule_step is also passed.")
    parser.add_argument("--keep_schedule_step", action="store_true",
                        help="When --resume_from is set, also restore state.step (the counter "
                             "consumed by the hamiltonian warmup ramp). The LR schedule itself "
                             "reads from opt_state[2].count and is always restored.")
    parser.add_argument("--reset_opt_state", action="store_true",
                        help="When --resume_from is set, skip restoring Adam moments and start "
                             "the optimizer fresh (cosine LR also restarts from base). Use only "
                             "if you explicitly want a fresh warmup on top of saved params.")
    parser.add_argument("--input_noise_sigma", type=float, default=0.0,
                        help="Std of Gaussian noise added to the normalized eta input channel "
                             "during training. 0 disables. Targets and the xi channel are NOT "
                             "modified — xi feeds the analytical k*tanh(hk) linear-DNO baseline, "
                             "which amplifies any noise by ~k_max and would dominate the loss.")
    parser.add_argument("--pushforward_steps", type=int, default=0,
                        help="If >0, after the standard prediction take this many Euler steps "
                             "using the surrogate's own prediction for dη/dt and the Zakharov "
                             "RHS for dξ/dt, then evaluate the analytical G(η_k)ξ_k as a second "
                             "training target. 0 disables (baseline behavior).")
    parser.add_argument("--pushforward_weight", type=float, default=1.0,
                        help="Weight of the pushforward-step loss term relative to the standard "
                             "single-step loss. Only used when --pushforward_steps > 0.")
    parser.add_argument("--pushforward_dt", type=float, default=0.01,
                        help="Euler step size for the pushforward state advancement.")
    parser.add_argument("--pushforward_order", type=int, default=4,
                        help="Craig-Sulem series order for the pushforward analytical target. "
                             "Lower than data-generation order (6) for speed; still much higher "
                             "than the linear baseline.")
    parser.add_argument("--pushforward_pad_factor", type=int, default=4,
                        help="FFT pad factor for the pushforward analytical target.")
    parser.add_argument("--pushforward_gravity", type=float, default=1.0,
                        help="Gravity constant used in the Zakharov RHS for ξ advancement.")
    parser.add_argument("--pushforward_h_clip_max", type=float, default=5.0,
                        help="Maximum physical depth used when evaluating the analytical "
                             "pushforward target. Must match the model's h_clip_max.")
    parser.add_argument("--hamiltonian_weight", type=float, default=0.0,
                        help="Weight of the relative Hamiltonian conservation penalty "
                             "|H_1 - H_0| / |H_0| over a single Euler step using the "
                             "surrogate's own gxi prediction. 0 disables.")
    parser.add_argument("--hamiltonian_dt", type=float, default=0.01,
                        help="Euler step size for the Hamiltonian conservation penalty.")
    parser.add_argument("--hamiltonian_warmup_steps", type=int, default=10000,
                        help="Linear ramp of hamiltonian_weight from 0 to its target value "
                             "over this many optimizer steps. Lets the surrogate converge "
                             "before the H-penalty kicks in (it's catastrophic at random init).")
    parser.add_argument("--hamiltonian_clip", type=float, default=10.0,
                        help="Per-sample upper bound on the relative H drift before it enters "
                             "the loss. Floor against H_0 ~ 0 outliers; effectively a no-op "
                             "for trained models (drift is far below this).")
    parser.add_argument("--psd_hinge_weight", type=float, default=0.0,
                        help="Weight of the PSD hinge penalty mean[ReLU(-<xi, G_pred(eta) xi>)]. "
                             "Pushes the surrogate toward a positive-semi-definite operator (the "
                             "true G is PSD, since <xi, G(eta) xi> = 2 * surface kinetic energy >= 0). "
                             "0 disables.")
    parser.add_argument("--psd_hinge_warmup_steps", type=int, default=10000,
                        help="Linear ramp of psd_hinge_weight from 0 to its target value "
                             "over this many optimizer steps. Lets the surrogate converge "
                             "on the data loss before the hinge penalty kicks in.")
    parser.add_argument("--jac_reg_lambda", type=float, default=0.0,
                        help="Weight of band-restricted Jacobian penalty E_v[||P_hi · (∂M/∂η) · P_hi v||²] "
                             "where P_hi is the high-pass projector onto |k| >= jac_reg_kcut and v is a "
                             "random unit vector. Penalizes the model's mid/high-k Lyapunov exponent; "
                             "0 disables.")
    parser.add_argument("--jac_reg_kcut", type=float, default=32.0,
                        help="Band threshold |k| >= jac_reg_kcut for the Jacobian penalty's projector.")
    parser.add_argument("--jac_reg_warmup_steps", type=int, default=500,
                        help="Linear ramp of jac_reg_lambda from 0 to its target value over this many "
                             "optimizer steps. Lets the data-fit loss converge before the penalty kicks in.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

    if args.precision == "fp64" or args.cs_fft_fp64:
        jax.config.update("jax_enable_x64", True)
    compute_dtype = jnp.float64 if args.precision == "fp64" else jnp.float32

    backend, devices = require_jax_devices()
    n_devices = len(devices)
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by device count {n_devices}"
        )
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())
    eval_device = devices[0]

    dataset_path = REPO_ROOT / "data" / args.dataset
    outputs_dir = REPO_ROOT / args.output_root
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("fno_jax_10m_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_arrays(dataset_path)
    nx = int(dataset["x"].shape[0])
    stats = dict(load_or_compute_stats(dataset_path, dataset=dataset))
    stats["target_kind"] = "gxi"
    ns = NormStats.from_dict(stats, mode=args.norm)
    train_indices, val_indices, _ = build_split_indices(
        int(dataset["eta"].shape[0]),
        args.seed,
    )
    if args.data_fraction < 1.0:
        n_train_keep = int(train_indices.shape[0] * args.data_fraction)
        n_val_keep = int(val_indices.shape[0] * args.data_fraction)
        train_indices = train_indices[:n_train_keep]
        val_indices = val_indices[:n_val_keep]
    train_steps_per_epoch = train_indices.shape[0] // args.batch_size

    domain_length = float(stats.get("domain_length", dataset.get("domain_length", 2.0 * np.pi)))
    # FNO1d's linear-baseline path needs to recover physical xi from the normalized
    # input channel. That's only exact under norm=scale, where the channel is divided
    # by feature_absmax. norm=minmax shifts as well, so the baseline is approximate.
    xi_scale = float(np.asarray(ns.feature_absmax).reshape(-1)[1])
    eta_scale = float(np.asarray(ns.feature_absmax).reshape(-1)[0])
    target_scale = float(ns.target_absmax)
    if args.model == "spectral_dno":
        model = SpectralDNO(
            modes=args.modes,
            width=args.width,
            n_blocks=args.n_blocks,
            latent=args.latent,
            domain_length=domain_length,
            xi_scale=xi_scale,
            target_scale=target_scale,
        )
    elif args.model == "cs_dno":
        model = CraigSulemDNO(
            modes=args.modes,
            width=args.width,
            n_blocks=args.n_blocks,
            latent=args.latent,
            n_polys=args.cs_n_polys,
            use_first_deriv=args.cs_use_first_deriv,
            use_second_deriv=args.cs_use_second_deriv,
            use_half_deriv=args.cs_use_half_deriv,
            use_hilbert=args.cs_use_hilbert,
            use_g0_eta=args.cs_use_g0_eta,
            use_g0_eta_dx=args.cs_use_g0_eta_dx,
            mult_hidden=args.cs_mult_hidden,
            use_g1_baseline=args.cs_use_g1_baseline,
            g1_k_cut=args.cs_g1_k_cut,
            fft_fp64=args.cs_fft_fp64,
            tie_xi_out_mult=args.cs_tie_xi_out_mult,
            phi_bias_free=args.cs_phi_bias_free,
            domain_length=domain_length,
            xi_scale=xi_scale,
            eta_scale=eta_scale,
            target_scale=target_scale,
        )
    else:
        model = FNO1d(
            modes=args.modes,
            width=args.width,
            n_blocks=args.n_blocks,
            domain_length=domain_length,
            xi_scale=xi_scale,
            target_scale=target_scale,
            eta_features=args.fno_eta_features,
        )
    with jax.default_device(eval_device):
        params = model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros((1, nx, 2), dtype=compute_dtype),
            jnp.zeros((1, 1), dtype=compute_dtype),
        )["params"]
    # Flax modules default to fp32 param_dtype regardless of input dtype, so a
    # naive init produces fp32 weights even when --precision fp64. Cast so the
    # actual matmuls/FFTs run fp64 instead of upcasting fp64 inputs onto fp32
    # kernels.
    if args.precision == "fp64":
        params = jax.tree_util.tree_map(lambda x: x.astype(compute_dtype), params)
    loss_fn = build_loss(sobolev_k=args.sobolev_k)

    schedule_epochs = args.total_epochs if args.total_epochs is not None else args.epochs
    total_steps = schedule_epochs * train_steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=total_steps,
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    training_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from).resolve()
        if not (resume_dir / "metadata.json").exists():
            raise FileNotFoundError(f"no metadata.json under {resume_dir}")
        # Restore against a target template built from the freshly-initialized state.
        # Without target=, orbax PyTreeCheckpointer cannot reconstruct optax NamedTuples
        # (ScaleByAdamState / ScaleByScheduleState / EmptyState); they would come back as
        # plain dicts and tree_structure(loaded_opt) != tree_structure(state.opt_state),
        # forcing --reset_opt_state and destroying Adam moments + the cosine schedule step.
        template = {
            "params": training_state.params,
            "opt_state": training_state.opt_state,
            "step": int(training_state.step),
        }
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=resume_dir, target=template, prefix="ckpt_",
            orbax_checkpointer=ocp.PyTreeCheckpointer(),
        )
        loaded_params = restored["params"]
        replace_kwargs: dict[str, object] = {"params": loaded_params}
        if not args.reset_opt_state:
            replace_kwargs["opt_state"] = restored["opt_state"]
            if args.keep_schedule_step:
                replace_kwargs["step"] = jnp.asarray(int(restored["step"]), dtype=jnp.int32)
            sched_step = int(restored["opt_state"][-1].count)
            print(f"resumed params + opt_state from {resume_dir} "
                  f"(LR schedule step={sched_step}; "
                  + (f"state.step={int(restored['step'])})" if args.keep_schedule_step
                     else "state.step reset to 0)"))
        else:
            print(f"resumed params only from {resume_dir} (--reset_opt_state: Adam state will rewarm)")
        training_state = training_state.replace(**replace_kwargs)

    training_state = jax.device_put(training_state, replicated)
    checkpoint_async_manager = checkpoints.AsyncManager(max_workers=1)

    config_payload: dict[str, object] = {
        "model": args.model,
        "norm": args.norm,
        "precision": args.precision,
        "dataset": args.dataset,
        "device": backend,
        "device_count": n_devices,
        "modes": args.modes,
        "width": args.width,
        "n_blocks": args.n_blocks,
        "sobolev_k": args.sobolev_k,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "total_epochs": schedule_epochs,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "param_count": count_params(params),
        "train_examples": int(train_indices.shape[0]),
        "val_examples": int(val_indices.shape[0]),
        "data_fraction": float(args.data_fraction),
    }
    config_payload["domain_length"] = domain_length
    config_payload["xi_scale"] = xi_scale
    config_payload["eta_scale"] = eta_scale
    config_payload["target_scale"] = target_scale
    config_payload["input_noise_sigma"] = float(args.input_noise_sigma)
    config_payload["pushforward_steps"] = int(args.pushforward_steps)
    config_payload["pushforward_weight"] = float(args.pushforward_weight)
    config_payload["pushforward_dt"] = float(args.pushforward_dt)
    config_payload["pushforward_order"] = int(args.pushforward_order)
    config_payload["pushforward_pad_factor"] = int(args.pushforward_pad_factor)
    config_payload["pushforward_gravity"] = float(args.pushforward_gravity)
    config_payload["pushforward_h_clip_max"] = float(args.pushforward_h_clip_max)
    config_payload["hamiltonian_weight"] = float(args.hamiltonian_weight)
    config_payload["hamiltonian_dt"] = float(args.hamiltonian_dt)
    config_payload["hamiltonian_warmup_steps"] = int(args.hamiltonian_warmup_steps)
    config_payload["hamiltonian_clip"] = float(args.hamiltonian_clip)
    config_payload["psd_hinge_weight"] = float(args.psd_hinge_weight)
    config_payload["psd_hinge_warmup_steps"] = int(args.psd_hinge_warmup_steps)
    config_payload["jac_reg_lambda"] = float(args.jac_reg_lambda)
    config_payload["jac_reg_kcut"] = float(args.jac_reg_kcut)
    config_payload["jac_reg_warmup_steps"] = int(args.jac_reg_warmup_steps)
    config_payload["filter_gxi_fraction"] = float(args.filter_gxi_fraction)
    config_payload["filter_shape"] = str(args.filter_shape)
    config_payload["houli_a"] = float(args.houli_a)
    config_payload["houli_m"] = float(args.houli_m)
    if args.model == "fno":
        config_payload["fno_eta_features"] = bool(args.fno_eta_features)
    if args.model == "spectral_dno":
        config_payload["latent"] = args.latent
    if args.model == "cs_dno":
        config_payload["latent"] = args.latent
        config_payload["cs_n_polys"] = args.cs_n_polys
        config_payload["cs_use_first_deriv"] = bool(args.cs_use_first_deriv)
        config_payload["cs_use_second_deriv"] = bool(args.cs_use_second_deriv)
        config_payload["cs_use_half_deriv"] = bool(args.cs_use_half_deriv)
        config_payload["cs_use_hilbert"] = bool(args.cs_use_hilbert)
        config_payload["cs_use_g0_eta"] = bool(args.cs_use_g0_eta)
        config_payload["cs_use_g0_eta_dx"] = bool(args.cs_use_g0_eta_dx)
        config_payload["cs_mult_hidden"] = args.cs_mult_hidden
        config_payload["cs_use_g1_baseline"] = bool(args.cs_use_g1_baseline)
        config_payload["cs_g1_k_cut"] = int(args.cs_g1_k_cut)
        config_payload["cs_fft_fp64"] = bool(args.cs_fft_fp64)
        config_payload["cs_tie_xi_out_mult"] = bool(args.cs_tie_xi_out_mult)
        config_payload["cs_phi_bias_free"] = bool(args.cs_phi_bias_free)
    if args.resume_from is not None:
        config_payload["resume_from"] = str(Path(args.resume_from).resolve())
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    norm_inputs_jax, norm_targets_jax, denorm_targets_jax = make_normalizers(ns)

    input_noise_sigma = float(args.input_noise_sigma)
    pushforward_steps = int(args.pushforward_steps)
    pushforward_weight = float(args.pushforward_weight)
    pushforward_dt = float(args.pushforward_dt)
    pushforward_order = int(args.pushforward_order)
    pushforward_pad_factor = int(args.pushforward_pad_factor)
    pushforward_gravity = float(args.pushforward_gravity)
    pushforward_h_clip_max = float(args.pushforward_h_clip_max)
    pushforward_log_h_max = float(np.log(pushforward_h_clip_max))
    hamiltonian_weight = float(args.hamiltonian_weight)
    hamiltonian_dt = float(args.hamiltonian_dt)
    hamiltonian_warmup_steps = int(args.hamiltonian_warmup_steps)
    hamiltonian_clip = float(args.hamiltonian_clip)
    psd_hinge_weight = float(args.psd_hinge_weight)
    psd_hinge_warmup_steps = int(args.psd_hinge_warmup_steps)
    jac_reg_lambda = float(args.jac_reg_lambda)
    jac_reg_kcut = float(args.jac_reg_kcut)
    jac_reg_warmup_steps = int(args.jac_reg_warmup_steps)
    filter_gxi_fraction = float(args.filter_gxi_fraction)
    filter_shape = str(args.filter_shape)
    houli_a = float(args.houli_a)
    houli_m = float(args.houli_m)
    dx_phys = float(domain_length / nx)

    # Wavenumbers k_n for the Zakharov RHS and the analytical pushforward target.
    _x_grid, _k_grid = build_grid(nx, domain_length)
    k_grid_jax = jnp.asarray(_k_grid, dtype=compute_dtype)

    def _filter_predictions(predictions: jax.Array) -> jax.Array:
        # Lowpass commutes with the linear target scaling, so filtering in normalized
        # space is exact. Predictions are (B, nx, 1); apply_filter acts on the last axis.
        if filter_gxi_fraction >= 1.0:
            return predictions
        filtered = apply_filter(
            predictions[..., 0], k_grid_jax,
            shape=filter_shape, filter_fraction=filter_gxi_fraction,
            houli_a=houli_a, houli_m=houli_m,
        )
        return filtered[..., None]

    def _zakharov_step(
        eta_phys: jax.Array,
        xi_phys: jax.Array,
        gxi_pred_phys: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """One Euler step of the Zakharov system using the surrogate's gxi prediction."""
        k = k_grid_jax  # (nx,)
        eta_hat = jnp.fft.fft(eta_phys, axis=-1)
        xi_hat = jnp.fft.fft(xi_phys, axis=-1)
        eta_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * eta_hat, axis=-1))
        xi_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * xi_hat, axis=-1))
        eta_t = gxi_pred_phys
        numerator = gxi_pred_phys + eta_x * xi_x
        xi_t = -pushforward_gravity * eta_phys - 0.5 * xi_x ** 2 + 0.5 * numerator ** 2 / (1.0 + eta_x ** 2)
        eta_next = eta_phys + pushforward_dt * eta_t
        xi_next = xi_phys + pushforward_dt * xi_t
        return eta_next, xi_next

    def _analytical_gxi(eta_phys: jax.Array, xi_phys: jax.Array, h: jax.Array) -> jax.Array:
        """Per-sample analytical Craig-Sulem G(eta) xi target. Maps (B, nx), (B,) -> (B, nx)."""
        def one_sample(eta_i: jax.Array, xi_i: jax.Array, h_i: jax.Array) -> jax.Array:
            return dno_series_eval(eta_i, xi_i, k_grid_jax, h_i,
                                   pushforward_order, pad_factor=pushforward_pad_factor)
        return jax.vmap(one_sample, in_axes=(0, 0, 0))(eta_phys, xi_phys, h)

    def _train_step_body(
        current_state: train_state.TrainState,
        rng_key: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ):
        eta = eta.astype(compute_dtype)
        xi = xi.astype(compute_dtype)
        gxi = gxi.astype(compute_dtype)
        batch_depth = batch_depth.astype(compute_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        batch_targets = _filter_predictions(norm_targets_jax(gxi))
        if input_noise_sigma > 0.0:
            # Noise on eta only (channel 0). xi (channel 1) feeds the analytical linear-DNO
            # baseline, which multiplies by k*tanh(hk) and amplifies white noise by ~k_max.
            local_key = jax.random.fold_in(rng_key, jax.lax.axis_index("batch"))
            eta_shape = (batch_inputs.shape[0], batch_inputs.shape[1])
            eta_noise = jax.random.normal(local_key, eta_shape, dtype=batch_inputs.dtype)
            noise_field = jnp.stack(
                (input_noise_sigma * eta_noise, jnp.zeros_like(eta_noise)),
                axis=-1,
            )
            batch_inputs = batch_inputs + noise_field

        # Physical depth per sample (B,), clipped exactly the way the model clips it
        # so that pushforward targets match the operator the surrogate is approximating.
        log_h = jnp.minimum(batch_depth[:, 0], pushforward_log_h_max)
        h_phys = jnp.exp(log_h)

        def _hamiltonian(eta_phys, xi_phys, gxi_phys):
            # H(eta, xi) = 0.5 * sum_x (xi * G(eta) xi + g * eta^2) * dx, per sample
            return 0.5 * jnp.sum(
                xi_phys * gxi_phys + pushforward_gravity * eta_phys ** 2,
                axis=-1,
            ) * dx_phys

        def loss_for_params(current_params):
            predictions_0 = current_state.apply_fn(
                {"params": current_params}, batch_inputs, batch_depth
            )
            predictions_0 = _filter_predictions(predictions_0)
            loss_0 = loss_fn(predictions_0, batch_targets)
            extra = jnp.asarray(0.0, dtype=compute_dtype)

            if psd_hinge_weight > 0.0:
                # PSD hinge: <xi, G_pred(eta) xi> >= 0 (proportional to 2 * surface KE).
                # Penalize the negative part with ramp warmup. Only tests PSD in the
                # training-xi direction — cheap but tight enough to drive the gradient
                # when the surrogate produces a negative-eigenvalue mode aligned with xi.
                gxi_pred_0_phys_psd = denorm_targets_jax(predictions_0)[..., 0]
                ke_per_sample = jnp.sum(xi * gxi_pred_0_phys_psd, axis=-1) * dx_phys
                psd_per_sample = jax.nn.relu(-ke_per_sample)
                step_f = jnp.asarray(current_state.step, dtype=compute_dtype)
                psd_warmup = jnp.minimum(
                    step_f / jnp.asarray(max(psd_hinge_warmup_steps, 1), dtype=compute_dtype), 1.0,
                )
                loss_psd = jnp.mean(psd_per_sample)
                extra = extra + psd_hinge_weight * psd_warmup * loss_psd

            if hamiltonian_weight > 0.0:
                # H_0 uses surrogate's own gxi at (eta, xi); advance one Euler step;
                # H_1 uses surrogate's gxi at the advanced state.
                gxi_pred_0_phys = denorm_targets_jax(predictions_0)[..., 0]
                H_0 = _hamiltonian(eta, xi, gxi_pred_0_phys)
                # Use a dedicated hamiltonian_dt so it can differ from pushforward_dt.
                k = k_grid_jax
                eta_hat = jnp.fft.fft(eta, axis=-1)
                xi_hat = jnp.fft.fft(xi, axis=-1)
                eta_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * eta_hat, axis=-1))
                xi_x = jnp.real(jnp.fft.ifft(1j * k[None, :] * xi_hat, axis=-1))
                numerator = gxi_pred_0_phys + eta_x * xi_x
                xi_t = -pushforward_gravity * eta - 0.5 * xi_x ** 2 + 0.5 * numerator ** 2 / (1.0 + eta_x ** 2)
                eta_1 = jax.lax.stop_gradient(eta + hamiltonian_dt * gxi_pred_0_phys)
                xi_1 = jax.lax.stop_gradient(xi + hamiltonian_dt * xi_t)
                inputs_1 = norm_inputs_jax(eta_1, xi_1)
                predictions_1 = current_state.apply_fn(
                    {"params": current_params}, inputs_1, batch_depth
                )
                predictions_1 = _filter_predictions(predictions_1)
                gxi_pred_1_phys = denorm_targets_jax(predictions_1)[..., 0]
                H_1 = _hamiltonian(eta_1, xi_1, gxi_pred_1_phys)
                # Relative absolute drift, clipped per-sample so untrained-surrogate outliers
                # can't blow up the gradient. Linear warmup keeps the loss inert while the
                # standard term is still moving the model out of random init.
                per_sample = jnp.abs(H_1 - H_0) / (jnp.abs(H_0) + 1e-6)
                per_sample = jnp.minimum(per_sample, jnp.asarray(hamiltonian_clip, dtype=compute_dtype))
                step_f = jnp.asarray(current_state.step, dtype=compute_dtype)
                warmup = jnp.minimum(step_f / jnp.asarray(max(hamiltonian_warmup_steps, 1), dtype=compute_dtype), 1.0)
                loss_h = jnp.mean(per_sample)
                extra = extra + hamiltonian_weight * warmup * loss_h

            if jac_reg_lambda > 0.0:
                # Band-restricted Hutchinson estimator of ||P_hi · (∂M/∂η) · P_hi||_F^2.
                # P_hi = projector onto |k| >= jac_reg_kcut. Penalizes the model's
                # mid/high-k Lyapunov exponent (the cascade driver per JCP09 §3.3).
                local_key = jax.random.fold_in(rng_key, jax.lax.axis_index("batch") * 7 + 13)
                eta_norm = batch_inputs[..., 0]
                xi_norm = batch_inputs[..., 1]
                v = jax.random.normal(local_key, eta_norm.shape, dtype=eta_norm.dtype)
                v_hat = jnp.fft.fft(v, axis=-1)
                k_abs = jnp.abs(k_grid_jax)[None, :]
                hi_mask = (k_abs >= jnp.asarray(jac_reg_kcut, dtype=k_abs.dtype)).astype(v_hat.dtype)
                v_hi = jnp.real(jnp.fft.ifft(v_hat * hi_mask, axis=-1))
                # Normalize per-sample so the penalty is dimensionless in v.
                v_hi = v_hi / (jnp.linalg.norm(v_hi, axis=-1, keepdims=True) + 1e-12)

                def _model_of_eta(eta_field: jnp.ndarray) -> jnp.ndarray:
                    inputs_v = jnp.stack((eta_field, xi_norm), axis=-1)
                    preds = current_state.apply_fn({"params": current_params}, inputs_v, batch_depth)
                    return _filter_predictions(preds)[..., 0]

                _, jvp_out = jax.jvp(_model_of_eta, (eta_norm,), (v_hi,))
                jvp_hat = jnp.fft.fft(jvp_out, axis=-1)
                jvp_hi = jnp.real(jnp.fft.ifft(jvp_hat * hi_mask, axis=-1))
                loss_jac = jnp.mean(jnp.sum(jvp_hi ** 2, axis=-1))
                step_f = jnp.asarray(current_state.step, dtype=compute_dtype)
                jac_warmup = jnp.minimum(
                    step_f / jnp.asarray(max(jac_reg_warmup_steps, 1), dtype=compute_dtype), 1.0,
                )
                extra = extra + jac_reg_lambda * jac_warmup * loss_jac

            if pushforward_steps == 0:
                return loss_0 + extra

            eta_k = eta
            xi_k = xi
            predictions_k = predictions_0
            loss_pf = jnp.asarray(0.0, dtype=compute_dtype)
            for _ in range(pushforward_steps):
                # Advance state with the surrogate's gxi; detach so the pushforward loss
                # only optimizes the (eta_k, xi_k) -> G prediction, not the trajectory itself.
                gxi_pred_phys = denorm_targets_jax(predictions_k)[..., 0]
                eta_k, xi_k = _zakharov_step(eta_k, xi_k, gxi_pred_phys)
                eta_k = jax.lax.stop_gradient(eta_k)
                xi_k = jax.lax.stop_gradient(xi_k)

                gxi_target_k = jax.lax.stop_gradient(_analytical_gxi(eta_k, xi_k, h_phys))
                inputs_k = norm_inputs_jax(eta_k, xi_k)
                targets_k = _filter_predictions(norm_targets_jax(gxi_target_k))
                predictions_k = current_state.apply_fn(
                    {"params": current_params}, inputs_k, batch_depth
                )
                predictions_k = _filter_predictions(predictions_k)
                loss_pf = loss_pf + loss_fn(predictions_k, targets_k)
            return loss_0 + pushforward_weight * loss_pf + extra

        loss_value, grads = jax.value_and_grad(loss_for_params)(current_state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_value = jax.lax.pmean(loss_value, axis_name="batch")
        next_state = current_state.apply_gradients(grads=grads)
        next_state = next_state.replace(
            params=jax.tree.map(
                lambda x: jax.lax.all_gather(x, axis_name="batch", tiled=False)[0],
                next_state.params,
            )
        )
        return next_state, loss_value

    train_step = jax.jit(shard_map(
        _train_step_body,
        mesh=mesh,
        in_specs=(P(), P(), P("batch"), P("batch"), P("batch"), P("batch")),
        out_specs=(P(), P()),
        check_rep=False,
    ))

    @jax.jit
    def eval_step(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ):
        eta = eta.astype(compute_dtype)
        xi = xi.astype(compute_dtype)
        gxi = gxi.astype(compute_dtype)
        batch_depth = batch_depth.astype(compute_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        batch_targets = _filter_predictions(norm_targets_jax(gxi))
        predictions = _filter_predictions(model.apply({"params": current_params}, batch_inputs, batch_depth))
        loss_value = loss_fn(predictions, batch_targets)
        # Return predictions denormalized so callers can compute raw-space metrics directly.
        return loss_value, denorm_targets_jax(predictions)

    @jax.jit
    def predict_batch(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        batch_depth: jax.Array,
    ) -> jax.Array:
        eta = eta.astype(compute_dtype)
        xi = xi.astype(compute_dtype)
        batch_depth = batch_depth.astype(compute_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        predictions = model.apply({"params": current_params}, batch_inputs, batch_depth)
        return denorm_targets_jax(predictions)

    def _eval_loss_body(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ):
        eta = eta.astype(compute_dtype)
        xi = xi.astype(compute_dtype)
        gxi = gxi.astype(compute_dtype)
        batch_depth = batch_depth.astype(compute_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        batch_targets = _filter_predictions(norm_targets_jax(gxi))
        predictions = _filter_predictions(model.apply({"params": current_params}, batch_inputs, batch_depth))
        loss_value = loss_fn(predictions, batch_targets)
        return jax.lax.pmean(loss_value, axis_name="batch")

    eval_loss_step = jax.jit(shard_map(
        _eval_loss_body,
        mesh=mesh,
        in_specs=(P(), P("batch"), P("batch"), P("batch"), P("batch")),
        out_specs=P(),
        check_rep=False,
    ))

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    train_loss = float("inf")
    start_epoch = 1

    # Auto-resume from per-epoch checkpoint when present (so Modal preemption restarts
    # don't lose progress). The current run_dir's latest_ckpt takes priority over
    # --resume_from, because preempt-restart re-invokes with the same CLI (--resume_from
    # still pointing at the original warm-start ckpt) and we want to pick up where we
    # left off, not rewind to the warm-start ckpt.
    latest_ckpt_dir = run_dir / "latest_ckpt"
    if (latest_ckpt_dir / "metadata.json").exists():
        template = {
            "params": training_state.params,
            "opt_state": training_state.opt_state,
            "step": int(training_state.step),
        }
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=latest_ckpt_dir, target=template, prefix="ckpt_",
            orbax_checkpointer=ocp.PyTreeCheckpointer(),
        )
        training_state = training_state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"],
            step=jnp.asarray(int(restored["step"]), dtype=jnp.int32),
        )
        training_state = jax.device_put(training_state, replicated)
        with open(latest_ckpt_dir / "metadata.json", "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        history = list(meta.get("history", []))
        best_val_loss = float(meta.get("best_val_loss", float("inf")))
        best_epoch = int(meta.get("best_epoch", 0))
        start_epoch = int(meta["epoch"]) + 1
        print(f"auto-resume: restored from {latest_ckpt_dir} at epoch {meta['epoch']}, "
              f"resuming at epoch {start_epoch}")

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)
    train_rng = jax.random.PRNGKey(args.seed + 1)

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_iter = get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"],
            train_indices, args.batch_size, epoch_rng,
            shuffle=True,
            drop_last=True,
        )
        train_destinations = (data_sharding, data_sharding, data_sharding, data_sharding, None)
        batch_iter = device_prefetch(batch_iter, destinations=train_destinations, depth=2)
        batch_losses: list[float] = []
        train_bar = tqdm(
            batch_iter,
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for eta_b, xi_b, gxi_b, depth_b, _ in train_bar:
            train_rng, step_key = jax.random.split(train_rng)
            training_state, batch_loss = train_step(
                training_state, step_key, eta_b, xi_b, gxi_b, depth_b,
            )
            batch_loss_value = float(jax.device_get(batch_loss))
            batch_losses.append(batch_loss_value)
            train_bar.set_postfix(loss=batch_loss_value)

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")

        val_batches = get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"],
            val_indices, args.batch_size, None,
            shuffle=False,
            drop_last=True,
        )
        val_destinations = (data_sharding, data_sharding, data_sharding, data_sharding, None)
        val_batches = device_prefetch(val_batches, destinations=val_destinations, depth=2)
        val_losses: list[float] = []
        for eta_b, xi_b, gxi_b, depth_b, _ in val_batches:
            bl = eval_loss_step(training_state.params, eta_b, xi_b, gxi_b, depth_b)
            val_losses.append(float(jax.device_get(bl)))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")

        if not args.skip_plots and epoch % 10 == 0:
            save_loss_history_plot(run_dir, history)

        # Per-epoch checkpoint for preemption-safe resume. Overwrites prior latest_ckpt.
        save_checkpoint(
            run_dir / "latest_ckpt",
            epoch=epoch,
            state=training_state,
            train_loss=train_loss,
            val_loss=val_loss,
            history=history,
            best_val_loss=best_val_loss if val_loss >= best_val_loss else val_loss,
            best_epoch=best_epoch if val_loss >= best_val_loss else epoch,
            stats=stats,
            async_manager=checkpoint_async_manager,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best_val_ckpt",
                epoch=epoch,
                state=training_state,
                train_loss=train_loss,
                val_loss=val_loss,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                stats=stats,
                async_manager=checkpoint_async_manager,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            break

    save_checkpoint(
        run_dir / "final_ckpt",
        epoch=history[-1]["epoch"],
        state=training_state,
        train_loss=train_loss,
        val_loss=history[-1]["val_loss"],
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        stats=stats,
        async_manager=checkpoint_async_manager,
    )
    checkpoint_async_manager.wait_previous_save()

    if not args.skip_plots:
        save_loss_history_plot(run_dir, history)
        eval_params = jax.device_put(training_state.params, eval_device)
        final_batches = get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"],
            val_indices, args.batch_size, None,
            shuffle=False,
            drop_last=False,
        )
        final_destinations = (eval_device, eval_device, eval_device, eval_device, None)
        final_batches = device_prefetch(final_batches, destinations=final_destinations, depth=2)
        _, final_representative_payload = evaluate(
            eval_params,
            batches=final_batches,
            dataset=dataset,
            ns=ns,
            device=eval_device,
            eval_step_fn=eval_step,
            predict_batch_fn=predict_batch,
            collect_representatives=True,
            representative_seed=args.seed + history[-1]["epoch"],
        )
        save_final_representative_plot(
            run_dir,
            history[-1]["epoch"],
            np.asarray(dataset["x"]),
            final_representative_payload,
        )

    dno_eval_summary: dict[str, object] | None = None
    dno_eval_error: str | None = None
    if not args.skip_dno_eval:
        try:
            dno_eval_summary, _ = evaluate_run_on_dno_dataset(
                run_dir=run_dir,
                checkpoint="best",
                dataset=args.dno_eval_dataset,
                seed=args.seed,
                batch_size=args.batch_size,
                allow_cpu=False,
                keep_xi_mean=False,
            )
        except Exception as exc:  # pragma: no cover - operational guard
            dno_eval_error = f"{type(exc).__name__}: {exc}"
            print(f"dno_dataset evaluation failed: {dno_eval_error}")

    summary_payload = {
        "dataset": args.dataset,
        "device": backend,
        "run_dir": str(run_dir),
        "hparams": config_payload,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"],
        "runtime_seconds": perf_counter() - start_time,
        "dno_eval_dataset": args.dno_eval_dataset if not args.skip_dno_eval else None,
        "dno_eval_summary": dno_eval_summary,
        "dno_eval_error": dno_eval_error,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
