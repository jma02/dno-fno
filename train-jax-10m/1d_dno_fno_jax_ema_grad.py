from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from util import (  # noqa: E402
    FlatParams,
    assert_pytree_replicated,
    build_dataset_split_indices,
    device_prefetch,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    make_normalizers,
    replicate_pytree_from_host,
)
from solver.solvers.dno_series_jax import build_grid  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from hadamard_shape_regularizer import compute_hadamard_loss  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_translation_tangent_loss,
)
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402

from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, DepthAwareMultiplier  # noqa: E402
from fno1d import FNO1d  # noqa: E402
from checkpoint_util import (  # noqa: E402
    CheckpointMetadata,
    save_checkpoint,
    training_counter_values,
)


def main() -> None:
    # Model, dataset, and optimizer options.
    parser = argparse.ArgumentParser(description="Train a 1D JAX neural DNO surrogate with gradient EMA.")
    parser.add_argument("--model", choices=("fno", "cs_dno"), default="fno")
    parser.add_argument(
        "--cs_n_polys",
        type=int,
        default=3,
        help="Highest power of η used in the CS-DNO spatial features. "
        "n_polys=3 includes eta, eta^2, eta^3.",
    )
    for feature in ("first_deriv", "second_deriv", "half_deriv", "hilbert"):
        parser.add_argument(
            f"--cs_use_{feature}", action=argparse.BooleanOptionalAction, default=True
        )
    parser.add_argument(
        "--cs_use_relative_elevation", action="store_true",
        help="Add physical eta/h to the surface features.",
    )
    parser.add_argument(
        "--cs_baseline_order", type=int, choices=(1, 2), default=1,
        help="Analytic baseline order: G0+G1 or G0+G1+G2.",
    )
    parser.add_argument("--cs_baseline_true_depth", action="store_true",
                        help="Use the actual depth for the analytic baseline, without clipping at h=5.")
    parser.add_argument(
        "--cs_mult_hidden",
        type=int,
        default=32,
        help="Hidden size of the depth-aware Fourier-multiplier MLP.",
    )
    parser.add_argument("--norm", choices=("minmax", "scale"), default="minmax")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory containing the saved .npy arrays.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Final epoch number and learning-rate schedule length. "
        "Keep the same value when resuming; completed epochs are not repeated.",
    )
    parser.add_argument(
        "--stop_after_epoch", type=int, default=1,
        help="Stop after this epoch without shortening the learning-rate schedule; 0 runs all epochs.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Linear learning-rate warmup from zero before cosine decay. "
        "Useful when a zero-initialized residual sits on top of a strong "
        "analytic baseline. 0 preserves the original schedule.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--gradient_ema_decay", type=float, default=0.9,
        help="Bias-corrected gradient EMA decay before AdamW; must be in [0, 1).",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--init_checkpoint", default=None,
                        help="Initialize weights from this checkpoint, with fresh optimizer state.")
    parser.add_argument("--nonlinear_reference", type=float, default=0.002,
                        help="Relative correction beyond G0+G1 at which the nonlinear weight is 1.5.")
    parser.add_argument("--nonlinear_weight_mean", type=float, default=1.463313,
                        help="Fixed training mean of the nonlinear weight; default calibrated on hard128 training rows.")
    parser.add_argument("--data_normalization", choices=("nonlinear", "residual"), default="nonlinear",
                        help="Weight by the M1 correction, or divide by the model's own baseline residual.")
    parser.add_argument("--residual_floor", type=float, default=1e-6,
                        help="Residual denominator floor as a fraction of the target norm.")
    parser.add_argument("--metric_interval", type=int, default=16,
                        help="Copy metrics to the host every N training or validation batches.")
    # Additional losses; a zero weight disables each one.
    parser.add_argument(
        "--translation_tangent_weight",
        type=float,
        default=0.0,
        help="Weight of the localized translation-tangent error on selected rows. Projects "
        "the DNO error onto eta_x in periodic windows and normalizes "
        "by sqrt(g*h). 0 disables.",
    )
    parser.add_argument(
        "--translation_tangent_scope", choices=("tanaka", "all"), default="tanaka",
        help="Wave families contributing to the translation-tangent loss.",
    )
    parser.add_argument(
        "--stokes_relative_tangent", action="store_true",
        help="Replace Stokes local tangent loss with target-normalized global translation error.",
    )
    parser.add_argument(
        "--translation_tangent_smoothing_scale",
        type=float,
        default=1.0,
        help="Gaussian smoothing standard deviation = this value * water depth.",
    )
    parser.add_argument(
        "--translation_tangent_denominator_eps",
        type=float,
        default=1e-3,
        help="Add eps * max_x(smoothed eta_x^2) to the speed-error denominator "
        "for each sample, preventing large estimates in nearly flat regions.",
    )
    parser.add_argument(
        "--translation_tangent_loss_norm",
        choices=("squared", "charbonnier"),
        default="squared",
        help="Per-row tangent error norm: squared phase pressure, or Charbonnier "
        "sqrt(row_loss + eps^2) - eps for absolute global / RMS local phase error.",
    )
    parser.add_argument(
        "--translation_tangent_charbonnier_eps",
        type=float,
        default=1e-7,
        help="Charbonnier smoothing epsilon; must be finite and positive.",
    )
    parser.add_argument(
        "--mode_balanced_weight",
        type=float,
        default=0.0,
        help="Weight of the universal mode-balanced complex spectral loss. 0 disables.",
    )
    parser.add_argument(
        "--mode_balanced_warmup_steps",
        type=int,
        default=500,
        help="Linear ramp of mode_balanced_weight over this many optimizer steps.",
    )
    parser.add_argument(
        "--mode_balanced_k_max",
        type=float,
        default=128.0,
        help="Largest positive physical wavenumber included in the mode-balanced loss.",
    )
    parser.add_argument(
        "--mode_balanced_activity_threshold",
        type=float,
        default=1e-4,
        help="Downweight weak Fourier modes. At 1e-4, a mode with 0.01%% of the "
        "strongest mode's strength in the same sample gets about half weight.",
    )
    parser.add_argument(
        "--mode_balanced_denominator_eps",
        type=float,
        default=1e-6,
        help="Add eps times the strongest mode's strength in each sample to every "
        "mode's error denominator.",
    )
    parser.add_argument(
        "--hadamard_weight",
        type=float,
        default=0.0,
        help="Weight of the randomized finite-secant DNO Hadamard shape-identity "
        "loss. 0 disables the regularizer.",
    )
    parser.add_argument(
        "--hadamard_interval",
        type=int,
        default=4,
        help="Evaluate the Hadamard regularizer every this-many optimizer steps.",
    )
    parser.add_argument(
        "--val_hadamard_interval", type=int, default=16,
        help="Evaluate the sampled validation Hadamard diagnostic every N batches; 1 evaluates all.",
    )
    parser.add_argument(
        "--hadamard_microbatch",
        type=int,
        default=8,
        help="Global Hadamard microbatch size, split evenly across devices.",
    )
    parser.add_argument(
        "--hadamard_warmup_steps",
        type=int,
        default=500,
        help="Linear ramp of hadamard_weight from zero to its configured value.",
    )
    parser.add_argument(
        "--hadamard_k_max",
        type=float,
        default=128.0,
        help="Maximum retained |k| in the Hadamard defect and probe.",
    )
    parser.add_argument(
        "--hadamard_sobolev_order",
        type=int,
        choices=(0, 1),
        default=1,
        help="Hadamard norm: 0 = L2, 1 = H1 (includes the first derivative).",
    )
    parser.add_argument(
        "--hadamard_fd_step_min",
        type=float,
        default=1e-3,
        help="Minimum relative finite-difference step (fraction of surface RMS).",
    )
    parser.add_argument(
        "--hadamard_fd_step_max",
        type=float,
        default=3e-3,
        help="Maximum relative finite-difference step (fraction of surface RMS).",
    )
    parser.add_argument(
        "--hadamard_min_surface_rms",
        type=float,
        default=1e-3,
        help="Minimum surface RMS used to set the perturbation's scale.",
    )
    parser.add_argument(
        "--hadamard_denominator_eps",
        type=float,
        default=1e-12,
        help="Epsilon added to the squared norm in the Hadamard loss denominator.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.gradient_ema_decay < 1.0:
        parser.error("--gradient_ema_decay must be in [0, 1)")
    if args.metric_interval < 1:
        parser.error("--metric_interval must be positive")
    if args.val_hadamard_interval < 1:
        parser.error("--val_hadamard_interval must be positive")
    if args.stokes_relative_tangent and args.translation_tangent_scope != "all":
        parser.error("--stokes_relative_tangent requires --translation_tangent_scope all")
    if (
        not np.isfinite(args.translation_tangent_charbonnier_eps)
        or args.translation_tangent_charbonnier_eps <= 0.0
    ):
        parser.error("--translation_tangent_charbonnier_eps must be finite and positive")
    if not np.isfinite(args.nonlinear_reference) or args.nonlinear_reference <= 0.0:
        parser.error("--nonlinear_reference must be finite and positive")
    if not 1.0 <= args.nonlinear_weight_mean <= 2.0:
        parser.error("--nonlinear_weight_mean must be in [1, 2]")
    if not np.isfinite(args.residual_floor) or args.residual_floor <= 0.0:
        parser.error("--residual_floor must be finite and positive")
    if args.data_normalization == "residual" and args.model != "cs_dno":
        parser.error("--data_normalization residual requires --model cs_dno")
    if args.model == "cs_dno" and args.norm != "scale":
        parser.error("CS residual normalization requires --norm scale")
    start_time = perf_counter()

    training_dtype = jnp.float32

    # Split batches across GPUs, but keep a full model/optimizer copy on each GPU.
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"JAX GPU backend is required. Found {backend!r}.")
    devices = jax.local_devices()
    n_devices = len(devices)
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by device count {n_devices}"
        )
    if args.hadamard_weight > 0.0 and args.hadamard_microbatch % n_devices != 0:
        raise ValueError(
            f"hadamard_microbatch {args.hadamard_microbatch} must be divisible by "
            f"device count {n_devices}"
        )
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())

    dataset_path = Path(args.dataset).expanduser().resolve()
    outputs_dir = REPO_ROOT / args.output_root
    run_name = args.run_name or datetime.now().strftime("fno_jax_10m_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Use the saved simulation splits and fit normalization on training samples only.
    dataset = load_dataset_arrays(dataset_path)
    family_ids = np.load(dataset_path / "family_id.npy", mmap_mode="r")
    nx = int(dataset["x"].shape[0])
    train_indices, val_indices, _ = build_dataset_split_indices(dataset)
    stats = load_or_compute_stats(dataset_path, dataset, indices=train_indices)
    full_batches, remainder = divmod(train_indices.size, args.batch_size)
    train_steps_per_epoch = (
        full_batches + bool(remainder // n_devices) + bool(remainder % n_devices)
    )

    domain_length = float(dataset["domain_length"])
    # Physical-unit scaling for both models' analytic DNO terms, matched to norm=scale.
    feature_scale = np.asarray(
        cast(list[float], stats["feature_absmax"]), dtype=np.float32
    )
    eta_scale, xi_scale = map(float, np.where(feature_scale > 0, feature_scale, 1.0))
    target_absmax = float(cast(float, stats["target_absmax"]))
    target_scale = target_absmax if target_absmax > 0 else 1.0
    # Construct the selected model and initialize its parameters.
    if args.model == "cs_dno":
        model = CraigSulemDNO(
            width=args.width,
            n_blocks=args.n_blocks,
            latent=args.latent,
            n_polys=args.cs_n_polys,
            use_first_deriv=args.cs_use_first_deriv,
            use_second_deriv=args.cs_use_second_deriv,
            use_half_deriv=args.cs_use_half_deriv,
            use_hilbert=args.cs_use_hilbert,
            use_relative_elevation=args.cs_use_relative_elevation,
            mult_hidden=args.cs_mult_hidden,
            baseline_order=args.cs_baseline_order,
            baseline_true_depth=args.cs_baseline_true_depth,
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
        )
    with jax.default_device(devices[0]):
        params = model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros((1, nx, 2), dtype=training_dtype),
            jnp.zeros((1, 1), dtype=training_dtype),
        )["params"]
        if args.init_checkpoint and not (run_dir / "latest_ckpt" / "metadata.json").exists():
            restored_weights = checkpoints.restore_checkpoint(
                ckpt_dir=Path(args.init_checkpoint).resolve(), target=None,
                prefix="ckpt_", orbax_checkpointer=ocp.PyTreeCheckpointer(),
            )
            params = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value, dtype=training_dtype), restored_weights["params"]
            )
            print(f"Initialized weights from {args.init_checkpoint}; optimizer starts fresh")

    def apply_model(variables: dict[str, Any], inputs: jax.Array, depth: jax.Array,
                    *, return_baseline: bool = False) -> Any:
        if args.model != "cs_dno":
            prediction = cast(jax.Array, model.apply(variables, inputs, depth))
            return (prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction)) if return_baseline else prediction
        multiplier_model = DepthAwareMultiplier(args.latent, domain_length, 5.0, args.cs_mult_hidden)

        def intercept(next_fun: Any, call_args: tuple[Any, ...], call_kwargs: dict[str, Any],
                      context: nn.module.InterceptorContext) -> Any:
            if isinstance(context.module, CraigSulemBlock) and context.method_name == "__call__":
                eta_features, xi_phys, log_depth = call_args
                if context.module.name != f"cs_block_{args.n_blocks - 1}":
                    return jnp.zeros_like(xi_phys)
                groups = [variables["params"][f"cs_block_{i}"] for i in range(args.n_blocks)]
                weights = eta_features @ jnp.concatenate([group["phi_proj"]["kernel"] for group in groups], axis=-1)
                filters = jnp.concatenate([
                    cast(jax.Array, multiplier_model.apply({"params": group["m_shared"]}, log_depth, nx // 2 + 1))
                    for group in groups
                ], axis=-1)
                xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")
                filtered = jnp.fft.irfft(filters * xi_hat[..., None], n=nx, axis=1, norm="forward")
                weighted_hat = jnp.fft.rfft(weights * filtered, axis=1, norm="forward")
                return jnp.fft.irfft(jnp.sum(filters * weighted_hat, axis=-1), n=nx, axis=-1, norm="forward")
            return next_fun(*call_args, **call_kwargs)

        with nn.intercept_methods(intercept):
            return model.apply(variables, inputs, depth, return_baseline=return_baseline)

    # Set the learning-rate schedule and initialize AdamW's update state.
    total_steps = args.epochs * train_steps_per_epoch
    if args.lr_warmup_steps > 0:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=args.lr_warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=total_steps,
        )
    optimizer = optax.chain(
        optax.ema(args.gradient_ema_decay, debias=True),
        optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay),
    )
    print(f"gradient EMA before AdamW: decay={args.gradient_ema_decay}, debias=True")
    training_state = train_state.TrainState.create(
        apply_fn=apply_model, params=params, tx=optimizer
    )

    training_state = replicate_pytree_from_host(training_state, replicated)
    checkpoint_async_manager = checkpoints.AsyncManager(max_workers=1)

    # Save the resolved settings and dataset sizes alongside the checkpoints.
    config_payload: dict[str, object] = {
        **vars(args),
        "run_name": run_name,
        "device": backend,
        "device_count": n_devices,
        "param_count": sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
        "train_examples": int(train_indices.shape[0]),
        "val_examples": int(val_indices.shape[0]),
        "domain_length": domain_length,
        "xi_scale": xi_scale,
        "eta_scale": eta_scale,
        "target_scale": target_scale,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    norm_inputs_jax, norm_targets_jax, denorm_targets_jax = make_normalizers(
        stats, args.norm
    )

    log_h_max = float(np.log(5.0))
    hadamard_microbatch_local = args.hadamard_microbatch // n_devices
    _, _k_grid = build_grid(nx, domain_length)
    k_grid_jax = jnp.asarray(_k_grid, dtype=training_dtype)
    k_rfft_jax = jnp.abs(k_grid_jax[: nx // 2 + 1])

    # Full-batch losses shared by training and validation.
    def compute_loss_components(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
        *,
        collect_family_metrics: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
        eta, xi, gxi, batch_depth = map(training_dtype, (eta, xi, gxi, batch_depth))
        batch_targets = norm_targets_jax(gxi)
        predictions, baseline, m1_baseline = apply_model(
            {"params": current_params}, norm_inputs_jax(eta, xi), batch_depth,
            return_baseline=True,
        )
        pred_hat = jnp.fft.rfft(predictions, axis=1)
        target_hat = jnp.fft.rfft(batch_targets, axis=1)
        baseline_hat = jnp.fft.rfft(baseline, axis=1)
        squared_error = jnp.sum(jnp.abs((pred_hat - target_hat).astype(jnp.complex128)) ** 2, axis=(1, 2))
        zero_error = squared_error == 0.0
        error_norm = jnp.where(zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error)))
        target_norm = jnp.sqrt(jnp.maximum(
            jnp.sum(jnp.abs(target_hat.astype(jnp.complex128)) ** 2, axis=(1, 2)), 1e-12,
        ))
        residual_squared = jnp.sum(
            jnp.abs((target_hat - baseline_hat).astype(jnp.complex128)) ** 2, axis=(1, 2),
        )
        residual_denominator = jnp.sqrt(residual_squared + (args.residual_floor * target_norm) ** 2)
        denominator = target_norm
        if args.model == "cs_dno":
            if args.data_normalization == "residual":
                denominator = residual_denominator
            else:
                # The control keeps its weighting based on the M1 correction.
                m1_hat = jnp.fft.rfft(m1_baseline, axis=1)
                residual_ratio = jnp.sqrt(jnp.sum(
                    jnp.abs((target_hat - m1_hat).astype(jnp.complex128)) ** 2, axis=(1, 2),
                )) / target_norm
                nonlinear_weight = 1.0 + residual_ratio / (residual_ratio + args.nonlinear_reference)
                denominator = target_norm * args.nonlinear_weight_mean / nonlinear_weight
        denominator = jax.lax.stop_gradient(denominator)
        data_loss = jnp.mean(error_norm / denominator)
        full_l2_loss = jnp.mean(error_norm / target_norm)
        family_metrics: tuple[jax.Array, ...] = ()
        if collect_family_metrics:
            row_metrics = jnp.stack((
                jnp.ones_like(error_norm), error_norm / target_norm,
                error_norm / residual_denominator, squared_error < residual_squared,
                error_norm / denominator,
            ))
            # Masked means, not means of each shard's family means: pmean remains exact.
            family_metrics = tuple(
                value for family_id in PhysicalFamilyId
                for value in jnp.mean(jnp.where(family == family_id, row_metrics, 0.0), axis=1)
            )
        # Data fit uses normalized Gxi; the physics losses below use physical fields.
        h_phys = jnp.exp(jnp.minimum(batch_depth[:, 0], log_h_max))
        gxi_prediction = denorm_targets_jax(predictions)[..., 0]
        mode_loss = jnp.float32(0.0)
        if args.mode_balanced_weight > 0.0:
            mode_loss = compute_mode_balanced_loss(
                eta=eta,
                gxi_prediction=gxi_prediction,
                gxi_target=denorm_targets_jax(batch_targets)[..., 0],
                depth=h_phys,
                k_rfft=k_rfft_jax,
                k_max=args.mode_balanced_k_max,
                activity_threshold=args.mode_balanced_activity_threshold,
                denominator_eps=args.mode_balanced_denominator_eps,
            )
        tangent_loss = jnp.float32(0.0)
        local_selected = jnp.float32(0.0)
        if args.translation_tangent_weight > 0.0:
            tangent_loss, local_selected = compute_translation_tangent_loss(
                eta=eta,
                gxi_prediction=gxi_prediction,
                gxi_target=gxi,
                depth=h_phys,
                k=k_grid_jax,
                sample_mask=(family == PhysicalFamilyId.TANAKA) if args.translation_tangent_scope == "tanaka" else None,
                global_relative_mask=(family == PhysicalFamilyId.STOKES) if args.stokes_relative_tangent else None,
                smoothing_scale=args.translation_tangent_smoothing_scale,
                denominator_eps=args.translation_tangent_denominator_eps,
                loss_norm=args.translation_tangent_loss_norm,
                charbonnier_eps=args.translation_tangent_charbonnier_eps,
            )
            # Weight each GPU's contribution by its number of selected nonflat samples.
            global_selected = jax.lax.psum(local_selected, axis_name="batch")
            device_count = jax.lax.psum(jnp.float32(1.0), axis_name="batch")
            shard_weight = (
                device_count
                * local_selected
                / jnp.maximum(global_selected, jnp.float32(1.0))
            )
            tangent_loss = tangent_loss * shard_weight
        return data_loss, mode_loss, tangent_loss, local_selected / eta.shape[0], full_l2_loss, family_metrics

    def hadamard_loss_on_subset(
        rng: jax.Array,
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        batch_depth: jax.Array,
    ) -> jax.Array:
        rng_perm, rng_probe = jax.random.split(rng)
        sample_indices = jax.random.permutation(rng_perm, eta.shape[0])[
            :hadamard_microbatch_local
        ]
        eta_sub = eta.astype(training_dtype)[sample_indices]
        xi_sub = xi.astype(training_dtype)[sample_indices]
        batch_depth_sub = batch_depth.astype(training_dtype)[sample_indices].astype(jnp.float64)
        return compute_hadamard_loss(
            rng=rng_probe,
            apply_fn=apply_model,
            model_params=current_params,
            eta_phys=eta_sub,
            xi_phys=xi_sub,
            batch_depth_local=batch_depth_sub,
            norm_inputs_fn=norm_inputs_jax,
            denorm_targets_fn=denorm_targets_jax,
            k=k_grid_jax,
            dtype=jnp.float64,
            k_max=args.hadamard_k_max,
            sobolev_order=args.hadamard_sobolev_order,
            fd_step_min=args.hadamard_fd_step_min,
            fd_step_max=args.hadamard_fd_step_max,
            min_surface_rms=args.hadamard_min_surface_rms,
            denominator_eps=args.hadamard_denominator_eps,
        )

    def train_step(
        current_state: train_state.TrainState,
        rng_key: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
    ) -> tuple[train_state.TrainState, jax.Array, dict[str, jax.Array]]:
        def compute_training_loss(
            current_params: FlatParams,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            data_loss, mode_loss, tangent_loss, tangent_fraction, full_l2_loss, _ = compute_loss_components(
                current_params, eta, xi, gxi, batch_depth, family
            )
            physics_loss = jnp.asarray(0.0, dtype=training_dtype)
            metrics: dict[str, jax.Array] = {
                "train_l2_loss": full_l2_loss, "train_data_loss": data_loss,
            }
            if args.mode_balanced_weight > 0.0:
                mode_weight = args.mode_balanced_weight * jnp.minimum(
                    jnp.asarray(current_state.step, dtype=training_dtype)
                    / max(args.mode_balanced_warmup_steps, 1),
                    1.0,
                )
                physics_loss += mode_weight * mode_loss
                metrics["mode_balanced_loss"] = mode_loss
            if args.translation_tangent_weight > 0.0:
                physics_loss += args.translation_tangent_weight * tangent_loss
                metrics["translation_tangent_loss"] = tangent_loss
                metrics["translation_tangent_fraction"] = tangent_fraction
            if args.hadamard_weight > 0.0:
                # Evaluate Hadamard on a small random subset, only on scheduled steps.
                hadamard_scheduled = current_state.step % args.hadamard_interval == 0

                rng_hadamard = jax.random.fold_in(
                    jax.random.fold_in(rng_key, jax.lax.axis_index("batch") * 53 + 127),
                    current_state.step,
                )
                hadamard_loss = jax.lax.cond(
                    hadamard_scheduled,
                    lambda rng: hadamard_loss_on_subset(
                        rng, current_params, eta, xi, batch_depth
                    ),
                    lambda _: jnp.float64(0.0),
                    rng_hadamard,
                )
                hadamard_weight = args.hadamard_weight * jnp.minimum(
                    jnp.asarray(current_state.step, dtype=jnp.float64)
                    / max(args.hadamard_warmup_steps, 1),
                    1.0,
                )
                physics_loss += training_dtype(hadamard_weight * hadamard_loss)
                metrics.update(
                    hadamard_active=training_dtype(hadamard_scheduled),
                    hadamard_loss=training_dtype(hadamard_loss),
                )

            return data_loss + physics_loss, metrics

        # Differentiate the total loss, average gradients across GPUs, then update weights.
        (loss_value, metrics), grads = jax.value_and_grad(
            compute_training_loss, has_aux=True
        )(current_state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_value = jax.lax.pmean(loss_value, axis_name="batch")
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        next_state = current_state.apply_gradients(grads=grads)
        return next_state, loss_value, metrics

    # Split divisible batches across GPUs; replicate tiny remainder batches instead.
    # Creating these callables does not train the model; the epoch loop calls them.
    train_steps = {
        partition: jax.jit(
            shard_map(
                train_step,
                mesh=mesh,
                in_specs=(P(), P(), partition, partition, partition, partition, partition),
                out_specs=(P(), P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }

    # Validation uses fixed Hadamard samples/probes and takes no parameter gradients.
    def eval_step(
        current_params: FlatParams,
        rng_key: jax.Array,
        batch_index: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
        *,
        run_hadamard: bool,
    ) -> tuple[jax.Array, ...]:
        hadamard_loss = jnp.float32(0.0)
        if args.hadamard_weight > 0.0 and run_hadamard:
            hadamard_loss = training_dtype(hadamard_loss_on_subset(
                jax.random.fold_in(jax.random.fold_in(rng_key, batch_index), jax.lax.axis_index("batch")),
                current_params, eta, xi, batch_depth,
            ))
        data_loss, mode_loss, tangent_loss, tangent_fraction, full_l2_loss, family_metrics = compute_loss_components(
            current_params, eta, xi, gxi, batch_depth, family, collect_family_metrics=True,
        )
        return jax.lax.pmean(
            (data_loss, mode_loss, tangent_loss, hadamard_loss, tangent_fraction, full_l2_loss, *family_metrics),
            axis_name="batch",
        )

    eval_steps = {
        (partition, run_hadamard): jax.jit(
            shard_map(
                partial(eval_step, run_hadamard=run_hadamard),
                mesh=mesh,
                in_specs=(P(), P(), P(), partition, partition, partition, partition, partition),
                out_specs=(P(),) * (6 + 5 * len(PhysicalFamilyId)),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
        for run_hadamard in (False, True)
    }

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    train_loss = float("inf")
    start_epoch = 1

    # Resume the last saved epoch, including optimizer state and metric history.
    latest_ckpt_dir = run_dir / "latest_ckpt"
    metadata_path = latest_ckpt_dir / "metadata.json"
    if metadata_path.exists():
        metadata: CheckpointMetadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        resume_epoch = metadata["epoch"]
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=latest_ckpt_dir,
            target={
                "params": training_state.params,
                "opt_state": training_state.opt_state,
                "step": int(training_state.step),
            },
            step=resume_epoch,
            prefix="ckpt_",
            orbax_checkpointer=ocp.PyTreeCheckpointer(),
        )
        training_state = training_state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"],
            step=jnp.asarray(int(restored["step"]), dtype=jnp.int32),
        )
        training_state = replicate_pytree_from_host(training_state, replicated)
        history = metadata["history"]
        train_loss = metadata["train_loss"]
        best_val_loss = metadata["best_val_loss"]
        best_epoch = metadata["best_epoch"]
        start_epoch = resume_epoch + 1
        print(
            f"auto-resume: restored from {latest_ckpt_dir} at epoch {resume_epoch}, "
            f"resuming at epoch {start_epoch}"
        )

    # Verify all GPU copies agree before training, including after checkpoint restore.
    # The counter helper reads AdamW; checkpoints retain the full EMA + AdamW state.
    training_counter_values(
        training_state.replace(opt_state=training_state.opt_state[1]),
        context="training startup",
        announce=True,
    )
    assert_pytree_replicated(training_state, name="training startup state")

    # Repair a best checkpoint interrupted after its latest checkpoint was saved.
    if history and history[-1]["epoch"] == best_epoch and np.isfinite(best_val_loss):
        save_checkpoint(
            run_dir / "best_val_ckpt",
            state=training_state,
            history=history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            stats=stats,
            async_manager=checkpoint_async_manager,
        )
    # Discard log rows not backed by the restored checkpoint, including partial rows.
    with open(run_dir / "train_log.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in history)

    # Load batches and copy the next one to GPUs in the background.
    def prefetch_batches(
        indices: np.ndarray, rng: np.random.Generator | None
    ) -> Iterator[tuple[jax.Array, ...]]:
        batches = get_batches(
            dataset["eta"],
            dataset["xi"],
            dataset["gxi"],
            dataset["depth"],
            indices,
            args.batch_size,
            rng,
            device_count=n_devices,
        )
        return device_prefetch(
            ((*batch[:4], family_ids[batch[4]]) for batch in batches),
            sharding=data_sharding,
        )

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)
    # Hadamard folds this fixed key with the GPU index and restored optimizer step.
    train_rng = replicate_pytree_from_host(jax.random.PRNGKey(args.seed + 1), replicated)
    validation_rng = replicate_pytree_from_host(jax.random.PRNGKey(args.seed + 2), replicated)

    # Train on every sample once per epoch, in a newly shuffled order.
    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        if args.stop_after_epoch and epoch > args.stop_after_epoch:
            break
        if 0 < args.early_stopping_patience <= epoch - 1 - best_epoch:
            break
        epoch_start_step, epoch_start_optimizer_count = training_counter_values(
            training_state.replace(opt_state=training_state.opt_state[1]),
            context=f"epoch {epoch} start",
        )
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_losses: list[float] = []
        train_batch_sizes: list[int] = []
        loss_sums: dict[str, float] = {}
        tangent_loss_sum = tangent_sample_count = 0.0
        hadamard_loss_sum = 0.0
        hadamard_batch_evaluation_count = 0.0
        pending_train_metrics: list[tuple[jax.Array, dict[str, jax.Array], int]] = []
        train_bar = tqdm(
            prefetch_batches(train_indices, epoch_rng),
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for batch_index, (eta_b, xi_b, gxi_b, depth_b, family_b) in enumerate(train_bar, 1):
            batch_size = int(eta_b.shape[0])
            compiled_train_step = train_steps[P("batch") if batch_size % n_devices == 0 else P()]
            # Apply one optimizer update; the returned metrics are only for logging.
            training_state, batch_loss, batch_metrics = compiled_train_step(
                training_state, train_rng, eta_b, xi_b, gxi_b, depth_b, family_b
            )
            pending_train_metrics.append((batch_loss, batch_metrics, batch_size))
            if len(pending_train_metrics) < args.metric_interval and batch_index < train_steps_per_epoch:
                continue
            for batch_loss, batch_metrics, buffered_size in jax.device_get(pending_train_metrics):
                batch_loss_value = float(batch_loss)
                if not np.isfinite(batch_loss_value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
                batch_losses.append(batch_loss_value)
                train_batch_sizes.append(buffered_size)
                # Skipped Hadamard batches add zero loss and zero to the evaluation count.
                hadamard_loss_sum += float(batch_metrics.pop("hadamard_loss", 0.0))
                hadamard_batch_evaluation_count += float(batch_metrics.pop("hadamard_active", 0.0))
                # Translation loss is a mean over selected nonflat rows.
                selected_count = float(batch_metrics.pop("translation_tangent_fraction", 0.0)) * buffered_size
                tangent_sample_count += selected_count
                tangent_loss_sum += float(batch_metrics.pop("translation_tangent_loss", 0.0)) * selected_count
                # Other losses are batch means; convert them to sums over samples.
                for name, batch_mean in batch_metrics.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(batch_mean) * buffered_size
            pending_train_metrics.clear()
            train_bar.set_postfix(loss=batch_losses[-1], refresh=False)

        # Require one optimizer update per batch and matching counters across GPUs.
        epoch_end_step, epoch_end_optimizer_count = training_counter_values(
            training_state.replace(opt_state=training_state.opt_state[1]),
            context=f"epoch {epoch} end",
        )
        state_updates = epoch_end_step - epoch_start_step
        optimizer_updates = epoch_end_optimizer_count - epoch_start_optimizer_count
        expected_updates = len(batch_losses)
        if state_updates != expected_updates or optimizer_updates != expected_updates:
            raise RuntimeError(
                f"epoch {epoch}: expected {expected_updates} updates; "
                f"TrainState={state_updates}, optimizer={optimizer_updates}"
            )
        train_loss = float(np.average(batch_losses, weights=train_batch_sizes))

        # Evaluate validation samples without shuffling or changing model parameters.
        val_batch_losses: list[tuple[float, ...]] = []
        val_batch_sizes: list[int] = []
        pending_val_losses: list[tuple[jax.Array, ...]] = []
        for batch_index, (eta_b, xi_b, gxi_b, depth_b, family_b) in enumerate(
            prefetch_batches(val_indices, None)
        ):
            val_batch_sizes.append(int(eta_b.shape[0]))
            compiled_eval_step = eval_steps[
                P("batch") if eta_b.shape[0] % n_devices == 0 else P(),
                args.hadamard_weight > 0.0 and batch_index % args.val_hadamard_interval == 0,
            ]
            batch_val_losses = compiled_eval_step(
                training_state.params, validation_rng, batch_index,
                eta_b, xi_b, gxi_b, depth_b, family_b,
            )
            pending_val_losses.append(batch_val_losses)
            if len(pending_val_losses) == args.metric_interval:
                val_batch_losses.extend(tuple(map(float, losses)) for losses in jax.device_get(pending_val_losses))
                pending_val_losses.clear()
        if pending_val_losses:
            val_batch_losses.extend(tuple(map(float, losses)) for losses in jax.device_get(pending_val_losses))

        # Ordinary losses use every batch; HD is a diagnostic on fixed sampled batches.
        val_data_loss = val_mode_loss = val_tangent_loss = val_hadamard_loss = float("inf")
        val_full_l2_loss = float("inf")
        losses = np.asarray(val_batch_losses)
        if val_batch_losses:
            means = np.average(losses[:, :4], axis=0, weights=val_batch_sizes)
            # Weight translation loss by its selected nonflat rows across batches.
            tangent_counts = losses[:, 4] * val_batch_sizes
            means[2] = np.dot(losses[:, 2], tangent_counts) / max(tangent_counts.sum(), 1.0)
            hadamard_selected = np.arange(len(losses)) % args.val_hadamard_interval == 0
            means[3] = np.average(losses[hadamard_selected, 3], weights=np.asarray(val_batch_sizes)[hadamard_selected])
            val_data_loss, val_mode_loss, val_tangent_loss, val_hadamard_loss = map(float, means)
            val_full_l2_loss = float(np.average(losses[:, 5], weights=val_batch_sizes))
        val_loss = (
            val_data_loss
            + args.mode_balanced_weight * val_mode_loss
            + args.translation_tangent_weight * val_tangent_loss
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l2_loss": val_full_l2_loss,
            "val_data_loss": val_data_loss,
            "val_mode_balanced_loss": val_mode_loss,
            "val_translation_tangent_loss": val_tangent_loss,
            "val_hadamard_loss": val_hadamard_loss,
        }
        if val_batch_losses:
            family_totals = np.sum(losses[:, 6:] * np.asarray(val_batch_sizes)[:, None], axis=0).reshape(-1, 5)
            for family_id, (count, relative, residual, wins, data_sum) in zip(PhysicalFamilyId, family_totals):
                prefix = f"val_{family_id.name.lower()}"
                epoch_record[f"{prefix}_l2_loss"] = float(relative / count)
                epoch_record[f"{prefix}_baseline_ratio_floored"] = float(residual / count)
                epoch_record[f"{prefix}_baseline_win_fraction"] = float(wins / count)
                epoch_record[f"{prefix}_data_loss_share"] = float(data_sum / family_totals[:, 4].sum())
        # Ordinary losses: total / samples. Hadamard: total / evaluated batches.
        for name, total in loss_sums.items():
            epoch_record[name] = total / sum(train_batch_sizes)
        if args.translation_tangent_weight > 0.0:
            epoch_record["translation_tangent_loss"] = tangent_loss_sum / max(tangent_sample_count, 1.0)
        if hadamard_batch_evaluation_count:
            epoch_record["hadamard_loss"] = hadamard_loss_sum / hadamard_batch_evaluation_count
        history.append(epoch_record)
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")

        # Per-epoch checkpoint for preemption-safe resume. Overwrites prior latest_ckpt.
        save_checkpoint(
            run_dir / "latest_ckpt",
            state=training_state,
            history=history,
            best_val_loss=best_val_loss if val_loss >= best_val_loss else val_loss,
            best_epoch=best_epoch if val_loss >= best_val_loss else epoch,
            stats=stats,
            async_manager=checkpoint_async_manager,
        )

        # Save a separate copy whenever validation improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best_val_ckpt",
                state=training_state,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                stats=stats,
                async_manager=checkpoint_async_manager,
            )

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
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
