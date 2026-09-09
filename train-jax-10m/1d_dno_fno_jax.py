from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast

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
from hadamard_shape_regularizer import compute_hadamard_reg  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_translation_tangent_loss,
)
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402

from dno_net_v2 import CraigSulemDNO  # noqa: E402
from fno1d import FNO1d  # noqa: E402
from losses import count_params, relative_l2_loss  # noqa: E402


class CheckpointMetadata(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    history: list[dict[str, float]]
    best_val_loss: float
    best_epoch: int
    stats: dict[str, object]


def save_checkpoint(
    output_dir: Path,
    state: train_state.TrainState,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
    stats: dict[str, object],
    async_manager: checkpoints.AsyncManager,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    host_state = jax.device_get(state)

    payload = {
        "params": jax.tree_util.tree_map(np.asarray, host_state.params),
        "opt_state": jax.tree_util.tree_map(np.asarray, host_state.opt_state),
        "step": int(host_state.step),
    }

    metadata = {
        "epoch": int(history[-1]["epoch"]),
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stats": stats,
    }
    checkpoints.save_checkpoint(
        ckpt_dir=output_dir,
        target=payload,
        step=metadata["epoch"],
        prefix="ckpt_",
        keep=2,
        overwrite=True,
        async_manager=async_manager,
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    # Do not advertise an epoch until its Orbax files have been written. Keeping
    # previous payload and restoring the exact metadata epoch makes interruption
    # during this wait recoverable instead of silently mixing epochs.
    async_manager.wait_previous_save()
    metadata_path = output_dir / "metadata.json"
    metadata_tmp = output_dir / "metadata.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)


def read_parameter_update_count(update_counter: jax.Array, *, counter_name: str) -> int:
    """Read the model parameter update counter from every local device, requiring equal counts.

    For example, [100, 100] returns 100; [100, 99] stops training with an error.
    This checks the counters; it does not advance or synchronize them.
    """
    update_counts_per_device = [
        int(np.asarray(device_copy.data))
        for device_copy in update_counter.addressable_shards
    ]
    if len(set(update_counts_per_device)) != 1:
        raise RuntimeError(
            f"{counter_name} differs across devices: {update_counts_per_device}"
        )
    return update_counts_per_device[0]


def training_counter_values(
    state: train_state.TrainState,
    *,
    context: str,
    announce: bool = False,
) -> tuple[int, int]:
    """Validate replicated TrainState/Adam/schedule counters."""
    state_step = read_parameter_update_count(
        cast(jax.Array, state.step),
        counter_name=f"{context} TrainState.step",
    )
    optimizer_state = cast(
        tuple[optax.ScaleByAdamState, object, optax.ScaleByScheduleState],
        state.opt_state,
    )
    adam_step = read_parameter_update_count(
        cast(jax.Array, optimizer_state[0].count),
        counter_name=f"{context} Adam count",
    )
    schedule_step = read_parameter_update_count(
        cast(jax.Array, optimizer_state[-1].count),
        counter_name=f"{context} LR schedule count",
    )
    if adam_step != schedule_step:
        raise RuntimeError(
            f"{context} optimizer counters disagree: Adam={adam_step}, "
            f"schedule={schedule_step}"
        )
    if announce:
        print(
            f"{context}: replicated counters verified "
            f"(state.step={state_step}, optimizer_count={schedule_step})"
        )
    return state_step, schedule_step


def main() -> None:
    # Model, dataset, and optimizer options.
    parser = argparse.ArgumentParser(description="Train a 1D JAX neural DNO surrogate.")
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=None,
        help="LR-schedule budget in epochs (decay_steps = total_epochs * "
        "steps_per_epoch). Defaults to --epochs. Use when resuming a "
        "partial run to keep the cosine shape matched to the original "
        "budget — e.g. resume from ep 13 of a 40-epoch run with "
        "--epochs 27 --total_epochs 40.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Linear learning-rate warmup from zero before cosine decay. "
        "Useful when a zero-initialized residual sits on top of a strong "
        "analytic baseline. 0 preserves the original schedule.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    # Additional losses; a zero weight disables each one.
    parser.add_argument(
        "--translation_tangent_weight",
        type=float,
        default=0.0,
        help="Weight of the localized translation-tangent error. Projects "
        "the DNO error onto eta_x in periodic windows and normalizes "
        "by sqrt(g*h). 0 disables.",
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
        "--hadamard_eta_scale_floor",
        type=float,
        default=1e-3,
        help="Physical RMS floor used when scaling a probe relative to eta.",
    )
    parser.add_argument(
        "--hadamard_denominator_floor",
        type=float,
        default=1e-12,
        help="Floor in the normalized Hadamard response denominator.",
    )
    args = parser.parse_args()
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
    nx = int(dataset["x"].shape[0])
    train_indices, val_indices, _ = build_dataset_split_indices(dataset)
    stats = load_or_compute_stats(dataset_path, dataset, indices=train_indices)
    stats["target_kind"] = "gxi"
    full_batches, remainder = divmod(train_indices.size, args.batch_size)
    train_steps_per_epoch = (
        full_batches + bool(remainder // n_devices) + bool(remainder % n_devices)
    )

    domain_length = float(dataset["domain_length"])
    # FNO1d's linear-baseline path needs to recover physical xi from the normalized
    # input channel. That's only exact under norm=scale, where the channel is divided
    # by feature_absmax. norm=minmax shifts as well, so the baseline is approximate.
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
            mult_hidden=args.cs_mult_hidden,
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

    # Set the learning-rate schedule and initialize AdamW's update state.
    schedule_epochs = (
        args.total_epochs if args.total_epochs is not None else args.epochs
    )
    total_steps = schedule_epochs * train_steps_per_epoch
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
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    training_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    training_state = replicate_pytree_from_host(training_state, replicated)
    checkpoint_async_manager = checkpoints.AsyncManager(max_workers=1)

    # Save the resolved settings and dataset sizes alongside the checkpoints.
    config_payload: dict[str, object] = {
        **vars(args),
        "run_name": run_name,
        "total_epochs": schedule_epochs,
        "device": backend,
        "device_count": n_devices,
        "param_count": count_params(params),
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

    # Losses shared by training and validation; Hadamard is added only during training.
    def loss_components(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        eta, xi, gxi, batch_depth = map(training_dtype, (eta, xi, gxi, batch_depth))
        batch_targets = norm_targets_jax(gxi)
        predictions = cast(
            jax.Array,
            model.apply(
                {"params": current_params}, norm_inputs_jax(eta, xi), batch_depth
            ),
        )
        data_loss = relative_l2_loss(predictions, batch_targets)
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
        if args.translation_tangent_weight > 0.0:
            tangent_loss, local_selected = compute_translation_tangent_loss(
                eta=eta,
                gxi_prediction=gxi_prediction,
                gxi_target=gxi,
                depth=h_phys,
                k=k_grid_jax,
                smoothing_scale=args.translation_tangent_smoothing_scale,
                denominator_eps=args.translation_tangent_denominator_eps,
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
        return data_loss, mode_loss, tangent_loss

    def _train_step_body(
        current_state: train_state.TrainState,
        rng_key: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ) -> tuple[train_state.TrainState, jax.Array, dict[str, jax.Array]]:
        def loss_for_params(
            current_params: FlatParams,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            data_loss, mode_loss, tangent_loss = loss_components(
                current_params, eta, xi, gxi, batch_depth
            )
            physics_loss = jnp.asarray(0.0, dtype=training_dtype)
            metrics: dict[str, jax.Array] = {"train_l2_loss": data_loss}
            if args.mode_balanced_weight > 0.0:
                mode_weight_eff = args.mode_balanced_weight * jnp.minimum(
                    jnp.asarray(current_state.step, dtype=training_dtype)
                    / max(args.mode_balanced_warmup_steps, 1),
                    1.0,
                )
                physics_loss = physics_loss + mode_weight_eff * mode_loss
                metrics["mode_balanced_loss"] = mode_loss
            if args.translation_tangent_weight > 0.0:
                physics_loss = (
                    physics_loss + args.translation_tangent_weight * tangent_loss
                )
                metrics["translation_tangent_loss"] = tangent_loss
            if args.hadamard_weight > 0.0:
                # Evaluate Hadamard on a small random subset, only on scheduled steps.
                step_active = current_state.step % args.hadamard_interval == 0

                def _hadamard_active_branch(_operand: None) -> dict[str, jax.Array]:
                    rng_base = jax.random.fold_in(
                        rng_key,
                        jax.lax.axis_index("batch") * 53 + 127,
                    )
                    rng_hadamard = jax.random.fold_in(rng_base, current_state.step)
                    rng_perm, rng_probe = jax.random.split(rng_hadamard)
                    h_phys = jnp.exp(
                        jnp.minimum(batch_depth.astype(training_dtype)[:, 0], log_h_max)
                    )
                    sample_indices = jax.random.permutation(rng_perm, eta.shape[0])[
                        :hadamard_microbatch_local
                    ]
                    eta_sub = eta.astype(training_dtype)[sample_indices]
                    xi_sub = xi.astype(training_dtype)[sample_indices]
                    depth_sub = h_phys[sample_indices]
                    batch_depth_sub = jnp.log(depth_sub)[:, None].astype(jnp.float64)
                    loss_hadamard = compute_hadamard_reg(
                        rng=rng_probe,
                        apply_fn=current_state.apply_fn,
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
                        eta_scale_floor=args.hadamard_eta_scale_floor,
                        denominator_floor=args.hadamard_denominator_floor,
                    )
                    warmup = jnp.minimum(
                        jnp.asarray(current_state.step, dtype=jnp.float64)
                        / max(args.hadamard_warmup_steps, 1),
                        1.0,
                    )
                    weight_eff = args.hadamard_weight * warmup
                    return {
                        "hadamard_active": jnp.float32(1.0),
                        "hadamard_loss": training_dtype(loss_hadamard),
                        "hadamard_extra": training_dtype(weight_eff * loss_hadamard),
                    }

                # Both branches must return the same metric keys and scalar dtypes.
                metrics.update(
                    jax.lax.cond(
                        step_active,
                        _hadamard_active_branch,
                        lambda _: {
                            "hadamard_active": jnp.float32(0.0),
                            "hadamard_loss": jnp.float32(0.0),
                            "hadamard_extra": jnp.float32(0.0),
                        },
                        operand=None,
                    )
                )
                physics_loss = physics_loss + metrics.pop("hadamard_extra")

            return data_loss + physics_loss, metrics

        # Differentiate the total loss, average gradients across GPUs, then update weights.
        (loss_value, metrics), grads = jax.value_and_grad(
            loss_for_params, has_aux=True
        )(current_state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_value = jax.lax.pmean(loss_value, axis_name="batch")
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        next_state = current_state.apply_gradients(grads=grads)
        # Copy the first GPU's updated parameters to every GPU to keep them identical.
        next_state = next_state.replace(
            params=jax.tree.map(
                lambda x: jax.lax.all_gather(x, axis_name="batch", tiled=False)[0],
                next_state.params,
            )
        )
        return next_state, loss_value, metrics

    # Split divisible batches across GPUs; replicate tiny remainder batches instead.
    # Creating these callables does not train the model; the epoch loop calls them.
    train_steps = {
        partition: jax.jit(
            shard_map(
                _train_step_body,
                mesh=mesh,
                in_specs=(P(), P(), partition, partition, partition, partition),
                out_specs=(P(), P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }

    # Validation evaluates the shared losses without gradients or Hadamard.
    def _eval_loss_body(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.lax.pmean(
            loss_components(current_params, eta, xi, gxi, batch_depth),
            axis_name="batch",
        )

    eval_loss_steps = {
        partition: jax.jit(
            shard_map(
                _eval_loss_body,
                mesh=mesh,
                in_specs=(P(), partition, partition, partition, partition),
                out_specs=(P(), P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    train_loss = float("inf")
    start_epoch = 1

    # Resume the last saved epoch, including optimizer state and metric history.
    latest_ckpt_dir = run_dir / "latest_ckpt"
    metadata_path = latest_ckpt_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid checkpoint metadata object: {metadata_path}")

        for key in ("epoch", "best_epoch"):
            if type(metadata.get(key)) is not int:
                raise ValueError(f"checkpoint metadata {key!r} must be an integer")
        for key in ("train_loss", "val_loss", "best_val_loss"):
            if type(metadata.get(key)) not in (int, float):
                raise ValueError(f"checkpoint metadata {key!r} must be numeric")

        checkpoint_history = metadata.get("history")
        if not isinstance(checkpoint_history, list):
            raise ValueError("checkpoint metadata 'history' must be a list")
        for index, entry in enumerate(checkpoint_history):
            if not isinstance(entry, dict) or any(
                type(value) not in (int, float) for value in entry.values()
            ):
                raise ValueError(
                    f"checkpoint history entry {index} must be an object with numeric values"
                )

        if not isinstance(metadata.get("stats"), dict):
            raise ValueError("checkpoint metadata 'stats' must be an object")

        metadata = cast(CheckpointMetadata, metadata)
        resume_epoch = metadata["epoch"]
        payload_path = latest_ckpt_dir / f"ckpt_{resume_epoch}"
        if not payload_path.is_dir():
            raise RuntimeError(
                f"checkpoint metadata advertises epoch {resume_epoch}, but {payload_path} "
                "does not exist; refusing a potentially torn checkpoint"
            )
        template = {
            "params": training_state.params,
            "opt_state": training_state.opt_state,
            "step": int(training_state.step),
        }
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=latest_ckpt_dir,
            target=template,
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
    training_counter_values(
        training_state,
        context="training startup",
        announce=True,
    )
    assert_pytree_replicated(training_state, name="training startup state")

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
        return device_prefetch((batch[:4] for batch in batches), sharding=data_sharding)

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)
    train_rng = jax.random.PRNGKey(args.seed + 1)

    # Train on every sample once per epoch, in a newly shuffled order.
    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        epoch_start_step, epoch_start_optimizer_count = training_counter_values(
            training_state,
            context=f"epoch {epoch} start",
        )
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_losses: list[float] = []
        train_batch_sizes: list[int] = []
        loss_sums: dict[str, float] = {}
        hadamard_loss_sum = 0.0
        hadamard_evaluation_count = 0.0
        train_bar = tqdm(
            prefetch_batches(train_indices, epoch_rng),
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for eta_b, xi_b, gxi_b, depth_b in train_bar:
            batch_size = int(eta_b.shape[0])
            train_batch_sizes.append(batch_size)
            train_step = train_steps[P("batch") if batch_size % n_devices == 0 else P()]
            train_rng, step_key = jax.random.split(train_rng)
            # Apply one optimizer update; the returned metrics are only for logging.
            training_state, batch_loss, batch_metrics = train_step(
                training_state, step_key, eta_b, xi_b, gxi_b, depth_b
            )
            batch_loss_value = float(jax.device_get(batch_loss))
            batch_losses.append(batch_loss_value)
            batch_metrics = jax.device_get(batch_metrics)
            # Skipped Hadamard batches add zero loss and zero to the evaluation count.
            hadamard_loss_sum += float(batch_metrics.pop("hadamard_loss", 0.0))
            hadamard_evaluation_count += float(batch_metrics.pop("hadamard_active", 0.0))
            # Other losses are batch means; convert them to sums over samples.
            for name, batch_mean in batch_metrics.items():
                loss_sums[name] = (
                    loss_sums.get(name, 0.0) + float(batch_mean) * batch_size
                )
            train_bar.set_postfix(loss=batch_loss_value)

        # Require one optimizer update per batch and matching counters across GPUs.
        epoch_end_step, epoch_end_optimizer_count = training_counter_values(
            training_state,
            context=f"epoch {epoch} end",
        )
        completed_steps = len(batch_losses)
        for name, advanced in (
            ("TrainState.step", epoch_end_step - epoch_start_step),
            (
                "optimizer count",
                epoch_end_optimizer_count - epoch_start_optimizer_count,
            ),
        ):
            if advanced != completed_steps:
                raise RuntimeError(
                    f"epoch {epoch} {name} advanced by {advanced}, expected {completed_steps}"
                )
        train_loss = (
            float(np.average(batch_losses, weights=train_batch_sizes))
            if batch_losses
            else float("inf")
        )

        # Evaluate validation samples without shuffling or changing model parameters.
        val_batch_losses: list[tuple[float, ...]] = []
        val_batch_sizes: list[int] = []
        for eta_b, xi_b, gxi_b, depth_b in prefetch_batches(val_indices, None):
            val_batch_sizes.append(int(eta_b.shape[0]))
            eval_loss_step = eval_loss_steps[
                P("batch") if eta_b.shape[0] % n_devices == 0 else P()
            ]
            batch_val_losses = eval_loss_step(
                training_state.params, eta_b, xi_b, gxi_b, depth_b
            )
            val_batch_losses.append(tuple(map(float, jax.device_get(batch_val_losses))))

        val_data_loss, val_mode_loss, val_tangent_loss = (
            tuple(
                float(np.average(losses, weights=val_batch_sizes))
                for losses in zip(*val_batch_losses)
            )
            if val_batch_losses
            else (float("inf"),) * 3
        )
        val_loss = (
            val_data_loss
            + args.mode_balanced_weight * val_mode_loss
            + args.translation_tangent_weight * val_tangent_loss
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l2_loss": val_data_loss,
            "val_mode_balanced_loss": val_mode_loss,
            "val_translation_tangent_loss": val_tangent_loss,
        }
        # Ordinary losses: total / samples. Hadamard: total / evaluated batches.
        for name, total in loss_sums.items():
            epoch_record[name] = total / sum(train_batch_sizes)
        if hadamard_evaluation_count:
            epoch_record["hadamard_loss"] = hadamard_loss_sum / hadamard_evaluation_count
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

        # Save a separate best checkpoint and track epochs without validation improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best_val_ckpt",
                state=training_state,
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

    # Save the final state, whether training reached the epoch limit or stopped early.
    save_checkpoint(
        run_dir / "final_ckpt",
        state=training_state,
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        stats=stats,
        async_manager=checkpoint_async_manager,
    )
    checkpoint_async_manager.wait_previous_save()

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
