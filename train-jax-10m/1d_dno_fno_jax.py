from __future__ import annotations

import argparse
import json
import os
import sys
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
    NormStats,
    assert_pytree_replicated,
    build_dataset_split_indices,
    device_prefetch,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    make_normalizers,
    replicate_pytree_from_host,
    require_jax_devices,
)
from solver.solvers.dno_series_jax import build_grid  # noqa: E402
from hadamard_shape_regularizer import (  # noqa: E402
    HadamardRegConfig,
    compute_hadamard_reg,
    sample_microbatch,
)
from translation_tangent_regularizer import (  # noqa: E402
    TranslationTangentConfig,
    compute_translation_tangent_loss,
)
from mode_balanced_regularizer import (  # noqa: E402
    ModeBalancedConfig,
    compute_mode_balanced_loss,
)

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

    def to_np(value: object) -> np.ndarray:
        return np.asarray(value)

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
    checkpoints.save_checkpoint(
        ckpt_dir=output_dir,
        target=payload,
        step=epoch,
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


def read_committed_checkpoint_metadata(checkpoint_dir: Path) -> CheckpointMetadata:
    """Read metadata only when its exact, committed Orbax payload exists."""
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"no metadata.json under {checkpoint_dir}")
    decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"invalid checkpoint metadata object: {metadata_path}")
    record = cast(dict[str, object], decoded)

    def require_int(key: str) -> int:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"checkpoint metadata {key!r} must be an integer")
        return value

    def require_float(key: str) -> float:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"checkpoint metadata {key!r} must be numeric")
        return float(value)

    history_value = record.get("history")
    if not isinstance(history_value, list):
        raise ValueError("checkpoint metadata 'history' must be a list")
    history: list[dict[str, float]] = []
    for index, entry in enumerate(history_value):
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint history entry {index} must be an object")
        parsed_entry: dict[str, float] = {}
        for key, value in entry.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"checkpoint history entry {index} must contain numeric values"
                )
            parsed_entry[key] = value
        history.append(parsed_entry)

    stats_value = record.get("stats")
    if not isinstance(stats_value, dict) or not all(
        isinstance(key, str) for key in stats_value
    ):
        raise ValueError("checkpoint metadata 'stats' must be an object")

    metadata = CheckpointMetadata(
        epoch=require_int("epoch"),
        train_loss=require_float("train_loss"),
        val_loss=require_float("val_loss"),
        history=history,
        best_val_loss=require_float("best_val_loss"),
        best_epoch=require_int("best_epoch"),
        stats=cast(dict[str, object], stats_value),
    )
    epoch = metadata["epoch"]
    payload_path = checkpoint_dir / f"ckpt_{epoch}"
    if not payload_path.is_dir():
        raise RuntimeError(
            f"checkpoint metadata advertises epoch {epoch}, but {payload_path} "
            "does not exist; refusing a potentially torn checkpoint"
        )
    return metadata


def replicated_scalar_value(value: jax.Array, *, name: str) -> int:
    """Return a replicated scalar after verifying every local device agrees."""
    replica_values = [int(np.asarray(shard.data)) for shard in value.addressable_shards]
    if not replica_values:
        raise RuntimeError(f"{name} has no addressable replicas")
    if len(set(replica_values)) != 1:
        raise RuntimeError(f"{name} replicas disagree: {replica_values}")
    return replica_values[0]


def training_counter_values(
    state: train_state.TrainState,
    *,
    context: str,
    announce: bool = False,
) -> tuple[int, int]:
    """Validate replicated TrainState/Adam/schedule counters."""
    state_step = replicated_scalar_value(
        cast(jax.Array, state.step),
        name=f"{context} TrainState.step",
    )
    optimizer_state = cast(
        tuple[optax.ScaleByAdamState, object, optax.ScaleByScheduleState],
        state.opt_state,
    )
    adam_step = replicated_scalar_value(
        cast(jax.Array, optimizer_state[0].count),
        name=f"{context} Adam count",
    )
    schedule_step = replicated_scalar_value(
        cast(jax.Array, optimizer_state[-1].count),
        name=f"{context} LR schedule count",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D JAX neural DNO surrogate.")
    parser.add_argument("--model", choices=("fno", "cs_dno"), default="fno")
    parser.add_argument(
        "--cs_n_polys",
        type=int,
        default=3,
        help="Highest power of η used in the CS-DNO spatial features. "
        "n_polys=3 includes η, η², η³.",
    )
    parser.add_argument(
        "--cs_use_first_deriv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cs_use_second_deriv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cs_use_half_deriv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cs_use_hilbert", action=argparse.BooleanOptionalAction, default=True
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
    parser.add_argument(
        "--translation_tangent_weight",
        type=float,
        default=0.0,
        help="Weight of the localized translation-tangent error. Projects "
        "the DNO error onto eta_x in periodic windows and normalizes "
        "by sqrt(g*h). 0 disables.",
    )
    parser.add_argument(
        "--translation_tangent_window_depths",
        type=float,
        default=1.0,
        help="Gaussian localization width as a multiple of physical depth h.",
    )
    parser.add_argument(
        "--translation_tangent_energy_floor_relative",
        type=float,
        default=1e-3,
        help="Local eta_x-energy denominator floor relative to its sample maximum.",
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
        "--mode_balanced_active_scale_relative",
        type=float,
        default=1e-4,
        help="Soft activity threshold relative to each sample's strongest modal energy.",
    )
    parser.add_argument(
        "--mode_balanced_denominator_floor_relative",
        type=float,
        default=1e-6,
        help="Relative floor in the physical modal-error denominator.",
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
        default=1,
        help="Sobolev order used to scale probes and weight the Hadamard defect.",
    )
    parser.add_argument(
        "--hadamard_relative_eps_min",
        type=float,
        default=1e-3,
        help="Minimum relative surface perturbation for the finite secant.",
    )
    parser.add_argument(
        "--hadamard_relative_eps_max",
        type=float,
        default=3e-3,
        help="Maximum relative surface perturbation for the finite secant.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

    training_dtype = jnp.float32

    backend, devices = require_jax_devices()
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
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("fno_jax_10m_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_arrays(dataset_path)
    nx = int(dataset["x"].shape[0])
    train_indices, val_indices, _ = build_dataset_split_indices(dataset)
    stats = dict(
        load_or_compute_stats(
            dataset_path,
            dataset=dataset,
            indices=train_indices,
        )
    )
    stats["target_kind"] = "gxi"
    ns = NormStats.from_dict(stats, mode=args.norm)
    full_batches, remainder = divmod(train_indices.size, args.batch_size)
    train_steps_per_epoch = (
        full_batches + bool(remainder // n_devices) + bool(remainder % n_devices)
    )

    domain_length = float(dataset["domain_length"])
    # FNO1d's linear-baseline path needs to recover physical xi from the normalized
    # input channel. That's only exact under norm=scale, where the channel is divided
    # by feature_absmax. norm=minmax shifts as well, so the baseline is approximate.
    xi_scale = float(np.asarray(ns.feature_absmax).reshape(-1)[1])
    eta_scale = float(np.asarray(ns.feature_absmax).reshape(-1)[0])
    target_scale = float(ns.target_absmax)
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
    loss_fn = relative_l2_loss

    schedule_epochs = (
        args.total_epochs if args.total_epochs is not None else args.epochs
    )
    total_steps = schedule_epochs * train_steps_per_epoch
    if args.lr_warmup_steps > 0:
        if args.lr_warmup_steps >= total_steps:
            raise ValueError(
                f"lr_warmup_steps {args.lr_warmup_steps} must be smaller than "
                f"total optimizer steps {total_steps}"
            )
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

    config_payload: dict[str, object] = {
        "model": args.model,
        "norm": args.norm,
        "dataset": args.dataset,
        "device": backend,
        "device_count": n_devices,
        "width": args.width,
        "n_blocks": args.n_blocks,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "epochs": args.epochs,
        "total_epochs": schedule_epochs,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "param_count": count_params(params),
        "train_examples": int(train_indices.shape[0]),
        "val_examples": int(val_indices.shape[0]),
    }
    config_payload["domain_length"] = domain_length
    config_payload["xi_scale"] = xi_scale
    config_payload["eta_scale"] = eta_scale
    config_payload["target_scale"] = target_scale
    config_payload["translation_tangent_weight"] = float(
        args.translation_tangent_weight
    )
    config_payload["translation_tangent_window_depths"] = float(
        args.translation_tangent_window_depths
    )
    config_payload["translation_tangent_energy_floor_relative"] = float(
        args.translation_tangent_energy_floor_relative
    )
    config_payload["translation_tangent_scope"] = "all_nonflat_rows"
    config_payload["mode_balanced_weight"] = float(args.mode_balanced_weight)
    config_payload["mode_balanced_warmup_steps"] = int(args.mode_balanced_warmup_steps)
    config_payload["mode_balanced_k_max"] = float(args.mode_balanced_k_max)
    config_payload["mode_balanced_active_scale_relative"] = float(
        args.mode_balanced_active_scale_relative
    )
    config_payload["mode_balanced_denominator_floor_relative"] = float(
        args.mode_balanced_denominator_floor_relative
    )
    config_payload["hadamard_weight"] = float(args.hadamard_weight)
    config_payload["hadamard_interval"] = int(args.hadamard_interval)
    config_payload["hadamard_microbatch"] = int(args.hadamard_microbatch)
    config_payload["hadamard_warmup_steps"] = int(args.hadamard_warmup_steps)
    config_payload["hadamard_k_max"] = float(args.hadamard_k_max)
    config_payload["hadamard_sobolev_order"] = int(args.hadamard_sobolev_order)
    config_payload["hadamard_relative_eps_min"] = float(args.hadamard_relative_eps_min)
    config_payload["hadamard_relative_eps_max"] = float(args.hadamard_relative_eps_max)
    config_payload["hadamard_eta_scale_floor"] = float(args.hadamard_eta_scale_floor)
    config_payload["hadamard_denominator_floor"] = float(
        args.hadamard_denominator_floor
    )
    if args.model == "fno":
        config_payload["modes"] = args.modes
    if args.model == "cs_dno":
        config_payload["latent"] = args.latent
        config_payload["cs_n_polys"] = args.cs_n_polys
        config_payload["cs_use_first_deriv"] = bool(args.cs_use_first_deriv)
        config_payload["cs_use_second_deriv"] = bool(args.cs_use_second_deriv)
        config_payload["cs_use_half_deriv"] = bool(args.cs_use_half_deriv)
        config_payload["cs_use_hilbert"] = bool(args.cs_use_hilbert)
        config_payload["cs_mult_hidden"] = args.cs_mult_hidden
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    norm_inputs_jax, norm_targets_jax, denorm_targets_jax = make_normalizers(ns)

    gravity = 1.0
    log_h_max = float(np.log(5.0))
    translation_tangent_weight = float(args.translation_tangent_weight)
    translation_tangent_cfg = TranslationTangentConfig(
        window_depths=float(args.translation_tangent_window_depths),
        energy_floor_relative=float(args.translation_tangent_energy_floor_relative),
        gravity=gravity,
    )
    mode_balanced_weight = float(args.mode_balanced_weight)
    mode_balanced_warmup_steps = int(args.mode_balanced_warmup_steps)
    mode_balanced_cfg = ModeBalancedConfig(
        k_max=float(args.mode_balanced_k_max),
        gravity=gravity,
        active_scale_relative=float(args.mode_balanced_active_scale_relative),
        denominator_floor_relative=float(args.mode_balanced_denominator_floor_relative),
    )
    hadamard_weight = float(args.hadamard_weight)
    hadamard_interval = int(args.hadamard_interval)
    hadamard_microbatch_local = int(args.hadamard_microbatch) // n_devices
    hadamard_warmup_steps = int(args.hadamard_warmup_steps)
    hadamard_cfg = HadamardRegConfig(
        k_max=float(args.hadamard_k_max),
        sobolev_order=int(args.hadamard_sobolev_order),
        relative_eps_min=float(args.hadamard_relative_eps_min),
        relative_eps_max=float(args.hadamard_relative_eps_max),
        eta_scale_floor=float(args.hadamard_eta_scale_floor),
        denominator_floor=float(args.hadamard_denominator_floor),
    )
    _, _k_grid = build_grid(nx, domain_length)
    k_grid_jax = jnp.asarray(_k_grid, dtype=training_dtype)
    k_rfft_jax = jnp.abs(k_grid_jax[: nx // 2 + 1])

    hadamard_metric_names = (
        "hadamard_active",
        "hadamard_loss",
        "hadamard_defect_rms",
        "hadamard_residual_hs_rms",
        "hadamard_forcing_hs_rms",
        "hadamard_secant_hs_rms",
        "hadamard_relative_eps",
        "hadamard_eta_scale",
        "hadamard_extra",
        "hadamard_warmup",
        "hadamard_weight_eff",
    )
    translation_tangent_metric_names = (
        "translation_tangent_active",
        "translation_tangent_loss",
        "translation_tangent_extra",
        "translation_tangent_speed_abs_error",
        "translation_tangent_speed_rel_error",
        "translation_tangent_selected_samples",
    )
    mode_balanced_metric_names = (
        "mode_balanced_active",
        "mode_balanced_loss",
        "mode_balanced_extra",
        "mode_balanced_weight_eff",
        "mode_balanced_unweighted_loss",
        "mode_balanced_effective_frequency_squared_mean",
        "mode_balanced_relative_error_rms",
        "mode_balanced_active_modes",
        "mode_balanced_clipped_mode_fraction",
    )

    def _train_step_body(
        current_state: train_state.TrainState,
        rng_key: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ):
        eta = eta.astype(training_dtype)
        xi = xi.astype(training_dtype)
        gxi = gxi.astype(training_dtype)
        batch_depth = batch_depth.astype(training_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        batch_targets = norm_targets_jax(gxi)
        log_h = jnp.minimum(batch_depth[:, 0], log_h_max)
        h_phys = jnp.exp(log_h)

        def loss_for_params(current_params):
            predictions_0 = current_state.apply_fn(
                {"params": current_params}, batch_inputs, batch_depth
            )
            data_loss = loss_fn(predictions_0, batch_targets)
            extra = jnp.asarray(0.0, dtype=training_dtype)
            zero = jnp.asarray(0.0, dtype=training_dtype)
            hadamard_metrics = jnp.zeros(
                (len(hadamard_metric_names),),
                dtype=training_dtype,
            )
            translation_tangent_metrics = jnp.zeros(
                (len(translation_tangent_metric_names),),
                dtype=training_dtype,
            )
            mode_balanced_metrics = jnp.zeros(
                (len(mode_balanced_metric_names),),
                dtype=training_dtype,
            )

            if mode_balanced_weight > 0.0:
                gxi_pred_phys_mode = denorm_targets_jax(predictions_0)[..., 0]
                gxi_target_phys_mode = denorm_targets_jax(batch_targets)[..., 0]
                loss_mode, mode_diagnostics = compute_mode_balanced_loss(
                    eta=eta,
                    gxi_prediction=gxi_pred_phys_mode,
                    gxi_target=gxi_target_phys_mode,
                    depth=h_phys,
                    k_rfft=k_rfft_jax,
                    config=mode_balanced_cfg,
                )
                step_f = jnp.asarray(current_state.step, dtype=training_dtype)
                mode_warmup = jnp.minimum(
                    step_f
                    / jnp.asarray(
                        max(mode_balanced_warmup_steps, 1), dtype=training_dtype
                    ),
                    1.0,
                )
                mode_weight_eff = (
                    jnp.asarray(mode_balanced_weight, dtype=training_dtype)
                    * mode_warmup
                )
                mode_extra = mode_weight_eff * loss_mode
                extra = extra + mode_extra
                mode_balanced_metrics = jnp.stack(
                    (
                        jnp.asarray(1.0, dtype=training_dtype),
                        loss_mode,
                        mode_extra,
                        mode_weight_eff,
                        mode_diagnostics["unweighted_loss"],
                        mode_diagnostics["effective_frequency_squared_mean"],
                        mode_diagnostics["relative_error_rms"],
                        mode_diagnostics["active_modes"],
                        mode_diagnostics["clipped_mode_fraction"],
                    )
                )

            if translation_tangent_weight > 0.0:
                gxi_pred_phys_tan = denorm_targets_jax(predictions_0)[..., 0]
                loss_tangent, tangent_diagnostics = compute_translation_tangent_loss(
                    eta=eta,
                    gxi_prediction=gxi_pred_phys_tan,
                    gxi_target=gxi,
                    depth=h_phys,
                    k=k_grid_jax,
                    config=translation_tangent_cfg,
                )
                local_selected = tangent_diagnostics["selected_samples"]
                global_selected = jax.lax.psum(local_selected, axis_name="batch")
                device_count = jax.lax.psum(
                    jnp.asarray(1.0, dtype=training_dtype), axis_name="batch"
                )
                shard_weight = (
                    device_count
                    * local_selected
                    / jnp.maximum(
                        global_selected, jnp.asarray(1.0, dtype=training_dtype)
                    )
                )
                loss_tangent = loss_tangent * shard_weight
                tangent_extra = translation_tangent_weight * loss_tangent
                extra = extra + tangent_extra
                translation_tangent_metrics = jnp.stack(
                    (
                        jnp.asarray(1.0, dtype=training_dtype),
                        loss_tangent,
                        tangent_extra,
                        tangent_diagnostics["speed_abs_error"] * shard_weight,
                        tangent_diagnostics["speed_relative_error"] * shard_weight,
                        global_selected / device_count,
                    )
                )
            if hadamard_weight > 0.0:
                current_step = jnp.asarray(current_state.step)
                step_active = jnp.equal(
                    current_step
                    % jnp.asarray(hadamard_interval, dtype=current_step.dtype),
                    jnp.asarray(0, dtype=current_step.dtype),
                )

                def _hadamard_active_branch(_):
                    rng_base = jax.random.fold_in(
                        rng_key,
                        jax.lax.axis_index("batch") * 53 + 127,
                    )
                    rng_hadamard = jax.random.fold_in(rng_base, current_state.step)
                    rng_perm, rng_probe = jax.random.split(rng_hadamard)
                    eta_sub, xi_sub, depth_sub = sample_microbatch(
                        rng_perm,
                        eta,
                        xi,
                        h_phys,
                        hadamard_microbatch_local,
                    )
                    batch_depth_sub = jnp.log(depth_sub)[:, None].astype(jnp.float64)
                    loss_hadamard, diagnostics = compute_hadamard_reg(
                        rng=rng_probe,
                        apply_fn=current_state.apply_fn,
                        model_params=current_params,
                        eta_phys=eta_sub,
                        xi_phys=xi_sub,
                        batch_depth_local=batch_depth_sub,
                        norm_inputs_fn=norm_inputs_jax,
                        denorm_targets_fn=denorm_targets_jax,
                        k=k_grid_jax,
                        cfg=hadamard_cfg,
                        dtype=jnp.float64,
                    )
                    step_f = jnp.asarray(current_state.step, dtype=jnp.float64)
                    warmup = jnp.minimum(
                        step_f
                        / jnp.asarray(max(hadamard_warmup_steps, 1), dtype=jnp.float64),
                        jnp.asarray(1.0, dtype=jnp.float64),
                    )
                    weight_eff = (
                        jnp.asarray(hadamard_weight, dtype=jnp.float64) * warmup
                    )
                    hadamard_extra = weight_eff * loss_hadamard
                    metrics = jnp.stack(
                        [
                            jnp.asarray(1.0, dtype=jnp.float64),
                            diagnostics["hadamard_loss"],
                            diagnostics["hadamard_defect_rms"],
                            diagnostics["hadamard_residual_hs_rms"],
                            diagnostics["hadamard_forcing_hs_rms"],
                            diagnostics["hadamard_secant_hs_rms"],
                            diagnostics["hadamard_relative_eps"],
                            diagnostics["hadamard_eta_scale"],
                            hadamard_extra,
                            warmup,
                            weight_eff,
                        ]
                    ).astype(training_dtype)
                    return hadamard_extra.astype(training_dtype), metrics

                def _hadamard_skip_branch(_):
                    return zero, jnp.zeros(
                        (len(hadamard_metric_names),),
                        dtype=training_dtype,
                    )

                hadamard_extra, hadamard_metrics = jax.lax.cond(
                    step_active,
                    _hadamard_active_branch,
                    _hadamard_skip_branch,
                    operand=None,
                )
                extra = extra + hadamard_extra

            return data_loss + extra, (
                hadamard_metrics,
                translation_tangent_metrics,
                mode_balanced_metrics,
            )

        (
            (
                loss_value,
                (
                    hadamard_metrics,
                    translation_tangent_metrics,
                    mode_balanced_metrics,
                ),
            ),
            grads,
        ) = jax.value_and_grad(loss_for_params, has_aux=True)(current_state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_value = jax.lax.pmean(loss_value, axis_name="batch")
        hadamard_metrics = jax.lax.pmean(hadamard_metrics, axis_name="batch")
        translation_tangent_metrics = jax.lax.pmean(
            translation_tangent_metrics,
            axis_name="batch",
        )
        mode_balanced_metrics = jax.lax.pmean(
            mode_balanced_metrics,
            axis_name="batch",
        )
        next_state = current_state.apply_gradients(grads=grads)
        next_state = next_state.replace(
            params=jax.tree.map(
                lambda x: jax.lax.all_gather(x, axis_name="batch", tiled=False)[0],
                next_state.params,
            )
        )
        return (
            next_state,
            loss_value,
            hadamard_metrics,
            translation_tangent_metrics,
            mode_balanced_metrics,
        )

    train_steps = {
        partition: jax.jit(
            shard_map(
                _train_step_body,
                mesh=mesh,
                in_specs=(P(), P(), partition, partition, partition, partition),
                out_specs=(P(), P(), P(), P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }

    def _eval_loss_body(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
    ):
        eta = eta.astype(training_dtype)
        xi = xi.astype(training_dtype)
        gxi = gxi.astype(training_dtype)
        batch_depth = batch_depth.astype(training_dtype)
        batch_inputs = norm_inputs_jax(eta, xi)
        batch_targets = norm_targets_jax(gxi)
        predictions = cast(
            jax.Array,
            model.apply({"params": current_params}, batch_inputs, batch_depth),
        )
        data_loss = loss_fn(predictions, batch_targets)
        log_h = jnp.minimum(batch_depth[:, 0], log_h_max)
        h_phys = jnp.exp(log_h)
        mode_loss = jnp.asarray(0.0, dtype=training_dtype)
        if mode_balanced_weight > 0.0:
            mode_loss, _ = compute_mode_balanced_loss(
                eta=eta,
                gxi_prediction=denorm_targets_jax(predictions)[..., 0],
                gxi_target=denorm_targets_jax(batch_targets)[..., 0],
                depth=h_phys,
                k_rfft=k_rfft_jax,
                config=mode_balanced_cfg,
            )
        tangent_loss = jnp.asarray(0.0, dtype=training_dtype)
        if translation_tangent_weight > 0.0:
            tangent_loss, tangent_diagnostics = compute_translation_tangent_loss(
                eta=eta,
                gxi_prediction=denorm_targets_jax(predictions)[..., 0],
                gxi_target=gxi,
                depth=h_phys,
                k=k_grid_jax,
                config=translation_tangent_cfg,
            )
            local_selected = tangent_diagnostics["selected_samples"]
            global_selected = jax.lax.psum(local_selected, axis_name="batch")
            device_count = jax.lax.psum(
                jnp.asarray(1.0, dtype=training_dtype), axis_name="batch"
            )
            shard_weight = (
                device_count
                * local_selected
                / jnp.maximum(
                    global_selected,
                    jnp.asarray(1.0, dtype=training_dtype),
                )
            )
            tangent_loss = tangent_loss * shard_weight
        return (
            jax.lax.pmean(data_loss, axis_name="batch"),
            jax.lax.pmean(mode_loss, axis_name="batch"),
            jax.lax.pmean(tangent_loss, axis_name="batch"),
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

    # Auto-resume from the current run's per-epoch checkpoint so Modal preemption
    # does not lose progress.
    latest_ckpt_dir = run_dir / "latest_ckpt"
    if (latest_ckpt_dir / "metadata.json").exists():
        meta = read_committed_checkpoint_metadata(latest_ckpt_dir)
        resume_epoch = int(meta["epoch"])
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
        history = list(meta["history"])
        best_val_loss = meta["best_val_loss"]
        best_epoch = meta["best_epoch"]
        start_epoch = int(meta["epoch"]) + 1
        print(
            f"auto-resume: restored from {latest_ckpt_dir} at epoch {meta['epoch']}, "
            f"resuming at epoch {start_epoch}"
        )

    training_counter_values(
        training_state,
        context="training startup",
        announce=True,
    )
    assert_pytree_replicated(training_state, name="training startup state")

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)
    train_rng = jax.random.PRNGKey(args.seed + 1)

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        epoch_start_step, epoch_start_optimizer_count = training_counter_values(
            training_state,
            context=f"epoch {epoch} start",
        )
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_iter = get_batches(
            dataset["eta"],
            dataset["xi"],
            dataset["gxi"],
            dataset["depth"],
            train_indices,
            args.batch_size,
            epoch_rng,
            device_count=n_devices,
        )
        batch_iter = (
            (eta_b, xi_b, gxi_b, depth_b)
            for eta_b, xi_b, gxi_b, depth_b, _ in batch_iter
        )
        batch_iter = device_prefetch(batch_iter, sharding=data_sharding, depth=2)
        batch_losses: list[float] = []
        train_batch_sizes: list[int] = []
        hadamard_metric_sums = np.zeros(
            (len(hadamard_metric_names),),
            dtype=np.float64,
        )
        translation_tangent_metric_sums = np.zeros(
            (len(translation_tangent_metric_names),),
            dtype=np.float64,
        )
        mode_balanced_metric_sums = np.zeros(
            (len(mode_balanced_metric_names),),
            dtype=np.float64,
        )
        train_bar = tqdm(
            batch_iter,
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for eta_b, xi_b, gxi_b, depth_b in train_bar:
            train_batch_sizes.append(int(eta_b.shape[0]))
            train_step = train_steps[
                P("batch") if eta_b.shape[0] % n_devices == 0 else P()
            ]
            train_rng, step_key = jax.random.split(train_rng)
            (
                training_state,
                batch_loss,
                batch_hadamard_metrics,
                batch_translation_tangent_metrics,
                batch_mode_balanced_metrics,
            ) = train_step(training_state, step_key, eta_b, xi_b, gxi_b, depth_b)
            batch_loss_value = float(jax.device_get(batch_loss))
            batch_losses.append(batch_loss_value)
            hadamard_metrics_np = np.asarray(
                jax.device_get(batch_hadamard_metrics),
                dtype=np.float64,
            )
            tangent_metrics_np = np.asarray(
                jax.device_get(batch_translation_tangent_metrics),
                dtype=np.float64,
            )
            if eta_b.shape[0] % n_devices:
                tangent_metrics_np[-1] /= n_devices
            mode_metrics_np = np.asarray(
                jax.device_get(batch_mode_balanced_metrics),
                dtype=np.float64,
            )
            if translation_tangent_weight > 0.0:
                tangent_metrics_np[:-1] *= eta_b.shape[0]
                translation_tangent_metric_sums += tangent_metrics_np
            if mode_balanced_weight > 0.0:
                mode_balanced_metric_sums += mode_metrics_np * eta_b.shape[0]
            active_hadamard = bool(
                hadamard_weight > 0.0 and hadamard_metrics_np[0] > 0.5
            )
            batch_index = len(batch_losses) - 1
            current_step = epoch_start_step + batch_index
            if hadamard_weight > 0.0:
                expected_hadamard = current_step % hadamard_interval == 0
                if active_hadamard != expected_hadamard:
                    raise RuntimeError(
                        "Hadamard regularizer firing mismatch at "
                        f"epoch={epoch}, batch={batch_index}, step={current_step}: "
                        f"expected={expected_hadamard}, observed={active_hadamard}"
                    )
            if active_hadamard:
                hadamard_metric_sums += hadamard_metrics_np
            if active_hadamard:
                train_bar.set_postfix(
                    loss=batch_loss_value,
                    shape=float(hadamard_metrics_np[1]),
                    defect=float(hadamard_metrics_np[2]),
                )
            else:
                train_bar.set_postfix(loss=batch_loss_value)

        epoch_end_step, epoch_end_optimizer_count = training_counter_values(
            training_state,
            context=f"epoch {epoch} end",
        )
        completed_steps = len(batch_losses)
        if epoch_end_step - epoch_start_step != completed_steps:
            raise RuntimeError(
                f"epoch {epoch} TrainState.step advanced by "
                f"{epoch_end_step - epoch_start_step}, expected {completed_steps}"
            )
        if epoch_end_optimizer_count - epoch_start_optimizer_count != completed_steps:
            raise RuntimeError(
                f"epoch {epoch} optimizer count advanced by "
                f"{epoch_end_optimizer_count - epoch_start_optimizer_count}, "
                f"expected {completed_steps}"
            )
        train_loss = (
            float(np.average(batch_losses, weights=train_batch_sizes))
            if batch_losses
            else float("inf")
        )

        val_batches = get_batches(
            dataset["eta"],
            dataset["xi"],
            dataset["gxi"],
            dataset["depth"],
            val_indices,
            args.batch_size,
            None,
            device_count=n_devices,
        )
        val_batches = (
            (eta_b, xi_b, gxi_b, depth_b)
            for eta_b, xi_b, gxi_b, depth_b, _ in val_batches
        )
        val_batches = device_prefetch(val_batches, sharding=data_sharding, depth=2)
        val_data_losses: list[float] = []
        val_mode_losses: list[float] = []
        val_tangent_losses: list[float] = []
        val_batch_sizes: list[int] = []
        for eta_b, xi_b, gxi_b, depth_b in val_batches:
            val_batch_sizes.append(int(eta_b.shape[0]))
            eval_loss_step = eval_loss_steps[
                P("batch") if eta_b.shape[0] % n_devices == 0 else P()
            ]
            (
                data_bl,
                mode_bl,
                tangent_bl,
            ) = eval_loss_step(
                training_state.params,
                eta_b,
                xi_b,
                gxi_b,
                depth_b,
            )
            val_data_losses.append(float(jax.device_get(data_bl)))
            val_mode_losses.append(float(jax.device_get(mode_bl)))
            val_tangent_losses.append(float(jax.device_get(tangent_bl)))

        def mean_validation_batches(losses: list[float]) -> float:
            if not losses:
                return float("inf")
            return float(np.average(losses, weights=val_batch_sizes))

        val_data_loss = mean_validation_batches(val_data_losses)
        val_mode_loss = mean_validation_batches(val_mode_losses)
        val_tangent_loss = mean_validation_batches(val_tangent_losses)
        val_loss = (
            val_data_loss
            + mode_balanced_weight * val_mode_loss
            + translation_tangent_weight * val_tangent_loss
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l2_loss": val_data_loss,
            "val_mode_balanced_loss": val_mode_loss,
            "val_translation_tangent_loss": val_tangent_loss,
        }
        if hadamard_weight > 0.0:
            active_hadamard_batches = (
                float(hadamard_metric_sums[0]) if batch_losses else 0.0
            )
            epoch_record["hadamard_active_batches"] = active_hadamard_batches
            epoch_record["hadamard_active_fraction"] = (
                active_hadamard_batches / float(len(batch_losses))
                if batch_losses
                else 0.0
            )
            if active_hadamard_batches > 0.0:
                hadamard_means = hadamard_metric_sums / active_hadamard_batches
                for name, value in zip(hadamard_metric_names[1:], hadamard_means[1:]):
                    epoch_record[name] = float(value)
        if translation_tangent_weight > 0.0:
            tangent_means = translation_tangent_metric_sums / sum(train_batch_sizes)
            for name, value in zip(
                translation_tangent_metric_names[1:-1], tangent_means[1:-1]
            ):
                epoch_record[name] = float(value)
            epoch_record["translation_tangent_selected_samples"] = float(
                translation_tangent_metric_sums[-1] * n_devices
            )
        if mode_balanced_weight > 0.0:
            mode_means = mode_balanced_metric_sums / sum(train_batch_sizes)
            for name, value in zip(mode_balanced_metric_names[1:], mode_means[1:]):
                epoch_record[name] = float(value)
        history.append(epoch_record)
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")

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
        epoch=int(history[-1]["epoch"]),
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
