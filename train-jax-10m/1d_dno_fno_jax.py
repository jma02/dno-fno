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
MODEL_DIR = REPO_ROOT / "models" / "fno-jax"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fno1d import FNO1d
from losses import build_loss, count_params
from util import (
    FlatParams,
    NormStats,
    build_split_indices,
    evaluate,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    require_jax_devices,
    save_final_representative_plot,
    save_loss_history_plot,
)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    params: FlatParams,
    train_loss: float,
    val_loss: float,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
    stats: dict[str, object],
    async_manager: checkpoints.AsyncManager,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.tree_util.tree_map(
            lambda value: np.asarray(value),
            jax.device_get(params),
        ),
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
    parser.add_argument("--dataset", default="tanaka_1_clean_shards01_half_shuffled.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--loss", choices=("mse", "relative_l2", "lp", "sobolev"), default="relative_l2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

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
    stats = load_or_compute_stats(dataset_path, dataset=dataset)
    ns = NormStats.from_dict(stats)
    train_indices, val_indices, _ = build_split_indices(
        int(dataset["eta"].shape[0]),
        args.seed,
    )
    train_steps_per_epoch = train_indices.shape[0] // args.batch_size

    model = FNO1d(args.modes, args.width, args.n_blocks)
    with jax.default_device(eval_device):
        params = model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros((1, nx, 2), dtype=jnp.float32),
        )["params"]
    loss_fn = build_loss(args.loss)

    total_steps = args.epochs * train_steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=total_steps,
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    training_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    training_state = jax.device_put(training_state, replicated)
    checkpoint_async_manager = checkpoints.AsyncManager(max_workers=1)

    config_payload = {
        "dataset": args.dataset,
        "device": backend,
        "device_count": n_devices,
        "modes": args.modes,
        "width": args.width,
        "n_blocks": args.n_blocks,
        "loss": args.loss,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "param_count": count_params(params),
        "train_examples": int(train_indices.shape[0]),
        "val_examples": int(val_indices.shape[0]),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    def _train_step_body(current_state: train_state.TrainState, batch_inputs: jax.Array, batch_targets: jax.Array):
        def loss_for_params(current_params):
            predictions = current_state.apply_fn({"params": current_params}, batch_inputs)
            return loss_fn(predictions, batch_targets)

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
        in_specs=(P(), P("batch"), P("batch")),
        out_specs=(P(), P()),
        check_rep=False,
    ))

    @jax.jit
    def eval_step(current_params: FlatParams, batch_inputs: jax.Array, batch_targets: jax.Array):
        predictions = model.apply({"params": current_params}, batch_inputs)
        return loss_fn(predictions, batch_targets), predictions

    @jax.jit
    def predict_batch(current_params: FlatParams, batch_inputs: jax.Array) -> jax.Array:
        return model.apply({"params": current_params}, batch_inputs)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    train_loss = float("inf")

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_iter = get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"],
            train_indices, args.batch_size, ns, epoch_rng,
            shuffle=True, drop_last=True,
        )
        batch_losses: list[float] = []
        train_bar = tqdm(
            batch_iter,
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for batch_inputs, batch_targets, _ in train_bar:
            batch_inputs = jax.device_put(batch_inputs, data_sharding)
            batch_targets = jax.device_put(batch_targets, data_sharding)
            training_state, batch_loss = train_step(training_state, batch_inputs, batch_targets)
            batch_loss_value = float(jax.device_get(batch_loss))
            batch_losses.append(batch_loss_value)
            train_bar.set_postfix(loss=batch_loss_value)

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")

        eval_params = jax.device_put(training_state.params, eval_device)
        val_batches = get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"],
            val_indices, args.batch_size, ns, None,
            shuffle=False, drop_last=False,
        )
        val_loss, _ = evaluate(
            eval_params,
            batches=val_batches,
            dataset=dataset,
            ns=ns,
            device=eval_device,
            eval_step_fn=eval_step,
            predict_batch_fn=predict_batch,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")

        if not args.skip_plots and epoch % 10 == 0:
            save_loss_history_plot(run_dir, history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best_val_ckpt",
                epoch=epoch,
                params=jax.device_get(training_state.params),
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
        params=jax.device_get(training_state.params),
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
            dataset["eta"], dataset["xi"], dataset["gxi"],
            val_indices, args.batch_size, ns, None,
            shuffle=False, drop_last=False,
        )
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
