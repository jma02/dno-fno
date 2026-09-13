"""Fine-tune C27 on a dataset, applying translation loss only to Tanaka rows."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, cast

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.evals.model_rollout import load_run  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from solver.solvers.dno_series_jax import build_grid  # noqa: E402
from checkpoint_util import save_checkpoint, training_counter_values  # noqa: E402
from hadamard_shape_regularizer import ApplyFn, compute_hadamard_loss  # noqa: E402
from losses import relative_l2_loss  # noqa: E402
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402
from translation_tangent_regularizer import compute_translation_tangent_loss  # noqa: E402
from util import (  # noqa: E402
    FlatParams,
    assert_pytree_replicated,
    build_dataset_split_indices,
    device_prefetch,
    get_batches,
    load_dataset_arrays,
    make_normalizers,
    replicate_pytree_from_host,
)

# eta, xi, Gxi, log-depth, and the Tanaka selection mask.
Batch = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def average_losses(records: list[tuple[int, np.ndarray]]) -> dict[str, float]:
    """Average each loss over the samples on which it was evaluated."""
    sizes = np.asarray([size for size, _ in records])
    values = np.stack([values for _, values in records])
    weights = np.column_stack(
        (sizes, sizes, sizes * values[:, 4], sizes * values[:, 5])
    )
    means = (weights * values[:, :4]).sum(axis=0) / np.maximum(weights.sum(axis=0), 1)
    return dict(
        zip(
            ("l2_loss", "mode_loss", "tangent_loss", "hadamard_loss"), map(float, means)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--translation_tangent_weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Fine-tuning requires a GPU")

    # Keep the source model and normalization; start a new optimizer and schedule.
    loaded = load_run(args.run_dir, checkpoint="final")
    config = loaded.config
    dataset_path = (args.dataset or Path(config["dataset"])).resolve()
    dataset = load_dataset_arrays(dataset_path)
    families = np.load(dataset_path / "family_id.npy", mmap_mode="r")
    train_indices, val_indices, _ = build_dataset_split_indices(dataset)
    norm_inputs, norm_targets, denorm_targets = make_normalizers(
        loaded.stats, loaded.norm_mode
    )
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices), ("batch",))
    sharding = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())
    microbatch = int(config["hadamard_microbatch"])
    if args.batch_size % len(devices) or microbatch % len(devices):
        raise ValueError("Batch and Hadamard microbatch sizes must divide across GPUs")
    full, remainder = divmod(train_indices.size, args.batch_size)
    steps_per_epoch = (
        full + bool(remainder // len(devices)) + bool(remainder % len(devices))
    )
    state = train_state.TrainState.create(
        apply_fn=loaded.model.apply,
        params=jax.tree.map(lambda value: value.astype(jnp.float32), loaded.params),
        tx=optax.adamw(
            optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch),
            weight_decay=float(config["weight_decay"]),
        ),
    )
    state = replicate_pytree_from_host(state, replicated)
    assert_pytree_replicated(state, name="fine-tuning initial state")
    args.output_dir.mkdir(parents=True)
    config = {
        **config,
        **vars(args),
        "run_dir": str(args.output_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "run_name": args.output_dir.name,
        "dataset": str(dataset_path),
        "train_examples": int(train_indices.size),
        "val_examples": int(val_indices.size),
        "finetune_from": str(args.run_dir.resolve()),
        "source_epoch": loaded.epoch,
        "total_epochs": args.epochs,
        "optimizer_reset": True,
        "lr_warmup_steps": 0,
        "mode_balanced_warmup_steps": 0,
        "hadamard_warmup_steps": 0,
        "translation_tangent_scope": "tanaka",
        "device_count": len(devices),
        "validation_precision": "float64",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    _, k = build_grid(dataset["x"].size, float(config["domain_length"]))
    mode_weight = float(config["mode_balanced_weight"])
    hadamard_weight = float(config["hadamard_weight"])
    hadamard_interval = int(config["hadamard_interval"])

    def batches(
        indices: np.ndarray, rng: np.random.Generator | None
    ) -> Iterator[tuple[np.ndarray, ...]]:
        raw = get_batches(
            dataset["eta"],
            dataset["xi"],
            dataset["gxi"],
            dataset["depth"],
            indices,
            args.batch_size,
            rng,
            device_count=len(devices),
        )
        for eta, xi, gxi, log_depth, rows in raw:
            if rng is None:
                # Validation starts from raw fields, before any float32 centering/log.
                eta, gxi = eta.astype(np.float64), gxi.astype(np.float64)
                xi = np.asarray(dataset["xi"][rows], dtype=np.float64)
                xi -= xi.mean(axis=-1, keepdims=True)
                log_depth = np.log(
                    np.maximum(dataset["depth"][rows].astype(np.float64), 1e-12)
                )[:, None]
            yield eta, xi, gxi, log_depth, families[rows] == PhysicalFamilyId.TANAKA

    def loss_components(params: FlatParams, batch: Batch) -> jax.Array:
        eta, xi, gxi, log_depth, tanaka = batch
        prediction = cast(
            jax.Array,
            loaded.model.apply({"params": params}, norm_inputs(eta, xi), log_depth),
        )
        target = norm_targets(gxi)
        physical_prediction = denorm_targets(prediction)[..., 0]
        depth = jnp.exp(jnp.minimum(log_depth[:, 0], np.log(5.0)))
        grid = k.astype(eta.dtype)
        mode = compute_mode_balanced_loss(
            eta,
            physical_prediction,
            denorm_targets(target)[..., 0],
            depth,
            jnp.abs(grid[: eta.shape[-1] // 2 + 1]),
        )
        tangent, selected = compute_translation_tangent_loss(
            eta,
            physical_prediction,
            gxi,
            depth,
            grid,
            sample_mask=tanaka,
        )
        # pmean below must average Tanaka samples, not equally weight GPU means.
        global_selected = jax.lax.psum(selected, "batch")
        tangent *= (
            jax.lax.psum(jnp.asarray(1, eta.dtype), "batch")
            * selected
            / jnp.maximum(global_selected, 1)
        )
        return jnp.stack(
            (
                relative_l2_loss(prediction, target),
                mode,
                tangent,
                selected / eta.shape[0],
            )
        )

    def hadamard_subset(
        params: FlatParams,
        rng: jax.Array,
        batch: Batch,
    ) -> jax.Array:
        eta, xi, _, log_depth, _ = batch
        permutation_key, probe_key = jax.random.split(rng)
        selected = jax.random.permutation(permutation_key, eta.shape[0])[
            : microbatch // len(devices)
        ]
        depth = jnp.exp(jnp.minimum(log_depth[selected, 0], np.log(5.0)))
        return compute_hadamard_loss(
            probe_key,
            cast(ApplyFn, loaded.model.apply),
            params,
            eta[selected],
            xi[selected],
            jnp.log(depth)[:, None].astype(jnp.float64),
            norm_inputs,
            denorm_targets,
            k.astype(eta.dtype),
            jnp.float64,
            sobolev_order=cast(Literal[0, 1], int(config["hadamard_sobolev_order"])),
        )

    def train_step(
        current: train_state.TrainState,
        rng: jax.Array,
        batch: Batch,
    ) -> tuple[train_state.TrainState, jax.Array]:
        def objective(params: FlatParams) -> tuple[jax.Array, jax.Array]:
            l2, mode, tangent, selected_fraction = loss_components(params, batch)
            active = (current.step % hadamard_interval == 0) & (hadamard_weight > 0)
            key = jax.random.fold_in(
                jax.random.fold_in(rng, jax.lax.axis_index("batch") * 53 + 127),
                current.step,
            )
            hadamard = jax.lax.cond(
                active,
                lambda key: hadamard_subset(params, key, batch),
                lambda _: jnp.float64(0),
                key,
            )
            physics = mode_weight * mode + args.translation_tangent_weight * tangent
            physics += jnp.float32(hadamard_weight * hadamard)
            # Selected fractions weight summaries; retain the actual training objective.
            return l2 + physics, jnp.stack(
                (l2, mode, tangent, hadamard, selected_fraction, active, l2 + physics)
            )

        (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
            current.params
        )
        current = current.apply_gradients(grads=jax.lax.pmean(grads, "batch"))
        current = current.replace(
            params=jax.tree.map(
                lambda value: jax.lax.all_gather(value, "batch")[0],
                current.params,
            )
        )
        return current, jax.lax.pmean(metrics, "batch")

    def eval_step(params: FlatParams, rng: jax.Array, batch: Batch) -> jax.Array:
        l2, mode, tangent, selected_fraction = loss_components(params, batch)
        hadamard = hadamard_subset(
            params, jax.random.fold_in(rng, jax.lax.axis_index("batch")), batch
        )
        return jax.lax.pmean(
            jnp.stack((l2, mode, tangent, hadamard, selected_fraction, 1)), "batch"
        )

    train_steps = {
        p: jax.jit(
            shard_map(
                train_step, mesh, (P(), P(), (p,) * 5), (P(), P()), check_rep=False
            )
        )
        for p in (P("batch"), P())
    }
    eval_steps = {
        p: jax.jit(
            shard_map(eval_step, mesh, (P(), P(), (p,) * 5), P(), check_rep=False)
        )
        for p in (P("batch"), P())
    }
    history: list[dict[str, float]] = []
    best_loss, best_epoch = float("inf"), 0
    checkpoint_manager = checkpoints.AsyncManager(max_workers=1)
    epoch_seeds = np.random.SeedSequence(args.seed).spawn(args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_records: list[tuple[int, np.ndarray]] = []
        progress = tqdm(
            device_prefetch(
                batches(train_indices, np.random.default_rng(epoch_seeds[epoch - 1])),
                sharding=sharding,
            ),
            total=steps_per_epoch,
            desc=f"Train {epoch}",
        )
        for batch in progress:
            partition = P("batch") if batch[0].shape[0] % len(devices) == 0 else P()
            state, metrics = train_steps[partition](
                state, jax.random.PRNGKey(args.seed + 1), batch
            )
            values = np.asarray(jax.device_get(metrics))
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite fine-tuning loss")
            train_records.append((batch[0].shape[0], values))
            progress.set_postfix(l2=values[0], tangent=values[2])

        training_counter_values(
            state, context=f"fine-tuning epoch {epoch}", announce=True
        )
        assert_pytree_replicated(state, name=f"fine-tuning epoch {epoch}")
        eval_params = jax.tree.map(
            lambda value: value.astype(jnp.float64), state.params
        )
        val_records: list[tuple[int, np.ndarray]] = []
        for index, batch in enumerate(
            tqdm(
                device_prefetch(batches(val_indices, None), sharding=sharding),
                desc=f"Validate {epoch}",
            )
        ):
            partition = P("batch") if batch[0].shape[0] % len(devices) == 0 else P()
            metrics = eval_steps[partition](
                eval_params,
                jax.random.fold_in(jax.random.PRNGKey(args.seed + 2), index),
                batch,
            )
            values = np.asarray(jax.device_get(metrics))
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite fine-tuning validation loss")
            val_records.append((batch[0].shape[0], values))
        train_metrics, val_metrics = map(average_losses, (train_records, val_records))
        record = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        record["train_loss"] = float(
            np.average(
                [values[6] for _, values in train_records],
                weights=[size for size, _ in train_records],
            )
        )
        record["val_loss"] = (
            val_metrics["l2_loss"]
            + mode_weight * val_metrics["mode_loss"]
            + args.translation_tangent_weight * val_metrics["tangent_loss"]
        )
        history.append(record)
        if record["val_loss"] < best_loss:
            best_loss, best_epoch = record["val_loss"], epoch
        for name in (
            "latest_ckpt",
            *(["best_val_ckpt"] if best_epoch == epoch else []),
            *(["final_ckpt"] if epoch == args.epochs else []),
        ):
            save_checkpoint(
                args.output_dir / name,
                state,
                history,
                best_loss,
                best_epoch,
                loaded.stats,
                checkpoint_manager,
            )
        with (args.output_dir / "train_log.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "epochs_completed": len(history),
                "runtime_seconds": perf_counter() - started,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
