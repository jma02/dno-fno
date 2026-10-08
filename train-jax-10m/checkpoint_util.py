from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import numpy as np
import optax
from flax import struct
from flax.training import checkpoints
from flax.training import train_state
import orbax.checkpoint as ocp


class CheckpointMetadata(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    history: list[dict[str, float]]
    best_val_loss: float
    best_epoch: int
    stats: dict[str, object]


class WeightEMATrainState(train_state.TrainState):
    weight_ema_params: Any = None
    weight_ema_decay: float = struct.field(pytree_node=False, default=0.0)
    weight_ema_start_step: int = struct.field(pytree_node=False, default=0)

    def apply_gradients(self, *, grads: Any, **kwargs: Any) -> WeightEMATrainState:
        state = cast(WeightEMATrainState, super().apply_gradients(grads=grads, **kwargs))
        if self.weight_ema_decay > 0.0:
            shadow = jax.lax.cond(
                state.step > self.weight_ema_start_step,
                lambda: jax.lax.cond(
                    self.step == self.weight_ema_start_step,
                    lambda: state.params,
                    lambda: optax.incremental_update(
                        state.params, self.weight_ema_params, 1.0 - self.weight_ema_decay,
                    ),
                ),
                lambda: self.weight_ema_params,
            )
            state = state.replace(weight_ema_params=shadow)
        return state


def save_checkpoint(
    output_dir: Path,
    state: train_state.TrainState,
    history: list[dict[str, float]],
    best_val_loss: float,
    best_epoch: int,
    stats: dict[str, object],
    async_manager: checkpoints.AsyncManager,
    *,
    weight_ema: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if weight_ema:
        ema_state = cast(WeightEMATrainState, state)
        payload = {"params": jax.device_get(ema_state.weight_ema_params), "step": int(state.step)}
        metadata = {
            "epoch": int(history[-1]["epoch"]), "stats": stats,
            "params_kind": "weight_ema", "weight_ema_decay": ema_state.weight_ema_decay,
            "weight_ema_start_step": ema_state.weight_ema_start_step,
        }
    else:
        host_state = jax.device_get(state)
        payload = {
            "params": host_state.params, "opt_state": host_state.opt_state,
            "step": int(host_state.step),
        }
        if isinstance(host_state, WeightEMATrainState) and host_state.weight_ema_params is not None:
            payload["weight_ema_params"] = host_state.weight_ema_params
        metadata = {
            "epoch": int(history[-1]["epoch"]),
            "train_loss": history[-1]["train_loss"], "val_loss": history[-1]["val_loss"],
            "history": history, "best_val_loss": best_val_loss,
            "best_epoch": best_epoch, "stats": stats,
        }
    payload = {
        name: value if name == "step" else jax.tree_util.tree_map(np.asarray, value)
        for name, value in payload.items()
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
