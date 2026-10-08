"""Regression test for replicated training-state broadcasts.

Run on at least two GPUs to exercise the failure mode directly::

    CUDA_VISIBLE_DEVICES=0,1 uv run python train-jax-10m/tests/tests_replication_safety.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    assert_pytree_replicated,
    device_prefetch,
    get_batches,
    replicate_pytree_from_host,
)
from checkpoint_util import WeightEMATrainState, save_checkpoint  # noqa: E402


def _replica_payloads(value: jax.Array) -> list[np.ndarray]:
    return [np.asarray(shard.data) for shard in value.addressable_shards]


def test_device_backed_pytree_is_broadcast_identically() -> None:
    devices = np.asarray(jax.local_devices())
    mesh = Mesh(devices, axis_names=("replica",))
    replicated = NamedSharding(mesh, P())
    source = {
        "params": jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        "step": jnp.asarray(238_560, dtype=jnp.int32),
    }

    result = replicate_pytree_from_host(source, replicated)
    assert_pytree_replicated(result, name="test tree")
    for name, expected in source.items():
        expected_host = np.asarray(jax.device_get(expected))
        payloads = _replica_payloads(result[name])
        assert len(payloads) == len(devices)
        assert all(np.array_equal(payload, expected_host) for payload in payloads), (
            name,
            payloads,
        )


def main() -> int:
    test_device_backed_pytree_is_broadcast_identically()
    test_remainder_batches_preserve_rows_and_losses()
    test_weight_ema_preserves_raw_updates_and_checkpoint_continuity()
    print(
        f"[PASS] replicated state, bounded remainder batches, and weight EMA across {len(jax.local_devices())} device(s)"
    )
    return 0


def test_remainder_batches_preserve_rows_and_losses() -> None:
    mesh = Mesh(np.asarray(jax.local_devices()), axis_names=("batch",))
    sharding = NamedSharding(mesh, P("batch"))

    # This same scalar objective is checked unsharded and through both partitions.
    def loss(weight: jax.Array, values: jax.Array) -> jax.Array:
        return jnp.mean((weight * values - values) ** 2)

    def loss_and_gradient(weight: jax.Array, values: jax.Array):
        return jax.lax.pmean(jax.value_and_grad(loss)(weight, values), "batch")

    mapped = {
        partition: jax.jit(
            shard_map(
                loss_and_gradient,
                mesh=mesh,
                in_specs=(P(), partition),
                out_specs=(P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }
    replicated_reference = jax.jit(
        shard_map(
            loss_and_gradient,
            mesh=mesh,
            in_specs=(P(), P()),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    weight = jnp.asarray(0.8, dtype=jnp.float32)
    for count, batch_size in ((5, 4), (7, 4), (1023, 1024)):
        rows = np.arange(1, count * 4 + 1, dtype=np.float32).reshape(count, 4)
        raw = get_batches(
            rows,
            rows,
            rows,
            np.ones(count),
            np.arange(count),
            batch_size=batch_size,
            rng=None,
            device_count=mesh.size,
        )
        batches = list(device_prefetch((batch[:4] for batch in raw), sharding=sharding))
        np.testing.assert_array_equal(
            np.concatenate([batch[0] for batch in batches]), rows
        )
        losses, sizes = [], []
        for values, *_ in batches:
            partition = P("batch") if values.shape[0] % mesh.size == 0 else P()
            value, gradient = mapped[partition](weight, values)
            expected = jax.value_and_grad(loss)(weight, jnp.asarray(values))
            np.testing.assert_allclose((value, gradient), expected, rtol=2e-6)
            np.testing.assert_allclose(
                replicated_reference(weight, values), expected, rtol=2e-6
            )
            losses.append(float(value))
            sizes.append(values.shape[0])
        assert sum(sizes) == count
        assert all(size % mesh.size == 0 or size < mesh.size for size in sizes)
        np.testing.assert_allclose(
            np.average(losses, weights=sizes),
            loss(weight, jnp.asarray(rows)),
            rtol=2e-6,
        )


def test_weight_ema_preserves_raw_updates_and_checkpoint_continuity() -> None:
    def update(state: Any, key: jax.Array, values: jax.Array) -> tuple[Any, jax.Array]:
        key, draw = jax.random.split(key)
        target = jax.random.normal(draw, ())
        grads = {"weight": jnp.mean(2 * (state.params["weight"] * values - target) * values)}
        return state.apply_gradients(grads=jax.lax.pmean(grads, "batch")), key

    for count in sorted({1, len(jax.local_devices())}):
        mesh = Mesh(np.asarray(jax.local_devices()[:count]), axis_names=("batch",))
        replicated = NamedSharding(mesh, P())
        step = jax.jit(shard_map(
            update, mesh=mesh, in_specs=(P(), P(), P("batch")),
            out_specs=(P(), P()), check_rep=False,
        ))
        for decay, start in ((0.0, 0), (0.9, 0), (0.9, 4)):
            params = {"weight": jnp.asarray(0.8, dtype=jnp.float32)}
            tx = optax.chain(optax.ema(0.9, debias=True), optax.adamw(2e-5))
            raw = train_state.TrainState.create(apply_fn=lambda: None, params=params, tx=tx)
            ema = WeightEMATrainState.create(
                apply_fn=raw.apply_fn, params=params, tx=tx,
                weight_ema_params=params if decay else None,
                weight_ema_decay=decay, weight_ema_start_step=start,
            )
            raw, ema = replicate_pytree_from_host((raw, ema), replicated)
            key = jax.device_put(jax.random.PRNGKey(17), replicated)
            values = jax.device_put(jnp.arange(1, 9, dtype=jnp.float32), NamedSharding(mesh, P("batch")))
            expected_shadow = np.asarray(params["weight"])
            for index in range(1, 8):
                raw, raw_key = step(raw, key, values)
                ema, key = step(ema, key, values)
                jax.tree_util.tree_map(np.testing.assert_array_equal,
                                       (ema.params, ema.opt_state, ema.step, key),
                                       (raw.params, raw.opt_state, raw.step, raw_key))
                if decay:
                    if index == start + 1:
                        expected_shadow = np.asarray(raw.params["weight"])
                    elif index > start:
                        expected_shadow = decay * expected_shadow + (1 - decay) * np.asarray(raw.params["weight"])
                    np.testing.assert_allclose(ema.weight_ema_params["weight"], expected_shadow, rtol=2e-7)
                else:
                    assert ema.weight_ema_params is None
                assert_pytree_replicated(ema, name="weight EMA state")

            with TemporaryDirectory() as directory:
                manager = checkpoints.AsyncManager(max_workers=1)
                history = [{"epoch": 1.0, "train_loss": 0.5, "val_loss": 0.4}]
                destination = Path(directory) / "latest_ckpt"
                save_checkpoint(destination, ema, history, 0.4, 1, {"scale": 2.0}, manager)
                target = {"params": ema.params, "opt_state": ema.opt_state, "step": int(ema.step)}
                if decay:
                    target["weight_ema_params"] = ema.weight_ema_params
                restored = checkpoints.restore_checkpoint(
                    destination, target=target, step=1, prefix="ckpt_",
                    orbax_checkpointer=ocp.PyTreeCheckpointer(),
                )
                resumed = replicate_pytree_from_host(ema.replace(**restored), replicated)
                next_live, live_key = step(ema, key, values)
                next_resumed, resumed_key = step(resumed, key, values)
                jax.tree_util.tree_map(np.testing.assert_array_equal,
                                       (next_live, live_key), (next_resumed, resumed_key))
                if decay:
                    destination = Path(directory) / "weight_ema_ckpt"
                    save_checkpoint(destination, ema, history, 0.4, 1, {"scale": 2.0}, manager, weight_ema=True)
                    exported = checkpoints.restore_checkpoint(
                        destination, target=None, step=1, prefix="ckpt_",
                        orbax_checkpointer=ocp.PyTreeCheckpointer(),
                    )
                    assert set(exported) == {"params", "step"}
                    jax.tree_util.tree_map(np.testing.assert_array_equal, exported["params"], ema.weight_ema_params)
                    metadata = json.loads((destination / "metadata.json").read_text())
                    assert not {"val_loss", "train_loss", "history", "best_val_loss"}.intersection(metadata)
                    assert metadata["params_kind"] == "weight_ema"
                    assert metadata["epoch"] == 1 and metadata["stats"] == {"scale": 2.0}


if __name__ == "__main__":
    raise SystemExit(main())
