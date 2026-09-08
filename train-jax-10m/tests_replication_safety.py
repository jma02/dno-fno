"""Regression test for replicated training-state broadcasts.

Run on at least two GPUs to exercise the failure mode directly::

    CUDA_VISIBLE_DEVICES=0,1 uv run python train-jax-10m/tests_replication_safety.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    assert_pytree_replicated,
    device_prefetch,
    get_batches,
    replicate_pytree_from_host,
)


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
    print(
        f"[PASS] host broadcast and bounded remainder batches across {len(jax.local_devices())} device(s)"
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


if __name__ == "__main__":
    raise SystemExit(main())
