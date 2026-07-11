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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from util import assert_pytree_replicated, replicate_pytree_from_host  # noqa: E402


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
    print(f"[PASS] identical host broadcast across {len(jax.local_devices())} device(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
