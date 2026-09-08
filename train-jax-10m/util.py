from __future__ import annotations

import hashlib
import json
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from solver.gen_data.pipeline.types import DatasetSplit

FlatParams = dict[str, jax.Array]
StatsDict = dict[str, object]

_TRAINING_ARRAY_NAMES = ("eta", "xi", "gxi", "depth", "time", "dataset_split", "x")


def replicate_pytree_from_host(tree: Any, sharding: jax.sharding.Sharding) -> Any:
    """Broadcast every array leaf through host memory before replication.

    Passing an already device-backed pytree directly to ``jax.device_put`` with
    replicated sharding can preserve only the primary device's local buffer and
    leave secondary replicas zero-filled.  A host round trip gives JAX a complete
    value to broadcast to every addressable device.
    """
    host_tree = jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)),
        tree,
    )
    return jax.device_put(host_tree, sharding)


def assert_pytree_replicated(tree: Any, *, name: str) -> None:
    """Raise when any addressable replica differs from its primary copy."""
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        if not isinstance(leaf, jax.Array) or len(leaf.addressable_shards) < 2:
            continue
        payloads = [np.asarray(shard.data) for shard in leaf.addressable_shards]
        reference = payloads[0]
        if not all(
            np.array_equal(payload, reference, equal_nan=True)
            for payload in payloads[1:]
        ):
            raise RuntimeError(f"{name} leaf {path} has divergent device replicas")


def require_jax_devices() -> tuple[str, list[Any]]:
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"JAX GPU backend is required. Found {backend!r}.")
    devices = list(jax.local_devices())
    return backend, devices


def load_dataset_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    """Memory-map the saved arrays without copying the dataset into RAM."""
    dataset = {
        name: np.load(dataset_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in _TRAINING_ARRAY_NAMES
    }
    eta = dataset["eta"]
    x = dataset["x"]
    if eta.ndim != 2 or x.shape != (eta.shape[1],):
        raise ValueError("Dataset eta and spatial grid have inconsistent shapes")
    if any(dataset[name].shape != eta.shape for name in ("xi", "gxi")):
        raise ValueError("Dataset fields have inconsistent shapes")
    if any(
        dataset[name].shape != (eta.shape[0],)
        for name in ("depth", "time", "dataset_split")
    ):
        raise ValueError("Dataset row metadata has inconsistent shapes")
    if not np.isin(
        dataset["dataset_split"], tuple(split.value for split in DatasetSplit)
    ).all():
        raise ValueError("Dataset contains unknown splits")
    length = float((x[1] - x[0]) * x.size)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Dataset spatial grid has an invalid domain length")
    dataset["domain_length"] = np.asarray(length)
    return dataset


def _zero_mean_xi(xi: np.ndarray) -> np.ndarray:
    # G(eta) annihilates xi's constant mode. Center only gathered batches/chunks
    # so the read-only memory map stays unchanged and the full array is not copied.
    return xi - xi.mean(axis=1, keepdims=True)


def build_dataset_split_indices(
    dataset: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all rows from the dataset's simulation-level splits."""
    dataset_split = np.asarray(dataset["dataset_split"])
    return (
        np.flatnonzero(dataset_split == DatasetSplit.TRAIN.value),
        np.flatnonzero(dataset_split == DatasetSplit.VALIDATION.value),
        np.flatnonzero(dataset_split == DatasetSplit.TEST.value),
    )


def _stats_valid(s: dict[str, object]) -> bool:
    feature_min = s.get("feature_min")
    feature_max = s.get("feature_max")
    if not (
        isinstance(feature_min, (list, tuple))
        and isinstance(feature_max, (list, tuple))
        and len(feature_min) == 2
        and len(feature_max) == 2
    ):
        return False
    values = (*feature_min, *feature_max, s.get("target_min"), s.get("target_max"))
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
        for value in values
    )


def _compute_selected_extrema(
    array: np.ndarray,
    indices: np.ndarray | None,
    *,
    chunk_size: int = 65_536,
    zero_mean: bool = False,
) -> tuple[float, float, float]:
    minimum = np.inf
    maximum = -np.inf
    absmax = 0.0
    number_of_rows = array.shape[0] if indices is None else indices.shape[0]
    for start in range(0, number_of_rows, chunk_size):
        selected = (
            array[start : start + chunk_size]
            if indices is None
            else array[indices[start : start + chunk_size]]
        )
        if zero_mean:
            selected = _zero_mean_xi(selected)
        minimum = min(minimum, float(np.min(selected)))
        maximum = max(maximum, float(np.max(selected)))
        absmax = max(absmax, float(np.max(np.abs(selected))))
    return minimum, maximum, absmax


def load_or_compute_stats(
    dataset_path: Path,
    dataset: dict[str, np.ndarray] | None = None,
    *,
    indices: np.ndarray | None = None,
) -> StatsDict:
    input_file_state: list[dict[str, int]] = []
    for name in _TRAINING_ARRAY_NAMES:
        state = (dataset_path / f"{name}.npy").stat()
        input_file_state.append(
            {
                "size_bytes": state.st_size,
                "modified_ns": state.st_mtime_ns,
            }
        )
    selection: dict[str, object] | None = None
    if indices is not None:
        indices = np.asarray(indices)
        if indices.size == 0:
            raise ValueError(
                "Cannot compute normalization statistics from an empty selection"
            )
        selection = {
            "count": int(indices.shape[0]),
            "sha256": hashlib.sha256(
                np.sort(indices.astype(np.int64)).tobytes()
            ).hexdigest(),
        }

    stats_path = dataset_path / "stats.json"
    if stats_path.exists():
        cached: object = None
        with suppress(json.JSONDecodeError, OSError):
            cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and _stats_valid(cached)
            and cached.get("index_selection") == selection
            and cached.get("input_file_state") == input_file_state
        ):
            return cached

    if dataset is None:
        dataset = load_dataset_arrays(dataset_path)
    num_examples = int(dataset["eta"].shape[0])

    eta_min, eta_max, eta_absmax = _compute_selected_extrema(dataset["eta"], indices)
    xi_min, xi_max, xi_absmax = _compute_selected_extrema(
        dataset["xi"], indices, zero_mean=True
    )
    target_min, target_max, target_absmax = _compute_selected_extrema(
        dataset["gxi"], indices
    )
    depth_arr = np.asarray(dataset["depth"], dtype=np.float64)
    depth_min, depth_max, _ = _compute_selected_extrema(depth_arr, indices)
    log_depth = np.log(np.clip(depth_arr, 1e-12, None))
    log_depth_min, log_depth_max, _ = _compute_selected_extrema(log_depth, indices)
    stats: StatsDict = {
        "dataset": str(dataset_path),
        "num_examples": int(indices.shape[0]) if indices is not None else num_examples,
        "storage_num_examples": num_examples,
        "feature_min": [eta_min, xi_min],
        "feature_max": [eta_max, xi_max],
        "feature_absmax": [eta_absmax, xi_absmax],
        "target_min": target_min,
        "target_max": target_max,
        "target_absmax": target_absmax,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "log_depth_min": log_depth_min,
        "log_depth_max": log_depth_max,
        "domain_length": float(dataset["domain_length"]),
        "index_selection": selection,
        "input_file_state": input_file_state,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


@dataclass(frozen=True)
class NormStats:
    feature_min: np.ndarray
    feature_max: np.ndarray
    feature_absmax: np.ndarray
    target_min: float
    target_max: float
    target_absmax: float
    mode: str  # "minmax" or "scale"

    @staticmethod
    def from_dict(stats: StatsDict, mode: str = "minmax") -> NormStats:
        feature_absmax = np.asarray(
            cast(Sequence[float], stats["feature_absmax"]),
            dtype=np.float32,
        ).reshape((1, 1, 2))
        target_absmax = float(cast(float | int, stats["target_absmax"]))
        return NormStats(
            feature_min=np.asarray(
                cast(Sequence[float], stats["feature_min"]),
                dtype=np.float32,
            ).reshape((1, 1, 2)),
            feature_max=np.asarray(
                cast(Sequence[float], stats["feature_max"]),
                dtype=np.float32,
            ).reshape((1, 1, 2)),
            feature_absmax=np.where(feature_absmax > 0, feature_absmax, 1.0),
            target_min=float(cast(float | int, stats["target_min"])),
            target_max=float(cast(float | int, stats["target_max"])),
            target_absmax=target_absmax if target_absmax > 0 else 1.0,
            mode=mode,
        )


RawBatchTuple = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
InputNormalizer = Callable[[jax.Array, jax.Array], jax.Array]
ArrayNormalizer = Callable[[jax.Array], jax.Array]
NormalizerFunctions = tuple[InputNormalizer, ArrayNormalizer, ArrayNormalizer]


def compute_log_depth(depth: np.ndarray) -> np.ndarray:
    """Convert physical depth (B,) -> log-depth (B, 1) float32 for FiLM input."""
    arr = np.asarray(depth, dtype=np.float32).reshape(-1, 1)
    return np.log(np.clip(arr, 1e-12, None)).astype(np.float32)


def make_normalizers(ns: NormStats) -> NormalizerFunctions:
    """Return (norm_inputs, norm_targets, denorm_targets) jit-compatible functions
    that bake the stats as constants. Inputs/targets stay in fp32."""
    if ns.mode == "scale":
        feat_scale = jnp.asarray(ns.feature_absmax, dtype=jnp.float32)
        tgt_scale = jnp.float32(ns.target_absmax)

        def norm_inputs(eta: jax.Array, xi: jax.Array) -> jax.Array:
            stacked = jnp.stack((eta, xi), axis=-1)
            return stacked / feat_scale

        def norm_targets(gxi: jax.Array) -> jax.Array:
            return (gxi[..., None]) / tgt_scale

        def denorm_targets(arr: jax.Array) -> jax.Array:
            return arr * tgt_scale

    else:
        feat_min = jnp.asarray(ns.feature_min, dtype=jnp.float32)
        feat_range = jnp.asarray(
            ns.feature_max - ns.feature_min + 1e-8, dtype=jnp.float32
        )
        tgt_min = jnp.float32(ns.target_min)
        tgt_range = jnp.float32(ns.target_max - ns.target_min + 1e-8)

        def norm_inputs(eta: jax.Array, xi: jax.Array) -> jax.Array:
            stacked = jnp.stack((eta, xi), axis=-1)
            return ((stacked - feat_min) / feat_range) * 2.0 - 1.0

        def norm_targets(gxi: jax.Array) -> jax.Array:
            return ((gxi[..., None] - tgt_min) / tgt_range) * 2.0 - 1.0

        def denorm_targets(arr: jax.Array) -> jax.Array:
            return ((arr + 1.0) * 0.5) * tgt_range + tgt_min

    return norm_inputs, norm_targets, denorm_targets


def get_batches(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    depth: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None,
    *,
    device_count: int = 1,
) -> Iterator[RawBatchTuple]:
    """Yield raw (eta, xi, gxi, depth_log, indices) batches. Normalization happens
    on-device inside the jitted step. An indivisible tail is split into a sharded
    prefix and fewer than device_count replicated rows."""
    ordered_indices = np.array(indices, copy=True)
    if rng is not None:
        rng.shuffle(ordered_indices)

    for start in range(0, ordered_indices.size, batch_size):
        end = min(start + batch_size, ordered_indices.size)
        divided_end = end - (end - start) % device_count
        for batch_indices in (
            ordered_indices[start:divided_end],
            ordered_indices[divided_end:end],
        ):
            if batch_indices.size:
                yield (
                    eta[batch_indices],
                    _zero_mean_xi(xi[batch_indices]),
                    gxi[batch_indices],
                    compute_log_depth(depth[batch_indices]),
                    batch_indices,
                )


def device_prefetch(
    iterator: Iterable[tuple[Any, ...]],
    *,
    sharding: jax.sharding.NamedSharding,
    depth: int = 2,
) -> Iterator[tuple[Any, ...]]:
    """Prefetch batches onto the GPU in a background thread.

    A tail indivisible by the device count is replicated, without dropping or
    padding rows. Producer runs ahead so copying overlaps with model compute.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel: object = object()
    replicated = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())

    def _producer() -> None:
        try:
            for item in iterator:
                destination = (
                    sharding
                    if item[0].shape[0] % sharding.mesh.size == 0
                    else replicated
                )
                q.put(tuple(jax.device_put(field, destination) for field in item))
        except BaseException as exc:  # surface in main thread
            q.put(("__prefetch_error__", exc))
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and item[0] == "__prefetch_error__"
        ):
            raise item[1]
        yield item
