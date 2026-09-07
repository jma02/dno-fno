from __future__ import annotations

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

_TRAINING_MAP_DTYPES = {
    "trajectory_index": np.dtype(np.int32),
    "trajectory_accepted": np.dtype(np.bool_),
}
_TRAINING_MAP_FIELDS = (*_TRAINING_MAP_DTYPES, "trajectory_dataset_split")


@dataclass(frozen=True)
class DatasetLocation:
    dataset_shard_paths: tuple[Path, ...]
    trajectory_map_path: Path
    manifest: dict[str, object]


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


def _resolve_dataset_location(dataset_path: Path) -> DatasetLocation:
    """Resolve a paper-dataset view."""
    if not dataset_path.name.endswith(".dataset.json"):
        raise ValueError(
            f"Training requires a *.dataset.json manifest, got {dataset_path}"
        )

    manifest_raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise ValueError(f"Dataset manifest must contain a JSON object: {dataset_path}")
    manifest: dict[str, object] = manifest_raw
    shard_records = manifest.get("dataset_shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise ValueError(
            f"Dataset manifest requires a nonempty dataset_shards list: {dataset_path}"
        )
    resolved_shards: list[Path] = []
    for record in shard_records:
        if not isinstance(record, dict):
            raise TypeError("every dataset_shards entry must be an object")
        shard_name = record.get("path")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("every dataset shard requires a nonempty path")
        shard_path = (dataset_path.parent / shard_name).resolve()
        if not shard_path.exists():
            raise FileNotFoundError(f"Dataset shard is missing: {shard_path}")
        resolved_shards.append(shard_path)

    trajectory_map_name = manifest.get("trajectory_map_npz")
    if not isinstance(trajectory_map_name, str) or not trajectory_map_name:
        raise ValueError(
            f"Dataset manifest requires a nonempty trajectory_map_npz: {dataset_path}"
        )
    trajectory_map_path = (dataset_path.parent / trajectory_map_name).resolve()
    if not trajectory_map_path.exists():
        raise FileNotFoundError(
            "Dataset manifest requires a trajectory map, but the sidecar is missing: "
            f"{trajectory_map_path}"
        )

    return DatasetLocation(
        dataset_shard_paths=tuple(resolved_shards),
        trajectory_map_path=trajectory_map_path,
        manifest=manifest,
    )


def _load_trajectory_map_arrays(
    trajectory_map_path: Path,
    *,
    num_examples: int,
) -> dict[str, np.ndarray]:
    """Load only the row ownership needed to apply release-defined splits."""
    with np.load(trajectory_map_path, allow_pickle=False) as archive:
        missing = sorted(set(_TRAINING_MAP_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(
                f"Trajectory map {trajectory_map_path} is missing required arrays: {missing}"
            )
        arrays = {name: np.asarray(archive[name]) for name in _TRAINING_MAP_FIELDS}

    for name, expected_dtype in _TRAINING_MAP_DTYPES.items():
        array = arrays[name]
        if array.ndim != 1:
            raise ValueError(f"Trajectory-map array {name!r} must be one-dimensional")
        if array.dtype != expected_dtype:
            raise TypeError(
                f"Trajectory-map array {name!r} has dtype {array.dtype}; "
                f"expected {expected_dtype}"
            )

    trajectory_index = arrays["trajectory_index"]
    if trajectory_index.shape[0] != num_examples:
        raise ValueError(
            f"trajectory_index must have {num_examples} entries; got "
            f"{trajectory_index.shape[0]}"
        )

    accepted = arrays["trajectory_accepted"]
    dataset_split = arrays["trajectory_dataset_split"]
    if dataset_split.ndim != 1 or dataset_split.dtype.kind not in {"U", "S"}:
        raise TypeError(
            "trajectory_dataset_split must be a one-dimensional string array"
        )
    num_trajectories = accepted.shape[0]
    if dataset_split.shape[0] != num_trajectories:
        raise ValueError(
            "Trajectory acceptance and split tables must have equal lengths"
        )
    if num_examples and num_trajectories == 0:
        raise ValueError("Nonempty datasets require a nonempty trajectory table")
    if trajectory_index.size and (
        int(trajectory_index.min()) < 0
        or int(trajectory_index.max()) >= num_trajectories
    ):
        raise ValueError("trajectory_index contains an out-of-range table index")
    unknown_splits = set(map(str, dataset_split)).difference(
        split.value for split in DatasetSplit
    )
    if unknown_splits:
        raise ValueError(f"unknown dataset splits: {sorted(unknown_splits)}")

    return arrays


def _zero_mean_xi(xi: np.ndarray) -> np.ndarray:
    # G(eta) annihilates the xi mode-0 (Dirichlet-Neumann of a constant is zero),
    # so projecting xi to zero-mean leaves gxi unchanged while keeping the model
    # input distribution consistent with eval/rollout, which both project too.
    return xi - xi.mean(axis=1, keepdims=True)


def _load_paper_dataset_shards(
    location: DatasetLocation,
) -> dict[str, np.ndarray]:
    manifest = location.manifest
    shard_records = cast(list[dict[str, object]], manifest["dataset_shards"])
    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Dataset manifest requires grid")
    nx = grid.get("nx")
    length = grid.get("length")
    if (
        not isinstance(nx, int)
        or isinstance(nx, bool)
        or nx <= 0
        or not isinstance(length, (int, float))
        or isinstance(length, bool)
        or not np.isfinite(length)
        or length <= 0.0
    ):
        raise ValueError("Dataset manifest has an invalid spatial grid")
    field_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("eta", "xi", "gxi", "depth", "time")
    }
    for path, record in zip(location.dataset_shard_paths, shard_records):
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(field_parts) - set(archive.files))
            if missing:
                raise ValueError(f"Dataset shard {path} is missing {missing}")
            arrays = {name: np.asarray(archive[name]) for name in field_parts}
        eta = arrays["eta"]
        if eta.ndim != 2 or eta.shape[1] != nx:
            raise ValueError(
                f"Dataset shard {path} has eta shape {eta.shape}; expected (rows, {nx})"
            )
        row_count = eta.shape[0]
        if record.get("n_rows") != row_count:
            raise ValueError(f"Dataset shard row count does not match: {path}")
        if arrays["xi"].shape != eta.shape or arrays["gxi"].shape != eta.shape:
            raise ValueError(f"Dataset shard fields have inconsistent shapes: {path}")
        if arrays["depth"].shape != (row_count,) or arrays["time"].shape != (
            row_count,
        ):
            raise ValueError(f"Dataset shard scalars have inconsistent shapes: {path}")
        for name, array in arrays.items():
            field_parts[name].append(np.asarray(array, dtype=np.float32))
    eta = np.concatenate(field_parts["eta"], axis=0)
    dataset = {
        "eta": eta,
        "xi": _zero_mean_xi(np.concatenate(field_parts["xi"], axis=0)),
        "gxi": np.concatenate(field_parts["gxi"], axis=0),
        "depth": np.concatenate(field_parts["depth"], axis=0),
        "time": np.concatenate(field_parts["time"], axis=0),
        "x": np.linspace(
            0.0,
            float(length),
            nx,
            endpoint=False,
            dtype=np.float32,
        ),
        "domain_length": float(length),
    }
    trajectory_map = _load_trajectory_map_arrays(
        location.trajectory_map_path,
        num_examples=eta.shape[0],
    )
    trajectory_index = trajectory_map["trajectory_index"]
    dataset["accepted_mask"] = trajectory_map["trajectory_accepted"][trajectory_index]
    dataset["dataset_split"] = trajectory_map["trajectory_dataset_split"][
        trajectory_index
    ]
    return dataset


def load_dataset_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    """Load one paper-dataset view."""
    location = _resolve_dataset_location(dataset_path)
    return _load_paper_dataset_shards(location)


def build_dataset_split_indices(
    dataset: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every accepted row from the dataset's preassigned splits."""
    dataset_split = np.asarray(dataset["dataset_split"])
    eligible = np.asarray(dataset["accepted_mask"])
    rng = np.random.default_rng(seed)
    return (
        rng.permutation(
            np.flatnonzero(eligible & (dataset_split == DatasetSplit.TRAIN.value))
        ),
        rng.permutation(
            np.flatnonzero(eligible & (dataset_split == DatasetSplit.VALIDATION.value))
        ),
        rng.permutation(
            np.flatnonzero(eligible & (dataset_split == DatasetSplit.TEST.value))
        ),
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
) -> tuple[float, float, float]:
    if indices is None:
        return (
            float(np.min(array)),
            float(np.max(array)),
            float(np.max(np.abs(array))),
        )

    minimum = np.inf
    maximum = -np.inf
    absmax = 0.0
    for start in range(0, indices.shape[0], chunk_size):
        selected = array[indices[start : start + chunk_size]]
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
    location = _resolve_dataset_location(dataset_path.resolve())
    input_file_state: list[dict[str, int]] = []
    for path in (
        dataset_path.resolve(),
        location.trajectory_map_path,
        *location.dataset_shard_paths,
    ):
        state = path.stat()
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
        selection = {"count": int(indices.shape[0])}

    stats_path = dataset_path.with_suffix(".stats.json")
    if stats_path.exists():
        cached: object = None
        with suppress(json.JSONDecodeError, OSError):
            cached = json.loads(stats_path.read_text(encoding="utf-8"))
        cached_selection = (
            cached.get("index_selection") if isinstance(cached, dict) else None
        )
        selection_matches = (
            cached_selection is None
            if selection is None
            else (
                isinstance(cached_selection, dict)
                and cached_selection.get("count") == selection["count"]
            )
        )
        if (
            isinstance(cached, dict)
            and _stats_valid(cached)
            and selection_matches
            and cached.get("input_file_state") == input_file_state
        ):
            return {
                **cached,
                "index_selection": selection,
            }

    if dataset is None:
        dataset = load_dataset_arrays(dataset_path)
    num_examples = int(dataset["eta"].shape[0])

    eta_min, eta_max, eta_absmax = _compute_selected_extrema(dataset["eta"], indices)
    xi_min, xi_max, xi_absmax = _compute_selected_extrema(dataset["xi"], indices)
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
    drop_last: bool,
) -> Iterator[RawBatchTuple]:
    """Yield raw (eta, xi, gxi, depth_log, indices) batches. Normalization happens
    on-device inside the jitted step."""
    ordered_indices = np.array(indices, copy=True)
    if rng is not None:
        rng.shuffle(ordered_indices)

    limit = ordered_indices.shape[0]
    if drop_last:
        limit = (limit // batch_size) * batch_size
    ordered_indices = ordered_indices[:limit]

    for start in range(0, limit, batch_size):
        end = start + batch_size
        batch_indices = ordered_indices[start:end]
        yield (
            eta[batch_indices],
            xi[batch_indices],
            gxi[batch_indices],
            compute_log_depth(depth[batch_indices]),
            batch_indices,
        )


def device_prefetch(
    iterator: Iterable[tuple[Any, ...]],
    *,
    destinations: Sequence[Any | None],
    depth: int = 2,
) -> Iterator[tuple[Any, ...]]:
    """Prefetch batches onto the GPU in a background thread.

    For each tuple yielded by ``iterator``, fields whose corresponding entry in
    ``destinations`` is non-None are pushed via ``jax.device_put``; others pass
    through unchanged. Producer runs ahead by up to ``depth`` batches so the
    H2D copy overlaps with model compute.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel: object = object()

    def _producer() -> None:
        try:
            for item in iterator:
                pushed = tuple(
                    jax.device_put(field, dest) if dest is not None else field
                    for field, dest in zip(item, destinations)
                )
                q.put(pushed)
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
