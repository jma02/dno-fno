from __future__ import annotations

import json
import queue
import threading
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

FlatParams = dict[str, jax.Array]
StatsDict = dict[str, object]


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


def require_jax_devices(*, allow_cpu: bool = False, min_device_count: int = 1) -> tuple[str, list[jax.Device]]:
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise RuntimeError(f"JAX GPU backend is required. Found {backend!r}.")
    devices = list(jax.local_devices())
    if len(devices) < min_device_count:
        raise RuntimeError(f"Expected at least {min_device_count} local JAX devices, found {len(devices)}.")
    return backend, devices


def _read_dataset_meta(dataset_path: Path) -> tuple[dict, bool]:
    """Return (meta, is_flat). Combined-flat datasets have a sidecar
    ``<name>.meta.json`` next to the npz; legacy shard datasets carry
    meta.json inside the zip."""
    sidecar = dataset_path.with_suffix(".meta.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8")), True
    with zipfile.ZipFile(dataset_path, mode="r") as zf:
        return json.loads(zf.read("meta.json")), False


def _zero_mean_xi(xi: np.ndarray) -> np.ndarray:
    # G(eta) annihilates the xi mode-0 (Dirichlet-Neumann of a constant is zero),
    # so projecting xi to zero-mean leaves gxi unchanged while keeping the model
    # input distribution consistent with eval/rollout, which both project too.
    return xi - xi.mean(axis=1, keepdims=True)


def load_dataset_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    meta, is_flat = _read_dataset_meta(dataset_path)
    domain_length = float(meta.get("length", 2.0 * np.pi))

    if is_flat:
        with np.load(dataset_path) as archive:
            eta = np.asarray(archive["eta"], dtype=np.float32)
            xi = np.asarray(archive["xi"], dtype=np.float32)
            gxi = np.asarray(archive["gxi"], dtype=np.float32)
            depth = np.asarray(archive["depth"], dtype=np.float32)
            time_arr = (
                np.asarray(archive["time"], dtype=np.float32)
                if "time" in archive.files
                else np.zeros((eta.shape[0],), dtype=np.float32)
            )
            source = (
                np.asarray(archive["source"], dtype=np.int8)
                if "source" in archive.files
                else np.zeros((eta.shape[0],), dtype=np.int8)
            )
            x = np.asarray(archive["x"], dtype=np.float32)
        return {
            "eta": eta, "xi": _zero_mean_xi(xi), "gxi": gxi,
            "depth": depth, "time": time_arr, "source": source,
            "case_id": np.arange(eta.shape[0], dtype=np.int64),
            "x": x, "domain_length": domain_length,
        }

    with np.load(dataset_path) as archive:
        eta_parts: list[np.ndarray] = []
        xi_parts: list[np.ndarray] = []
        gxi_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        case_id_parts: list[np.ndarray] = []
        depth_parts: list[np.ndarray] = []

        n_shards = int(meta["n_batches_planned"])
        for shard_id in range(n_shards):
            tag = f"{shard_id:04d}"
            eta_parts.append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            xi_parts.append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            gxi_parts.append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
            time_parts.append(np.asarray(archive[f"time_batch_{tag}"], dtype=np.float32))
            case_id_parts.append(np.asarray(archive[f"case_id_batch_{tag}"], dtype=np.int64))
            depth_key = f"depth_batch_{tag}"
            if depth_key in archive.files:
                depth_parts.append(np.asarray(archive[depth_key], dtype=np.float32))

        x = np.asarray(archive["x"], dtype=np.float32)

    eta_all = np.concatenate(eta_parts, axis=0)
    if depth_parts:
        depth = np.concatenate(depth_parts, axis=0)
    else:
        # Legacy Tanaka-style datasets predate the depth field; convention is h=1.
        depth = np.full((eta_all.shape[0],), float(meta.get("depth", 1.0)), dtype=np.float32)

    return {
        "eta": eta_all,
        "xi": _zero_mean_xi(np.concatenate(xi_parts, axis=0)),
        "gxi": np.concatenate(gxi_parts, axis=0),
        "time": np.concatenate(time_parts, axis=0),
        "case_id": np.concatenate(case_id_parts, axis=0),
        "depth": depth,
        "x": x,
        "domain_length": domain_length,
    }

def build_split_indices(num_examples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(num_examples)
    val_count = int(num_examples * 0.1)
    test_count = int(num_examples * 0.1)
    train_count = num_examples - val_count - test_count
    train_indices = permutation[:train_count]
    val_indices = permutation[train_count : train_count + val_count]
    test_indices = permutation[train_count + val_count :]
    return train_indices, val_indices, test_indices


def _stats_valid(s: dict[str, object]) -> bool:
    valid = False
    with suppress(KeyError, TypeError):
        valid = (
            len(s["feature_min"]) == 2
            and len(s["feature_max"]) == 2
            and all(np.isfinite(v) for v in (*s["feature_min"], *s["feature_max"], s["target_min"], s["target_max"]))
        )
    return valid


def load_or_compute_stats(
    dataset_path: Path,
    dataset: dict[str, np.ndarray] | None = None,
) -> StatsDict:
    stats_path = dataset_path.with_suffix(".stats.json")
    if stats_path.exists():
        cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if _stats_valid(cached):
            return cached

    if dataset is None:
        dataset = load_dataset_arrays(dataset_path)
    depth_arr = np.asarray(dataset["depth"], dtype=np.float64)
    log_depth = np.log(np.clip(depth_arr, 1e-12, None))
    stats: StatsDict = {
        "dataset": str(dataset_path),
        "num_examples": int(dataset["eta"].shape[0]),
        "feature_min": [
            float(np.min(dataset["eta"])),
            float(np.min(dataset["xi"])),
        ],
        "feature_max": [
            float(np.max(dataset["eta"])),
            float(np.max(dataset["xi"])),
        ],
        "feature_absmax": [
            float(np.max(np.abs(dataset["eta"]))),
            float(np.max(np.abs(dataset["xi"]))),
        ],
        "target_min": float(np.min(dataset["gxi"])),
        "target_max": float(np.max(dataset["gxi"])),
        "target_absmax": float(np.max(np.abs(dataset["gxi"]))),
        "depth_min": float(np.min(depth_arr)),
        "depth_max": float(np.max(depth_arr)),
        "log_depth_min": float(np.min(log_depth)),
        "log_depth_max": float(np.max(log_depth)),
        "domain_length": float(dataset.get("domain_length", 2.0 * np.pi)),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def normalize_minmax(
    array: np.ndarray,
    lower: float,
    upper: float,
    value_min: np.ndarray | float,
    value_max: np.ndarray | float,
) -> np.ndarray:
    value_min = np.asarray(value_min, dtype=np.float32)
    value_max = np.asarray(value_max, dtype=np.float32)
    scaled = (array - value_min) / (value_max - value_min + 1e-8)
    return (scaled * (upper - lower) + lower).astype(np.float32)


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
            stats.get("feature_absmax", [1.0, 1.0]), dtype=np.float32,
        ).reshape((1, 1, 2))
        target_absmax = float(stats.get("target_absmax", 1.0))
        return NormStats(
            feature_min=np.asarray(stats["feature_min"], dtype=np.float32).reshape((1, 1, 2)),
            feature_max=np.asarray(stats["feature_max"], dtype=np.float32).reshape((1, 1, 2)),
            feature_absmax=np.where(feature_absmax > 0, feature_absmax, 1.0),
            target_min=float(stats["target_min"]),
            target_max=float(stats["target_max"]),
            target_absmax=target_absmax if target_absmax > 0 else 1.0,
            mode=mode,
        )


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: NormStats) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32, copy=False)
    if ns.mode == "scale":
        return (stacked / ns.feature_absmax).astype(np.float32)
    return normalize_minmax(stacked, lower=-1.0, upper=1.0, value_min=ns.feature_min, value_max=ns.feature_max)


def normalize_targets(gxi: np.ndarray, ns: NormStats) -> np.ndarray:
    targets = gxi[..., None].astype(np.float32, copy=False)
    if ns.mode == "scale":
        return (targets / ns.target_absmax).astype(np.float32)
    return normalize_minmax(
        targets,
        lower=-1.0,
        upper=1.0,
        value_min=ns.target_min,
        value_max=ns.target_max,
    )


def denormalize_targets(array: np.ndarray, ns: NormStats) -> np.ndarray:
    if ns.mode == "scale":
        return array * ns.target_absmax
    return ((array + 1.0) * 0.5) * (ns.target_max - ns.target_min + 1e-8) + ns.target_min


RawBatchTuple = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def compute_log_depth(depth: np.ndarray) -> np.ndarray:
    """Convert physical depth (B,) -> log-depth (B, 1) float32 for FiLM input."""
    arr = np.asarray(depth, dtype=np.float32).reshape(-1, 1)
    return np.log(np.clip(arr, 1e-12, None)).astype(np.float32)


def make_normalizers(ns: NormStats):
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
        feat_range = jnp.asarray(ns.feature_max - ns.feature_min + 1e-8, dtype=jnp.float32)
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
    shuffle: bool,
    drop_last: bool,
) -> Iterator[RawBatchTuple]:
    """Yield raw (eta, xi, gxi, depth_log, indices) batches. Normalization happens
    on-device inside the jitted step."""
    ordered_indices = np.array(indices, copy=True)
    if shuffle:
        if rng is None:
            raise ValueError("rng is required when shuffle=True")
        rng.shuffle(ordered_indices)

    limit = ordered_indices.shape[0]
    if drop_last:
        limit = (limit // batch_size) * batch_size
    ordered_indices = ordered_indices[:limit]

    # Gather into contiguous arrays once so batch iteration is sequential slicing.
    eta_epoch = eta[ordered_indices]
    xi_epoch = xi[ordered_indices]
    gxi_epoch = gxi[ordered_indices]
    log_depth_epoch = compute_log_depth(depth[ordered_indices])

    for start in range(0, limit, batch_size):
        end = start + batch_size
        yield (
            eta_epoch[start:end],
            xi_epoch[start:end],
            gxi_epoch[start:end],
            log_depth_epoch[start:end],
            ordered_indices[start:end],
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
                if len(item) != len(destinations):
                    raise ValueError(
                        f"prefetch: item arity {len(item)} != destinations {len(destinations)}"
                    )
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
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__prefetch_error__":
            raise item[1]
        yield item


EvalStepFn = Callable[..., tuple[jax.Array, jax.Array]]
PredictFn = Callable[..., jax.Array]


def evaluate(
    current_params: FlatParams,
    *,
    batches: Iterator[RawBatchTuple],
    dataset: dict[str, np.ndarray],
    ns: NormStats,
    device: jax.Device,
    eval_step_fn: EvalStepFn,
    predict_batch_fn: PredictFn,
    collect_representatives: bool = False,
    representative_seed: int = 0,
) -> tuple[float, dict[str, np.ndarray] | None]:
    """Run validation. ``batches`` yields ``(eta, xi, gxi, depth_log, indices)``;
    eta/xi/gxi/depth may already be on-device (caller can prefetch). The
    eval_step_fn is responsible for normalization + loss in fp32 jax."""
    losses: list[float] = []
    rel_l2_all: list[np.ndarray] = []
    rel_l1_all: list[np.ndarray] = []
    example_idx_all: list[np.ndarray] = []

    for eta_b, xi_b, gxi_b, depth_b, batch_indices in batches:
        if not isinstance(eta_b, jax.Array):
            eta_b = jax.device_put(eta_b, device)
            xi_b = jax.device_put(xi_b, device)
            gxi_b = jax.device_put(gxi_b, device)
            depth_b = jax.device_put(depth_b, device)
        batch_loss, predicted_raw_jax = eval_step_fn(current_params, eta_b, xi_b, gxi_b, depth_b)
        batch_loss_value = float(np.asarray(jax.device_get(batch_loss)).reshape(-1)[0])
        losses.append(batch_loss_value)
        if collect_representatives:
            predicted_raw = np.asarray(jax.device_get(predicted_raw_jax))
            target_raw = dataset["gxi"][batch_indices, :, None]
            flat_pred = predicted_raw.reshape((predicted_raw.shape[0], -1))
            flat_target = target_raw.reshape((target_raw.shape[0], -1))
            rel_l2_all.append(
                (
                    np.linalg.norm(flat_pred - flat_target, axis=1)
                    / (np.linalg.norm(flat_target, axis=1) + 1e-12)
                ).astype(np.float32)
            )
            rel_l1_all.append(
                (
                    np.sum(np.abs(flat_pred - flat_target), axis=1)
                    / (np.sum(np.abs(flat_target), axis=1) + 1e-12)
                ).astype(np.float32)
            )
            example_idx_all.append(np.asarray(batch_indices, dtype=np.int64))

    val_loss = float(np.mean(losses)) if losses else float("inf")
    if not collect_representatives:
        return val_loss, None

    rel_l2 = np.concatenate(rel_l2_all, axis=0)
    rel_l1 = np.concatenate(rel_l1_all, axis=0)
    example_idx_vec = np.concatenate(example_idx_all, axis=0)
    sorted_indices = np.argsort(rel_l2)
    representative_indices = np.asarray(
        [
            int(sorted_indices[0]),
            int(sorted_indices[len(sorted_indices) // 2]),
            int(sorted_indices[-1]),
            int(np.random.default_rng(representative_seed).integers(0, len(rel_l2))),
        ],
        dtype=np.int32,
    )
    representative_rows = example_idx_vec[representative_indices]
    rep_eta = dataset["eta"][representative_rows]
    rep_xi = dataset["xi"][representative_rows]
    rep_targets = dataset["gxi"][representative_rows, :, None]
    rep_depth = compute_log_depth(dataset["depth"][representative_rows])
    rep_predictions = np.asarray(jax.device_get(predict_batch_fn(
        current_params,
        jax.device_put(rep_eta, device),
        jax.device_put(rep_xi, device),
        jax.device_put(rep_depth, device),
    )))
    representative_payload = {
        "eta": rep_eta,
        "xi": rep_xi,
        "targets": rep_targets,
        "predictions": rep_predictions,
        "rel_l2": rel_l2[representative_indices],
        "rel_l1": rel_l1[representative_indices],
    }
    return val_loss, representative_payload


def save_loss_history_plot(
    run_dir: Path,
    history: list[dict[str, float]],
) -> None:
    from jax_training_util import plot_loss_history

    plot_loss_history(history, run_dir / "loss_curve.png")


def save_final_representative_plot(
    run_dir: Path,
    epoch: int,
    x: np.ndarray,
    representative_payload: dict[str, np.ndarray] | None,
) -> None:
    from jax_training_util import plot_labeled_samples

    if representative_payload is None:
        return

    plot_labeled_samples(
        run_dir / "val_representative_samples_final.png",
        f"Tanaka Validation Epoch {epoch:03d}",
        x,
        representative_payload["eta"],
        representative_payload["xi"],
        representative_payload["targets"],
        representative_payload["predictions"],
        representative_payload["rel_l2"],
        representative_payload["rel_l1"],
        ("best", "median", "worst", "random"),
    )
