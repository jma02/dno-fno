from __future__ import annotations

import json
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import jax
import numpy as np

FlatParams = dict[str, jax.Array]
StatsDict = dict[str, object]


def require_jax_devices(*, allow_cpu: bool = False, min_device_count: int = 1) -> tuple[str, list[jax.Device]]:
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise RuntimeError(f"JAX GPU backend is required. Found {backend!r}.")
    devices = list(jax.local_devices())
    if len(devices) < min_device_count:
        raise RuntimeError(f"Expected at least {min_device_count} local JAX devices, found {len(devices)}.")
    return backend, devices


def load_dataset_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(dataset_path, mode="r") as zf:
        meta = json.loads(zf.read("meta.json"))

    with np.load(dataset_path) as archive:
        eta_parts: list[np.ndarray] = []
        xi_parts: list[np.ndarray] = []
        gxi_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        case_id_parts: list[np.ndarray] = []

        n_shards = int(meta["n_batches_planned"])
        for shard_id in range(n_shards):
            tag = f"{shard_id:04d}"
            eta_parts.append(np.asarray(archive[f"eta_batch_{tag}"], dtype=np.float32))
            xi_parts.append(np.asarray(archive[f"xi_batch_{tag}"], dtype=np.float32))
            gxi_parts.append(np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32))
            time_parts.append(np.asarray(archive[f"time_batch_{tag}"], dtype=np.float32))
            case_id_parts.append(np.asarray(archive[f"case_id_batch_{tag}"], dtype=np.int64))

        x = np.asarray(archive["x"], dtype=np.float32)

    return {
        "eta": np.concatenate(eta_parts, axis=0),
        "xi": np.concatenate(xi_parts, axis=0),
        "gxi": np.concatenate(gxi_parts, axis=0),
        "time": np.concatenate(time_parts, axis=0),
        "case_id": np.concatenate(case_id_parts, axis=0),
        "x": x,
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
        "target_min": float(np.min(dataset["gxi"])),
        "target_max": float(np.max(dataset["gxi"])),
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
    target_min: float
    target_max: float

    @staticmethod
    def from_dict(stats: StatsDict) -> NormStats:
        return NormStats(
            feature_min=np.asarray(stats["feature_min"], dtype=np.float32).reshape((1, 1, 2)),
            feature_max=np.asarray(stats["feature_max"], dtype=np.float32).reshape((1, 1, 2)),
            target_min=float(stats["target_min"]),
            target_max=float(stats["target_max"]),
        )


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: NormStats) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32, copy=False)
    return normalize_minmax(stacked, lower=-1.0, upper=1.0, value_min=ns.feature_min, value_max=ns.feature_max)


def normalize_targets(gxi: np.ndarray, ns: NormStats) -> np.ndarray:
    targets = gxi[..., None].astype(np.float32, copy=False)
    return normalize_minmax(
        targets,
        lower=-1.0,
        upper=1.0,
        value_min=ns.target_min,
        value_max=ns.target_max,
    )


def denormalize_targets(array: np.ndarray, ns: NormStats) -> np.ndarray:
    return ((array + 1.0) * 0.5) * (ns.target_max - ns.target_min + 1e-8) + ns.target_min


def get_batches(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    ns: NormStats,
    rng: np.random.Generator | None,
    *,
    shuffle: bool,
    drop_last: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ordered_indices = np.array(indices, copy=True)
    if shuffle:
        if rng is None:
            raise ValueError("rng is required when shuffle=True")
        rng.shuffle(ordered_indices)

    limit = ordered_indices.shape[0]
    if drop_last:
        limit = (limit // batch_size) * batch_size

    for start in range(0, limit, batch_size):
        batch_indices = ordered_indices[start : start + batch_size]
        batch_inputs = normalize_features(eta[batch_indices], xi[batch_indices], ns)
        batch_targets = normalize_targets(gxi[batch_indices], ns)
        yield batch_inputs, batch_targets, batch_indices


EvalStepFn = Callable[..., tuple[jax.Array, jax.Array]]
PredictFn = Callable[..., jax.Array]


def evaluate(
    current_params: FlatParams,
    *,
    batches: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dataset: dict[str, np.ndarray],
    ns: NormStats,
    device: jax.Device,
    eval_step_fn: EvalStepFn,
    predict_batch_fn: PredictFn,
    collect_representatives: bool = False,
    representative_seed: int = 0,
) -> tuple[float, dict[str, np.ndarray] | None]:
    losses: list[float] = []
    rel_l2_all: list[np.ndarray] = []
    rel_l1_all: list[np.ndarray] = []
    example_idx_all: list[np.ndarray] = []

    for batch_inputs, batch_targets, batch_indices in batches:
        batch_inputs = jax.device_put(batch_inputs, device)
        batch_targets = jax.device_put(batch_targets, device)
        batch_loss, predicted_norm = eval_step_fn(current_params, batch_inputs, batch_targets)
        batch_loss_value = float(np.asarray(jax.device_get(batch_loss)).reshape(-1)[0])
        losses.append(batch_loss_value)
        if collect_representatives:
            predicted_raw = denormalize_targets(
                np.asarray(jax.device_get(predicted_norm)),
                ns,
            )
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
            example_idx_all.append(batch_indices.astype(np.int64))

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
    rep_inputs = normalize_features(rep_eta, rep_xi, ns)
    rep_predictions = denormalize_targets(
        np.asarray(jax.device_get(predict_batch_fn(current_params, jax.device_put(rep_inputs, device)))),
        ns,
    )
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
