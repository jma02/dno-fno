from __future__ import annotations

import hashlib
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

DATASET_VIEW_SCHEMA_VERSIONS = frozenset((1, 2))
_SHA256_HEX_LENGTH = 64
_TRAJECTORY_MAP_BASE_DTYPES = {
    "trajectory_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
    "trajectory_family_id": np.dtype(np.int16),
    "trajectory_revision_id": np.dtype(np.int16),
    "trajectory_case_id": np.dtype(np.int64),
    "trajectory_accepted": np.dtype(np.bool_),
    "trajectory_evaluated_bits": np.dtype(np.uint32),
    "trajectory_failed_bits": np.dtype(np.uint32),
}
_TRAJECTORY_MAP_V2_DTYPES = {
    **_TRAJECTORY_MAP_BASE_DTYPES,
    "shard_index": np.dtype(np.int32),
    "shard_row": np.dtype(np.int64),
    "trajectory_split_id": np.dtype(np.uint8),
    "trajectory_cell_id": np.dtype(np.int32),
    "trajectory_required_bits": np.dtype(np.uint32),
    "trajectory_first_row": np.dtype(np.int64),
    "trajectory_row_count": np.dtype(np.int32),
}

@dataclass(frozen=True)
class DatasetLocation:
    dataset_path: Path
    dataset_shard_paths: tuple[Path, ...] = ()
    trajectory_map_path: Path | None = None
    manifest: dict[str, object] | None = None


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


def _resolve_dataset_location(dataset_path: Path) -> DatasetLocation:
    """Resolve a dataset view without changing direct ``.npz`` semantics.

    A ``*.dataset.json`` path is an explicit, versioned view over an immutable
    data archive.  Passing the archive itself remains the legacy path even when
    a manifest happens to exist next to it.
    """
    if not dataset_path.name.endswith(".dataset.json"):
        return DatasetLocation(dataset_path=dataset_path)

    manifest_raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise ValueError(f"Dataset manifest must contain a JSON object: {dataset_path}")
    manifest: dict[str, object] = manifest_raw
    manifest_version = manifest.get("schema_version")
    if (
        not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version not in DATASET_VIEW_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"Unsupported dataset manifest schema_version in {dataset_path}; "
            f"expected one of {sorted(DATASET_VIEW_SCHEMA_VERSIONS)}"
        )
    if manifest_version == 2:
        _validate_dataset_contract_fingerprint(manifest, dataset_path)

    dataset_shard_paths: tuple[Path, ...] = ()
    if manifest_version == 1:
        dataset_name = manifest.get("dataset_npz")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(
                f"Dataset manifest is missing string field 'dataset_npz': "
                f"{dataset_path}"
            )
        resolved_dataset_path = (dataset_path.parent / dataset_name).resolve()
    else:
        shard_records = manifest.get("dataset_shards")
        if not isinstance(shard_records, list) or not shard_records:
            raise ValueError(
                f"Schema-v2 manifest requires a nonempty dataset_shards list: "
                f"{dataset_path}"
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
        dataset_shard_paths = tuple(resolved_shards)
        resolved_dataset_path = dataset_shard_paths[0]

    requires_trajectory_map = manifest.get("requires_trajectory_map", False)
    if not isinstance(requires_trajectory_map, bool):
        raise ValueError(
            f"Dataset manifest field 'requires_trajectory_map' must be boolean: {dataset_path}"
        )

    trajectory_map_name = manifest.get("trajectory_map_npz")
    if trajectory_map_name is not None and (
        not isinstance(trajectory_map_name, str) or not trajectory_map_name
    ):
        raise ValueError(
            f"Dataset manifest field 'trajectory_map_npz' must be a nonempty string: "
            f"{dataset_path}"
        )
    trajectory_map_path = (
        (dataset_path.parent / trajectory_map_name).resolve()
        if isinstance(trajectory_map_name, str)
        else resolved_dataset_path.with_suffix(".trajectory_map.npz")
    )
    if requires_trajectory_map and not trajectory_map_path.exists():
        raise FileNotFoundError(
            "Dataset manifest requires a trajectory map, but the sidecar is missing: "
            f"{trajectory_map_path}"
        )
    if not trajectory_map_path.exists():
        trajectory_map_path = None

    return DatasetLocation(
        dataset_path=resolved_dataset_path,
        dataset_shard_paths=dataset_shard_paths,
        trajectory_map_path=trajectory_map_path,
        manifest=manifest,
    )


def _load_trajectory_map_arrays(
    trajectory_map_path: Path,
    *,
    num_examples: int,
    manifest: dict[str, object],
) -> dict[str, np.ndarray]:
    manifest_version = manifest.get("schema_version", 1)
    expected_dtypes = (
        _TRAJECTORY_MAP_V2_DTYPES
        if manifest_version == 2
        else _TRAJECTORY_MAP_BASE_DTYPES
    )
    with np.load(trajectory_map_path, allow_pickle=False) as archive:
        missing = sorted(set(expected_dtypes) - set(archive.files))
        if missing:
            raise ValueError(
                f"Trajectory map {trajectory_map_path} is missing required arrays: {missing}"
            )
        if "schema_version" in archive.files:
            schema_version = np.asarray(archive["schema_version"])
            if schema_version.ndim != 0 or int(schema_version) != manifest_version:
                raise ValueError(
                    f"Unsupported trajectory-map schema_version in {trajectory_map_path}; "
                    f"expected scalar {manifest_version}"
                )
        arrays = {
            name: np.asarray(archive[name])
            for name in expected_dtypes
        }

    for name, expected_dtype in expected_dtypes.items():
        array = arrays[name]
        if array.ndim != 1:
            raise ValueError(f"Trajectory-map array {name!r} must be one-dimensional")
        if array.dtype != expected_dtype:
            raise TypeError(
                f"Trajectory-map array {name!r} has dtype {array.dtype}; "
                f"expected {expected_dtype}"
            )

    trajectory_index = arrays["trajectory_index"]
    frame_index = arrays["frame_index"]
    if trajectory_index.shape[0] != num_examples or frame_index.shape[0] != num_examples:
        raise ValueError(
            f"Trajectory-map row arrays must have {num_examples} entries; got "
            f"trajectory_index={trajectory_index.shape[0]}, frame_index={frame_index.shape[0]}"
        )

    row_fields = {"trajectory_index", "frame_index"}
    if manifest_version == 2:
        row_fields.update(("shard_index", "shard_row"))
        for name in ("shard_index", "shard_row"):
            if arrays[name].shape[0] != num_examples:
                raise ValueError(
                    f"Trajectory-map row array {name!r} must have "
                    f"{num_examples} entries"
                )
    trajectory_fields = tuple(
        name
        for name in expected_dtypes
        if name not in row_fields
    )
    num_trajectories = arrays[trajectory_fields[0]].shape[0]
    mismatched_fields = [
        name for name in trajectory_fields
        if arrays[name].shape[0] != num_trajectories
    ]
    if mismatched_fields:
        raise ValueError(
            "Trajectory-map table arrays have inconsistent lengths: "
            f"{mismatched_fields}"
        )
    if num_examples and num_trajectories == 0:
        raise ValueError("Nonempty datasets require a nonempty trajectory table")
    if trajectory_index.size and (
        int(trajectory_index.min()) < 0 or int(trajectory_index.max()) >= num_trajectories
    ):
        raise ValueError("trajectory_index contains an out-of-range table index")
    if frame_index.size and int(frame_index.min()) < 0:
        raise ValueError("frame_index must be nonnegative")

    compound_ids = np.rec.fromarrays(
        (
            arrays["trajectory_family_id"],
            arrays["trajectory_revision_id"],
            arrays["trajectory_case_id"],
        ),
        names=("family", "revision", "case"),
    )
    if np.unique(compound_ids).shape[0] != num_trajectories:
        raise ValueError(
            "Compound trajectory IDs (family, revision, case) must be unique"
        )
    accepted = arrays["trajectory_accepted"]
    if manifest_version == 2:
        split_id = arrays["trajectory_split_id"]
        required_bits = arrays["trajectory_required_bits"]
        evaluated_bits = arrays["trajectory_evaluated_bits"]
        failed_bits = arrays["trajectory_failed_bits"]
        first_row = arrays["trajectory_first_row"]
        row_count = arrays["trajectory_row_count"]
        if np.any(split_id > 2):
            raise ValueError(
                "trajectory_split_id must use train=0, validation=1, or test=2"
            )
        expected_accepted = (
            (required_bits & ~evaluated_bits) == 0
        ) & ((required_bits & failed_bits) == 0)
        if not np.array_equal(accepted, expected_accepted):
            raise ValueError(
                "trajectory_accepted disagrees with the quality masks"
            )
        if np.any(arrays["shard_index"] < 0) or np.any(arrays["shard_row"] < 0):
            raise ValueError("shard coordinates must be nonnegative")
        if trajectory_index.size and not np.all(accepted[trajectory_index]):
            raise ValueError("rejected trajectories cannot own dataset rows")
        for trajectory in range(num_trajectories):
            rows = np.flatnonzero(trajectory_index == trajectory)
            if accepted[trajectory]:
                if (
                    row_count[trajectory] <= 0
                    or first_row[trajectory] < 0
                    or not np.array_equal(
                        rows,
                        np.arange(
                            first_row[trajectory],
                            first_row[trajectory] + row_count[trajectory],
                        ),
                    )
                ):
                    raise ValueError(
                        "accepted trajectory row ownership is inconsistent"
                    )
            elif (
                rows.size
                or first_row[trajectory] != -1
                or row_count[trajectory] != 0
            ):
                raise ValueError("rejected trajectory must own zero rows")
    expected_counts = {
        "n_rows": num_examples,
        "n_trajectories": num_trajectories,
        "n_accepted_rows": int(np.count_nonzero(accepted[trajectory_index])),
    }
    if manifest_version == 2:
        expected_counts["n_accepted_trajectories"] = int(
            np.count_nonzero(accepted)
        )
    for name, actual in expected_counts.items():
        declared = manifest.get(name)
        if declared is not None and (
            not isinstance(declared, int)
            or isinstance(declared, bool)
            or declared != actual
        ):
            raise ValueError(
                f"Dataset manifest {name}={declared!r} does not match trajectory-map "
                f"value {actual}"
            )

    return arrays


def _read_dataset_meta(dataset_path: Path) -> tuple[dict, bool]:
    """Return (meta, is_flat). Combined-flat datasets have a sidecar
    ``<name>.meta.json`` next to the npz; legacy shard datasets carry
    meta.json inside the zip."""
    sidecar = dataset_path.with_suffix(".meta.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8")), True
    with zipfile.ZipFile(dataset_path, mode="r") as zf:
        return json.loads(zf.read("meta.json")), False


def load_dataset_meta(dataset_path: Path) -> dict[str, object]:
    """Load storage metadata for either an archive or a dataset-manifest view."""
    location = _resolve_dataset_location(dataset_path)
    if location.dataset_shard_paths:
        manifest = location.manifest or {}
        grid = manifest.get("grid")
        if not isinstance(grid, dict):
            raise ValueError("Schema-v2 dataset manifest requires a grid object")
        return {
            "dataset_kind": "paper_corpus_shards",
            "length": grid["length"],
            "nx": grid["nx"],
            "configuration_fingerprint": manifest.get(
                "configuration_fingerprint"
            ),
        }
    meta, _ = _read_dataset_meta(location.dataset_path)
    return meta


def _zero_mean_xi(xi: np.ndarray) -> np.ndarray:
    # G(eta) annihilates the xi mode-0 (Dirichlet-Neumann of a constant is zero),
    # so projecting xi to zero-mean leaves gxi unchanged while keeping the model
    # input distribution consistent with eval/rollout, which both project too.
    return xi - xi.mean(axis=1, keepdims=True)


def _with_row_trajectory_map(
    dataset: dict[str, np.ndarray],
    trajectory_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    trajectory_index = trajectory_map["trajectory_index"]
    dataset.update(
        {
            "trajectory_index": trajectory_index,
            "frame_index": trajectory_map["frame_index"],
            "family_id": trajectory_map["trajectory_family_id"][trajectory_index],
            "revision_id": trajectory_map["trajectory_revision_id"][trajectory_index],
            "case_id": trajectory_map["trajectory_case_id"][trajectory_index],
            "accepted_mask": trajectory_map["trajectory_accepted"][trajectory_index],
            "quality_required_bits": trajectory_map[
                "trajectory_required_bits"
            ][trajectory_index]
            if "trajectory_required_bits" in trajectory_map
            else np.zeros_like(
                trajectory_map["trajectory_evaluated_bits"][trajectory_index]
            ),
            "quality_evaluated_bits": trajectory_map["trajectory_evaluated_bits"][trajectory_index],
            "quality_failed_bits": trajectory_map["trajectory_failed_bits"][trajectory_index],
        }
    )
    dataset["source"] = dataset["family_id"]
    if "trajectory_split_id" in trajectory_map:
        dataset["split_id"] = trajectory_map["trajectory_split_id"][
            trajectory_index
        ]
        dataset["cell_id"] = trajectory_map["trajectory_cell_id"][
            trajectory_index
        ]
    return dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_dataset_contract_fingerprint(
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    contract = manifest.get("dataset_contract")
    declared = manifest.get("dataset_contract_fingerprint")
    if contract is None and declared is None:
        return
    if not isinstance(contract, dict):
        raise ValueError(
            f"Schema-v2 manifest has an invalid dataset_contract: {manifest_path}"
        )
    if not _is_sha256(declared):
        raise ValueError(
            "Schema-v2 manifest requires a lowercase SHA-256 "
            f"dataset_contract_fingerprint: {manifest_path}"
        )
    actual = _canonical_json_sha256(contract)
    if actual != declared:
        raise RuntimeError(
            f"Dataset contract fingerprint does not match: {manifest_path}"
        )


def _verify_declared_file(
    path: Path,
    record: dict[str, object],
    *,
    sha256_field: str,
    bytes_field: str,
    context: str,
) -> None:
    declared_bytes = record.get(bytes_field)
    if declared_bytes is not None:
        if (
            not isinstance(declared_bytes, int)
            or isinstance(declared_bytes, bool)
            or declared_bytes < 0
        ):
            raise ValueError(f"{context} has an invalid {bytes_field}")
        if path.stat().st_size != declared_bytes:
            raise RuntimeError(f"{context} byte count does not match: {path}")

    declared_sha256 = record.get(sha256_field)
    if not _is_sha256(declared_sha256):
        raise ValueError(
            f"{context} requires a lowercase SHA-256 field {sha256_field!r}"
        )
    if _sha256_file(path) != declared_sha256:
        raise RuntimeError(f"{context} hash does not match: {path}")


def load_dataset_identity(dataset_path: Path) -> dict[str, object] | None:
    """Return the content identity of an explicit dataset-manifest view.

    Direct archive paths retain legacy behavior and return ``None``. For a
    manifest view, the manifest hash binds every listed shard hash and the
    trajectory-map hash; the contract fingerprint separately names the
    numerical and storage contract.
    """
    if not dataset_path.name.endswith(".dataset.json"):
        return None
    resolved_path = dataset_path.resolve()
    location = _resolve_dataset_location(resolved_path)
    manifest = location.manifest or {}
    return {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": _sha256_file(resolved_path),
        "dataset_contract_fingerprint": manifest.get(
            "dataset_contract_fingerprint"
        ),
        "trajectory_map_sha256": manifest.get("trajectory_map_sha256"),
    }


def _load_paper_corpus_shards(
    location: DatasetLocation,
) -> dict[str, np.ndarray]:
    manifest = location.manifest or {}
    shard_records = manifest.get("dataset_shards")
    grid = manifest.get("grid")
    if not isinstance(shard_records, list) or not isinstance(grid, dict):
        raise ValueError("Schema-v2 manifest requires dataset_shards and grid")
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
        raise ValueError("Schema-v2 manifest has an invalid spatial grid")
    if location.trajectory_map_path is None:
        raise FileNotFoundError("Schema-v2 dataset requires a trajectory map")
    _verify_declared_file(
        location.trajectory_map_path,
        manifest,
        sha256_field="trajectory_map_sha256",
        bytes_field="trajectory_map_bytes",
        context="Trajectory map",
    )

    field_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("eta", "xi", "gxi", "depth", "time")
    }
    expected_shard_index: list[np.ndarray] = []
    expected_shard_row: list[np.ndarray] = []
    for shard_index, (path, record) in enumerate(
        zip(location.dataset_shard_paths, shard_records)
    ):
        if not isinstance(record, dict):
            raise TypeError("every dataset shard record must be an object")
        _verify_declared_file(
            path,
            record,
            sha256_field="sha256",
            bytes_field="bytes",
            context="Dataset shard",
        )
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(field_parts) - set(archive.files))
            if missing:
                raise ValueError(f"Dataset shard {path} is missing {missing}")
            arrays = {
                name: np.asarray(archive[name])
                for name in field_parts
            }
        eta = arrays["eta"]
        if eta.ndim != 2 or eta.shape[1] != nx:
            raise ValueError(
                f"Dataset shard {path} has eta shape {eta.shape}; "
                f"expected (rows, {nx})"
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
            field_parts[name].append(
                np.asarray(array, dtype=np.float32)
            )
        expected_shard_index.append(
            np.full(row_count, shard_index, dtype=np.int32)
        )
        expected_shard_row.append(np.arange(row_count, dtype=np.int64))

    eta = np.concatenate(field_parts["eta"], axis=0)
    dataset = {
        "eta": eta,
        "xi": _zero_mean_xi(np.concatenate(field_parts["xi"], axis=0)),
        "gxi": np.concatenate(field_parts["gxi"], axis=0),
        "depth": np.concatenate(field_parts["depth"], axis=0),
        "time": np.concatenate(field_parts["time"], axis=0),
        "source": np.zeros(eta.shape[0], dtype=np.int16),
        "case_id": np.arange(eta.shape[0], dtype=np.int64),
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
        manifest=manifest,
    )
    if not np.array_equal(
        trajectory_map["shard_index"],
        np.concatenate(expected_shard_index),
    ) or not np.array_equal(
        trajectory_map["shard_row"],
        np.concatenate(expected_shard_row),
    ):
        raise ValueError(
            "Trajectory-map shard coordinates do not match manifest shard order"
        )
    return _with_row_trajectory_map(dataset, trajectory_map)


def load_dataset_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    location = _resolve_dataset_location(dataset_path)
    if location.dataset_shard_paths:
        return _load_paper_corpus_shards(location)
    storage_path = location.dataset_path
    meta, is_flat = _read_dataset_meta(storage_path)
    domain_length = float(meta.get("length", 2.0 * np.pi))

    if is_flat:
        with np.load(storage_path) as archive:
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
        dataset = {
            "eta": eta, "xi": _zero_mean_xi(xi), "gxi": gxi,
            "depth": depth, "time": time_arr, "source": source,
            "case_id": np.arange(eta.shape[0], dtype=np.int64),
            "x": x, "domain_length": domain_length,
        }
        if location.trajectory_map_path is not None:
            trajectory_map = _load_trajectory_map_arrays(
                location.trajectory_map_path,
                num_examples=eta.shape[0],
                manifest=location.manifest or {},
            )
            dataset = _with_row_trajectory_map(dataset, trajectory_map)
        return dataset

    if location.trajectory_map_path is not None:
        raise ValueError("Trajectory-map dataset views currently require a flat dataset archive")

    with np.load(storage_path) as archive:
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


def build_grouped_split_indices(
    trajectory_index: np.ndarray,
    family_id: np.ndarray,
    seed: int,
    *,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an 80/10/10 split of whole trajectories within each family.

    ``trajectory_index`` is the dense row-to-trajectory table index supplied by
    the trajectory-map sidecar. Rows from one trajectory are therefore
    inseparable even when compound case IDs overlap between generator revisions.
    """
    trajectory_index = np.asarray(trajectory_index)
    family_id = np.asarray(family_id)
    if trajectory_index.ndim != 1 or family_id.ndim != 1:
        raise ValueError("trajectory_index and family_id must be one-dimensional")
    if trajectory_index.shape != family_id.shape:
        raise ValueError("trajectory_index and family_id must have the same shape")
    if not np.issubdtype(trajectory_index.dtype, np.integer):
        raise TypeError("trajectory_index must have an integer dtype")
    if not np.issubdtype(family_id.dtype, np.integer):
        raise TypeError("family_id must have an integer dtype")
    if trajectory_index.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    if int(trajectory_index.min()) < 0:
        raise ValueError("trajectory_index must be nonnegative")

    eligible = (
        np.ones(trajectory_index.shape, dtype=np.bool_)
        if eligible_mask is None
        else np.asarray(eligible_mask)
    )
    if eligible.shape != trajectory_index.shape or eligible.dtype != np.bool_:
        raise TypeError("eligible_mask must be a boolean array matching trajectory_index")

    num_trajectory_slots = int(trajectory_index.max()) + 1
    row_counts = np.bincount(trajectory_index, minlength=num_trajectory_slots)
    eligible_counts = np.bincount(
        trajectory_index,
        weights=eligible.astype(np.int8),
        minlength=num_trajectory_slots,
    ).astype(np.int64)
    partial = (eligible_counts != 0) & (eligible_counts != row_counts)
    if np.any(partial):
        bad_index = int(np.flatnonzero(partial)[0])
        raise ValueError(
            "eligible_mask must accept or reject whole trajectories; "
            f"trajectory_index={bad_index} is only partially eligible"
        )

    family_min = np.full(num_trajectory_slots, np.iinfo(np.int64).max, dtype=np.int64)
    family_max = np.full(num_trajectory_slots, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(family_min, trajectory_index, family_id)
    np.maximum.at(family_max, trajectory_index, family_id)
    populated = row_counts > 0
    if np.any(family_min[populated] != family_max[populated]):
        bad_index = int(np.flatnonzero(populated & (family_min != family_max))[0])
        raise ValueError(
            f"trajectory_index={bad_index} is assigned to more than one physical family"
        )

    active_trajectories = np.flatnonzero(eligible_counts == row_counts)
    active_trajectories = active_trajectories[row_counts[active_trajectories] > 0]
    split_by_trajectory = np.full(num_trajectory_slots, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for family in np.unique(family_min[active_trajectories]):
        family_trajectories = active_trajectories[
            family_min[active_trajectories] == family
        ]
        family_trajectories = rng.permutation(family_trajectories)
        val_count = int(family_trajectories.shape[0] * 0.1)
        test_count = int(family_trajectories.shape[0] * 0.1)
        train_count = family_trajectories.shape[0] - val_count - test_count
        split_by_trajectory[family_trajectories[:train_count]] = 0
        split_by_trajectory[
            family_trajectories[train_count : train_count + val_count]
        ] = 1
        split_by_trajectory[family_trajectories[train_count + val_count :]] = 2

    row_split = split_by_trajectory[trajectory_index]
    split_indices = tuple(
        rng.permutation(np.flatnonzero(eligible & (row_split == split_id)))
        for split_id in range(3)
    )
    return split_indices


def build_dataset_split_indices(
    dataset: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use trajectory-grouped splits when a trajectory map is present."""
    if "trajectory_index" not in dataset:
        return build_split_indices(int(dataset["eta"].shape[0]), seed)

    required = {"family_id", "accepted_mask"}
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"Trajectory-map dataset is missing row arrays: {missing}")
    if "split_id" in dataset:
        split_id = np.asarray(dataset["split_id"])
        eligible = np.asarray(dataset["accepted_mask"])
        if split_id.shape != eligible.shape:
            raise ValueError("split_id and accepted_mask must have matching shapes")
        if split_id.dtype != np.uint8 or eligible.dtype != np.bool_:
            raise TypeError("split_id must be uint8 and accepted_mask must be bool")
        if np.any(split_id > 2):
            raise ValueError("split_id values must be train=0, validation=1, or test=2")
        rng = np.random.default_rng(seed)
        return tuple(
            rng.permutation(np.flatnonzero(eligible & (split_id == value)))
            for value in range(3)
        )
    return build_grouped_split_indices(
        dataset["trajectory_index"],
        dataset["family_id"],
        seed,
        eligible_mask=dataset["accepted_mask"],
    )


def build_hierarchical_epoch_indices(
    dataset: dict[str, np.ndarray],
    *,
    split_id: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Choose one stored time per accepted case, equally across families.

    The schema-v2 corpus is generated with equal accepted case quotas for each
    physical family.  This function requires that equality instead of silently
    reweighting a family.  Within an epoch every accepted case appears once,
    one of its stored times is drawn uniformly, and the resulting rows are
    shuffled.
    """

    if split_id not in (0, 1, 2):
        raise ValueError("split_id must be train=0, validation=1, or test=2")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    required = {
        "trajectory_index",
        "family_id",
        "split_id",
        "accepted_mask",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"hierarchical sampling requires row arrays: {missing}")

    trajectory_index = np.asarray(dataset["trajectory_index"])
    family_id = np.asarray(dataset["family_id"])
    row_split = np.asarray(dataset["split_id"])
    accepted = np.asarray(dataset["accepted_mask"])
    row_count = int(np.asarray(dataset["eta"]).shape[0])
    if any(
        array.ndim != 1 or array.shape[0] != row_count
        for array in (trajectory_index, family_id, row_split, accepted)
    ):
        raise ValueError("hierarchical row arrays must match the dataset length")
    eligible = accepted & (row_split == split_id)
    trajectories = np.unique(trajectory_index[eligible])
    if trajectories.size == 0:
        return np.empty((0,), dtype=np.int64)

    trajectories_by_family: dict[int, list[int]] = {}
    rows_by_trajectory: dict[int, np.ndarray] = {}
    for trajectory in trajectories:
        rows = np.flatnonzero(eligible & (trajectory_index == trajectory))
        families = np.unique(family_id[rows])
        if families.size != 1:
            raise ValueError("one trajectory cannot belong to multiple families")
        family = int(families[0])
        trajectories_by_family.setdefault(family, []).append(int(trajectory))
        rows_by_trajectory[int(trajectory)] = rows
    family_case_counts = {
        family: len(family_trajectories)
        for family, family_trajectories in trajectories_by_family.items()
    }
    if len(set(family_case_counts.values())) != 1:
        raise ValueError(
            "hierarchical sampling requires equal accepted case counts per "
            f"family, got {family_case_counts}"
        )

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected_rows: list[int] = []
    for family in sorted(trajectories_by_family):
        family_trajectories = rng.permutation(
            trajectories_by_family[family]
        )
        selected_rows.extend(
            int(rows_by_trajectory[int(trajectory)][
                rng.integers(rows_by_trajectory[int(trajectory)].size)
            ])
            for trajectory in family_trajectories
        )
    return np.asarray(rng.permutation(selected_rows), dtype=np.int64)


def build_epoch_sample_indices(
    dataset: dict[str, np.ndarray],
    fallback_indices: np.ndarray,
    *,
    split_id: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Use case/time sampling for schema-v2 and preserve legacy row sampling."""
    if "split_id" not in dataset:
        return fallback_indices
    return build_hierarchical_epoch_indices(
        dataset,
        split_id=split_id,
        seed=seed,
        epoch=epoch,
    )


def build_case_balanced_validation_indices(
    dataset: dict[str, np.ndarray],
    fallback_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return a fixed validation draw with one stored time per schema-v2 case.

    The fixed ``epoch=0`` key makes checkpoint losses directly comparable
    throughout training. Schema-v1 datasets retain their original all-row
    validation indices.
    """
    return build_epoch_sample_indices(
        dataset,
        fallback_indices,
        split_id=1,
        seed=seed,
        epoch=0,
    )


def _stats_valid(s: dict[str, object]) -> bool:
    valid = False
    with suppress(KeyError, TypeError):
        valid = (
            len(s["feature_min"]) == 2
            and len(s["feature_max"]) == 2
            and all(np.isfinite(v) for v in (*s["feature_min"], *s["feature_max"], s["target_min"], s["target_max"]))
        )
    return valid


def _index_selection(indices: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(indices, dtype="<i8")
    return {
        "count": int(canonical.shape[0]),
        "sha256": hashlib.sha256(memoryview(canonical).cast("B")).hexdigest(),
    }


def _selected_extrema(
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
    dataset_identity = load_dataset_identity(dataset_path)
    selection: dict[str, object] | None = None
    if indices is not None:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("indices must be a one-dimensional integer array")
        if indices.size == 0:
            raise ValueError("Cannot compute normalization statistics from an empty selection")
        selection = _index_selection(indices)

    stats_path = dataset_path.with_suffix(".stats.json")
    if stats_path.exists():
        cached: object = None
        with suppress(json.JSONDecodeError, OSError):
            cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and _stats_valid(cached)
            and cached.get("index_selection") == selection
            and cached.get("dataset_identity") == dataset_identity
        ):
            return cached

    if dataset is None:
        dataset = load_dataset_arrays(dataset_path)
    num_examples = int(dataset["eta"].shape[0])
    if indices is not None and (
        int(indices.min()) < 0 or int(indices.max()) >= num_examples
    ):
        raise IndexError("Statistics selection contains an out-of-range row index")

    eta_min, eta_max, eta_absmax = _selected_extrema(dataset["eta"], indices)
    xi_min, xi_max, xi_absmax = _selected_extrema(dataset["xi"], indices)
    target_min, target_max, target_absmax = _selected_extrema(dataset["gxi"], indices)
    depth_arr = np.asarray(dataset["depth"], dtype=np.float64)
    depth_min, depth_max, _ = _selected_extrema(depth_arr, indices)
    log_depth = np.log(np.clip(depth_arr, 1e-12, None))
    log_depth_min, log_depth_max, _ = _selected_extrema(log_depth, indices)
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
        "domain_length": float(dataset.get("domain_length", 2.0 * np.pi)),
        "index_selection": selection,
        "dataset_identity": dataset_identity,
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
