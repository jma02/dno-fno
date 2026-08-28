"""Load and atomically write generated dataset artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

import numpy as np
from numpy.typing import NDArray


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Load every array from a pickle-free NPZ before closing the file."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def npz_arrays_equal(
    left: Mapping[str, NDArray[Any]],
    right: Mapping[str, NDArray[Any]],
) -> bool:
    """Check that two NPZ array mappings have identical names, dtypes, and values."""

    if set(left) != set(right):
        return False

    for name, left_array in left.items():
        right_array = right[name]
        if left_array.dtype != right_array.dtype:
            return False
        if not np.array_equal(
            left_array,
            right_array,
            equal_nan=left_array.dtype.kind in {"f", "c"},
        ):
            return False
    return True


def deterministic_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize an object as deterministic, finite, human-readable JSON."""

    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _replace_atomically(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Atomically write a pickle-free NPZ with deterministic member order."""

    ordered = {name: np.asarray(arrays[name]) for name in sorted(arrays)}
    if any(array.dtype.kind == "O" for array in ordered.values()):
        raise TypeError("atomic NPZ arrays cannot use object dtype")

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            # NumPy's stub treats arbitrary NPZ member names as potential
            # ``allow_pickle`` arguments even though object arrays are rejected above.
            np.savez(handle, **ordered)  # pyright: ignore[reportArgumentType]

    _replace_atomically(path, writer)


def ensure_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    validate: Callable[[Mapping[str, NDArray[Any]]], object],
    artifact_name: str,
) -> None:
    """Write a valid NPZ or verify that an existing artifact is an exact replay."""

    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    validate(normalized)
    if path.exists():
        existing = load_npz(path)
        validate(existing)
        if not npz_arrays_equal(existing, normalized):
            raise RuntimeError(
                f"existing {artifact_name} differs from replayed {artifact_name}"
            )
    else:
        write_npz_atomic(path, normalized)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write deterministic, finite JSON."""

    encoded = deterministic_json_bytes(payload)

    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            handle.write(encoded)

    _replace_atomically(path, writer)


def ensure_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    artifact_name: str,
) -> None:
    """Write new JSON or verify that an existing artifact is an exact replay."""

    encoded = deterministic_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing {artifact_name} differs from replay")
    else:
        write_json_atomic(path, payload)
