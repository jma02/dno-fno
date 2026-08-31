"""Load and atomically write generated dataset artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from numpy.typing import NDArray


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Load every array from a pickle-free NPZ before closing the file."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def json_text(value: object, *, indent: int | None = None) -> str:
    """Serialize a JSON-like value deterministically and reject nonfinite numbers."""

    return json.dumps(
        value,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def parse_json(text: str) -> object:
    """Parse JSON, rejecting the nonstandard NaN and Infinity values."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON file and require an object at its root."""

    value = parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_file_atomically(
    path: Path,
    write_temporary_file: Callable[[Path], None],
) -> None:
    """Write a temporary file completely, then rename it to the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_temporary_file(temporary)
        temporary.replace(path)
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

    def write_temporary_file(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            # NumPy's stub treats arbitrary NPZ member names as potential
            # ``allow_pickle`` arguments even though object arrays are rejected above.
            np.savez(handle, **ordered)  # pyright: ignore[reportArgumentType]

    _write_file_atomically(path, write_temporary_file)


def ensure_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    validate: Callable[[Mapping[str, NDArray[Any]]], None],
) -> None:
    """Write a valid NPZ or verify that an existing artifact is an exact replay."""

    normalized = {name: np.asarray(array) for name, array in arrays.items()}
    validate(normalized)
    if path.exists():
        existing = load_npz(path)
        validate(existing)
        arrays_differ = existing.keys() != normalized.keys() or any(
            existing[name].dtype != array.dtype
            or not np.array_equal(
                existing[name],
                array,
                equal_nan=array.dtype.kind in {"f", "c"},
            )
            for name, array in normalized.items()
        )
        if arrays_differ:
            raise RuntimeError(f"existing {path} differs from replayed arrays")
    else:
        write_npz_atomic(path, normalized)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write deterministic, finite JSON."""

    encoded = (json_text(payload, indent=2) + "\n").encode("utf-8")

    def write_temporary_file(temporary: Path) -> None:
        temporary.write_bytes(encoded)

    _write_file_atomically(path, write_temporary_file)
