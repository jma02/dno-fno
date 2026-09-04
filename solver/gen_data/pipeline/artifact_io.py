"""Load and atomically write generated dataset artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from numpy.typing import NDArray


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Load every array from a pickle-free NPZ before closing the file."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


@contextmanager
def _atomic_output_path(
    path: Path,
    *,
    replace_existing: bool,
) -> Iterator[Path]:
    """Expose a temporary path and publish it only after a successful write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        if replace_existing:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    replace_existing: bool = True,
) -> None:
    """Atomically write a pickle-free NPZ with deterministic member order."""

    ordered = {name: np.asarray(arrays[name]) for name in sorted(arrays)}
    if any(array.dtype.kind == "O" for array in ordered.values()):
        raise TypeError("atomic NPZ arrays cannot use object dtype")

    with _atomic_output_path(path, replace_existing=replace_existing) as temporary:
        with temporary.open("wb") as handle:
            # NumPy's stub treats arbitrary NPZ member names as potential
            # ``allow_pickle`` arguments even though object arrays are rejected above.
            np.savez(handle, **ordered)  # pyright: ignore[reportArgumentType]


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write deterministic, finite JSON."""

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    with _atomic_output_path(path, replace_existing=True) as temporary:
        temporary.write_bytes(encoded)
