# for loading gts and testing our solver
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_SOLITON_ROOT = Path("/home/johnma/dno_locl/soliton_data")
README_NAME = "README.rtf"
EXPECTED_NX = 1024
EXPECTED_LENGTH = 164.0


def list_soliton_files(root: str | Path = DEFAULT_SOLITON_ROOT) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    files = sorted(path for path in root_path.iterdir() if path.is_file() and path.name != README_NAME)
    return files


def load_soliton_file(path: str | Path) -> dict[str, np.ndarray | int | float | str]:
    file_path = Path(path).expanduser().resolve()
    raw = np.loadtxt(file_path)
    nt = raw.shape[0] // EXPECTED_NX
    blocks = raw.reshape(nt, EXPECTED_NX, 6)

    x = np.asarray(blocks[0, :, 1], dtype=np.float64)
    t = np.asarray(blocks[:, 0, 2], dtype=np.float64)

    dx = float(x[1] - x[0])
    length = float(dx * EXPECTED_NX)

    return {
        "name": file_path.name,
        "path": str(file_path),
        "nt": nt,
        "nx": EXPECTED_NX,
        "length": length,
        "x": x,
        "t": t,
        "eta": np.asarray(blocks[:, :, 3], dtype=np.float64),
        "xi": np.asarray(blocks[:, :, 4], dtype=np.float64),
        "gxi": np.asarray(blocks[:, :, 5], dtype=np.float64),
    }


def load_soliton_dataset(root: str | Path = DEFAULT_SOLITON_ROOT) -> list[dict[str, np.ndarray | int | float | str]]:
    return [load_soliton_file(path) for path in list_soliton_files(root)]
