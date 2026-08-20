"""Canonical local entrypoint for DNO/FNO training.

The implementation remains in ``train-jax-10m`` because that source path and
its digest are part of the frozen paper-corpus release identity. User-facing
launchers call this stable entrypoint instead of reaching into the internal
training-engine directory.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ENGINE = REPO_ROOT / "train-jax-10m" / "1d_dno_fno_jax.py"


def main() -> None:
    """Execute the canonical training engine with unchanged CLI arguments."""
    training_directory = str(TRAINING_ENGINE.parent)
    if training_directory not in sys.path:
        sys.path.insert(0, training_directory)
    runpy.run_path(str(TRAINING_ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
