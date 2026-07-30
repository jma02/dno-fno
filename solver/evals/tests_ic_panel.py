"""CPU-only validation checks for compact held-out IC panels."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from solver.evals.eval_suite import (  # noqa: E402
    RegimeConfig,
    _load_ic_panel,
    _try_load_cached_truth,
    build_ics_and_truth_targets,
)


def _write_panel(
    root: Path,
    *,
    regime: str = "sample",
    n_ics: int = 2,
    nx: int = 8,
    length: float = 2.0 * np.pi,
) -> Path:
    meta = {
        "regime": regime,
        "n_ics": n_ics,
        "nx": nx,
        "length": length,
        "dt": 0.125,
        "tmax": 3.0,
        "provenance": {"generator": "test", "seed": 123},
    }
    path = root / f"{regime}_ics.npz"
    np.savez_compressed(
        path,
        eta=np.zeros((n_ics, nx)),
        xi=np.ones((n_ics, nx)),
        depth=np.linspace(1.0, 2.0, n_ics),
        case_ids=np.arange(10, 10 + n_ics, dtype=np.int64),
        **{"meta.json": np.asarray(json.dumps(meta).encode("utf-8"))},
    )
    return path


def _assert_rejected(operation: Callable[[], object], message_fragment: str) -> None:
    try:
        operation()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("invalid IC panel was accepted")


def test_panel_load_and_registry_timestep_precedence() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        root = Path(tmp)
        path = _write_panel(root)
        ics, _ = _load_ic_panel(path, "sample", 2, 8, 2.0 * np.pi)
        assert [ic.case_id for ic in ics] == [10, 11]
        assert len(ics[0].meta["panel_source"]["sha256"]) == 64

        inherited = RegimeConfig(name="sample", source=("unused",), n_ics=2)
        _, dt, tmax, _, _ = build_ics_and_truth_targets(
            inherited,
            ic_panel_dir=root,
            nx=8,
            length=2.0 * np.pi,
        )
        assert (dt, tmax) == (0.125, 3.0)

        overridden = RegimeConfig(
            name="sample",
            source=("unused",),
            dt=0.5,
            tmax=5.0,
            n_ics=2,
        )
        _, dt, tmax, _, _ = build_ics_and_truth_targets(
            overridden,
            ic_panel_dir=root,
            nx=8,
            length=2.0 * np.pi,
        )
        assert (dt, tmax) == (0.5, 5.0)


def test_panel_validation_rejects_wrong_count_and_identity() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        path = _write_panel(Path(tmp))
        _assert_rejected(
            lambda: _load_ic_panel(path, "sample", 1, 8, 2.0 * np.pi),
            "exactly (1, 8)",
        )
        _assert_rejected(
            lambda: _load_ic_panel(path, "other", 2, 8, 2.0 * np.pi),
            "expected 'other'",
        )
        _assert_rejected(
            lambda: _load_ic_panel(path, "sample", 2, 16, 2.0 * np.pi),
            "expected 16",
        )


def test_truth_cache_keys_on_panel_digest() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        root = Path(tmp)
        payload = np.zeros((3, 2, 4), dtype=np.float32)
        np.savez_compressed(
            root / "sample_trajs.npz",
            truth_eta=payload,
            truth_xi=payload,
            truth_gxi=payload,
            case_ids=np.asarray([1, 2]),
            ic_panel_sha256=np.asarray("right"),
            truth_protocol_sha256=np.asarray("protocol"),
        )
        assert _try_load_cached_truth(
            "sample", root, 2, 3, 4, [1, 2], "right", "protocol"
        ) is not None
        assert _try_load_cached_truth("sample", root, 2, 3, 4, [1, 2], "wrong") is None
        assert _try_load_cached_truth(
            "sample", root, 2, 3, 4, [1, 2], "right", "wrong"
        ) is None


if __name__ == "__main__":
    tests = (
        test_panel_load_and_registry_timestep_precedence,
        test_panel_validation_rejects_wrong_count_and_identity,
        test_truth_cache_keys_on_panel_digest,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
