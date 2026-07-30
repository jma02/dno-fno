"""Archive-contract tests for paper Stokes generation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import warnings
import zipfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from solver.gen_data.generate_stokes_dataset import (  # noqa: E402
    configuration_fingerprint,
    generation_configuration,
    make_batch_rng,
    validate_resume_artifacts,
)


def _arguments(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "regime": "deep",
        "target_samples": 3,
        "batch_size": 2,
        "seed": 42,
        "rng_stream_id": 0,
        "case_id_offset": 0,
        "n0_min": 1,
        "n0_max": 20,
        "a0_min": 0.000766,
        "a0_max": 0.011494,
        "steepness_max": 0.15,
        "rejection_attempts": 1000,
        "depth_min": 4.0,
        "depth_max": 50.0,
        "kh_min": 5.0,
        "kh_max": math.inf,
        "nx": 1024,
        "length": 2.0 * math.pi,
        "gravity": 1.0,
        "dno_order": 6,
        "pad_factor": 8,
        "rollout_dtype": "float64",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _write_archive(
    path: Path,
    *,
    configuration: dict[str, object],
    next_batch: int,
    fingerprint: str | None = None,
    omit: str | None = None,
    orphan: str | None = None,
    duplicate: str | None = None,
) -> None:
    if fingerprint is None:
        fingerprint = configuration_fingerprint(configuration)
    names = {"x.npy"}
    for batch_index in range(next_batch):
        tag = f"batch_{batch_index:04d}"
        names.update(
            {
                f"eta_{tag}.npy",
                f"xi_{tag}.npy",
                f"gxi_{tag}.npy",
                f"phase_{tag}.npy",
                f"case_id_{tag}.npy",
                f"depth_{tag}.npy",
                f"specs_{tag}.json",
            }
        )
    if omit is not None:
        names.remove(omit)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "meta.json",
            json.dumps(
                {
                    "config_fingerprint": fingerprint,
                    "configuration": configuration,
                },
                allow_nan=False,
            ),
        )
        for name in sorted(names):
            archive.writestr(name, b"test")
        if orphan is not None:
            archive.writestr(orphan, b"orphan")
        if duplicate is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate, b"duplicate")


class StokesArchiveContractTest(unittest.TestCase):
    def test_configuration_is_strict_json_and_content_addressed(self) -> None:
        configuration = generation_configuration(_arguments(), ichoi=0)
        encoded = json.dumps(configuration, allow_nan=False)
        changed = generation_configuration(_arguments(seed=43), ichoi=0)

        self.assertIn('"kh_max": "Infinity"', encoded)
        self.assertIn('"algorithm": "numpy_pcg64"', encoded)
        self.assertEqual(len(configuration_fingerprint(configuration)), 64)
        self.assertNotEqual(
            configuration_fingerprint(configuration),
            configuration_fingerprint(changed),
        )

    def test_pcg64_batch_key_is_reproducible_and_separates_batches(self) -> None:
        first = make_batch_rng(17, 3, 5)
        repeated = make_batch_rng(17, 3, 5)
        next_batch = make_batch_rng(17, 4, 5)

        self.assertIsInstance(first.bit_generator, np.random.PCG64)
        np.testing.assert_array_equal(
            first.random(16),
            repeated.random(16),
        )
        self.assertFalse(
            np.array_equal(
                make_batch_rng(17, 3, 5).random(16),
                next_batch.random(16),
            )
        )

    def test_valid_resume_archive_passes(self) -> None:
        configuration: dict[str, object] = {"fixture": "valid"}
        fingerprint = configuration_fingerprint(configuration)
        state: dict[str, object] = {
            "config_fingerprint": fingerprint,
            "next_batch": 1,
            "samples_written": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stokes.npz"
            _write_archive(
                path,
                configuration=configuration,
                next_batch=1,
            )

            validate_resume_artifacts(
                output_path=path,
                state=state,
                expected_fingerprint=fingerprint,
                target_samples=3,
                batch_size=2,
            )

    def test_resume_refuses_mismatches_before_append(self) -> None:
        configuration: dict[str, object] = {"fixture": "mismatch"}
        fingerprint = configuration_fingerprint(configuration)
        state: dict[str, object] = {
            "config_fingerprint": fingerprint,
            "next_batch": 1,
            "samples_written": 2,
        }
        cases = (
            ("wrong fingerprint", {"config_fingerprint": "b" * 64}, None, None),
            (
                "missing entry",
                {},
                "phase_batch_0000.npy",
                None,
            ),
            (
                "orphan entry",
                {},
                None,
                "eta_batch_0001.npy",
            ),
        )
        for label, state_update, omit, orphan in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "stokes.npz"
                    _write_archive(
                        path,
                        configuration=configuration,
                        next_batch=1,
                        omit=omit,
                        orphan=orphan,
                    )
                    with self.assertRaisesRegex(RuntimeError, "overwrite"):
                        validate_resume_artifacts(
                            output_path=path,
                            state={**state, **state_update},
                            expected_fingerprint=fingerprint,
                            target_samples=3,
                            batch_size=2,
                        )

    def test_resume_refuses_duplicate_entries_and_missing_archive(self) -> None:
        configuration: dict[str, object] = {"fixture": "duplicate"}
        fingerprint = configuration_fingerprint(configuration)
        state: dict[str, object] = {
            "config_fingerprint": fingerprint,
            "next_batch": 1,
            "samples_written": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.npz"
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                validate_resume_artifacts(
                    output_path=missing,
                    state=state,
                    expected_fingerprint=fingerprint,
                    target_samples=3,
                    batch_size=2,
                )

            duplicate = root / "duplicate.npz"
            _write_archive(
                duplicate,
                configuration=configuration,
                next_batch=1,
                duplicate="eta_batch_0000.npy",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                validate_resume_artifacts(
                    output_path=duplicate,
                    state=state,
                    expected_fingerprint=fingerprint,
                    target_samples=3,
                    batch_size=2,
                )

    def test_resume_refuses_corrupted_stored_configuration(self) -> None:
        configuration: dict[str, object] = {"fixture": "original"}
        fingerprint = configuration_fingerprint(configuration)
        state: dict[str, object] = {
            "config_fingerprint": fingerprint,
            "next_batch": 0,
            "samples_written": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stokes.npz"
            _write_archive(
                path,
                configuration={"fixture": "corrupted"},
                fingerprint=fingerprint,
                next_batch=0,
            )
            with self.assertRaisesRegex(RuntimeError, "corrupted"):
                validate_resume_artifacts(
                    output_path=path,
                    state=state,
                    expected_fingerprint=fingerprint,
                    target_samples=3,
                    batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
