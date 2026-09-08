"""Combine real batch artifacts and reject invalid family populations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.build_paper_dataset_view import FAMILY_IDS, build_combined_view
from solver.gen_data.pipeline.artifact_io import (
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.types import DatasetSplit, SimulationRows


def _write_run(
    root: Path,
    family: str,
    split: DatasetSplit,
    *,
    accepted_count: int = 1,
) -> Path:
    run_root = root / f"{split.value}_{family}"
    batch = run_root / "batches" / "batch_000000.npz"
    frame_count = int(FAMILY_IDS[family])
    eta = np.arange(frame_count * 4, dtype=np.float64).reshape(frame_count, 4)
    rows = SimulationRows(
        eta, eta + 1.0, eta - 1.0, 1.0, np.arange(frame_count, dtype=np.float64)
    )
    save_completed_batch(
        batch,
        ("group",) * (accepted_count + 1),
        (None, *((rows,) * accepted_count)),
        family_id=FAMILY_IDS[family],
        dataset_split=split,
    )
    summary_path = run_root / f"paper_dataset_{family}_{split.value}.summary.json"
    write_json_atomic(
        summary_path,
        {
            "status": "complete",
            "output_root": str(run_root),
            "run_spec": {
                "family_name": family,
                "dataset_split": split.value,
                "accepted_simulation_count": accepted_count,
            },
            "counts": {"accepted": accepted_count, "attempted": accepted_count + 1},
            "batch_paths": [str(batch.relative_to(run_root))],
        },
    )
    return summary_path


def _write_split(
    root: Path, split: DatasetSplit = DatasetSplit.TRAIN
) -> tuple[Path, ...]:
    return tuple(_write_run(root, family, split) for family in FAMILY_IDS)


class CombinedViewTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_combines_sparse_batches_in_split_and_family_order(self) -> None:
        paths = (*_write_split(self.root), *_write_split(self.root, DatasetSplit.TEST))
        summary_path = build_combined_view(
            tuple(reversed(paths)), output_root=self.root / "view"
        )
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["run_summaries"], list(map(str, paths)))
        self.assertEqual(summary["attempted_simulations"], 16)
        self.assertEqual(summary["accepted_simulations"], 8)
        self.assertEqual(summary["rows"], 20)
        manifest = json.loads(Path(summary["dataset_view"]["manifest"]).read_text())
        self.assertEqual(manifest["n_rows"], 20)
        self.assertEqual(
            manifest["split_counts"]["train"], {"attempted": 8, "accepted": 4}
        )
        self.assertEqual(
            manifest["split_counts"]["test"], {"attempted": 8, "accepted": 4}
        )
        with np.load(
            summary["dataset_view"]["trajectory_map"], allow_pickle=False
        ) as index:
            np.testing.assert_array_equal(
                index["trajectory_family_id"], [1, 1, 2, 2, 3, 3, 4, 4] * 2
            )
            np.testing.assert_array_equal(
                index["trajectory_accepted"], [False, True] * 8
            )
            np.testing.assert_array_equal(index["trajectory_first_row"][::2], [-1] * 8)
            np.testing.assert_array_equal(
                index["trajectory_row_count"][1::2], [1, 2, 3, 4] * 2
            )

    def test_requires_all_families_and_equal_accepted_counts(self) -> None:
        paths = _write_split(self.root)
        with self.assertRaisesRegex(ValueError, "all four"):
            build_combined_view(paths[:-1], output_root=self.root / "missing")
        unequal = _write_run(
            self.root / "unequal", "stokes", DatasetSplit.TRAIN, accepted_count=2
        )
        with self.assertRaisesRegex(ValueError, "equal accepted counts"):
            build_combined_view((unequal, *paths[1:]), output_root=self.root / "view")

    def test_rejects_duplicate_runs_and_batches(self) -> None:
        paths = _write_split(self.root)
        duplicate = _write_run(self.root / "duplicate", "stokes", DatasetSplit.TRAIN)
        with self.assertRaisesRegex(ValueError, "only one generation run"):
            build_combined_view((*paths, duplicate), output_root=self.root / "view")
        summary = json.loads(paths[0].read_text())
        alias = paths[0].parent / "alias.npz"
        alias.symlink_to(paths[0].parent / summary["batch_paths"][0])
        summary["batch_paths"].append(alias.name)
        write_json_atomic(paths[0], summary)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            build_combined_view(paths, output_root=self.root / "view")
        self.assertFalse((self.root / "view").exists())

    def test_cli_rejects_path_traversal_names_before_loading_runs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_paper_dataset_view",
                f"--run-summary={self.root / 'missing.json'}",
                f"--output-root={self.root / 'view'}",
                "--name=../escape",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--name must contain only", result.stderr)
        self.assertFalse((self.root / "view").exists())

    def test_rejects_inaccurate_attempted_or_accepted_counts(self) -> None:
        for count in (1, 2):
            with self.subTest(actual_accepted=count):
                root = self.root / str(count)
                paths = (
                    _write_run(
                        root, "stokes", DatasetSplit.TRAIN, accepted_count=count
                    ),
                    *(
                        _write_run(root, family, DatasetSplit.TRAIN)
                        for family in tuple(FAMILY_IDS)[1:]
                    ),
                )
                summary = json.loads(paths[0].read_text())
                summary["run_spec"]["accepted_simulation_count"] = 1
                summary["counts"]["accepted"] = 1
                summary["counts"]["attempted"] = 3
                write_json_atomic(paths[0], summary)
                with self.assertRaisesRegex(RuntimeError, "view counts disagree"):
                    build_combined_view(
                        paths, output_root=root / "view", name="combined"
                    )
                self.assertFalse((root / "view" / "combined.summary.json").exists())

    def test_rejects_misassigned_batches_and_undeclared_attempts(self) -> None:
        for corruption in ("swapped_families", "extra_split"):
            with self.subTest(corruption=corruption):
                root = self.root / corruption
                paths = _write_split(root)
                if corruption == "swapped_families":
                    for path, family_id in zip(paths[:2], (2, 1), strict=True):
                        batch = path.parent / "batches" / "batch_000000.npz"
                        arrays = load_npz(batch)
                        arrays["family_id"] = np.asarray(family_id, dtype=np.int16)
                        write_npz_atomic(batch, arrays)
                else:
                    extra = paths[-1].parent / "batches" / "extra.npz"
                    save_completed_batch(
                        extra,
                        ("extra",),
                        (None,),
                        family_id=FAMILY_IDS["jonswap_tma"],
                        dataset_split=DatasetSplit.TEST,
                    )
                    summary = json.loads(paths[-1].read_text())
                    summary["batch_paths"].append(
                        str(extra.relative_to(paths[-1].parent))
                    )
                    write_json_atomic(paths[-1], summary)
                with self.assertRaisesRegex(RuntimeError, "built view"):
                    build_combined_view(paths, output_root=root / "view")


if __name__ == "__main__":
    unittest.main()
