"""Import and dispatch tests for the paper-dataset generation launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_paper_dataset import main
from solver.gen_data.pipeline.types import PhysicalFamilyId
from solver.gen_data.stokes_sampling import STOKES_PARAMETER_GROUPS


class PaperDatasetGenerationTests(unittest.TestCase):
    def test_completed_generation_only_saves_batches_and_summary(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("jax.config.update"),
            patch("jax.default_backend", return_value="cpu"),
            patch("builtins.print"),
            patch("scripts.generate_paper_dataset.generate_simulations") as generate,
        ):
            root = Path(directory)
            batches = [root / "batches" / "batch_000000.npz"]
            generate.return_value = (
                {group: 1 for group in STOKES_PARAMETER_GROUPS},
                batches,
            )
            main(
                (
                    "--family",
                    "stokes",
                    "--seed",
                    "42",
                    "--num-simulations",
                    "4",
                    "--output-root",
                    directory,
                    "--batch-size",
                    "2",
                )
            )
            generate.assert_called_once()
            self.assertEqual(
                generate.call_args.kwargs["family_id"], PhysicalFamilyId.STOKES
            )
            self.assertEqual(generate.call_args.kwargs["seed"], 42)
            files = list(root.iterdir())
            self.assertEqual(len(files), 1)
            summary = json.loads(files[0].read_text())
            self.assertEqual(files[0].name, "paper_dataset_stokes.summary.json")
            self.assertEqual(summary["run_spec"]["seed"], 42)
            self.assertNotIn("dataset_split", summary["run_spec"])
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["batch_paths"], ["batches/batch_000000.npz"])
            self.assertEqual(summary["counts"]["accepted"], 4)

    def test_import_sets_float64_without_initializing_backends(self) -> None:
        subprocess.run(
            (
                sys.executable,
                "-c",
                "import scripts.generate_paper_dataset; "
                "from jax._src.xla_bridge import backends_are_initialized; "
                "from solver.tanaka_ICs.modified_tanaka import TANAKA_DTYPE_NAME; "
                "assert not backends_are_initialized(); "
                "assert TANAKA_DTYPE_NAME == 'float64'",
            ),
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "DNO_TANAKA_DTYPE": "float32",
                "JAX_PLATFORMS": "unused",
            },
            check=True,
        )

    def test_generation_starts_without_execute_flag(self) -> None:
        for flags, backend in (((), "cpu"), (("--gpu",), "gpu")):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}),
                patch("jax.config.update") as configure,
                patch("jax.default_backend", return_value=backend),
                patch(
                    "scripts.generate_paper_dataset.generate_simulations",
                    side_effect=RuntimeError("generation started"),
                ) as generate,
            ):
                with self.assertRaisesRegex(RuntimeError, "generation started"):
                    main(
                        (
                            *flags,
                            "--family",
                            "stokes",
                            "--num-simulations",
                            "4",
                            "--output-root",
                            directory,
                            "--batch-size",
                            "2",
                        )
                    )

                configure.assert_any_call("jax_enable_x64", True)
                self.assertEqual(generate.call_args.kwargs["seed"], 2026072210)
                configure.assert_any_call("jax_platforms", "cuda" if flags else "cpu")
                self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"], "7" if flags else ""
                )
                self.assertEqual(
                    tuple(
                        generate.call_args.kwargs[
                            "requested_simulations_per_group"
                        ].values()
                    ),
                    (1, 1, 1, 1),
                )


if __name__ == "__main__":
    unittest.main()
