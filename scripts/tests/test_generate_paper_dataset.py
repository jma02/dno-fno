"""Import and dispatch tests for the paper-dataset generation launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_paper_dataset import main


class PaperDatasetGenerationTests(unittest.TestCase):
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
                            "--split",
                            "validation",
                            "--num-simulations",
                            "4",
                            "--output-root",
                            directory,
                            "--batch-size",
                            "2",
                        )
                    )

                configure.assert_any_call("jax_enable_x64", True)
                configure.assert_any_call("jax_platforms", "cuda" if flags else "cpu")
                self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"], "7" if flags else ""
                )
                self.assertEqual(
                    tuple(generate.call_args.kwargs["accepted_targets"].values()),
                    (1, 1, 1, 1),
                )


if __name__ == "__main__":
    unittest.main()
