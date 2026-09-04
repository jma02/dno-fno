"""Import and dispatch tests for the paper-dataset generation launcher."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_paper_dataset import main


class PaperDatasetGenerationTests(unittest.TestCase):
    def test_import_does_not_initialize_jax(self) -> None:
        subprocess.run(
            (
                sys.executable,
                "-c",
                "import sys; import scripts.generate_paper_dataset; "
                "assert 'jax' not in sys.modules",
            ),
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )

    def test_generation_starts_without_execute_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "solver.gen_data.pipeline.dataset_generation.generate_simulations",
                side_effect=RuntimeError("generation started"),
            ) as generate:
                with self.assertRaisesRegex(RuntimeError, "generation started"):
                    main(
                        (
                            "--family",
                            "stokes",
                            "--split",
                            "validation",
                            "--accepted-simulations",
                            "4",
                            "--output-root",
                            directory,
                            "--batch-size",
                            "2",
                        )
                    )

        self.assertEqual(
            tuple(generate.call_args.kwargs["accepted_targets"].values()),
            (1, 1, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
