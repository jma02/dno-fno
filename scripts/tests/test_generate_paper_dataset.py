"""No-rollout tests for the paper-dataset generation launcher."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

from scripts.generate_paper_dataset import main


def _dry_run(
    output_root: Path,
    *,
    family: str,
    accepted_simulations: int,
    batch_size: int,
    solver_batch_size: int | None = None,
) -> dict[str, Any]:
    arguments = [
        "--family",
        family,
        "--split",
        "validation",
        "--accepted-simulations",
        str(accepted_simulations),
        "--output-root",
        str(output_root),
        "--batch-size",
        str(batch_size),
    ]
    if solver_batch_size is not None:
        arguments.extend(("--solver-batch-size", str(solver_batch_size)))
    stdout = StringIO()
    with redirect_stdout(stdout):
        main(arguments)
    return json.loads(stdout.getvalue())


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

    def test_balanced_quotas_for_every_family(self) -> None:
        accepted_by_family = {
            "stokes": 9,
            "tanaka": 11,
            "benjamin_feir": 68,
            "jonswap_tma": 29,
        }
        parameter_group_count = {
            "stokes": 4,
            "tanaka": 11,
            "benjamin_feir": 66,
            "jonswap_tma": 27,
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, accepted_simulations in accepted_by_family.items():
                with self.subTest(family=family):
                    plan = _dry_run(
                        Path(directory) / family,
                        family=family,
                        accepted_simulations=accepted_simulations,
                        batch_size=3,
                        solver_batch_size=3 if family == "jonswap_tma" else None,
                    )
                    targets = tuple(
                        quota["target_accepted"] for quota in plan["run_spec"]["quotas"]
                    )
                    self.assertEqual(len(targets), parameter_group_count[family])
                    self.assertEqual(sum(targets), accepted_simulations)
                    self.assertLessEqual(max(targets) - min(targets), 1)
                    if family == "tanaka":
                        self.assertEqual(targets, (1,) * 11)

    def test_default_cli_mode_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            plan = _dry_run(
                output,
                family="tanaka",
                accepted_simulations=5,
                batch_size=2,
            )
            self.assertFalse(output.exists())
            self.assertEqual(plan["mode"], "dry_run")
            self.assertEqual(plan["completed_batches"], 0)
            self.assertEqual(sum(plan["accepted_simulation_counts"].values()), 0)
            run_spec = plan["run_spec"]
            self.assertEqual(run_spec["accepted_simulation_count"], 5)
            self.assertEqual(
                [quota["target_accepted"] for quota in run_spec["quotas"]],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            )

    def test_jonswap_config_uses_solver_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = _dry_run(
                Path(directory) / "jonswap",
                family="jonswap_tma",
                accepted_simulations=27,
                batch_size=27,
                solver_batch_size=9,
            )
            self.assertEqual(plan["run_spec"]["solver_batch_size"], 9)

    def test_solver_batch_size_is_only_valid_for_jonswap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only used for JONSWAP"):
                _dry_run(
                    Path(directory),
                    family="tanaka",
                    accepted_simulations=4,
                    batch_size=2,
                    solver_batch_size=1,
                )


if __name__ == "__main__":
    unittest.main()
