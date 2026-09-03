"""No-rollout tests for the paper-dataset generation launcher."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.generate_paper_dataset import (
    BOOTSTRAP_PLATFORM,
    FAMILY_PARAMETER_GROUP_IDS,
    GenerationRequest,
    build_chunk_config,
    main,
)
from solver.gen_data.pipeline.simulation_allocation import DatasetSplit


class PaperDatasetGenerationTests(unittest.TestCase):
    def test_balanced_quotas_for_every_family(self) -> None:
        accepted_by_family = {
            "stokes": 9,
            "tanaka": 11,
            "benjamin_feir": 68,
            "jonswap_tma": 29,
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, accepted_simulations in accepted_by_family.items():
                with self.subTest(family=family):
                    request = GenerationRequest(
                        output_root=Path(directory) / family,
                        family=family,  # type: ignore[arg-type]
                        split=DatasetSplit.VALIDATION,
                        accepted_simulations=accepted_simulations,
                        batch_size=3,
                        platform=BOOTSTRAP_PLATFORM,
                        solver_batch_size=3 if family == "jonswap_tma" else None,
                    )
                    chunk_config = build_chunk_config(request)
                    targets = tuple(chunk_config.simulation_targets.values())
                    self.assertEqual(sum(targets), accepted_simulations)
                    self.assertLessEqual(max(targets) - min(targets), 1)
                    if family == "tanaka":
                        self.assertEqual(targets, (1,) * 11)
                    self.assertEqual(
                        tuple(chunk_config.simulation_targets),
                        FAMILY_PARAMETER_GROUP_IDS[family],  # type: ignore[index]
                    )

    def test_default_cli_mode_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(
                    (
                        "--family",
                        "tanaka",
                        "--split",
                        "validation",
                        "--accepted-simulations",
                        "5",
                        "--output-root",
                        str(output),
                        "--batch-size",
                        "2",
                    )
                )

            plan = json.loads(stdout.getvalue())
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
            request = GenerationRequest(
                output_root=Path(directory) / "jonswap",
                family="jonswap_tma",
                split=DatasetSplit.TEST,
                accepted_simulations=27,
                batch_size=27,
                platform=BOOTSTRAP_PLATFORM,
                solver_batch_size=9,
            )
            chunk_config = build_chunk_config(request)

            self.assertEqual(chunk_config.solver_batch_size, 9)
            self.assertEqual(chunk_config.to_json_record()["solver_batch_size"], 9)

    def test_solver_batch_size_is_only_valid_for_jonswap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only used for JONSWAP"):
                GenerationRequest(
                    output_root=Path(directory),
                    family="tanaka",
                    split=DatasetSplit.TEST,
                    accepted_simulations=4,
                    batch_size=2,
                    platform=BOOTSTRAP_PLATFORM,
                    solver_batch_size=1,
                )


if __name__ == "__main__":
    unittest.main()
