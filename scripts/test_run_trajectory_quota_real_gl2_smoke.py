"""Focused CPU tests for the real-GL2 trajectory quota smoke."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_trajectory_quota_real_gl2_smoke import (
    SMOKE_CONTRACT,
    build_execution,
    run_family,
)


def _batch_hashes(summary: dict[str, object]) -> tuple[str, ...]:
    batches = summary["batches"]
    assert isinstance(batches, list)
    hashes: list[str] = []
    for batch in batches:
        assert isinstance(batch, dict)
        artifacts = batch["artifacts"]
        assert isinstance(artifacts, dict)
        for name in ("proposal", "shard", "result"):
            artifact = artifacts[name]
            assert isinstance(artifact, dict)
            value = artifact["sha256"]
            assert isinstance(value, str)
            hashes.append(value)
    return tuple(hashes)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


class TrajectoryQuotaRealGL2SmokeTests(unittest.TestCase):
    def test_real_bf_complete_resume_and_config_refusal(self) -> None:
        """Exercise the production arm, immutable resume, and fail-closed scan."""

        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "smoke"
            first = run_family(output, "benjamin_feir")
            self.assertTrue(first.state.complete)
            self.assertEqual(
                first.execution.role,
                "reduced_wiring_evidence_only",
            )
            counts = first.summary["counts"]
            self.assertIsInstance(counts, dict)
            assert isinstance(counts, dict)
            self.assertGreaterEqual(counts["attempted"], 1)
            self.assertEqual(counts["accepted"], 1)
            self.assertEqual(
                counts["rejected"],
                counts["attempted"] - counts["accepted"],
            )

            view = first.summary["dataset_view"]
            assert isinstance(view, dict)
            loader = view["loader_validation"]
            assert isinstance(loader, dict)
            self.assertEqual(loader["loaded_rows"], 3)
            self.assertEqual(loader["spatial_size"], 64)
            self.assertTrue(loader["all_fields_and_scalars_finite"])

            accepted_cases = [
                case
                for case in first.summary["cases"]
                if isinstance(case, dict) and case["accepted"]
            ]
            self.assertEqual(len(accepted_cases), 1)
            diagnostics = accepted_cases[0]["diagnostics"]
            assert isinstance(diagnostics, dict)
            self.assertTrue(diagnostics["complete_admissible_trajectory"])
            self.assertTrue(diagnostics["all_stages_solved"])
            self.assertIsNotNone(diagnostics["maximum_stage_residual"])
            self.assertEqual(
                diagnostics["production_dt"],
                SMOKE_CONTRACT.production_dt,
            )

            json.loads(
                json.dumps(first.summary, allow_nan=False),
                parse_constant=_reject_nonfinite,
            )
            immutable_hashes = _batch_hashes(first.summary)

            resumed = run_family(output, "benjamin_feir")
            self.assertTrue(resumed.state.complete)
            resume = resumed.summary["resume"]
            assert isinstance(resume, dict)
            self.assertTrue(resume["had_existing_batches"])
            self.assertGreaterEqual(resume["initial_committed_batches"], 1)
            self.assertEqual(_batch_hashes(resumed.summary), immutable_hashes)

            incompatible_contract = replace(
                SMOKE_CONTRACT,
                gl2_residual_tolerance=5.0e-9,
            )
            incompatible = build_execution(
                "benjamin_feir",
                contract=incompatible_contract,
            )
            self.assertNotEqual(incompatible, first.execution)
            with self.assertRaisesRegex(
                RuntimeError,
                "configuration fingerprint does not match",
            ):
                run_family(
                    output,
                    "benjamin_feir",
                    contract=incompatible_contract,
                )


if __name__ == "__main__":
    unittest.main()
