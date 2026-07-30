"""Focused CPU tests for the resumable static Stokes quota pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.run_static_stokes_quota_pilot import run_pilot
from solver.gen_data.pipeline.reference import DiscreteDnoTarget
from solver.gen_data.stokes_population import STOKES_POPULATION_CELLS
from solver.gen_data.stokes_static_pipeline import StaticStokesContract


def _contract(*, dno_order: int = 2) -> StaticStokesContract:
    return StaticStokesContract.reduced_wiring_evidence(
        DiscreteDnoTarget(
            nx=64,
            length=2.0 * math.pi,
            dno_order=dno_order,
            pad_factor=2,
            maximum_wavenumber=24.0,
        )
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


class StaticStokesQuotaPilotTests(unittest.TestCase):
    def test_complete_resume_and_incompatible_config_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "pilot"
            first = run_pilot(
                output,
                contract=_contract(),
                maximum_ursell_redraws=0,
            )

            self.assertTrue(first.state.complete)
            counts = first.summary["counts"]
            self.assertIsInstance(counts, dict)
            assert isinstance(counts, dict)
            self.assertEqual(
                {
                    "attempted": counts["attempted"],
                    "accepted": counts["accepted"],
                    "rejected": counts["rejected"],
                },
                {"attempted": 4, "accepted": 4, "rejected": 0},
            )
            by_cell = counts["by_cell"]
            self.assertIsInstance(by_cell, dict)
            assert isinstance(by_cell, dict)
            self.assertEqual(
                set(by_cell),
                {cell.cell_id for cell in STOKES_POPULATION_CELLS},
            )
            self.assertTrue(
                all(
                    value
                    == {
                        "target_accepted": 1,
                        "attempted": 1,
                        "accepted": 1,
                        "rejected": 0,
                    }
                    for value in by_cell.values()
                )
            )
            loader = first.summary["dataset_view"]
            assert isinstance(loader, dict)
            loader_validation = loader["loader_validation"]
            assert isinstance(loader_validation, dict)
            self.assertEqual(loader_validation["loaded_rows"], 4)
            self.assertEqual(loader_validation["spatial_size"], 64)
            self.assertTrue(loader_validation["all_fields_finite"])

            strict_summary = json.loads(
                first.summary_path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite,
            )
            self.assertEqual(
                strict_summary["configuration_fingerprint"],
                first.summary["configuration_fingerprint"],
            )
            immutable_batch_hashes = _batch_hashes(first.summary)

            resumed = run_pilot(
                output,
                contract=_contract(),
                maximum_ursell_redraws=0,
            )
            self.assertTrue(resumed.state.complete)
            resume = resumed.summary["resume"]
            assert isinstance(resume, dict)
            self.assertTrue(resume["had_existing_batches"])
            self.assertEqual(resume["initial_committed_batches"], 1)
            self.assertEqual(_batch_hashes(resumed.summary), immutable_batch_hashes)

            with self.assertRaisesRegex(
                RuntimeError,
                "configuration fingerprint does not match",
            ):
                run_pilot(
                    output,
                    contract=_contract(dno_order=1),
                    maximum_ursell_redraws=0,
                )


if __name__ == "__main__":
    unittest.main()
