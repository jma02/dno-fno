from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_jonswap_relative_band_fresh_gate import (
    AGGREGATE_NAME,
    SUMMARY_NAME,
    audit,
)
from solver.gen_data.jonswap_tma_sampling import (
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)


def _summary(stream_id: int, *, rejected: int) -> dict[str, object]:
    cells = JONSWAP_TMA_SAMPLE_CELL_IDS
    by_cell = {
        cell_id: {
            "accepted": 10,
            "attempted": 10 + (rejected if index == 0 else 0),
            "rejected": rejected if index == 0 else 0,
            "target_accepted": 10,
        }
        for index, cell_id in enumerate(cells)
    }
    return {
        "status": "complete",
        "configuration_fingerprint": f"fingerprint-{stream_id}",
        "run_spec": {
            "family_name": "jonswap_tma",
            "family_id": 4,
            "revision_id": 4,
            "split_id": "validation",
            "stream_id": stream_id,
            "batch_size": 32,
            "maximum_attempts_per_accepted_case": 4,
            "quotas": [
                {"cell_id": cell_id, "target_accepted": 10} for cell_id in cells
            ],
        },
        "counts": {
            "accepted": 270,
            "attempted": 270 + rejected,
            "rejected": rejected,
            "by_cell": by_cell,
            "rejection_reasons": ({"GL2_STAGE_RESIDUAL": rejected} if rejected else {}),
        },
    }


class FreshGateAuditTests(unittest.TestCase):
    def _write_summaries(self, root: Path, *, rejected_per_stream: int) -> None:
        for stream_id in (941, 942):
            stream_root = root / f"stream_{stream_id}"
            stream_root.mkdir(parents=True)
            (stream_root / SUMMARY_NAME).write_text(
                json.dumps(
                    _summary(stream_id, rejected=rejected_per_stream),
                    allow_nan=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def test_ten_rejections_in_550_attempts_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summaries(root, rejected_per_stream=5)
            result = audit(root, write=True)
            self.assertEqual(result["status"], "pass")
            observed = result["observed"]
            assert isinstance(observed, dict)
            self.assertEqual(observed["attempted"], 550)
            self.assertEqual(observed["rejected"], 10)
            self.assertTrue((root / AGGREGATE_NAME).is_file())

    def test_twelve_rejections_in_552_attempts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summaries(root, rejected_per_stream=6)
            result = audit(root, write=False)
            self.assertEqual(result["status"], "fail")
            observed = result["observed"]
            assert isinstance(observed, dict)
            self.assertEqual(observed["attempted"], 552)
            self.assertEqual(observed["rejected"], 12)

    def test_changed_stream_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summaries(root, rejected_per_stream=0)
            path = root / "stream_942" / SUMMARY_NAME
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["run_spec"]["revision_id"] = 3
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run identity"):
                audit(root, write=False)


if __name__ == "__main__":
    unittest.main()
