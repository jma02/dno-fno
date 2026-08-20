"""Verify the inert shared-source snapshot for Benjamin--Feir revision 4."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[1]
SNAPSHOT: Final = (
    ROOT / "reproducibility/source_snapshots/benjamin_feir_revision4_e16773f"
)
EXPECTED: Final = {
    "production.py": (
        "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4"
    ),
    "trajectory_family_adapters.py": (
        "afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a"
    ),
}


class BenjaminFeirRevision4SourceSnapshotTest(unittest.TestCase):
    def test_manifest_and_archived_bytes_have_recorded_hashes(self) -> None:
        manifest_entries = {
            path: digest
            for digest, path in map(
                str.split,
                (SNAPSHOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines(),
            )
        }
        self.assertEqual(manifest_entries, EXPECTED)

        actual = {
            path: sha256((SNAPSHOT / path).read_bytes()).hexdigest()
            for path in EXPECTED
        }
        self.assertEqual(actual, EXPECTED)


if __name__ == "__main__":
    unittest.main()
