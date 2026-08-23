# Stokes revision-2 frozen generation sources

This directory preserves the exact historical bytes of the three generation
files recorded by every completed Stokes revision-2 chunk that subsequently
changed in the working tree. The bytes were recovered from Git commit
`60a28ffae394465c6ea295eb4ed6c075fbc756a4` and verified against all six
immutable chunk source maps before being added here.

| Archived file | Original run path | Bytes | SHA-256 |
|---|---|---:|---|
| `run_paper_corpus_quota.py` | `scripts/run_paper_corpus_quota.py` | 33,974 | `ab99c067c2f22823fce861a69fd03bd8cc4524effbfd5ab1a5360bfa5e615d98` |
| `manifest.py` | `solver/gen_data/pipeline/manifest.py` | 28,988 | `e2cc1a20cc01fef9dd0bb84b99485b5cf90da71f7ee035a813893f2f6a37f421` |
| `production.py` | `solver/gen_data/pipeline/production.py` | 7,854 | `8ed36cf1bd57494f344096e4a508bcfb37b9103cf03ace8a63df2656c11dc735` |

Verify the archived bytes from this directory with:

```sh
sha256sum --check SHA256SUMS
```

This is an inert historical snapshot. It is deliberately outside `solver/`
and `scripts/`, has no package initializer, and must not be imported or used
as the current generator. The strengthened Stokes completion binding
authenticates this checksum file and all three snapshots against all six chunk
source maps, while physically rehashing the other 11 recorded sources against
the current repository.

The resulting release binding is
`outputs/paper_corpus_cap4_revision2_20260728/stokes_completion_binding.json`,
SHA-256
`d2223fddeedebd541ecdb19752ca547bbd0325ef886c6d97de821b24935dd4f5`.
It reports 18,432 attempted and accepted cases, zero rejections, 18,432 rows,
and exact per-cell counts across all three splits.

This directory is not a self-contained executable checkout. The immutable run
records, the legacy full numerical audit, and the strengthened completion
binding remain authoritative for the run specifications, dependencies, corpus
transactions, arrays, and current-source identity.
