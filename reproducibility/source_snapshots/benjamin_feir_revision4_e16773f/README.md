# Benjamin--Feir revision-4 frozen shared sources

This directory preserves the exact bytes of the two shared generator modules
recorded by every completed Benjamin--Feir revision-4 chunk.  The files were
recovered from the isolated checkout at `/tmp/dno-fno-bf-frozen-e16773f` and
verified byte-for-byte before being added here.

| Archived file | Original run path | Bytes | SHA-256 |
|---|---|---:|---|
| `production.py` | `solver/gen_data/pipeline/production.py` | 8,494 | `2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4` |
| `trajectory_family_adapters.py` | `solver/gen_data/trajectory_family_adapters.py` | 28,852 | `afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a` |

The six completed chunk summaries under
`outputs/paper_corpus_bf_revision4_jonswap_revision3_literature_aligned_v1`
record these two hashes.  The strengthened source-bound completion audit is
`benjamin_feir_completion_audit.json` in that output root, with SHA-256
`c155b35276d0cfef6844f6b0db3e91da3ce15579ad8b0f0ca482f44cc912753c`,
status `pass`, and counts 18,432 accepted, 18,455 attempted, 23 rejected, and
3,686,400 retained rows.  It authenticates this checksum file and both
snapshots against all six chunk source maps, physically rehashes the other 17
recorded sources against the current repository, and records full-source
fingerprint
`d1ba68f08a5316d0b9ce0e399d48896d766044615504e461592eaee7ff3325ce`.

Verify the archived bytes from this directory with:

```sh
sha256sum --check SHA256SUMS
```

This is an inert historical snapshot.  It is deliberately outside `solver/`
and `scripts/`, has no package initializer, and must not be imported or used
as the current generator.  It does not alter any live source map.

This directory is also not a self-contained executable checkout.  It
preserves only the two shared files whose current working-tree versions later
diverged.  The immutable run records and completion audit remain authoritative
for the other source files, dependency versions, run specifications, and
corpus artifacts.
