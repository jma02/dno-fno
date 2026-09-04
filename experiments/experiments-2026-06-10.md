# Experiments — 2026-06-10

| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-10 | --:-- | v3 single-step accuracy on soliton vs off-manifold states | Diagnose whether the shallow-steep rollout failures are learnable (single-step error there is fine, only rollouts fail) or a coverage gap. | v3 dataset has 2M soliton rows, half below `h=0.05`; "linear" training family stops at `h=0.1`. Single-step error on exact soliton states (including steepest failing rollouts): `~2e-4` (well-learned). Below `h=0.02` on-manifold single-step error is 10–70× elevated. Failing eval regime sits at `h=0.038` with only soliton coverage. | Confirms coverage-gap: soliton single-step fine, near-soliton off-manifold states poorly covered. | short | BACKFILL from notes/v4_dataset_report/v4_dataset_report.tex — ACCEPTED as motivation for v4 shallow-steep wavetrain family (not more solitons). |
