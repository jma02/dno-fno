### Experiments

Session rule: append every meaningful experiment to the dated log for the day it
ran, `experiments/experiments-<YYYY-MM-DD>.md`, with correctness, timing, and
merge decision. Create that file from the header below if the day has no log yet,
and add its row to the index in this file. This file is the index only — it holds
no experiment rows.

Logging format:
- Use local `HH:MM` time from `date` when adding new rows.
- Use `--:--` only for older rows where exact time was not recorded; do not invent precision.
- Fill both `Motivation` and `What Tried / Evidence`.
- `Motivation` should say why this was plausible before running it.
- `What Tried / Evidence` should say what changed and what the result taught us.
- Keep timings in `HH:MM` format.
- Rows within a dated file run chronologically, earliest first.

Every dated file uses the same eight-column header:

```
| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
```

**Note on backfilled rows:** rows whose `Decision` column begins with `BACKFILL
from notes/<file> —` were reconstructed post-hoc by scanning `notes/` (design
docs, session logs, LaTeX reports) rather than logged as they ran. Numbers and
dates in those rows are as reported in the source note, with no re-verification;
`Time` is `--:--` because exact clock time was not recorded. They sit in the
dated file matching the date the source note reported. The three rows whose date
could not be recovered at all are in
[`experiments/experiments-unknown-date.md`](experiments/experiments-unknown-date.md).

## Dated logs

63 files, 919 rows.

| Date | Rows | File | Size |
| --- | --- | --- | --- |
| 2026-09-08 | 14 | [`experiments/experiments-2026-09-08.md`](experiments/experiments-2026-09-08.md) | 14 KB |
| 2026-09-06 | 3 | [`experiments/experiments-2026-09-06.md`](experiments/experiments-2026-09-06.md) | 4 KB |
| 2026-09-05 | 9 | [`experiments/experiments-2026-09-05.md`](experiments/experiments-2026-09-05.md) | 12 KB |
| 2026-09-04 | 2 | [`experiments/experiments-2026-09-04.md`](experiments/experiments-2026-09-04.md) | 3 KB |
| 2026-09-02 | 3 | [`experiments/experiments-2026-09-02.md`](experiments/experiments-2026-09-02.md) | 4 KB |
| 2026-08-20 | 10 | [`experiments/experiments-2026-08-20.md`](experiments/experiments-2026-08-20.md) | 18 KB |
| 2026-08-19 | 1 | [`experiments/experiments-2026-08-19.md`](experiments/experiments-2026-08-19.md) | 1 KB |
| 2026-08-18 | 2 | [`experiments/experiments-2026-08-18.md`](experiments/experiments-2026-08-18.md) | 2 KB |
| 2026-08-17 | 2 | [`experiments/experiments-2026-08-17.md`](experiments/experiments-2026-08-17.md) | 2 KB |
| 2026-08-13 | 94 | [`experiments/experiments-2026-08-13.md`](experiments/experiments-2026-08-13.md) | 135 KB |
| 2026-08-12 | 96 | [`experiments/experiments-2026-08-12.md`](experiments/experiments-2026-08-12.md) | 126 KB |
| 2026-08-11 | 107 | [`experiments/experiments-2026-08-11.md`](experiments/experiments-2026-08-11.md) | 138 KB |
| 2026-08-10 | 79 | [`experiments/experiments-2026-08-10.md`](experiments/experiments-2026-08-10.md) | 112 KB |
| 2026-08-09 | 121 | [`experiments/experiments-2026-08-09.md`](experiments/experiments-2026-08-09.md) | 193 KB |
| 2026-08-08 | 1 | [`experiments/experiments-2026-08-08.md`](experiments/experiments-2026-08-08.md) | 1 KB |
| 2026-08-07 | 1 | [`experiments/experiments-2026-08-07.md`](experiments/experiments-2026-08-07.md) | 1 KB |
| 2026-08-06 | 2 | [`experiments/experiments-2026-08-06.md`](experiments/experiments-2026-08-06.md) | 3 KB |
| 2026-08-05 | 1 | [`experiments/experiments-2026-08-05.md`](experiments/experiments-2026-08-05.md) | 1 KB |
| 2026-08-04 | 4 | [`experiments/experiments-2026-08-04.md`](experiments/experiments-2026-08-04.md) | 7 KB |
| 2026-08-02 | 36 | [`experiments/experiments-2026-08-02.md`](experiments/experiments-2026-08-02.md) | 65 KB |
| 2026-08-01 | 14 | [`experiments/experiments-2026-08-01.md`](experiments/experiments-2026-08-01.md) | 29 KB |
| 2026-07-31 | 23 | [`experiments/experiments-2026-07-31.md`](experiments/experiments-2026-07-31.md) | 47 KB |
| 2026-07-30 | 6 | [`experiments/experiments-2026-07-30.md`](experiments/experiments-2026-07-30.md) | 16 KB |
| 2026-07-29 | 1 | [`experiments/experiments-2026-07-29.md`](experiments/experiments-2026-07-29.md) | 2 KB |
| 2026-07-28 | 14 | [`experiments/experiments-2026-07-28.md`](experiments/experiments-2026-07-28.md) | 26 KB |
| 2026-07-27 | 6 | [`experiments/experiments-2026-07-27.md`](experiments/experiments-2026-07-27.md) | 11 KB |
| 2026-07-26 | 16 | [`experiments/experiments-2026-07-26.md`](experiments/experiments-2026-07-26.md) | 37 KB |
| 2026-07-25 | 47 | [`experiments/experiments-2026-07-25.md`](experiments/experiments-2026-07-25.md) | 86 KB |
| 2026-07-24 | 3 | [`experiments/experiments-2026-07-24.md`](experiments/experiments-2026-07-24.md) | 7 KB |
| 2026-07-23 | 9 | [`experiments/experiments-2026-07-23.md`](experiments/experiments-2026-07-23.md) | 23 KB |
| 2026-07-22 | 5 | [`experiments/experiments-2026-07-22.md`](experiments/experiments-2026-07-22.md) | 12 KB |
| 2026-07-21 | 3 | [`experiments/experiments-2026-07-21.md`](experiments/experiments-2026-07-21.md) | 7 KB |
| 2026-07-20 | 5 | [`experiments/experiments-2026-07-20.md`](experiments/experiments-2026-07-20.md) | 11 KB |
| 2026-07-19 | 3 | [`experiments/experiments-2026-07-19.md`](experiments/experiments-2026-07-19.md) | 6 KB |
| 2026-07-17 | 14 | [`experiments/experiments-2026-07-17.md`](experiments/experiments-2026-07-17.md) | 30 KB |
| 2026-07-16 | 16 | [`experiments/experiments-2026-07-16.md`](experiments/experiments-2026-07-16.md) | 28 KB |
| 2026-07-15 | 15 | [`experiments/experiments-2026-07-15.md`](experiments/experiments-2026-07-15.md) | 41 KB |
| 2026-07-14 | 8 | [`experiments/experiments-2026-07-14.md`](experiments/experiments-2026-07-14.md) | 16 KB |
| 2026-07-13 | 11 | [`experiments/experiments-2026-07-13.md`](experiments/experiments-2026-07-13.md) | 22 KB |
| 2026-07-12 | 7 | [`experiments/experiments-2026-07-12.md`](experiments/experiments-2026-07-12.md) | 15 KB |
| 2026-07-11 | 2 | [`experiments/experiments-2026-07-11.md`](experiments/experiments-2026-07-11.md) | 3 KB |
| 2026-07-10 | 7 | [`experiments/experiments-2026-07-10.md`](experiments/experiments-2026-07-10.md) | 13 KB |
| 2026-07-09 | 23 | [`experiments/experiments-2026-07-09.md`](experiments/experiments-2026-07-09.md) | 40 KB |
| 2026-07-08 | 9 | [`experiments/experiments-2026-07-08.md`](experiments/experiments-2026-07-08.md) | 18 KB |
| 2026-07-07 | 7 | [`experiments/experiments-2026-07-07.md`](experiments/experiments-2026-07-07.md) | 13 KB |
| 2026-07-06 | 9 | [`experiments/experiments-2026-07-06.md`](experiments/experiments-2026-07-06.md) | 15 KB |
| 2026-07-04 | 2 | [`experiments/experiments-2026-07-04.md`](experiments/experiments-2026-07-04.md) | 4 KB |
| 2026-07-01 | 5 | [`experiments/experiments-2026-07-01.md`](experiments/experiments-2026-07-01.md) | 7 KB |
| 2026-06-29 | 1 | [`experiments/experiments-2026-06-29.md`](experiments/experiments-2026-06-29.md) | 1 KB |
| 2026-06-25 | 1 | [`experiments/experiments-2026-06-25.md`](experiments/experiments-2026-06-25.md) | 1 KB |
| 2026-06-24 | 7 | [`experiments/experiments-2026-06-24.md`](experiments/experiments-2026-06-24.md) | 10 KB |
| 2026-06-22 | 2 | [`experiments/experiments-2026-06-22.md`](experiments/experiments-2026-06-22.md) | 2 KB |
| 2026-06-18 | 3 | [`experiments/experiments-2026-06-18.md`](experiments/experiments-2026-06-18.md) | 3 KB |
| 2026-06-16 | 1 | [`experiments/experiments-2026-06-16.md`](experiments/experiments-2026-06-16.md) | 1 KB |
| 2026-06-11 | 1 | [`experiments/experiments-2026-06-11.md`](experiments/experiments-2026-06-11.md) | 1 KB |
| 2026-06-10 | 1 | [`experiments/experiments-2026-06-10.md`](experiments/experiments-2026-06-10.md) | 0 KB |
| 2026-06-09 | 3 | [`experiments/experiments-2026-06-09.md`](experiments/experiments-2026-06-09.md) | 3 KB |
| 2026-06-05 | 4 | [`experiments/experiments-2026-06-05.md`](experiments/experiments-2026-06-05.md) | 4 KB |
| 2026-06-04 | 4 | [`experiments/experiments-2026-06-04.md`](experiments/experiments-2026-06-04.md) | 4 KB |
| 2026-05-03 | 3 | [`experiments/experiments-2026-05-03.md`](experiments/experiments-2026-05-03.md) | 2 KB |
| 2026-05-02 | 3 | [`experiments/experiments-2026-05-02.md`](experiments/experiments-2026-05-02.md) | 3 KB |
| 2026-03-31 | 4 | [`experiments/experiments-2026-03-31.md`](experiments/experiments-2026-03-31.md) | 3 KB |
| undated | 3 | [`experiments/experiments-unknown-date.md`](experiments/experiments-unknown-date.md) | 2 KB |

## Notes-with-no-extractable-experiments (backfill scan reported)

- `notes/cs_dno_jcp09_improvement_plan.md` — pure design/roadmap doc; completed tiers already logged in the dated files.
- `notes/tanaka_h1_design.md` — design doc, no experiments attached.
- `notes/linear_exact_design.md` — design doc.
- `notes/adaptive_sampling/adaptive_sampling.tex` — design + structural smoke only, no numeric error metrics.
- `notes/rollout_stability_approaches/rollout_stability_approaches.tex` — narrative summary superseded by `rollout_accuracy_report.tex` per-regime numbers.
- `notes/dno_nls_derivation/dno_nls_derivation.tex` — pure derivation.
- `notes/rescaling/rescaling.tex` — pure dimensional analysis.
- `notes/random_sea_generation/random_sea_generation.tex` — dataset generation procedure + statistics, no hypothesis-and-test structure.
- `notes/family_coverage_v5.json` — dataset composition only.
