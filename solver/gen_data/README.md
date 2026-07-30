# Data-generation contract

The physical families have different initial-condition mathematics, but they
must share one source-record, reference-solver, quality, target, and splitting
contract.  A generator shard is not a physical family: historical names such
as `g0`, `g1`, and `modal` identify independent runs only.

Five terms are used consistently here:

- a **constructor** maps one stored case specification to its periodic initial
  state;
- a **parameter category** is a predeclared part of one family's parameter
  range with its own accepted-case quota (the code stores its name as
  `cell_id`);
- a **validation panel** is a small, predeclared set of cases used to test a
  numerical contract and does not produce training data;
- a **writer** stores proposals, immutable field shards, decisions, and failure
  records;
- the **corpus** is the collection of accepted cases exposed to the loader by
  a dataset view.

## Order of operations

For rollout-derived data, the unit of generation and rejection is a complete
candidate trajectory:

1. assign the attempted case its physical family, generator revision,
   parameter category, train/validation/test split, and random-stream key;
2. sample its family-specific parameters and atomically store the complete
   proposal before constructing a field;
3. construct the periodic initial state on the declared grid, rejecting only
   a declared failure of the family's mathematical construction domain;
4. run the reference integrator once with GL2 step \(0.01\), save every
   \(0.08\), evaluate the common target on that saved grid, and record
   numerical diagnostics;
5. reject the case if and only if no complete admissible trajectory exists on
   its requested interval;
6. select training times only from the retained accepted trajectory;
7. take the selected
   \((\eta,\xi,G_{\mathrm{ref}}(\eta)\xi)\) rows;
8. commit only complete accepted trajectories, while retaining a decision for
   every attempted case;
9. build a dataset view that uses the split assigned in step 1 rather than
   making a row-level split.

Static Stokes cases use the same target and source-record schema, but each
independent state is its own quality unit.  Airy states appear only in
method-validation tests; they are not a fifth corpus family.  Activity-based
temporal sampling changes which accepted frames are stored; it is not a
quality filter.  The paper corpus target is the float64 order-six, pad-eight
Craig--Sulem value projected to the fixed delivered band \(|k|\leq128\), then
stored in float32.

## Quality semantics

`pipeline.QualityDecision` keeps three masks:

- `required`: checks that define acceptance under the declared policy;
- `evaluated`: checks that were actually run;
- `failed`: evaluated checks that failed.

A unit is accepted only when every required check was evaluated and no
required check failed.  Separate masks prevent old, unevaluated data from
being described as having passed a newer audit.  The stable reason vocabulary
is defined in `pipeline/quality.py`.

After a proposal is stored, acceptance has two stages.  First, the
family-specific formulas must produce an initial state in their declared
mathematical construction domain.  Second, a trajectory family must produce
one complete admissible trajectory under the frozen numerical method.
Unexpected constructor or solver errors stop generation; they are not
converted into rejected cases.

Every initial condition is sampled from a parameter set declared before
generation.  For finite-depth fifth-order Stokes states, let \(H\) be the
physical crest-to-trough height of the complete implemented series and
\(\lambda\) its carrier wavelength.  The code evaluates the series on 4096
equally spaced phases and adds the Taylor bound
\(\frac14(2\pi/4096)^2\sum_{m=1}^5m^2|E_m|\), where \(E_m\) is the
\(m\)-th elevation harmonic.  Denote the resulting upper bound by \(H_+\).
The support condition is

\[
\mathrm{Ur}_+=\frac{H_+\lambda^2}{h^3}\leq26.
\]

Taylor's theorem gives \(H\leq H_+\), so this conservatively enforces the
physical Ursell condition.  Every rejected amplitude and its
\(\mathrm{Ur}_+\) value are stored with the accepted case.  This is part of
the definition of the Stokes family, not an outcome-dependent cleaning rule.
Harmonic ordering remains a recorded diagnostic but is not a support gate.

The Stokes sampler draws the spatial phase directly as
\(\phi\sim\operatorname{Unif}[0,2\pi)\) and constructs the snapshot at
\(kx+\phi\); it does not use a sampled time as a proxy for phase.  Before the
draw, the common scheduler assigns one of four parameter categories:
finite/deep crossed with
\(ka\in[0.005,0.03)\) or \(ka\in[0.03,0.15]\).  The sampler chooses a
feasible integer carrier mode, a mode-conditioned depth, the phase, and then
an amplitude inside the assigned category.  An Ursell rejection redraws only the
amplitude, so the carrier, depth, phase, split, and category do not change.  The
float64 constructed surface potential has zero spatial mean; float32 storage
preserves this only to rounding error.

NumPy PCG64 uses all five coordinates of the stored case key: root seed,
physical-family identifier, generator revision, stream identifier, and
attempt index.  This is a reproducible stratified pseudorandom sampler, not a
low-discrepancy sampler.  If all 1000 same-category amplitude draws fail, the
sampler raises an exception whose strict record contains the fixed parameters
and every rejected amplitude and \(\mathrm{Ur}_+\) value.  This is a declared
case rejection, not a fatal batch error: `stokes_quota_executor.py` puts that
record in the proposal, commits a zero-row `OUTSIDE_SUPPORT` decision, keeps
valid siblings, and schedules a replacement in the same category.  An
unclassified sampler error propagates and is not relabeled as a physical
rejection.  The analytic deep-water branch
additionally requires \(kh\geq5\); the finite-depth branch uses the
conservative Ursell condition above.

For Tanaka data, let \(m\in\{1,2,3\}\) be the number of crests in a main
case and let

\[
q=\#\{j:s_j=+1\}
\]

be the number of right-moving crests.  The main pairs
\((m,q)\), \(q=0,\ldots,m\), define \(2+3+4=9\) parameter categories.
The steep one-crest branch adds the two categories \(q=0,1\).  Conditional
on \(q\), the sampler chooses uniformly which \(q\) of the \(m\) exchangeable
crest labels move right.  Accepted quotas are divided as evenly as possible
across all eleven categories, so their counts differ by at most one.  The
possible direction compositions receive equal coverage in the ideal category
law.  The quotient--remainder
assignment can introduce a one-case direction imbalance for a finite quota;
with the fixed category order, a quota of 1024 has one extra left-moving
one-crest case.  This replaces the old four-group independent-sign mixture
and deliberately gives more aggregate weight to two- and three-crest cases.

This direction-aware law advances the global generator revision to 2 and uses
`tanaka_population_spec_v2`.  Revision-1 parameter smokes, dry runs, and all
prior family pilot artifacts are historical evidence only; they cannot resume
into revision 2.
The 104-category trajectory gate and final corpus use revision 2.

The traveling-wave surface-potential formula contains a
square root.  When every wave speed and radicand entry is finite, a component
with a negative radicand is outside that formula's declared construction
domain and is recorded as a zero-row `OUTSIDE_SUPPORT` case before the square
root is taken.  A nonfinite radicand, a nonpositive or nonfinite wave speed,
or an unexpected constructor exception instead stops generation.  Such
failures are not treated as new random draws from the Tanaka population.

After construction succeeds, every production trajectory is integrated
once.  The saved interval is
\(\Delta t_s=0.08\), and GL2 takes eight equal substeps between saved states.
Thus the actual GL2 step is

\[
\Delta t_{\mathrm{GL2}}=\frac{0.08}{8}=0.01.
\]

This is the same step frozen by the May time-step study.  That study varied
the saved interval while retaining eight substeps, so its reported choice
\(0.08\) corresponds to an actual GL2 step of \(0.01\).  Production does not
repeat that solver-selection study for every sampled case.

A production trajectory is admissible when it reaches the requested terminal
time, every \(\eta\), \(\xi\), and delivered target value is finite, every
saved state satisfies \(h+\eta>0\), and every implicit GL2 stage is solved to
the declared residual tolerance.  The one post-rollout production condition
is therefore:

\[
\text{reject the case if and only if no complete admissible trajectory
exists on }[0,T].
\]

A nonfinite field, nonfinite target, unsolved stage, or loss of the graph
domain is a recorded cause of that single failure condition.  These are not
independent empirical shape filters.  A rejected trajectory owns zero rows;
no finite prefix or isolated frame is kept.

Time-step comparison remains a method-level validation calculation.  It is
run on a small, predeclared panel rather than on every production case.  If
\(P_K\) denotes projection to the delivered band and \(P_K^\circ\)
additionally removes the mean, define

\[
G_{\mathrm{ref}}(\eta;h)\xi
=P_K^\circ\!\left[
\mathcal G_6^{1024,8}(P_K\eta;h)(P_K\xi)
\right].
\]

The dimensionless delivered fields are

\[
Y_r(t)=\left(
\frac{P_K\eta_r(t)}{h},
\frac{P_K\xi_r(t)}{h\sqrt{gh}},
\frac{G_{\mathrm{ref}}(\eta_r(t);h)\xi_r(t)}{\sqrt{gh}}
\right),
\qquad \Delta t_r=2^{-r}\Delta t_0.
\]

Writing \(\mathcal U_r\) for the three components of \(Y_r\), the single
validation statistic is

\[
E_r=
\max_{u\in\mathcal U_r}
\frac{\max_j\lVert u_{r+1}(t_j)-u_r(t_j)\rVert_{L^2}}
     {\max_j\lVert u_{r+1}(t_j)\rVert_{L^2}+10^{-12}}.
\]

The validation pair uses actual GL2 steps \(0.01\) and \(0.005\).
The \(0.0025\) calculation is available only as a diagnostic retry in this
method-level utility; neither the second arm nor the retry is part of
production generation or either production acceptance stage.

The completed 12-case full-horizon panel corroborates the frozen \(0.01\)
production step.  All 10 cases that completed at both steps had
\(E_0\leq1.126337\times10^{-5}\), almost two orders of magnitude below
\(10^{-3}\).  The steep upper-seam Tanaka case and the equation-(33)
Benjamin--Feir stress case became incomplete at essentially the same physical
times under both steps.  No case changed decision under refinement, and no
\(0.0025\) retry ran.  This panel tests the common reference method at
selected support points; it is not an estimate of population rejection
rates.

Static Stokes cases do not undergo a fictitious temporal check.  They retain
only \(t=0\) after their declared support and static finite-state,
positive-water-column, and finite-target checks pass.
A returned nonfinite array therefore gives a zero-row decision, whereas an
unexpected exception from the Stokes constructor or target evaluator
propagates and stops generation.

Hamiltonian drift, mass, spectra, and stage iteration counts are reported on
the numerical validation panels as diagnostics of the common reference
method.  They are not required production-case fields or additional per-case
rejection thresholds.  In particular, the arbitrary half-depth margin is
removed.

Independent fixed-band spatial refinement is evaluated on a predeclared,
family-stratified panel for every generator revision.  It freezes the
supported parameter range before production; it is not an outcome-dependent
row filter.

The old Tanaka sign-transition count is historical evidence for the clipped
zero-extension defect.  It is not part of the production acceptance rule.
Neither a slope/extremum count nor an absolute cap on
\(\lvert G(\eta;h)\xi\rvert\) is a numerical-health test.

The CPU smoke test exercises the finite-Stokes constructor and two actual
GL2 refinement pairs as method-level validation:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_paper_acceptance_smoke
```

This older smoke verifies the refinement-audit decision layer.  It does not
describe the one-rollout production path.  The production executor records
whether every implicit stage was solved; its focused tests are listed below.

Two additional method-level smokes are intentionally separate from
per-trajectory acceptance:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_paper_acceptance_cross_family

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_stokes_spatial_smoke
```

The first checks constructor finiteness, graph validity, band limitation,
translation covariance of \(G(\eta;h)\xi\), and one actual \(\Delta t\) versus
\(\Delta t/2\) GL2 pair for linear, finite-Stokes, tangent-Hermite Tanaka, and
historical Benjamin--Feir code paths.  The separate public paper constructor
is `benjamin_feir_jcp09.py`.  The second test independently
constructs one Stokes state on \(N=32\) and \(N=64\) grids and compares
consistently normalized Fourier coefficients on the fixed band
\(|m|\leq12\).  It validates a generator revision; it does not reject
individual samples.

The random-sea paper constructor is `jonswap_tma.py`.  It uses a
cell-integrated JONSWAP density, the standard TMA depth factor, and a frozen
cosine-squared density window from wavenumber 96 to 128.  This placement is
tied to the declared support rather than to a realized profile.  Every case
has \(k_p\leq24\), and for \(r\geq1\),

\[
\frac{\omega(rk_p,h)}{\omega(k_p,h)}
=
\left(
r\frac{\tanh(rk_ph)}{\tanh(k_ph)}
\right)^{1/2}
\geq\sqrt r.
\]

Consequently \(96=4\max k_p\) leaves every frequency through
\(2\omega_p\) untapered.  The remaining 32 wavenumber cells form the smooth
transition to the common cutoff \(K=128\).  The production proposal records
the window name, transition, and quadrature order.  These are fixed
constructor settings, not case parameters or realized-profile tests.  The
constructor does not use realization-dependent slope, height, or appearance
gates.

The GL2 production mode is residual-controlled:

```python
rollout(
    ...,
    implicit_iterations=4,
    implicit_residual_tolerance=1e-8,
)
```

It stores per-step residual, convergence, iteration-cap, and finiteness
telemetry.  The frozen target implementation is
`pipeline.reference.PAPER_DNO_TARGET`, namely
\((L,M,N,p,K)=(2\pi,6,1024,8,128)\).  Its evaluator requires JAX float64
mode and promotes stored float32 inputs before applying the target.

The executable method-level pair/retry utility is
`pipeline.refinement.execute_residual_controlled_refinement`.  It is used for
the predeclared validation panel, not for routine corpus generation.  The
fixed- and variable-horizon production entry points are
`pipeline.refinement.execute_production_trajectory` and
`pipeline.refinement.execute_variable_horizon_production_trajectory`.
They take one residual-controlled GL2 rollout with actual step \(0.01\),
record complete-run telemetry, and expose fields through
`pipeline.trajectory_writer.outcomes_from_production` only when the complete
trajectory is admissible.  A rejected case retains its quality masks and GL2
telemetry but cannot expose a numerical prefix.

Post-acceptance time selection is implemented in
`pipeline.time_selection`.  Tanaka uses relative change in surface-gradient
energy; Benjamin--Feir uses the fourth power of the surface-envelope
excursion above its trajectory minimum.  Midpoint-quantile indices are
projected onto the endpoint-pinned strictly increasing integer grid.  Random
seas have peak period \(T_p=2\pi/\omega_p\), intended horizon
\(H=16T_p\), and saved spacing \(\Delta t_s=0.08\).  They are integrated to
the last saved-grid time
\(T_s=\Delta t_s\lfloor H/\Delta t_s\rfloor\leq H\).  With
\(J=T_s/\Delta t_s\), the 16 retained indices are the nearest integers
\(j_\ell=\lfloor \ell J/15+1/2\rfloor\), \(0\leq\ell\leq15\).
These functions do not inspect model error.

## Artifact layout

Large field arrays remain immutable.  A dataset view is a JSON manifest that
points to those arrays and to a compact row-to-case map.  Passing a historical
raw NPZ to the loader preserves its historical row-split behavior; passing a
schema-v2 manifest uses the case split fixed in the proposal.

Each attempted batch is a transaction:

```text
proposal NPZ -> optional accepted-case shard NPZ -> result JSON
```

The proposal is written before numerical work.  The result JSON is the commit
marker.  A proposal without a result, or a proposal and shard without a result,
can be replayed exactly.  A result that names a missing or hash-mismatched
shard is corruption.  A rejected case owns zero rows; every accepted case owns
one complete, contiguous row block.

The schema-v2 row map contains

```text
trajectory_index, frame_index, shard_index, shard_row
```

and its attempted-case table contains

```text
trajectory_family_id, trajectory_revision_id, trajectory_split_id
trajectory_case_id, trajectory_cell_id
trajectory_accepted, trajectory_required_bits
trajectory_evaluated_bits, trajectory_failed_bits
trajectory_first_row, trajectory_row_count
```

The manifest records every proposal, result, and immutable shard path and
SHA-256 digest.  Each batch retains its run-configuration fingerprint; these
fingerprints normally differ across families and splits because they bind the
quota, random stream, and split assignment.  A separate dataset-contract
  fingerprint binds the common DNO target, the shared production-integration
  contract, and stored dtypes.  The persisted trajectory contract records the
  single production step \(0.01\); audit-only half steps and discrepancy
  tolerances are not production settings.  View construction rejects batches
  whose common
contracts differ.  Rejected candidate specifications and reasons remain in
the proposal and result records even though they contribute no training rows.
The schema-v2 training sampler uses exactly one stored time from every
accepted case in an epoch and requires equal accepted-case counts across
physical families.

The family-independent quota, transaction, view, and loader checks are
reproduced by

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest \
    solver.gen_data.pipeline.test_production \
    solver.gen_data.pipeline.test_archive \
    solver.gen_data.pipeline.test_manifest \
    solver.gen_data.pipeline.test_writer \
    solver.gen_data.pipeline.test_quota_driver \
    solver.gen_data.pipeline.test_refinement \
    solver.gen_data.pipeline.test_time_selection \
    solver.gen_data.pipeline.test_trajectory_writer

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python train-jax-10m/tests_paper_corpus_dataset_view.py
```

These are tests of the common storage and sampling foundation.  The common
accepted-quota driver reconstructs exact per-category counts from committed
artifacts, replays proposal-only and proposal-plus-shard interruptions, and
uses a nonblocking single-writer lock.  A rejected attempt does not advance
its accepted-category count.  No sampled rollout family has yet completed a
population-scale paper-contract production batch.  The three rollout-family
adapters in
`trajectory_family_adapters.py` enforce the first three operations:

```text
sample complete specifications
-> persist the proposal
-> construct initial fields
```

A constructor accepts only a durable proposal token and rechecks the proposal
state, file hash, case identities, and stored specifications before numerical
construction.  The shared executor, time selector, and whole-case writer
enforce the remaining integration, selection, and commit operations.
`trajectory_quota_executor.py` connects those operations to accepted
per-category quotas and replay.  A corrected real-GL2 CPU smoke at
\((N,M,p,K)=(64,0,1,16)\) took one Tanaka, one Benjamin--Feir, and one
JONSWAP/TMA case through proposal, the then-current paired validation path,
selection, whole-case
commit, schema-v2 view, and the training loader.  All three cases were
accepted and contributed three finite rows each.  This historical run is
wiring evidence for those components, not a production timing measurement.
The Tanaka executor records only the finite-negative-radicand failure
described above as a zero-row `OUTSIDE_SUPPORT` case.  Its classifier verifies
the recorded identities \(c^2=|c|^2\) and \(R_{\min}/c^2\), and verifies that
the negative-entry count agrees with the sign of the finite minimum.  A
nonfinite, malformed, or otherwise unclassified construction failure leaves
the durable transaction pending for replay, while only an explicit
`DeclaredTrajectoryFatalError` writes a fatal sidecar.  These reduced
calculations establish software wiring only; they are not spatial,
full-horizon, or population validation.

Exact-contract generation for all four families is launched one family and
split at a time with `scripts/run_paper_corpus_quota.py`.  Its default action
is a read-only preflight: it prints the ordered parameter-category quotas, exact numerical
contract, expected retained-row count, source and dependency hashes, output
namespace, and any resumable state.  Numerical work requires the explicit
`--execute` flag.  For example, this first prints a CPU preflight and then
runs or resumes an eleven-case Tanaka validation pilot with one case from
every direction-aware parameter category:

```bash
uv run python scripts/run_paper_corpus_quota.py \
  --family tanaka \
  --split validation \
  --accepted-cases 11 \
  --output-root outputs/paper_corpus_exact_pilot_revision2 \
  --batch-size 1 \
  --maximum-attempts-per-accepted-case 4

uv run python scripts/run_paper_corpus_quota.py \
  --family tanaka \
  --split validation \
  --accepted-cases 11 \
  --output-root outputs/paper_corpus_exact_pilot_revision2 \
  --batch-size 1 \
  --maximum-attempts-per-accepted-case 4 \
  --execute
```

The separate all-category trajectory gate consists of
\(11+66+27=104\) cases: one from every Tanaka, Benjamin--Feir, and
JONSWAP/TMA parameter category.  The four static Stokes categories have
already passed their exact calculation under generator revision 2, with all
four cases accepted and `failed_bits=0`.  The output root is
`outputs/static_stokes_exact_target_pilot_revision2_20260727`.  The 104-case
gate and the final corpus use generator revision 2.

CPU is the fail-safe default.  `--platform gpu` selects the accelerator path
before JAX initializes.  Platform, quota, batch size, stream coordinates,
attempt ceiling, paper contract, family-specific source hashes,
Python/JAX/JAXLIB/NumPy versions, and the `pyproject.toml` and `uv.lock`
hashes are all bound by the configuration fingerprint.  Consequently a
resume must use the same command-defining values and dependency environment.
Use a different output root for a pilot and a later larger quota; an accepted
quota cannot be enlarged in place.  The same launcher accepts `--family stokes`,
`--family benjamin_feir`, and `--family jonswap_tma`.  Stokes uses the static
one-row transaction; the other three families use their declared
whole-trajectory integration and time-selection contracts.

Learning-curve corpora are additive chunks, not enlargements of an existing
run.  Let \(B_i(C)\) be the balanced quota of category \(i\) after \(C\) accepted
cases.  A chunk beginning at cumulative count \(C_0\) and containing \(A\)
new accepted cases receives

\[
  B_i(C_0+A)-B_i(C_0)
\]

cases from category \(i\).  The launcher implements this rule through
`--accepted-cases-before C0`; its default is zero for a standalone pilot.
This subtraction matters for the 66-category Benjamin--Feir and 27-category
JONSWAP/TMA families: balancing every chunk independently would repeatedly
favor the first remainder categories.  The planned nested learning curve is:

| Chunk | `--accepted-cases-before` | `--accepted-cases` | Cumulative count | Suggested `--stream-id` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 2048 | 2048 | 0 |
| 2 | 2048 | 2048 | 4096 | 1 |
| 3 | 4096 | 4096 | 8192 | 2 |
| 4 | 8192 | 8192 | 16384 | 3 |
| 5 | 16384 | 16384 | 32768 | 4 |

Every chunk must use a distinct output root and stream ID.  The output root
is the transaction namespace; the stream ID is part of every case identity
and random seed.  `--first-attempt-index` normally remains zero within each
new stream.  The preflight prints the cumulative before, chunk, and
cumulative after quota for every category so this assignment can be checked
without constructing a field.

Batch size changes transaction and memory granularity, not the population
law.  A larger batch can improve accelerator utilization, but increases peak
memory and the amount of work replayed after an interruption.  Use
`--batch-size 1` for the first exact CPU pilot, then benchmark a few
conservative batch sizes before fixing the production command.

The accepted-quota loop is also bounded.  If a category requires \(Q_i\) accepted
cases, the default `--maximum-attempts-per-accepted-case 4` permits at most
\(4Q_i\) durable case proposals in that category.  A pending proposal already
owns one of those attempt slots and is replayed rather than counted twice.  If
its result leaves the category short at the ceiling, the run stops with the
accepted, attempted, target, and ceiling counts instead of drawing forever.
This outer count does not include GL2 substeps or the finite-Stokes sampler's
internal amplitude redraws.  The multiplier is immutable within an output
root.

After the train, validation, and test runs are complete, build the single
loader-facing view from their completion summaries:

```bash
uv run python scripts/build_paper_corpus_view.py \
  --chunk-summary outputs/train_stokes_chunk_0/paper_corpus_stokes_train.summary.json \
  --chunk-summary outputs/train_tanaka_chunk_0/paper_corpus_tanaka_train.summary.json \
  --chunk-summary outputs/train_bf_chunk_0/paper_corpus_benjamin_feir_train.summary.json \
  --chunk-summary outputs/train_jonswap_chunk_0/paper_corpus_jonswap_tma_train.summary.json \
  --chunk-summary outputs/validation_stokes/paper_corpus_stokes_validation.summary.json \
  --chunk-summary outputs/validation_tanaka/paper_corpus_tanaka_validation.summary.json \
  --chunk-summary outputs/validation_bf/paper_corpus_benjamin_feir_validation.summary.json \
  --chunk-summary outputs/validation_jonswap/paper_corpus_jonswap_tma_validation.summary.json \
  --chunk-summary outputs/test_stokes/paper_corpus_stokes_test.summary.json \
  --chunk-summary outputs/test_tanaka/paper_corpus_tanaka_test.summary.json \
  --chunk-summary outputs/test_bf/paper_corpus_benjamin_feir_test.summary.json \
  --chunk-summary outputs/test_jonswap/paper_corpus_jonswap_tma_test.summary.json \
  --output-root outputs/paper_corpus_view
```

Repeat `--chunk-summary` for every later additive chunk.  This command is
also a read-only preflight by default; add `--execute` to write the manifest
and trajectory map.  Within every included split it requires exactly the four
physical families, gap-free cumulative intervals beginning at zero, distinct
stream IDs and roots within each family, and equal final accepted-case counts
across families.  It also requires one dependency environment and compatible
source mappings across the complete view.  It passes the exact set of run
fingerprints to `build_dataset_view` and validates every manifest split
count.  If \(C_{\rm tr},C_{\rm va},C_{\rm te}\) are the accepted counts per
family, the result contains
\(4(C_{\rm tr}+C_{\rm va}+C_{\rm te})\) accepted cases and
\(417(C_{\rm tr}+C_{\rm va}+C_{\rm te})\) rows.  Validation and test normally
use one fixed chunk per family rather than the five-stage training schedule.
A single-split view remains available for diagnostics, but training should
consume the combined all-split manifest.

Static Stokes is connected end to end by `stokes_static_pipeline.py`: it
writes the proposal before construction, evaluates the common target, stores
exactly one \(t=0\) row for an accepted case, and gives a rejected attempt
zero rows.  `stokes_quota_executor.py` connects that transaction to the
accepted-quota driver.  If the finite-depth sampler exhausts its declared
same-category amplitude redraws, the executor stores the complete draw ledger as
a zero-row `OUTSIDE_SUPPORT` attempt, keeps valid siblings in the batch, and
schedules a replacement only in the missing category.  Its production default is
the frozen paper target.  A reduced target must be labeled
`reduced_wiring_evidence_only`.  The exact-target pilot in
`scripts/run_static_stokes_quota_pilot.py` runs the accepted-quota path with
one validation case in each of the four Stokes categories at
\((N,M,p,K)=(1024,6,8,128)\).  The 2026-07-25 run accepted all four cases and
wrote a proposal, shard, result, manifest, and trajectory map with recorded
SHA-256 hashes; the training loader returned four finite one-row cases.  That
run used generator revision 1 and is historical transaction evidence.  The
current generator-revision-2 run on 2026-07-27 again accepted all four cases,
with `failed_bits=0`; it is stored at
`outputs/static_stokes_exact_target_pilot_revision2_20260727`.

The four declared population laws and both quota executors are checked by:

```bash
uv run python -m unittest \
  solver.gen_data.tests_stokes_population \
  solver.gen_data.tests_stokes_static_pipeline \
  solver.gen_data.tests_stokes_quota_executor \
  solver.gen_data.tests_tanaka_population \
  solver.gen_data.tests_tanaka_potential_radicand \
  solver.gen_data.tests_benjamin_feir_population \
  solver.gen_data.tests_jonswap_tma_population \
  solver.gen_data.tests_trajectory_family_adapters \
  solver.gen_data.tests_trajectory_quota_executor \
  scripts.test_run_paper_corpus_quota \
  scripts.test_build_paper_corpus_view \
  scripts.test_run_trajectory_quota_real_gl2_smoke
```

## Legacy v9

`combined_dataset_v9.npz` is frozen as a historical artifact.  Its first ten
sources were row-filtered and globally shuffled before four later source
additions were appended.  The flat archive discarded original case IDs, so it
must not itself be described as uniformly trajectory-filtered.

The trajectory-filtered v9 view reconstructs compound trajectory identity from the raw
source archives and the recorded shuffle/append lineage.  It masks complete
trajectories whenever a source row was historically rejected, and it splits
the retained trajectories by physical family.  Checks unavailable from the
saved artifacts remain explicitly unevaluated.  A future generator revision
must use this contract directly rather than reconstruct it after assembly.
