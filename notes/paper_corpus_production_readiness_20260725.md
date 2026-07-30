# Paper-corpus production readiness

Date: 2026-07-25

> **Historical note — superseded production contract.** This note records
> revision-1 measurements and terminology and is retained only as historical
> evidence. The current generator uses revision 2. In particular, Tanaka has
> eleven parameter categories indexed by structural regime, crest count
> \(m\), and right-moving crest count
> \(q=\#\{j:s_j=+1\}\); direction is therefore part of the category. No
> revision-1 artifact can be resumed into the revision-2 gate or final
> corpus. Use the
> [current readiness note](paper_corpus_generation_readiness_20260726.md)
> and the
> [condensed construction contract](paper_corpus_generation_condensed_20260726.tex)
> for production decisions. The measurements below have not been rewritten.

## Meaning of “production-ready”

A family is production-ready only if one executable path does all of the
following:

1. assigns an attempted case to a declared parameter cell and to
   train/validation/test before any numerical outcome is known;
2. stores the complete case specification before construction or integration;
3. constructs the declared periodic initial condition and checks its
   parameter-only support;
4. for a trajectory family, computes one residual-controlled float64 GL2
   trajectory with internal step \(h=0.01\), saves a state every
   \(\Delta t_{\rm save}=0.08\), evaluates the frozen order-six, pad-eight,
   \(|k|\leq128\) target at every saved state, and accepts the case only if a
   complete admissible trajectory exists on the declared interval;
5. selects times only from the retained accepted trajectory and takes the
   corresponding \((\eta,\xi,G(\eta)\xi)\) rows;
6. commits only complete cases to an archive that has deterministic restart,
   integrity checks, and an explicit failure record; and
7. emits case identities and split information that the training loader uses
   without separating temporal siblings or weighting families by their row
   counts.

The intended order is stated in the
[data-generation contract](../solver/gen_data/README.md#L8-L26). The plan
explicitly says that the new corpus has not been generated
([plan, lines 77–115](parameterized_paper_corpus_plan_20260722.tex#L77-L115)).
The shared numerical pieces already exist: the
[frozen target](../solver/gen_data/pipeline/reference.py#L13-L106), the
[single-arm GL2 kernel and separate refinement audit](../solver/gen_data/pipeline/refinement.py),
and residual-controlled GL2
([time integrator, lines 860–980](../solver/solvers/time_integrator.py#L860-L980)).
The full-horizon runner composes these pieces for twelve fixed stress cases
([case definitions, lines 219–417](../scripts/run_full_horizon_refinement_panel.py#L219-L417));
it is a validation panel, not a population generator.

## Time-step policy

The saved-state interval and the GL2 step are different quantities. Between
two consecutive saved times, the solver takes eight equal GL2 steps:
\[
  \Delta t_{\rm save}=0.08,\qquad
  h=\frac{\Delta t_{\rm save}}{8}=0.01.
\]
This is the production calculation. It is performed once per proposed case.

This choice was already made in the May 3 Tanaka time-step sweep. That report
called the saved interval `dt` and selected `dt=0.08` with eight substeps.
In the present notation, the selected internal GL2 step was therefore
\(h=0.08/8=0.01\). On four representative cases, this choice agreed with the
denser `dt=0.01` saved-interval reference to about \(10^{-9}\) at \(t=50\).
The next coarser saved interval, `dt=0.16`, corresponds to \(h=0.02\) and
gave a relative error \(6.2\times10^{-2}\). Thus the old sweep selected the
same internal step used here; the change of notation only separates the
saved interval from the eight actual GL2 steps.

The pair \(h=0.01\) and \(h=0.005\) was used to decide whether \(h=0.01\)
was sufficiently small. It is a method-level audit, not a case-level
acceptance rule. In the completed twelve-case audit, all ten cases that
produced complete finite trajectories at both steps had
\[
  1.36\times10^{-8}
  \;\leq\;
  E_{0.01,0.005}
  \;\leq\;
  1.13\times10^{-5},
\]
where \(E_{0.01,0.005}\) is the maximum relative difference between the two
solutions over all saved times and all three stored fields. The largest
value was about \(89\) times smaller than the predeclared tolerance
\(10^{-3}\). The three exact family smoke cases had errors
\(3.20\times10^{-8}\), \(3.71\times10^{-8}\), and
\(7.66\times10^{-7}\). The other two stress cases became incomplete at both
steps; neither was rejected by a finite step-to-step discrepancy.

This evidence supports one production run at \(h=0.01\). Running both audit
steps for every proposed case would cost about \(2.94\) times as much as the
single \(h=0.01\) calculation in the measured panel. There is no
\(h=0.0025\) production retry.

The one full-trajectory production condition is:
\[
  \text{a complete admissible trajectory exists on the declared interval}.
\]
Here “admissible” means that every implicit GL2 stage satisfies the declared
residual tolerance, every saved \(\eta\) and \(\xi\) value is finite, the
water surface remains a valid graph over the periodic horizontal coordinate,
and every saved value of \(G(\eta)\xi\) is finite. Failure of an implicit
stage, a nonfinite state or target, or loss of the graph condition identifies
why this one condition failed. These are diagnostic causes, not separate
trajectory-selection criteria. Parameter support is checked before the
rollout and remains a separate pre-integration condition.

## Family-by-family status

The table below reports the current components, not the obsolete bulk
generators. “Ready” means that the component exists behind the common
transaction and has focused tests. It does not mean that a population-scale
corpus has been generated.

| Family | Sampling and construction | Common transaction | Numerical evidence | Work remaining before a large generation |
|---|---|---|---|---|
| Stokes | **Ready.** Four balanced finite/deep and low/moderate cells, the conservative \(\mathrm{Ur}_+\leq26\) support rule, same-cell amplitude redraw, fixed phase, and fifth-order construction are implemented. | **Ready.** The static path assigns the split and case identity, writes the proposal first, checks finite state, positive water column, fixed support, and finite target, commits one \(t=0\) row for acceptance and zero rows for rejection, and builds a schema-v2 view. Its thin executor stops by accepted per-cell quota and resumes interrupted batches. A redraw exhaustion is a stored zero-row `OUTSIDE_SUPPORT` attempt followed by a same-cell replacement, while valid siblings remain committed. | The exact accepted-quota pilot at \((N,M,p,K)=(1024,6,8,128)\) attempted and accepted `4/4`, exactly one per cell, and loaded four finite one-row cases through the training view. A larger 16,384-case finite-moderate sampling stress had no exhaustion. A forced reduced-contract quota test rejected attempt 0, retained sibling attempt 1, and accepted the finite-depth replacement at attempt 2. | Choose corpus case counts from the learning curve. |
| Tanaka | **Ready.** The four balanced main/steep and crest-count cells, tangent-Hermite profile reconstruction, explicit periodic gaps, fixed-band projection, and fail-closed real-root check are implemented. A negative or nonfinite radicand is recorded and rejected without a tolerance. | **Ready as tested wiring.** The accepted-quota path implements sampling, durable proposal, construction, one \(h=0.01\) trajectory, activity selection, whole-case commit, and case-level split. A declared radicand failure is a zero-row case rejection; malformed or unclassified errors remain pending rather than being mislabeled. | The radicand and tangent-Hermite suites pass `12/12`; the exact \(N=1024\) upper seam construction has minimum \(R/c^2=0.372151\). The reduced real-GL2 quota smoke reaches the schema-v2 loader. The separate full-horizon audit has three complete cases and one case that becomes incomplete at both \(h=0.01\) and \(0.005\). In the restart, \(p=16\) fails identically; \(N=2048\), \(M=5\), and \(M=4\) prevent terminal failure but retain the cascade; \(K=192\) alone suppresses accumulation in the former terminal band. | Keep the current common target. Under the production rule, the fixed stress case is rejected because no complete admissible trajectory exists. Run a small exact-contract single-arm pilot to measure how often the declared population reaches this behavior. |
| Benjamin--Feir | **Ready.** Each of the 66 admissible integer pairs \((n_c,\Delta n)\) is an allocation cell, and steepness, perturbation ratio, phase, equation-(33) sidebands, and the project carrier are sampled and constructed reproducibly. | **Ready as tested wiring.** The accepted-quota path connects the durable proposal, one \(h=0.01\) trajectory, envelope-activity selection, complete-case writer, and split-aware view. | The corrected reduced real-GL2 quota smoke accepts its case and reaches the schema-v2 loader. The separate full-horizon audit has two complete cases and an equation-(33) stress case that becomes incomplete near \(t=108.78\) at both \(h=0.01\) and \(0.005\). Padding and grid controls do not remove the onset; \(K=192\) fails earlier; \(M=5,4\) retain the same early growth while avoiding the terminal failure. | Keep the current common target. Under the production rule, the fixed stress case is rejected because no complete admissible trajectory exists. Run a small exact-contract single-arm pilot to measure accepted and rejected counts in the 66 declared cells. |
| JONSWAP/TMA | **Ready.** All 27 depth--peak-enhancement--direction cells have explicit continuous parameter laws, stored right/left phase arrays, the cell-integrated spectrum, TMA depth factor, and fixed resolved-band window. | **Ready as tested wiring.** The accepted-quota path preserves a separate horizon for every sampled peak and connects one \(h=0.01\) trajectory, 16-point selection, complete-case commit, and schema-v2 view. | The corrected reduced quota smoke accepts its case. In the separate full-horizon audit, the shallow, finite, and deep fixed cases all pass through the last \(0.08\)-spaced time \(T_s\leq16T_p\). Their maximum \(h=0.01\) versus \(0.005\) errors are \(4.52\times10^{-6}\), \(5.63\times10^{-6}\), and \(4.75\times10^{-6}\), and every implicit stage is solved. | Run a small exact-contract single-arm trajectory pilot. |

Thus the common mathematics and both quota executors are now executable. The
remaining distinction is important: static Stokes has an exact-target
accepted-quota pilot, whereas all three trajectory families have a corrected
reduced end-to-end quota smoke and separate exact full-horizon validation
cases. The reduced smoke is software evidence, not paper-contract or
population evidence. No replacement training corpus has yet been generated.

## Implementation update at 03:09 EDT

The first two family-independent pieces now exist:

1. `pipeline/production.py` assigns the physical-family revision, data split,
   parameter cell, and deterministic random-stream key before construction.
   For a requested accepted-case count \(A\) and \(C\) ordered cells, write
   \(A=qC+r\), where \(0\leq r<C\). The first \(r\) cells receive quota
   \(q+1\), and the others receive quota \(q\). A rejected attempt does not
   increment its cell count, so its replacement remains in the same cell.
2. `pipeline/archive.py` writes one proposal before numerical work, one
   immutable shard containing only complete accepted cases, and one result
   JSON as the commit marker. The result records every attempted case;
   rejected cases have zero rows. Proposal-only and proposal-plus-shard states
   are replayable after interruption. A result whose recorded shard is absent
   or has a different SHA-256 digest is corruption.
3. `pipeline/manifest.py` combines committed immutable shards into a
   schema-v2 dataset view without copying their field arrays. The trajectory
   map retains every attempted case and stores the exact shard row, family,
   revision, cell, split, and quality masks for every accepted row.
4. The schema-v2 loader verifies each shard digest, consumes the split assigned
   before numerical work, and samples one stored time from every accepted case
   per epoch. It requires equal accepted case counts across physical families
   instead of hiding an imbalance by row weighting. The historical v9 archive
   keeps its old loader and row split.

Twenty-three focused CPU tests cover quota balance, split assignment,
deterministic replay, same-cell replacement, short final batches, all
transaction boundaries, complete row ownership, fatal failures, manifest row
coordinates, quality-mask consistency, and corruption detection. The common
writer persists the root seed, stream, attempt, split, parameter cell, and
complete case specification before numerical construction. It writes rows
only for complete accepted cases, while retaining rejected attempts with zero
rows. Separate schema-v1 and schema-v2 loader tests pass. These modules do not
yet call a family constructor or the reference integrator. Thus they close the
allocation, storage, view, and sampling foundations, but they do not change
the family status table above or make any family production-ready.

## Numerical and family implementation

Seven additional pieces now exist.

1. `pipeline/refinement.py` is the common float64 GL2 executor. It runs the
   single production arm at \(h=0.01\), evaluates the frozen target, and
   records every implicit-stage result. A production result contains a
   trajectory only when that one trajectory is complete and admissible. The
   same single-arm kernel is used by a separate \(h=0.01\) versus \(0.005\)
   audit. The audit result is not passed to the quota executor, and the
   production path has no \(h=0.0025\) retry.
2. `pipeline/time_selection.py` implements the declared Tanaka
   surface-gradient activity and Benjamin--Feir envelope activity. It also
   implements the 16-point random-sea rule. All selected indices are
   deterministic, endpoint-pinned, and strictly increasing. Time selection is
   a separate function whose input is an already accepted trajectory.
3. `tanaka_population.py` implements the four cells `main_m1`, `main_m2`,
   `main_m3`, and `steep_m1`. Multi-crest centers are sampled through explicit
   periodic gaps
   \[
     g_i=3h+(2\pi-3mh)w_i,\qquad
     (w_1,\ldots,w_m)\sim\operatorname{Dirichlet}(1,\ldots,1),
   \]
   followed by one uniform rotation. Thus the minimum separation and
   translation balance hold by construction.
4. `jonswap_tma_population.py` implements all 27
   depth-stratum--peak-enhancement--direction cells. It records the two
   realized phase arrays in the pre-numerical case specification.
5. `pipeline/trajectory_writer.py` is the boundary between numerical
   acceptance and storage. It selects times only for an accepted result,
   converts the accepted \(h=0.01\) trajectory into rows, and converts
   nonfinite diagnostic scalars to JSON `null`. A CPU integration test runs an
   actual GL2 trajectory through proposal, post-acceptance selection,
   complete-case commit, and schema-v2 view construction.
6. `benjamin_feir_population.py` makes each of the 66 admissible integer
   pairs \((n_c,\Delta n)\) an allocation cell. Conditional steepness,
   perturbation ratio, and phase are then sampled inside that fixed cell. The
   common quota scheduler therefore gives exact attempted-count balance over
   the discrete pairs rather than balance only in expectation.
7. `stokes_static_pipeline.py` is the first physical-family common
   transaction. It writes the proposal before construction, evaluates the
   fixed-phase fifth-order state and common target, stores one \(t=0\) row
   only when all required static checks pass, commits rejected attempts with
   zero rows, and builds a schema-v2 view. Its default is the exact paper
   target; a reduced test target must carry the explicit role
   `reduced_wiring_evidence_only`.

The single-arm kernel and the separate two-step audit have focused tests. The
production tests assert that the quota path invokes exactly one arm at
\(h=0.01\), retains that arm when it is complete and admissible, and never
requests \(h=0.005\) or \(0.0025\). The time selector has five focused tests.
The Tanaka sampler has six focused tests and passed a 16,384-case support
stress calculation. The JONSWAP/TMA sampler has six focused tests. The writer
bridge has one actual-trajectory integration test. The Benjamin--Feir
sampler, constructor, and scheduler tests pass 24/24, including an
8,448-case support stress. Ruff passed. Thirty-nine related Stokes tests
pass, including a real reduced CPU transaction with two accepted states and
injected zero-row rejections.

## Implementation update at 04:12 EDT

`trajectory_family_adapters.py` now enforces the order

1. sample a complete specification;
2. persist its proposal;
3. construct the initial field.

There is no public function that samples and constructs in one call. A
constructor accepts only a durable proposal token and rechecks the proposal
state, SHA-256 digest, case IDs, and strict JSON specifications. Tests verify
that deleting the proposal or changing a sampled phase after the proposal was
written prevents construction. Reduced real-GL2 Benjamin--Feir and
JONSWAP/TMA cases then passed through the numerical executor,
post-acceptance time selector, whole-case writer, and schema-v2 view. Together
with the sampler and static suites, that historical independent CPU selection
passed 35/35 tests in 17.57 seconds. It tested the former multi-arm wiring; it
does not replace the single-arm regression check listed below.

The exact-target static pilot
`scripts/run_static_stokes_exact_target_pilot.py` uses
\((N,M,p,K)=(1024,6,8,128)\) in float64. One validation case from each of the
four Stokes cells was proposed before construction; all four were accepted
with no failed required quality bit. The proposal, shard, result, manifest,
trajectory map, strict configuration fingerprint, and source hashes are in
`outputs/static_stokes_exact_target_pilot_20260725`. The complete CPU
transaction took 2.584 seconds.

At 04:12, the remaining method decision concerned the two failed fixed stress
cases and the trajectory quota executor had not yet been added. The completed
12-case panel produced ten complete trajectories at both audited steps. The
steep Tanaka and equation-(33) Benjamin--Feir stress cases became incomplete
at both steps. That panel is the completed time-step validation, not corpus
generation. The later 07:01 update below supersedes the earlier executor
status.

## Implementation update at 04:50 EDT

`pipeline/quota_driver.py` now binds the family and revision, split, root
seed, random stream, first attempt, ordered cell quotas, cell codes, batch
size, and complete numerical configuration into one strict JSON fingerprint.
It reconstructs accepted counts only from checked committed artifacts. A
proposal-only or proposal-plus-shard final batch is replayed before any new
case is scheduled, and a nonblocking file lock prevents two writers from
advancing the same run.

`stokes_quota_executor.py` connects this recovery loop to the static
transaction. In a forced test, finite-depth attempt 0 exhausted its amplitude
redraw allowance and was stored with zero rows and `OUTSIDE_SUPPORT`.
Deep-water sibling attempt 1 was still evaluated and accepted with its
original local index. The driver then scheduled finite-depth attempt 2 and
stopped with exactly one accepted case in each requested cell. A separate
interruption after the durable proposal replayed the identical sampled
records and proposal hash. An independent CPU selection passed 49 of 49
allocation, archive, quota-recovery, Stokes-sampling, target, and executor
tests in 4.75 seconds.

## Implementation update at 05:09 EDT

The exact static accepted-quota pilot uses the full
\((N,M,p,K)=(1024,6,8,128)\) float64 target and the normal 1,000-redraw
sampler limit. It attempted and accepted four cases, exactly one in each
Stokes cell, and the schema-v2 loader returned four unique finite
1,024-point rows. The immutable proposal, shard, result, manifest, and
trajectory-map hashes were rechecked. The quota work took 3.60 seconds and
the complete run, view construction, and validation took 3.76 seconds.

The Tanaka constructor now evaluates the complete radicand before the square
root and rejects any negative or nonfinite value without a tolerance. The
strict failure record identifies the local case, crest within that case,
flattened crest, parameters, speed, minimum radicand and normalized
radicand, grid location, and negative/nonfinite counts. Exact zero remains
admissible. The new six-test domain suite and existing six-test
tangent-Hermite suite both pass. An independent exact \(N=1024\) upper-seam
construction remains finite; its previously audited minimum normalized
radicand is \(0.372151\).

## Implementation update at 07:01 EDT

`trajectory_quota_executor.py` now connects Tanaka, Benjamin--Feir, and
JONSWAP/TMA to accepted per-cell quotas, proposal and shard replay,
same-cell replacement, family-specific horizons, post-acceptance selection,
whole-case commit, and schema-v2 views. Every archived trajectory decision
uses the same required mask:
`OUTSIDE_SUPPORT | INCOMPLETE_TRAJECTORY`. The first bit is the
pre-integration parameter-support check. The second bit is the one
full-trajectory production condition defined above. `NONFINITE_STATE`,
`NONFINITE_TARGET`, `GL2_STAGE_RESIDUAL`, and `BOTTOM_CLEARANCE` record the
cause when that condition fails. `TEMPORAL_DEFECT` belongs only to the
completed two-step audit and is not required by the production executor. The
Tanaka classifier accepts only the exact versioned radicand record, maps
constructor-subset component indices back to the original proposal, and
rejects malformed records. In particular, it checks \(c^2=|c|^2\), checks
that the recorded normalized minimum equals \(R_{\min}/c^2\) when \(c^2>0\),
and checks that a finite minimum is negative exactly when the negative-entry
count is positive. An unclassified exception propagates and leaves the
proposal or shard pending for replay. Only an explicit
`DeclaredTrajectoryFatalError` writes a terminal sidecar. Paper-labeled
executions require the production constructors, the single \(h=0.01\)
numerical arm, failure
classifiers, and, for Stokes, the 1,000-redraw production sampler.

The earlier reduced CPU smoke used float64
\((N,M,p,K)=(64,0,1,16)\),
\((\Delta t_c,\Delta t_f,\Delta t_r)=(.04,.02,.01)\), and saved spacing
\(.08\). It attempted and accepted one case from each trajectory family,
stored three rows per case, and loaded all nine finite rows through the
schema-v2 training view. For the random-sea case,
\(H=16T_p=30.03410688\) and
\(T_s=.08\lfloor H/.08\rfloor=30.0\). The total wall time was
17.99 seconds; the summary SHA-256 is
`3c944234e5e859af16ccbc78640c60e2fb958bb4f55a01ae13e81c2c8062456b`.
This smoke remains useful evidence for sampling, construction, storage, and
loading, but its pair/retry numerical policy is superseded. A current reduced
regression test asserts one call at \(h=0.01\) and no call at \(0.005\) or
\(0.0025\).
The exact Stokes pilot was rerun after binding the production hooks and
1,000-redraw setting; it again accepted `4/4` in 3.44 seconds. Its summary
SHA-256 is
`6e787c1996950b76f091cbf997bc92a3b52f0bcb07d62d2e6c0a2c8c2b3a1098`.
The post-change `solver/gen_data` discovery suite passes `185/185` in
107.617 seconds, and the focused production tests pass `27/27` after the
final contract rename. An
independent adversarial audit generated valid negative-radicand and
invalid-speed records and injected inconsistent speed, normalized-minimum,
and sign-count records. Valid records classify; every inconsistent record
raises while the durable proposal remains pending, with no result or fatal
sidecar.

## Remaining dependency-ordered work

The allocator, four population samplers, constructors, dense target,
single-arm production executor, separate two-step audit, time selectors,
complete-case writer, multi-shard view, and split-aware loader are
implemented. The remaining work is shorter:

The time-step decision is complete. Keep
\((N,M,p,K)=(1024,6,8,128)\), use one \(h=0.01\) GL2 trajectory, and save
every \(0.08\). The two incomplete full-horizon stress cases fail the
complete-trajectory condition. Their failure at both audited steps does not
justify rerunning every production case at \(h=0.005\), and there is no
production \(h=0.0025\) retry.

1. **Run a small exact-contract trajectory pilot.** Report attempted and
   accepted counts by cell and classify every rejection by the diagnostic
   cause under `INCOMPLETE_TRAJECTORY`. The old reduced three-arm smoke cannot
   be substituted for this calculation.

3. **Run the case-count learning curve.** Only after the exact-contract pilot
   should a population-scale allocation begin.
