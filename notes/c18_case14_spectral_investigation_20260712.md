# C18 Tanaka case-14 investigation

Date: 2026-07-12

## Conclusion

The remaining `tanaka_g1` case 14 divergence is not the same defect as the
`tanaka_g0` case 27 phase error.  Case 27 is a single shallow solitary wave
whose final discrepancy is almost entirely a translation.  Case 14 is a rare
two-crest interaction that develops a transverse, finite spectral cascade near
the production cutoff.  The scalar C18 translation loss is therefore the wrong
rank-one target for case 14.

The implemented numerical response is a soliton-selective spectral-envelope
guard.  It activates only for positive-elevation solitary-wave ICs, only after
the evolving `eta[k=64:128]` amplitude exceeds its initial value by 100 times
and an absolute numerical floor, and only damps above a depth-scaled Hou--Li
scale.  Oscillatory BF, random-sea, linear, and Stokes ICs are excluded by an
initial-state predicate rather than by a regime label.

## Initial condition and data coverage

Case `1000014` has

- depth `h = 0.1776005775`;
- two left-going crests with dimensionless amplitudes `0.12514314` and
  `0.11279759`;
- center separation `0.5789553 = 3.25987 h`.

It is already present in `combined_dataset_v9.npz`.  Its initial `eta` is
bit-identical to the source-6 training trajectory, and 163 of its 200 saved
states fall in the row-wise training split.  Thus the failure is not explained
by missing truth-orbit snapshots.  The close, nearly equal two-crest geometry
is rare: its separation is in approximately the lowest percentile of source-6
two-crest cases.

## C16 versus C18 trajectory evidence

At `t=200`:

| quantity | truth | C16 | C18 |
| --- | ---: | ---: | ---: |
| raw relative eta error | 0 | 1.1987 | 1.2875 |
| translation-aligned eta error | 0 | 0.5062 | 0.5871 |
| principal crest amplitude | 0.04485 | 0.05746 | 0.05752 |
| maximum absolute eta slope | 0.0777 | 1.124 | 1.532 |

Both predictions have a large phase displacement, but alignment leaves a
substantial shape error.  Learned-Hamiltonian drift peaks near 22--23% around
`t=176`, so this is also not a trajectory that conserves the correct shape
while merely translating at the wrong speed.

The teacher-forced DNO error on all 251 truth frames remains extremely small:
the maximum relative `gxi` error is about `1.05e-3` for C16 and `1.17e-3` for
C18.  Moreover, an exact order-6, pad-8 DNO evaluated on the *predicted* states
agrees with the learned DNO to roughly 1--7% through `t=184`, including its
high-band norm.  The high-frequency output is therefore mostly the DNO's
response to an already contaminated state, not a direct output hallucination.

## Why the C18 scalar loss was ill posed

C18 fitted one scalar

\[
c=-\frac{\langle G(\eta)\xi,\eta_x\rangle}
        {\|\eta_x\|_2^2}
\]

on every source-5/6 snapshot and divided its error by the fitted truth speed.
This is meaningful for a rigid traveling wave but not for counter-propagating
or interacting crests.  Approximately 41% of the selected source-5/6 rows are
mixed-direction multicrest states.  Near cancellation makes the fitted speed
small and the inverse-speed normalization assigns the greatest weight to the
states for which the scalar interpretation is least valid.

For case 14 itself, C16's initial scalar speed error is only about `9.6e-5`
relative, while 99.2% of the squared DNO error is orthogonal to the global
translation direction.  C18 lowers the dataset-average scalar diagnostic but
worsens the case-specific localized tangent error and the transverse rollout.

The training implementation has therefore been corrected to use an
`h`-localized projection of the DNO *error*, normalized by the characteristic
speed `sqrt(g h)`, and to include source 14.  This removes the cancellation
singularity.  It is a correction to the C18 objective, not by itself the
case-14 spectral remedy.

## Spectral precursor

Define the production-band amplitude

\[
A(t)=\frac{1}{N}
\sqrt{\operatorname{mean}_k
\left[\mathbf 1_{64\le |k|<128}|\widehat\eta_k(t)|^2\right]}.
\]

For case 14, `A(0)=4.5345e-9`, and truth never grows materially above this
value.  In contrast:

| t | C16 A(t) | C18 A(t) |
| ---: | ---: | ---: |
| 20 | 2.34e-7 | 1.94e-7 |
| 40 | 8.99e-7 | 7.58e-7 |
| 120 | 3.39e-6 | 3.04e-6 |
| 140 | 1.11e-5 | 1.04e-5 |
| 200 | 5.30e-5 | 7.54e-5 |

The existing guard used

\[
A(t)>\max(10^{-4},100A(0)),
\]

so it never activated.  A floor of `5e-7` activates near `t=30`, while the
relative eta error is still below `0.01`.

## Implemented guard

The initial-state soliton selector is

\[
r_- = \frac{\|\min(\eta_0,0)\|_2^2}{\|\eta_0\|_2^2},
\qquad r_-<10^{-3},\qquad \operatorname{mean}(\eta_0)>0.
\]

All 64 Tanaka ICs satisfy `r_- < 2.2e-4`; every inspected non-Tanaka IC has
`r_- >= 0.257`.  This makes the selector intrinsic and separates the class for
which the horizontal scale is proportional to depth.

After the temporal envelope trigger activates, the state is multiplied by

\[
\sigma_h(k)=\exp\left[-0.69
\left(\frac{|k|}{k_{\mathrm{eff}}(h)}\right)^8\right],
\qquad
k_{\mathrm{eff}}(h)=\min\left(128,\frac{12}{h}\right).
\]

For case 14, `k_eff=67.57`: mode 64 is retained at about 0.64, mode 80 at
0.070, mode 100 at `1.3e-7`, and the observed runaway modes 115--128 are
effectively eliminated.  For shallow case 27, `12/h > 128`, so the existing
production ceiling is unchanged.  The trigger also remains inactive because
its physical high-band energy does not grow by 100 times.

Implementation:

- `solver/evals/model_rollout.py`
- `solver/evals/eval_suite.py`
- `solver/evals/tests_soliton_spectral_guard.py`
- `scripts/eval_c16_soliton_spectral_guard.sh`

Focused CPU tests verify the IC selector, exact identity when inactive, and
per-depth attenuation at equal Fourier mode.  Full matched rollout results are
recorded separately in `EXPERIMENTS.md`.

## Follow-up: why Tanaka-g1 case 24 ends at 0.996

Case 24 is not a second spectral cascade.  It has depth `h=0.01549048` and two
same-direction crests with dimensionless amplitudes `0.21406` and `0.09417`.
At the final time, the guarded C18 raw surface error is `0.99624`, but a
continuous circular translation reduces it to `0.20001`; this removes 95.97%
of the squared discrepancy.  Aligning the two crests separately reduces their
combined local shape error to approximately `0.0859`.  Their final offsets are
11 and 7 grid points, so the dominant defect is differential phase speed, not
one common rigid shift.

The predicted final crest amplitudes are `0.0030707` and `0.0014880`, compared
with truth values `0.0034015` and `0.0014335`.  The maximum slope remains
bounded (`0.05598` predicted versus `0.06522` truth), while the learned
Hamiltonian drift is `-10.59%`.  The amplitude depletion and energy drift are
therefore material but do not constitute instability.

The soliton spectral guard is effectively the identity on this trajectory:

\[
  A(0)=8.9922\times10^{-7},\qquad
  A_{\mathrm{tr}}=8.9922\times10^{-5},\qquad
  \max_t A(t)=9.7008\times10^{-7}.
\]

The maximum sigmoid activation is only `2.14e-20`, and guarded and unguarded
C18 predictions differ by `3.35e-5` relative at the final time.  Increasing
the guard would therefore address the wrong mechanism.

C16 handled the same case better: raw/aligned errors `0.72539/0.07219`, shift
`6.58` grid points, and Hamiltonian drift `-7.66%`.  On the truth trajectory,
the C18 fine-tune improves the old *global* scalar speed diagnostic but worsens
the `h`-localized diagnostic: aggregate local relative speed RMS changes from
`1.574e-3` to `1.626e-3`, while mean teacher-forced DNO error changes from
`0.004015` to `0.004061`.  This is the exact failure mode hidden by fitting one
velocity to multiple crests.  The corrected localized tangent objective now
in the trainer is the appropriate next experiment; it has not yet been used
to train a checkpoint, so case 24 should remain a stated tail-margin caveat.
