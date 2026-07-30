# C20 mode-balanced CS-DNO: rollout evaluation and movie package

## Suggested subject

C20 CS-DNO evaluation: 0/320 nonfinite rollouts and substantially reduced long-time error tails

## Email-ready summary

We completed the fixed 32-initial-condition evaluation of the new C20
mode-balanced CS-DNO on all ten rollout families.  C20 is a fresh 40-epoch
retraining of the full CS-DNO architecture.  In addition to the ordinary
relative Sobolev loss, it was trained with a physically normalized complex
Fourier-mode loss, so that a weak but dynamically important sideband is not
hidden by the much larger carrier modes.  This changes training only; it adds
no operator evaluations or architectural cost at inference.

All 320 predicted trajectories remained finite.  Of the 305 trajectories for
which the reference solution passed the prescribed Hamiltonian-drift validity
test, the final relative surface-elevation error exceeded 0.25 in 10 cases,
compared with 16 for C19.  The corresponding counts above
0.25/0.50/0.75/1.00 changed from 16/4/2/0 for C19 to 10/1/1/1 for C20.  C20
improves the familywise median final error in 9 of 10 families and the 95th
percentile in 7 of 10.  The median improvements are especially clear for the
two Tanaka families and for all three Benjamin--Feir families.

The single C20 case whose raw final error exceeds one is not a numerical
blow-up.  It is Tanaka-g0 case 27: the trajectory is finite, smooth, and has a
bounded slope throughout, but the solitary wave has accumulated a position
error.  Its raw final relative error is 1.127; optimally translating the
prediction on the periodic domain reduces this to 0.0506, with a shift of 7.45
grid points.  Every Tanaka-g0 case has translation-aligned final error below
0.25.  Thus it would be inaccurate to call the result literally zero raw
divergences, but the remaining threshold event is qualitatively different
from the former mid-wavenumber cascade and NaN mechanism.

The Benjamin--Feir evidence supports the intended mechanism of the new loss.
For the previous principal modal-transfer outlier, bf-g0 case 23, final error
falls from 0.912 with C19 to 0.364 with C20, and final Hamiltonian drift improves
from -2.47% to -1.51%.  At the active modes k=17,19,21, the final complex modal
errors change from 0.515/0.546/6.35 to 0.171/0.336/2.57.  The weakest k=21
sideband is still over-amplified, but much less severely.  This is direct
evidence that balancing errors mode by mode improves the in-band energy
transfer that was poorly represented by a carrier-dominated global loss.

The evaluation used the same soliton-selective spectral-envelope guard as C19.
A saved-frame recomputation found a maximum guard weight below 0.0041 and a
maximum monitored-band amplitude below 0.58 of the activation threshold.  The
guard is analytically disabled on all non-Tanaka families.  Therefore the
observed 0/320 nonfinite result is principally model behavior rather than a
guard rescue, although saved frames cannot exclude a very small between-frame
activation.

The honest limitations are also visible.  Tanaka-g1 case 24 retains a genuine
aligned shape/phase error (raw 0.459, aligned 0.146).  The linear-family median
improves, but its p95 worsens from 0.136 to 0.165; moreover, the current
"linear" registry is known to use nonlinear reference dynamics because of a
truth-kind semantic mismatch.  Both Stokes p95 values regress slightly, but
remain small in absolute terms (0.00132 and 0.0334).  These caveats do not
change the broad conclusion that C20 is the strongest learned-operator
accuracy/stability checkpoint so far.

Five short GIFs are attached below.  Each uses the existing six-panel format:
ground truth and prediction for eta, xi, and G(eta)xi, together with relative
error histories.  The case-27 movie is the clearest illustration of a large
raw norm caused mainly by accumulated translation; bf-g0 case 23 shows the
remaining but substantially reduced sideband-transfer error.

## Fixed-suite results

All entries are final-time relative L2 errors in eta.  Statistics use only
truth-valid cases.

| Family | C19 median | C20 median | C19 p95 | C20 p95 | C20 nonfinite / 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tanaka g0 | 0.07090 | 0.02281 | 0.37295 | 0.29847 | 0 |
| Tanaka g1 | 0.12107 | 0.05212 | 0.59621 | 0.40361 | 0 |
| Benjamin--Feir g0 | 0.01140 | 0.00729 | 0.20966 | 0.14971 | 0 |
| Benjamin--Feir g1 | 0.01678 | 0.01613 | 0.12953 | 0.07210 | 0 |
| Benjamin--Feir modal | 0.01361 | 0.01150 | 0.15528 | 0.10898 | 0 |
| Random sea, deep | 0.00285 | 0.00145 | 0.02107 | 0.01354 | 0 |
| Random sea, finite depth | 0.00547 | 0.00251 | 0.03965 | 0.02460 | 0 |
| Linear registry | 0.02784 | 0.01676 | 0.13569 | 0.16453 | 0 |
| Stokes, deep | 0.000467 | 0.000462 | 0.00115 | 0.00132 | 0 |
| Stokes, finite depth | 0.00368 | 0.00386 | 0.02228 | 0.03338 | 0 |

The C20 Tanaka translation-aligned summaries are:

| Family | Raw median / p95 | Aligned median / p95 |
| --- | ---: | ---: |
| Tanaka g0 | 0.02281 / 0.29847 | 0.00518 / 0.04630 |
| Tanaka g1 | 0.05212 / 0.40361 | 0.00909 / 0.09723 |

## Movies

- [Tanaka g0 case 27: finite translation-dominated raw divergence](../outputs/c20_mode_balanced_full_20260713_050000/email_movies_20260714/tanaka_g0_idx=27_case27_h0.012.gif)
- [Tanaka g1 case 24: remaining aligned shape/phase tail](../outputs/c20_mode_balanced_full_20260713_050000/email_movies_20260714/tanaka_g1_idx=24_case1000024_h0.015.gif)
- [Tanaka g1 case 29: formerly severe phase case](../outputs/c20_mode_balanced_full_20260713_050000/email_movies_20260714/tanaka_g1_idx=29_case1000029_h0.012.gif)
- [Tanaka g1 representative median case](../outputs/c20_mode_balanced_full_20260713_050000/email_movies_20260714/tanaka_g1_median_case1000001_h0.015.gif)
- [Benjamin--Feir g0 case 23: modal-transfer outlier](../outputs/c20_mode_balanced_full_20260713_050000/email_movies_20260714/bf_g0_idx=23_case23_h3.035.gif)

The GIFs are 1150 x 700 pixels, contain 126 frames each, and were rendered on
CPU with every second saved rollout frame at 18 frames per second.

## Reproducibility pointers

- C20 checkpoint: `outputs/c20_mode_balanced_full_20260713_050000`, epoch-40
  `best` (identical to `final`).
- Tanaka outputs:
  `outputs/c20_mode_balanced_full_20260713_050000/eval_best_soliton_spectral_guard_20260714_042851`.
- Other eight families:
  `outputs/c20_mode_balanced_full_20260713_050000/eval_best_guarded_non_tanaka_suite_n32_20260714_044418`.
- Protocol: 32 initial conditions per family, cached matched reference
  trajectories, batched surrogate rollout, float64 numerical harness, 251
  saved frames.
