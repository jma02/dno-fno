# C19 full-suite tail-threshold audit

This audit filters the final raw relative elevation error
`rel_l2_eta[-1]` from the guarded C19 10-family, 32-IC suite. The official
truth-valid mask from each regime summary is applied before counting model
tails. Thresholds are strict (`error > threshold`) and nested.

## Truth-valid counts

| Threshold | Final-time count | Any-time count | Fraction of 305 valid cases |
|---|---:|---:|---:|
| `> 0.25` | 16 | 16 | 5.25% |
| `> 0.50` | 4 | 4 | 1.31% |
| `> 0.75` | 2 | 2 | 0.66% |
| `> 1.00` | 0 | 0 | 0.00% |

Equivalently, 12 cases lie in `(0.25, 0.50]`, two in `(0.50, 0.75]`, two
in `(0.75, 1.00]`, and none above the divergence threshold.

## Cases with final error above 0.25

| Regime | Case ID | Final | Maximum | Final learned-H drift | First crossing times |
|---|---:|---:|---:|---:|---|
| bf_g0 | 23 | 0.911981 | 0.911981 | -2.469% | 0.25: 153.6; 0.50: 173.6; 0.75: 186.4 |
| tanaka_g1 | 1000029 | 0.792011 | 0.792011 | +1.867% | 0.25: 84.0; 0.50: 140.0; 0.75: 192.0 |
| tanaka_g1 | 1000011 | 0.714079 | 0.714079 | +2.868% | 0.25: 98.4; 0.50: 156.8 |
| tanaka_g0 | 4 | 0.515510 | 0.515510 | +3.159% | 0.25: 133.6; 0.50: 196.8 |
| tanaka_g1 | 1000005 | 0.499775 | 0.499775 | +4.982% | 0.25: 150.4 |
| tanaka_g1 | 1000024 | 0.473434 | 0.473434 | -6.071% | 0.25: 161.6 |
| tanaka_g1 | 1000001 | 0.432614 | 0.432614 | +2.146% | 0.25: 152.0 |
| tanaka_g1 | 1000028 | 0.417780 | 0.417780 | +2.546% | 0.25: 150.4 |
| tanaka_g0 | 27 | 0.385958 | 0.385958 | +0.365% | 0.25: 167.2 |
| bf_g0 | 0 | 0.379928 | 0.379928 | -1.654% | 0.25: 178.4 |
| tanaka_g0 | 8 | 0.362310 | 0.362310 | +2.673% | 0.25: 160.8 |
| bf_g1 | 1000017 | 0.362032 | 0.364100 | -6.586% | 0.25: 184.8 |
| bf_modal | 2000004 | 0.333073 | 0.333073 | +0.033% | 0.25: 175.2 |
| tanaka_g0 | 9 | 0.314653 | 0.314653 | -2.335% | 0.25: 170.4 |
| linear | 14 | 0.296757 | 0.296757 | +0.140% | 0.25: 16.6 |
| tanaka_g0 | 17 | 0.267835 | 0.267835 | -2.839% | 0.25: 195.2 |

The four cases above 0.50 are therefore bf_g0 case 23, tanaka_g1 cases
1000029 and 1000011, and tanaka_g0 case 4. The two cases above 0.75 are
bf_g0 case 23 and tanaka_g1 case 1000029.

## Truth-invalid exclusions

Fifteen of the 320 generated reference trajectories fail the suite's truth
Hamiltonian-drift validity criterion and are excluded from official accuracy
and divergence rates. Nine of these have raw final error above 0.25:

| Regime | Case ID | Final | Maximum |
|---|---:|---:|---:|
| linear | 32 | 1.900755 | 1.900755 |
| linear | 107 | 1.475093 | 1.475093 |
| linear | 131 | 1.410078 | 1.487764 |
| linear | 227 | 1.308329 | 1.690138 |
| linear | 54 | 1.292977 | 1.680462 |
| linear | 101 | 1.122769 | 1.448185 |
| bf_modal | 2000006 | 1.026762 | 1.036458 |
| stokes_finite | 98 | 0.704332 | 0.864552 |
| linear | 147 | 0.313990 | 0.803518 |

These values are not counted as model divergences because their reference
solutions fail the predeclared validity test. Independently, an array-level
scan confirmed that C19's predicted eta, xi, and Gxi remain finite for all
320 trajectories, including these exclusions.
