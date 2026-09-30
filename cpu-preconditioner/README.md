# Tiny CPU neural-preconditioner experiment

This tests the inverse-solver idea on the **2D fluid interior**, separately from
the production DNO network. No production code or GPU jobs are involved.

For each Tanaka surface, linear triangular finite elements discretize Laplace's
equation on a periodic, surface-following mesh. Surface potential is prescribed;
the flat bottom has zero normal derivative. With the surface nodes eliminated:

```text
A = interior/interior block of the Laplace stiffness matrix
b = -interior/surface block × surface potential
solve A × interior potential = b
surface reaction / horizontal grid spacing = approximate DNO output
```

The tiny U-Net corrects a flat-water solution using the wavy geometry. A Fourier
transform in x reduces the flat inverse to 33 small vertical matrices; these
are precomputed at the mean fluid height. Input and output are scaled by the
flat solution's RMS. The combined inverse is nonlinear in the residual, so comparisons use **right-preconditioned
flexible GMRES**, not ordinary GMRES. PyAMG supplies the solver; the final true
residual is also checked explicitly. See the [PyAMG documentation](https://pyamg.readthedocs.io/en/latest/generated/pyamg.krylov.html#pyamg.krylov.fgmres).

## Run

From the repository root, using its existing CPU-capable PyTorch environment:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/codex-uv-cache \
  uv run --no-sync --with pyamg==5.3.0 cpu-preconditioner/run.py \
  --epochs 100 --learning-rate 1e-3 --output outputs/cpu_preconditioner_trial

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 uv run --no-sync --with pyamg==5.3.0 \
  python -m unittest discover -s cpu-preconditioner/tests -v
```

Use a fresh output directory. CPU execution is also enforced inside the script.
The source is `outputs/paper_dataset/arrays`; the nine fixed rows are in `ROWS`.
Six are actual training ICs, three actual test ICs, all at time zero and with
distinct simulation IDs. Depths range from 0.206 to 0.347. Fourier resampling
reduces 1,024 horizontal samples to 64; there are 16 vertical cells and 1,024
interior unknowns, including the bottom nodes.

The 5,609-parameter network trains from scratch using two CPU threads. The user
approved **100 epochs at initial learning rate `1e-3`** for this CPU experiment;
cosine decay ends at `1e-4`. This is an exception to the repo's one-epoch rule,
not a change to production training.

The six training geometries supply 6,144 fixed pairs `(r, A^-1 r)`: one quarter
white interior forcing, one quarter smooth manufactured corrections using all
16 vertical modes, and half surface-boundary forcing (smooth and broad-band).
Sparse LU supplies exact same-grid labels. Each pair is scaled to unit target
RMS. The loss is relative squared potential error plus `0.01` times relative
squared equation residual. Independent draws (128 per training geometry) select
the best checkpoint; they are RHS validation, not unseen-geometry validation.
The three test geometries and their actual Tanaka potentials never enter
training or selection. Fresh mixed RHS probes also test these unseen geometries.

`results.json` separates **standalone** inverse predictions from solutions
**after FGMRES**. The analytic flat inverse is always evaluated on its own:
the learned correction must improve on that baseline to deserve credit. DNO
error is measured separately because surface differentiation can amplify small
potential errors. Solver timings include every flat inverse and NN application;
matrix/factor setup is reported separately. Training time includes validation
and checkpoint selection.

## Original one-epoch result

The original Jacobi-plus-U-Net pilot used one epoch at `2e-5`, manufactured
corrections with only eight vertical modes, and no boundary-forcing examples.
Completed September 29, 2026; training took **5.12 seconds**. Raw weights,
timings, and short-rollout arrays are in `outputs/cpu_preconditioner_20260929/`.
All methods reached relative equation residual below `1e-9` on all three test
cases. Solve times below are medians of three warmed repeats, including NN
conversion/inference and Krylov work but excluding matrix/geometry setup.

| Method | Iterations across three cases | Solve time |
|---|---|---|
| No preconditioner | 143 / 170 / 173 | 10.0–12.2 ms |
| Jacobi | 135 / 173 / 183 | 9.6–13.0 ms |
| Jacobi + learned correction | 137 / 172 / 183 | 165–225 ms |
| ILU | 6 / 6 / 6 | 0.58–0.61 ms |
| Direct sparse LU | — | 0.048–0.056 ms after factorization |

ILU and direct LU setup each cost approximately 2.0–2.2 ms per geometry.
The learned preconditioner **did not improve convergence and was much slower**.
Its standalone potential error was **97.5%** on the three held-out ICs, almost
identical to Jacobi. The accurate corrected solve below was not evidence that
the network had learned an accurate inverse.

The NN-assisted solve's DNO output agrees with the same-grid direct result to
`3.76e-7` relative or better. Separately, refining from 64×16 to 256×64 changes
the three DNO outputs by 0.305–0.685%; iterative accuracy is not continuum
accuracy. Flat-mode analytic checks and real-IC refinements show approximately
second-order convergence.

A four-step RK4 water-wave smoke test (`dt=0.001`, horizon `0.004`) recomputes
the domain and preconditioner geometry at every stage. NN-assisted and direct
paths use the same coarse FEM discretization. Final eta/xi differences are
`3.62e-10` / `2.19e-12` relative; wall times are 2.67 / 0.043 seconds. This is
only a coupling check—not the production GL2/filter protocol or a long-rollout
stability/accuracy result.

## Trained flat-inverse correction

September 30: the approved 100-epoch run took **8 minutes 35 seconds**, including
validation and checkpoint selection. Epoch 100 had the best validation loss:
`4.35e-5`, versus `2.82e-3` for the untrained flat baseline. Results and weights:
`outputs/cpu_preconditioner_flat_20260930/`.

These are predictions **before GMRES**, compared with the same-grid direct solve:

| Held-out simulation | Flat potential error | Flat + NN potential error | Flat + NN DNO error |
|---|---:|---:|---:|
| 16477 | 1.241% | 0.927% | 124.8% |
| 16494 | 3.886% | 0.458% | 57.0% |
| 16602 | 4.633% | 0.377% | 27.9% |

On 128 fresh mixed RHSs per unseen geometry, learned potential error has medians
0.80–0.84% and 95th percentiles 1.04–1.16%. Thus the correction actually learned
an inverse approximation; it is not just exploiting the physical test RHSs.
However, potential accuracy still does **not** give an accurate standalone DNO.

FGMRES iterations fall from **6 / 8 / 8** with the flat inverse to **5 / 7 / 7**
with its learned correction. All solves converge, but the NN adds overhead:
**6.53–9.11 ms versus 0.60–0.76 ms** for flat-only FGMRES. ILU takes 0.57–0.59 ms.
The analytic flat inverse accounts for most of the improvement over the original
Jacobi pilot. This is progress in inverse prediction, **not a CPU acceleration
or a standalone DNO replacement**. No production model or GPU job was changed.

An independent four-step RK4 check **without GMRES**, using the first test IC,
gives final eta errors 0.197% (flat) and 0.120% (flat + NN) against the same-grid
direct rollout. All states stay finite, but the horizon is only `0.004`; this
does not establish useful long-time accuracy. The original GMRES-corrected
smoke agrees to `2.33e-12` in eta because it solves the equations accurately.
Both diagnostics are saved in the result directory.

Read `laplace.py` for the discrete operator and `run.py` for the network,
training loop, solver comparison, and short coupled test.
