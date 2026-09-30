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

The tiny U-Net predicts an interior correction from a residual and the geometry.
It adds that correction to Jacobi (division by the matrix diagonal). The network
is nonlinear in the residual, so all comparisons use **right-preconditioned
flexible GMRES**, not ordinary GMRES. PyAMG supplies the solver; the final true
residual is also checked explicitly. See the [PyAMG documentation](https://pyamg.readthedocs.io/en/latest/generated/pyamg.krylov.html#pyamg.krylov.fgmres).

## Run

From the repository root, using its existing CPU-capable PyTorch environment:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/codex-uv-cache \
  uv run --no-sync --with pyamg==5.3.0 cpu-preconditioner/run.py \
  --output outputs/cpu_preconditioner_trial

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 uv run --no-sync \
  python -m unittest discover -s cpu-preconditioner/tests -v
```

Use a fresh output directory. CPU execution is also enforced inside the script.
The source is `outputs/paper_dataset/arrays`; the nine fixed rows are in `ROWS`.
Six are actual training ICs, three actual test ICs, all at time zero and with
distinct simulation IDs. Depths range from 0.206 to 0.347. Fourier resampling
reduces 1,024 horizontal samples to 64; there are 16 vertical cells and 1,024
interior unknowns, including the bottom nodes.

The 5,609-parameter network trains from scratch for **one epoch**, Adam learning
rate `2e-5`, using two CPU threads. The six training geometries supply 6,144
synthetic pairs `(A e, e)`: the known correction `e` gives an exact training label
without an inverse solve. Half are smooth horizontally, half cover the retained
horizontal modes; both use eight vertical cosine modes. They are not uniformly
sampled over the entire interior vector space. The loss is mean squared error
in the unit-RMS correction, not the PDE residual. Test potentials are the actual
held-out Tanaka boundary potentials; no test data enter training.

## First result

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
The learned preconditioner **did not improve convergence and was much slower**
in this one-epoch pilot. This does not establish the performance of a fully
trained or larger model; no additional epochs or tuning runs were performed.

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

Read `laplace.py` for the discrete operator and `run.py` for the network,
single training pass, solver comparison, and short coupled test.
