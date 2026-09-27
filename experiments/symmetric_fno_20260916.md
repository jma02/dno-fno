# Four-layer linear, self-adjoint FNO option

Selected by the user after the benchmark: retain **four conditioning layers** and accept the measured latency increase. The existing canonical FNO and its saved checkpoints remain unchanged.

The surface-only FNO takes normalized eta and log-depth, and generates 32 real spatial weight fields. It retains the existing width32 Fourier blocks, all129 bins on the256-point grid, packed real GEMMs and backward-normalized cuFFT. The decoder is32→64→32; its final kernel is zero-initialized, so initialization still gives the analytic baseline.

The potential follows `sum_c K_c* [weight_c(eta, depth) * K_c xi] / sqrt(32)`. The same learned real Fourier filter appears on both sides, and no nonlinear activation or bias acts on xi. The32 filters retain every grid frequency except DC, preserving a mean-free correction and a constant-potential nullspace. These are32 channels, not a32-mode low-rank restriction. Depth enters through the spatial weights; the filters themselves are shared across examples.

This enforces linearity in xi and self-adjointness at fixed eta/depth, up to numerical rounding. At the1024-point output, upsampling uses the correctly scaled adjoint of restriction, including its coarse Nyquist factor. There is **no eta-squared gate or flat-surface-order constraint**. Full-grid G0+G1 stays unchanged.

| Same-run benchmark | Current FNO | Selected symmetric FNO |
| --- | ---: | ---: |
| Conditioning blocks | 4 | 4 |
| Parameters | 1,063,296 | 1,069,376 |
| Training batch latency | 11.827943 ms | 13.938088 ms |
| Compiler buffer estimate | 2.219 GiB | 2.345 GiB |

One NVIDIA H10080GB, batch4096, float32 learned parameters, resident real pilot data. Each variant had three warmups and21 synchronized measurements, followed by21 alternating-order measurements; the table uses alternating medians. Heads were nonzero to exercise the learned paths, mode-loss weight was fully active, and AdamW used constant1e-4. Ordinary forward/loss/backward/update is timed; compilation, data loading and periodic Hadamard are excluded. The selected option costs **17.84% more latency**, or about2.11ms/batch. This is not a validation-accuracy or full-epoch measurement.

Nine CPU model tests passed, including activated full-grid linearity/self-adjointness, coarse-Nyquist coverage, initial learning gradient and a finite-difference gradient check. GPU relative errors for the four-layer correction were3.478e-7 for linearity and2.660e-7 for the adjoint VJP; normalized bilinear defect was1.512e-8. Existing model tests still pass. No custom backward kernel was introduced.

Train this optional model with the existing trainer flags:

```text
--cs_correction symmetric_fno --n_blocks 4 --cs_learned_grid 256
```

A subsequent fresh four-epoch run at constant1e-4 completed: validation relative-L2 was0.000838658 versus0.000843668 for the unconstrained FNO at epoch4 (0.59% lower), while total validation loss was2.66% higher. This is effectively unchanged prediction accuracy in one seed. Full results and checkpoint checks are in [the training record](c27_symmetric_fno4_const1e4_equal_4epoch_20260916.json). The four-epoch run took464.73s including startup, compilation, validation and checkpoints.

The original completed benchmark also measured a two-block alternative (10.0114ms,538880 parameters); the user retained four layers. A later three-block follow-up was cancelled at the user's steering before a completed comparison. The benchmark command now compares only the current FNO and the selected four-layer option:

```sh
MODAL_PROFILE=sciml-at-ud modal run experiments/fewer_branches_pilot_20260916.py --prepared --symmetric-benchmark --run-name fewer_branches_20260916_1545
```

[Raw completed benchmark](fewer_branches_20260916_1545_symmetric_fno_benchmark_h100.json), source `eb3bea9`, [Modal app](https://modal.com/apps/sciml-at-ud/main/ap-TWNInVSwLTtRnC2xBFidlu).
