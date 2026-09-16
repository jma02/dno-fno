# Four-block FNO: per-batch performance inspection

The selected model takes **12.102 ms per batch of 4,096 on one H100 80GB** in this fresh run (21 synchronized trials, range 11.742–12.461 ms). The previous matched configuration measured 11.569 ms. The largest measured cost is activation processing around the transforms; matrix multiplication accounts for only 18% of recorded device time. The trace supports prioritizing those activation passes, but does not establish hardware bandwidth saturation.

This inspection leaves the architecture unchanged: four 32-channel FNO blocks, a 256-point learned grid, all 129 Fourier bins, folded spatial branches, packed real spectral GEMMs, backward-normalized cuFFT, a 32→64→1 decoder, and the full-resolution analytic G₀+G₁ baseline. There are 1,063,296 float32 learned parameters. CUDA Graphs and Tensor Core GEMMs are already active.

## Measurement scope

- Source commit `6e97220`; [Modal app](https://modal.com/apps/sciml-at-ud/main/ap-G0p7TXjByWsUbW9g4CdDOW), profile `sciml-at-ud`, NVIDIA H100 80GB HBM3, JAX 0.9.2.
- Resident real pilot data; three warmups, 21 separately synchronized timings, then a separate three-step profiler capture and optimized HLO dump. GPU function wall time was 29.35 seconds including setup, compilation and profiling.
- Full ordinary forward/loss/backward/AdamW executable, excluding loading, compilation, validation, periodic Hadamard and multi-GPU communication.
- The same initial state is reused, at step zero: the warmup learning rate and mode-loss coefficient are zero. They are runtime values, so this still executes the compiled ordinary step, but it is not a progressing epoch or a trained-checkpoint measurement. The pilot data source is `/data/outputs/paper_dataset/arrays`, not the requested full equal-family production dataset.
- The selected FNO is a benchmark candidate; this is not instrumentation of a currently running production epoch.

## Disjoint device-time breakdown

Times are means over the three profiled batches. Percentages refer to **10.650 ms summed GPU event duration**, not the independently measured 12.102 ms wall median.

| Category | ms/batch | Share of device event time |
| --- | ---: | ---: |
| Layout changes and real/complex packing fusions | 2.371 | 22.3% |
| Other activation, gradient, loss and update kernels | 2.271 | 21.3% |
| Actual FFT kernels | 2.003 | 18.8% |
| GEMM/GEMV kernels | 1.910 | 17.9% |
| Separate FFT scaling kernels | 1.183 | 11.1% |
| Device-to-device copies | 0.911 | 8.6% |
| Host-to-device transfer | 0.0009 | <0.1% |

Classification uses kernel names, checked against HLO and source metadata. Layout fusions can also perform scaling, conjugation or activation arithmetic; their time cannot be called pure memory-copy cost. Split-K reductions are included in “other.” Small profiler interval overlaps make the union of compute-stream activity 10.647 ms rather than the summed 10.649 ms on that stream. There are 217 device events per batch, including copies and memset; graph replay groups many launches.

Layout/packing, separate scaling and copies together cost **4.465 ms, 41.9%** of device event time. Combined with actual FFT kernels, these categories occupy 60.7%. This describes where time goes, not an assertion that all that work is redundant or removable.

## What the large kernels actually do

The optimized HLO and trace identify the following costs (these overlap the categories above and must not be added to that table):

| Trace kernel/group | What the HLO does | ms/batch |
| --- | --- | ---: |
| `input_transpose_fusion_16` and `_1` | Forward Fourier real/imag packing and unpacking, transposes between GEMM and FFT layouts, across the blocks | 0.851 |
| `input_transpose_fusion_12` and `_8` | Backward packing/unpacking, conjugation/adjoint scaling and transposes | 0.966 |
| `loop_multiply_tanh_fusion` | Decoder GELU forward; materializes two `[4096,256,64]` tensors including saved intermediates | 0.274 |
| `fusion_169` | Decoder output backward and hidden GELU derivative, fused into a `[4096,256,64]` gradient | 0.326 |
| `input_add_reduce_fusion_6` | Three block activation backpropagations, including FFT-adjoint scaling and bias-gradient partial reductions | 0.407 |
| `input_transpose_fusion_17` | Last-block activation backward and layout conversion | 0.132 |

Kernel names may be reused across equivalent HLO instructions in different blocks. For example `_1` aggregates four forward unpack/transpose operations; it is not just one block. Source mapping uses the called fusion computation as well as the entry instruction, because metadata on a fusion can name only one of several fused operations.

All **20 D2D copies** are associated with FFT HLO operations. Their payload totals **1,363,771,392 bytes per batch**. Eight FNO copies alone move 135,266,304 bytes (129 MiB) each: four forward and four backward, totaling **1.008 GiB of copy payload and 0.725 ms**. Copy payload is not measured HBM traffic; a copy logically reads and writes, and actual DRAM traffic depends on caching and implementation.

For the FNO transforms specifically, source-attributed forward FFT arithmetic costs 0.751 ms and backward costs 0.751 ms. Their separate scaling costs 0.486 ms in each direction and their copies cost 0.362 ms in each direction. Thus these FFT calls spend **1.696 ms on scaling/copies versus 1.501 ms on FFT kernels**, before accounting for packing and transposes.

The tensor sizes explain why this matters despite the small parameter count:

- One hidden activation `[4096,256,32]` float32: **128 MiB**.
- Its spectrum `[4096,32,129]` complex64, or packed real representation: **129 MiB**.
- A decoder hidden activation `[4096,256,64]` float32: **256 MiB**.
- All model weights together: **4.06 MiB**.

These are individual tensor sizes, not total peak allocation. The compiler buffer estimate is 2.219 GiB; allocator peak bytes in use is 2.649 GiB for the captured setup. The allocator's 6 GiB pool is a separate reservation statistic.

The trace contains explicit `sm90_xmma_gemm_f32f32_tf32f32` and CUTLASS `tensorop` kernels. Tensor Core GEMMs are already used; this does not imply every matrix operation uses Tensor Cores or achieves peak throughput. AdamW is present in small fused updates; some updates also reduce shared spatial-weight gradients. It is not valid to label the entire final command buffer “optimizer”: it also contains encoder and decoder gradient work.

## Launch gaps and limits of the inference

Within the profiled calls, mean compute-stream span is **10.934 ms**, of which **10.647 ms is active** and **0.287 ms is gaps**: 97.4% activity within that span. This is a timeline duty cycle, not SM, bandwidth or FLOP utilization. The largest internal gap is 26.7 microseconds in the first profile step and under 4.6 microseconds in the next two. There is no large recurring stall halfway through GPU computation.

The profiled host annotations average 12.906 ms: 1.252 ms before the first compute-stream event, 10.934 ms compute span, and 0.719 ms after the last event. Trace instrumentation, host dispatch and synchronization affect these boundaries. A tiny scalar H2D event precedes compute; including that separate stream would misleadingly describe some pre-compute delay as an internal GPU gap. The 12.906 ms profiled wall time is **not** interchangeable with the unprofiled 12.102 ms median. In particular, subtracting traced kernel time from the unprofiled median does not measure exact Python overhead.

No Nsight hardware counters were collected. XLA reports about 19.18 GB “bytes accessed” and 56.11 GFLOP, but library/custom-call coverage and fusion accounting differ from actual device traffic and arithmetic; its `optimal_seconds` field is negative. These compiler estimates are not a measured roofline or a reliable bandwidth-utilization percentage.

The evidence points to repeated activation movement, conversion and nonlinear processing as the largest performance constraint. Faster GEMMs alone address only 1.91 ms of the recorded work. Exact HBM-versus-instruction limits within the dominant fused kernels remain unmeasured. The preceding fused-FFT and channel-first experiments were slower, so identifying this cost does not establish that an available fusion can eliminate it profitably. No further optimization or architecture change was made for this inspection.

## Reproduction and artifacts

Run from the repository with the existing Modal environment:

```sh
MODAL_PROFILE=sciml-at-ud modal run experiments/fewer_branches_pilot_20260916.py --prepared --fno-transform-benchmark --profile-step --run-name fewer_branches_20260916_1545
```

- [Raw timings, compiler estimates and kernel totals](fewer_branches_20260916_1545_fno_inspect_h100.json).
- [Derived categories, step intervals and copies grouped by source](fno_per_batch_inspection_20260916.json).
- Volume `dno-fno-train-data`: `experiments/fewer_branches_20260916_1545/fno_inspect/plugins/profile/2026_09_16_22_54_06/modal.trace.json.gz` and sibling `modal.xplane.pb`.
- Same volume: `experiments/fewer_branches_20260916_1545/fno_inspect_optimized_hlo.txt`.
- Local trace/HLO copies: `../pilot-checkpoints/fno_profile_20260916/inspect.trace.json.gz` and `optimized_hlo.txt`.
