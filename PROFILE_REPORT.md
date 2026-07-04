# cs_dno single-step profile — 2× RTX 6000 Ada, batch=256

Command profiled (mirrors real train command):

```
XLA_FLAGS="--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true" \
JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno --norm scale --dataset combined_dataset_v9.npz --modes 64 \
  --width 512 --n_blocks 8 --latent 256 --sobolev_k 1 --batch_size 256 \
  --lr 2e-4 --weight_decay 1e-4 --epochs 40 --cs_n_polys 3 --cs_mult_hidden 128
```

Model: `CraigSulemDNO` from `models/dno-net/dno_net_v2.py`
(8 × `CraigSulemBlock`, latent=256, mult_hidden=128, nx=1024).
Params: **1,132,800**.
Sharding: `shard_map` over `Mesh(("batch",))`, 128 samples per GPU.
Loss: `sobolev_loss(k=1)`; all aux losses (pushforward, hamiltonian, PSD, jac-reg,
input-noise) OFF at defaults for this command.
Full step = fwd + value_and_grad + `pmean` + AdamW + `all_gather[0]` param sync.

## Wall-clock breakdown (median of 40 iters, `time.perf_counter` + `block_until_ready`)

| component                                          | median (ms) | p90     | min     |
|----------------------------------------------------|------------:|--------:|--------:|
| **full step (fwd + bwd + AdamW + all_gather)**     |   **80.70** |  80.82  |  80.44  |
| fwd + bwd (loss + grad, pmean, no opt)             |       78.41 |  78.51  |  78.32  |
| fwd only                                           |       28.18 |  28.20  |  28.13  |
| single `CraigSulemBlock` fwd (128×1024, dev0)      |        3.53 |   3.54  |   3.51  |
| rfft roundtrip `(128, 1024)`   dev0                |        0.08 |   0.08  |   0.07  |
| rfft roundtrip `(1024, 1024)`  dev0                |        0.08 |   0.09  |   0.08  |
| rfft branched `(128, 1024, 256)` dev0              |        2.41 |   2.42  |   2.39  |

### Derived

| quantity                              |  value      | share of full step |
|---------------------------------------|------------:|-------------------:|
| forward                               |    28.18 ms |          **34.9 %** |
| backward (fwdbwd − fwd)               |    50.24 ms |          **62.3 %** |
| opt + all_gather (full − fwdbwd)      |     2.56 ms |           **3.2 %** |
| implied it/s                          |  **12.40**  | (measured 12.35)   |

Reported training throughput was 12.27 it/s → this profile reproduces it to <1 %.

## nsys top kernels (aggregate over compile + warm + 40 iters)

Total GPU kernel time captured ≈ 8.77 s across 114 kernel types. Top 15 by share
of total GPU time:

|  share | ms      | kernel                                                 |
|-------:|--------:|--------------------------------------------------------|
| 14.2 % |  1245.9 | cuBLAS `scal_kernel_val<float,float>` (per-mode multiply of learned Fourier multiplier · rFFT tensor) |
| 14.2 % |  1241.6 | cuFFT `vector_fft_symm_c2r` (irfft, N=1024)            |
| 13.7 % |  1198.4 | cuFFT `vector_fft_symm_r2c` (rfft,  N=1024)            |
|  8.6 % |   756.8 | `input_transpose_fusion_3`  (FFT-adjacent layout fuse) |
|  8.6 % |   756.5 | `input_transpose_fusion_27` (FFT-adjacent layout fuse) |
|  7.9 % |   691.3 | `input_multiply_reduce_fusion_3` (post-FFT `Σ_i m_out·prods_hat` and phi·xi_branched contractions) |
|  5.9 % |   519.5 | `input_transpose_fusion_14`                            |
|  3.6 % |   315.8 | `input_transpose_fusion_17`                            |
|  3.5 % |   304.3 | `input_transpose_fusion_11`                            |
|  3.4 % |   295.6 | `input_transpose_fusion_9`                             |
|  2.9 % |   253.8 | `input_reduce_fusion_24`                               |
|  2.4 % |   213.3 | `gemm_fusion_dot_3` (Triton GEMM — one of the Dense trunks) |
|  1.7 % |   150.0 | `input_reduce_fusion`                                  |
|  0.8 % |    73.3 | NCCL `AllReduce_Sum_f32` (grad pmean)                  |
|  0.7 % |    62.2 | NCCL `AllGather`         (param sync)                  |

Rolled up by category:

| category                                        | share |
|-------------------------------------------------|------:|
| cuFFT r2c + c2r + FFT-adjacent transpose fusions | **~62 %** |
| element-wise / broadcast multiplies (`scal`, `input_multiply_reduce_fusion_*`) | ~22 % |
| dense/GEMM (Triton + a few cuBLAS-cutlass calls) | **~6–8 %** |
| NCCL comm (AllReduce grads + AllGather params)   | **~1.5 %** |
| everything else (elementwise reductions, memcpy, memsets) | ~2 % |

## Interpretation

### Where the time goes

- **Forward is 35 %; backward is 62 %; optimizer + comm is 3 %.** The backward-to-forward
  ratio (~1.8×) is normal for this architecture — each block's `rfft → mult → irfft → pointwise → rfft → mult → irfft`
  chain requires several saved activations, and gradients push equal-shape FFTs through
  the reverse graph. Nothing surprising or fixable here.
- **Optimizer + `all_gather` param sync is 2.6 ms out of 80.7 ms — 3.2 %.** With only
  1.1 M params, NCCL is not the bottleneck. (Grad pmean + param all-gather together
  are ~1.5 % of GPU kernel time in nsys.) Multi-GPU is essentially free at this size.
- **FFT + FFT-support transpose/reduce fusions = ~62 %** of GPU time. Raw
  rfft/irfft alone are ~28 %; the `input_transpose_fusion_*` family covers layout
  packing that only exists to feed the FFT (branch-axis moves, `(B,N,L)` ↔ `(B,L,N)` etc.).
- **Dense/GEMM is only ~7 %.** The DepthAwareMultiplier MLPs (`hidden`, `out` Dense
  layers, shape `(B, K=513, C ≤ 256)`) and the two eta-feature-trunk Dense layers
  are tiny compared to the spectral work. This is the opposite of a normal
  Transformer/FNO regime.

### FFT vs mem vs launch

- Raw batched rfft `(128, 1024)` is **0.08 ms**. Full step is **80.7 ms**, i.e.
  **≈ 1000× a raw rfft**. That factor is explained: each block does 3 FFTs on
  batched tensors, backward doubles them, ×8 blocks × 2 GPUs × the "branched"
  `(B, N, L=256)` shape (which alone is 2.4 ms on device 0, ≈ 30 raw-1D-rfft costs).
  In other words, the FFT work is **large-tensor FFT**, not raw 1D FFT.
- The `scal` and `input_multiply_reduce_fusion` kernels moving 14 % + 8 % of time
  are **HBM-bandwidth-bound** memory work: multiplying a `(B=128, K=513, C=256)`
  spectral tensor by a `(B, K, C)` multiplier and then reducing over C.
  That tensor is **128·513·256·4 B = 67 MiB per pass**, and each block does this
  twice forward, ~2× more in bwd — hundreds of MiB of streaming reads/writes
  per step. RTX 6000 Ada HBM tops out at ~960 GB/s, so a bandwidth-bound step
  budget of ~50 ms is exactly where we land.
- **Verdict: memory-bandwidth-bound spectral work, not launch-bound and not
  dense-compute-bound.** GEMM/Tensor-Core utilization is inherently low for
  this architecture because ~90 % of arithmetic is 1D FFTs and elementwise
  multiplies in k-space, neither of which uses tensor cores.

### Levers ranked

1. **Bigger local batch (biggest lever if VRAM allows).** With 1.1 M params, most
   activation memory is `(B_local, N, L=256)` spectral tensors ≈ 67 MiB each.
   Doubling `--batch_size` to 512 (→ B_local=256, ~134 MiB per stage) should
   fit in 48 GiB and improves FFT arithmetic intensity + amortizes the per-batch
   NCCL cost. **Expected: 20–40 % more it/s.** Cheapest to try.
2. **Reduce the branched spectral tensor.** The main spectral tensor shape is
   `(B, N, latent=256)` per block. Halving `--latent` from 256 → 128 cuts the
   dominant kernels almost linearly. If the accuracy delta is acceptable
   (dataset-load bearing feature note calls "50 effective branches/block" — so
   256 is already probably over-provisioned), this is a big win: **estimated
   ≈ 30–40 %** step-time reduction. Verify accuracy on a short run first.
3. **Fuse blocks / reduce n_blocks.** Same knob — `--n_blocks 4` halves nearly
   everything. Also load-bearing per the model design notes; test on a short run.
4. **Mixed-precision inside multiplier MLPs / trunk Dense.** Currently everything
   is fp32. Casting the two small Dense layers of `DepthAwareMultiplier` to bf16
   would save GEMM time but GEMM is only ~7 % of total — expected win **< 3 %.**
   Not worth pursuing.
5. **Compile flags.** `--xla_gpu_triton_gemm_any=true` is already on. Additionally
   `--xla_gpu_enable_command_buffer=""` (disable) sometimes hurts on Ada
   because CUDA graph capture overheads dominate; worth an A/B but expected
   effect is small (< 5 %).
6. **Reduce H2D copies / prefetch depth.** Not a bottleneck: nsys shows
   H2D bytes = 1 GB across the whole run, vs 500 GB of D2D. Already prefetched
   in `device_prefetch(depth=2)`. Skip.
7. **Custom fused FFT+multiplier kernel (`cutensor` / Triton).** Would attack
   the 22 % elementwise + 62 % FFT-adjacent cost by keeping the spectral
   tensor in registers/shared memory between rfft, multiplier, irfft.
   High effort, could be **another 15–25 %**, but only justified if the
   architecture is frozen.

## Recommendation

Training is **memory-bandwidth-bound on the spectral tensor
`(B_local, K, latent)`**, not launch- or comm-bound. NCCL is 1.5 % of GPU time
and dense matmul is <8 %, so multi-GPU is already efficient and there is no
Tensor-Core headroom to unlock without redesigning the kernel.

Realistic single-lever wins, in order of ROI:

- Try `--batch_size 512` first. Free experiment (no code change); if VRAM
  holds, expect ~1.2–1.4× throughput and no accuracy impact.
- Then evaluate `--latent 128` (or `--n_blocks 4`) for a further ~1.3–1.5×,
  contingent on val loss not regressing on a short run.
- Everything else (mixed precision on Dense, XLA flag sweeps, further NCCL
  tuning) is **≤ 3 %** and not worth the churn until the two above are
  saturated. A custom fused-FFT kernel is the only path to substantial
  further speedup (~1.5×) but is a real project, not a knob.

## Files

- Profile script: `/home/johnma/dno-fno/profile_cs_dno.py`
- Nsys report:   `/tmp/cs_dno_nsys.nsys-rep` (kernel table above extracted with
  `nsys stats --report cuda_gpu_kern_sum`)
