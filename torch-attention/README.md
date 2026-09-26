# Minimal PyTorch attention DNO

Three Python files, no JAX imports. `model.py` is the original model, `spectral.py`
is the optional smaller spectral model, and `train.py` handles training. Run from this directory:

```sh
uv run train.py --data /path/to/arrays --device mps
```

Defaults: batch256, constant LR1e-4, bias-corrected gradient EMA0.9 **before** AdamW,
weight decay1e-4, one epoch. Training samples reshuffle every epoch; validation
is fixed. `--device cpu` and `--device cuda` also work. Dependencies are declared
in the script and installed by uv into an isolated environment.

`--optimizer muon --lr 1e-5 --ema 0.8` uses Muon on the eight attention/MLP
weight matrices inside the two transformer blocks. Everything else uses AdamW,
including pointwise encoder/decoder, biases, normalization, position embeddings,
and Fourier filters. Muon uses `adjust_lr_fn="match_rms_adamw"` with the same
base learning rate (not identical updates), default momentum0.95 and five
Newton–Schulz iterations. Gradient EMA is applied before **both** optimizers.

`--bf16` autocasts the encoder, attention projections and transformer MLPs to BF16.
On PyTorch2.14 MPS, training SDPA accepts BF16 inputs but promotes its matrix
products and softmax to FP32 internally; this is not fully BF16 attention.
Master
weights, residual streams, FFTs, the entire final decoder, xi filters, baseline
addition, loss, gradient EMA and optimizer state retain FP32 (CPU/CUDA baseline
still uses FP64 internally). Muon's orthogonalization already uses BF16.
`--depth 2` stacks two blocks in each attention stage, four blocks total;
default depth1 is compatible with existing checkpoints. Checkpoint arguments
record depth and precision. No dropout; explicit SDPA is used in train and eval.

Input directory: `eta.npy`, `xi.npy`, `gxi.npy` with shape `[samples,1024]`,
positive physical `depth.npy`, periodic grid `x.npy`, and `dataset_split.npy`
containing `train` / `validation` labels. Inputs/targets are physical values,
without normalization. The final partial batch is retained.

Architecture: eta + log(depth) -> pointwise MLP128 -> **full1024 spatial attention**
-> FFT -> attention over all513 frequency bins (256 real/imag features) -> IFFT
-> pointwise decoder to32 spatial gates. Both attention blocks use four heads,
pre-layer normalization, residual connections and a two-layer MLP. No windows.
The separate xi path uses32 paired real Fourier filters around the spatial gates;
this preserves linearity in xi and self-adjointness. The analytic G0+G1 baseline
is added. The correction starts at zero; quadratic flatness is not enforced.

This scaffold trains only the relative-L2 data loss. It omits the earlier physics
regularizers and production normalization; losses are not a matched experiment.
MPS uses float32 for the analytic baseline because Metal has no float64 support;
CPU/CUDA evaluate that baseline in float64. FFTs remain native `torch.fft` calls.
Checkpoints include model, optimizer and gradient EMA state; automatic resume is
intentionally omitted.

Use the separate spectral option with the current pilot settings:

```sh
uv run train.py --data /path/to/arrays --device cuda --architecture spectral \
  --depth 2 --bf16 --batch-size 256 --optimizer muon --lr 1e-5 --ema 0.8
```

The spectral option fits RMS scales for seven physical surface features on the
training split only: eta, eta², eta³, first and second derivatives, half derivative,
and Hilbert transform. Scales are saved as model buffers; this initial pass is
additional setup work. Physical xi, targets and the analytic baseline stay unchanged.

A bias-free pointwise 7→64→64 encoder feeds FFT → two width-64, four-head
transformer blocks over all 513 bins → spectral projection → IFFT. Frequency and
depth features identify the tokens. There is no spatial attention or learned
absolute spatial position embedding. The encoder also feeds a direct local path,
`z*tanh(z)`, modulated by `1+tanh(transformer_context)`; a zero-initialized FP32
64→32 decoder produces the correction gates. This enforces O(eta²) behavior even
when the transformer has biases. Two blocks give 91,584 parameters.

The constrained head retains 32 paired real Fourier filters, now generated from
`k, h, tanh(k*h), k*tanh(k*h)` by a small depth-conditioned MLP. They preserve
linearity in xi and self-adjointness; masking DC also annihilates constant xi.
This is not a low-rank matrix projection or a literal single FFT/IFFT pair: physical
feature extraction and the constrained head require additional small FFTs.
Frequency attention does not enforce exact translation equivariance. BF16 covers
encoder/transformer matmuls; feature FFTs, decoder, gate products and filters use FP32.

Add `--fast-step` on CUDA to fetch entire batches together, compile/fuse the model and updates,
and replay forward, backward, gradient EMA and optimizer updates with a CUDA graph.
It keeps the same samples, shuffle order, batch size, model and optimizer settings.
Compilation/capture adds one-time setup work; capture warmup restores the original
parameters, EMA and optimizer state before the first actual update. Partial final
batches use eager model execution and fused updates with the same optimizer state and step counter.
Validation remains eager, with bulk data loading. Checkpoints keep the same model
keys and record the fast-step setting; floating-point fusion can cause small
numerical differences. The original execution path remains the default.

Fast training queues GPU work and reads accumulated loss every 32 updates and at
epoch end, rather than synchronizing after each batch. Nonfinite training is
therefore detected within 32 updates. Add `--prefetch` to prepare/pin the next
CPU batch on a background thread and transfer it on a separate CUDA stream.
This preserves shuffling and partial batches. Prefetch is optional: the cached
subset benchmark was faster with queued execution alone. Full-volume cold random
reads need separate measurement; cached-subset timings do not establish epoch I/O.
The pilot reports amortized batch time over synchronized groups, with compile
and validation time separate, so queue submission time is not mistaken for GPU time.

For a large dataset on a remote filesystem, `--cache-data /tmp/dno-arrays` stages
the six required arrays into local ephemeral storage before fitting scales or
training. Copies read files sequentially; batches still use the original global
shuffle and splits. The cache needs room for all six arrays. A completion manifest
checks source paths, sizes and modification times before reuse; incomplete files
are recopied. Staging time is reported separately. This is optional; the CPU-only production benchmark below measures its cost. Feature-scale fitting also
uses bulk batch reads, but remains an initial pass over the training split.

The spectral model uses contiguous spatial axes for learned FFTs and real/imaginary
views for filter products and reductions, allowing fusion of their intermediates.
These layout changes preserve the architecture, FP32 FFTs/head and checkpoint keys.

For the measured H100 matrix optimizations, use `--fast-step --autotune --optimizer muon-grouped` with the spectral pilot settings above. Autotuning
benchmarks GEMM kernels during compilation; grouped Muon batches equal-shaped
Newton–Schulz matrix products using the same update equations and checkpoint
state format. It uses PyTorch 2.14 optimizer internals and can differ in BF16
rounding. Both are optional; neither changes model capacity or precision policy.
A matched batch256 H100 trial improved amortized training from 7.39 to 7.01 ms
(5.4% higher throughput), with essentially identical validation after 80 updates.
Cold autotuning took about 67 seconds, separate from batch timing. Measurements
use the cached real subset and do not establish full-volume I/O performance.

Set `--width 256` to widen the encoder and transformer while keeping four heads
and 32 correction branches. With `--architecture spectral --depth 2`, this has
1,394,880 parameters. Width must be a positive multiple of four; omitting it
preserves spectral width64 or original attention width128. The width is recorded
in checkpoint arguments. Wider weights require a fresh model rather than loading
a width64 state dict. On one H100 at batch256 with the fast/tuned/grouped settings,
width256 measured17.48ms per training batch versus7.06ms at width64; compilation
was separate (106s for width256). This short benchmark checked finite updates,
not validation accuracy.

For parallel production reads, add `--loader-workers 4` alongside `--fast-step`.
This uses four bounded reader threads, preserves the exact sampler order/RNG state
and partial batches, and avoids nested CPU math thread pools. It can be combined
with `--prefetch` for pinned memory and asynchronous CUDA transfers; validation
continues to use its original loader. The default remains zero extra readers.

CPU-only tests on the full production training split measured warm loading at
1.09ms/batch with four threads versus1.60–1.66ms with one reader. Four separate
loader processes were slower at7.18ms. Fresh-worker access to256 globally shuffled
batches took49–53 seconds, mostly startup; these are not cache-flushed measurements.
On another65536 unseen examples, the integrated four-thread loader waited0.135s
in total while a CPU sleep simulated17.48ms of GPU work per batch. This is a loader
check, not a measured GPU training speedup. Exact sample order, tensors, generator
state and one-row tails were checked over multiple epochs.

Copying all181.9GB of required arrays to local SSD took54s after the initial read
probe. The warmed remote volume was faster than the local copy in that trial,
so local staging remains optional. Source files are never modified. Cache behavior
and startup costs can vary by worker; the short test does not establish full-epoch
I/O. Detailed results are in `experiments/torch_production_*cpu_20260926.json` and
`experiments/check_production_loader_cpu_20260926.json`.
