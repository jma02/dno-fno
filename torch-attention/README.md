# Minimal PyTorch attention DNO

Two Python files, no JAX imports. `model.py` is the model; `train.py` is the loader,
training loop and checkpoint writer. Run from this directory:

```sh
uv run train.py --data /path/to/arrays --device mps
```

Defaults: batch64, constant LR1e-4, bias-corrected gradient EMA0.9 **before** AdamW,
weight decay1e-4, one epoch. Training samples reshuffle every epoch; validation
is fixed. `--device cpu` and `--device cuda` also work. Dependencies are declared
in the script and installed by uv into an isolated environment.

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
