"""Experimental 256-point inverse FFT / multiply / FFT, including reverse mode.

Pallas keeps each butterfly's intermediates inside one GPU kernel. The default
model still uses cuFFT; this kernel is opt-in and only supports float32.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _fft(
    real: jax.Array, imag: jax.Array, *, inverse: bool,
) -> tuple[jax.Array, jax.Array]:
    """Unnormalized radix-two DIF FFT; shape (256, channels)."""
    channels = real.shape[1]
    for stage in range(8, 0, -1):
        size = 1 << stage
        shape = (256 // size, 2, size // 2, channels)
        ar, br = jnp.split(real.reshape(shape).transpose(0, 2, 3, 1), 2, axis=-1)
        ai, bi = jnp.split(imag.reshape(shape).transpose(0, 2, 3, 1), 2, axis=-1)
        angle = jnp.arange(size // 2, dtype=jnp.float32) * (
            (2 if inverse else -2) * jnp.pi / size
        )
        cosine, sine = jnp.cos(angle)[None, :, None, None], jnp.sin(angle)[None, :, None, None]
        dr, di = ar - br, ai - bi
        real = jnp.concatenate((ar + br, dr * cosine - di * sine), axis=-1).transpose(0, 3, 1, 2).reshape(256, channels)
        imag = jnp.concatenate((ai + bi, dr * sine + di * cosine), axis=-1).transpose(0, 3, 1, 2).reshape(256, channels)
    permutation = (*range(7, -1, -1), 8)
    shape = (*((2,) * 8), channels)
    return (
        real.reshape(shape).transpose(permutation).reshape(256, channels),
        imag.reshape(shape).transpose(permutation).reshape(256, channels),
    )


def _forward_kernel(zr: jax.Ref, zi: jax.Ref, a: jax.Ref, yr: jax.Ref, yi: jax.Ref) -> None:
    batch, start = pl.program_id(0), pl.program_id(1) * 8
    n, c = jnp.arange(256)[:, None], start + jnp.arange(8)[None, :]
    k = jnp.minimum(n, 256 - n)
    real = plt.load(zr.at[batch, k, c])
    imag = plt.load(zi.at[batch, k, c]) * jnp.where(n <= 128, 1.0, -1.0)
    imag = jnp.where((k == 0) | (k == 128), 0.0, imag)
    spatial, _ = _fft(real, imag, inverse=True)
    weighted = spatial * plt.load(a.at[batch, n, c])
    out_real, out_imag = _fft(weighted, jnp.zeros_like(weighted), inverse=False)
    plt.store(yr.at[batch, n, c], out_real / 256, mask=n <= 128)
    plt.store(yi.at[batch, n, c], jnp.where((n == 0) | (n == 128), 0.0, out_imag / 256), mask=n <= 128)


def _backward_kernel(
    zr: jax.Ref, zi: jax.Ref, a: jax.Ref, gr: jax.Ref, gi: jax.Ref,
    dzr: jax.Ref, dzi: jax.Ref, da: jax.Ref,
) -> None:
    batch, start = pl.program_id(0), pl.program_id(1) * 8
    n, c = jnp.arange(256)[:, None], start + jnp.arange(8)[None, :]
    k = jnp.minimum(n, 256 - n)
    edge = (k == 0) | (k == 128)
    sign = jnp.where(n <= 128, 1.0, -1.0)
    zr_values = plt.load(zr.at[batch, k, c])
    zi_values = jnp.where(edge, 0.0, plt.load(zi.at[batch, k, c]) * sign)
    spatial, _ = _fft(zr_values, zi_values, inverse=True)
    weight = jnp.where(edge, 1.0, 0.5)
    gr_values = plt.load(gr.at[batch, k, c]) * weight
    gi_values = jnp.where(edge, 0.0, -plt.load(gi.at[batch, k, c]) * weight * sign)
    adjoint, _ = _fft(gr_values, gi_values, inverse=True)
    adjoint /= 256
    plt.store(da.at[batch, n, c], adjoint * spatial)
    weighted = adjoint * plt.load(a.at[batch, n, c])
    out_real, out_imag = _fft(weighted, jnp.zeros_like(weighted), inverse=False)
    multiplicity = jnp.where(edge, 1.0, 2.0)
    plt.store(dzr.at[batch, n, c], out_real * multiplicity, mask=n <= 128)
    plt.store(dzi.at[batch, n, c], jnp.where(edge, 0.0, -out_imag * multiplicity), mask=n <= 128)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def fused_roundtrip(z: jax.Array, a: jax.Array, interpret: bool = False) -> jax.Array:
    """rfft(a * irfft(z)), with forward-normalized transforms along axis 1."""
    if a.shape[1] != 256 or a.shape[2] % 8 or a.dtype != jnp.float32 or z.dtype != jnp.complex64:
        raise ValueError("Fused FFT requires N=256, channels divisible by 8, float32/complex64")
    out = jax.ShapeDtypeStruct(z.shape, jnp.float32)
    real, imag = pl.pallas_call(
        _forward_kernel, out_shape=(out, out), grid=(a.shape[0], a.shape[2] // 8),
        interpret=interpret, compiler_params=plt.CompilerParams(num_warps=4),
        name="fused_irfft_multiply_rfft_256",
    )(z.real, z.imag, a)
    return jax.lax.complex(real, imag)


def _forward(
    z: jax.Array, a: jax.Array, interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return fused_roundtrip(z, a, interpret), (z, a)


def _backward(
    interpret: bool, saved: tuple[jax.Array, jax.Array], cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    z, a = saved
    spectrum_type = jax.ShapeDtypeStruct(z.shape, jnp.float32)
    spatial_type = jax.ShapeDtypeStruct(a.shape, jnp.float32)
    real, imag, spatial = pl.pallas_call(
        _backward_kernel, out_shape=(spectrum_type, spectrum_type, spatial_type),
        grid=(a.shape[0], a.shape[2] // 8), interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4), name="fused_fft_roundtrip_backward_256",
    )(z.real, z.imag, a, cotangent.real, cotangent.imag)
    return jax.lax.complex(real, imag), spatial


fused_roundtrip.defvjp(_forward, _backward)


def check_fused_roundtrip(*, interpret: bool) -> dict[str, float]:
    """Compare outputs and both complex/real cotangents with JAX's FFT rules."""
    keys = jax.random.split(jax.random.key(7), 5)
    z = jax.lax.complex(
        jax.random.normal(keys[0], (2, 129, 16), dtype=jnp.float32),
        jax.random.normal(keys[1], (2, 129, 16), dtype=jnp.float32),
    )
    a = jax.random.normal(keys[2], (2, 256, 16), dtype=jnp.float32)
    cotangent = jax.lax.complex(
        jax.random.normal(keys[3], z.shape, dtype=jnp.float32),
        jax.random.normal(keys[4], z.shape, dtype=jnp.float32),
    )

    def reference(spectrum: jax.Array, weights: jax.Array) -> jax.Array:
        spatial = jnp.fft.irfft(spectrum, n=256, axis=1, norm="forward")
        return jnp.fft.rfft(weights * spatial, axis=1, norm="forward")

    expected, expected_vjp = jax.vjp(reference, z, a)
    actual, actual_vjp = jax.vjp(partial(fused_roundtrip, interpret=interpret), z, a)
    errors = {}
    for name, left, right in zip(
        ("forward", "spectrum_gradient", "spatial_gradient"),
        (expected, *expected_vjp(cotangent)), (actual, *actual_vjp(cotangent)),
    ):
        relative = float(jnp.linalg.norm(left - right) / jnp.linalg.norm(left))
        assert bool(jnp.all(jnp.isfinite(right))) and relative < 2e-5, (name, relative)
        errors[name] = relative
    return errors


if __name__ == "__main__":
    print(check_fused_roundtrip(interpret=jax.default_backend() == "cpu"))
