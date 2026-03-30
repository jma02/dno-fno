from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer that keeps a fixed number of low modes."""

    in_channels: int
    out_channels: int
    modes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, grid_size, _ = x.shape
        kept_modes = min(self.modes, grid_size // 2 + 1)
        scale = 1.0 / (self.in_channels * self.out_channels)

        weights_real = self.param(
            "weights_real",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes),
        )
        weights_imag = self.param(
            "weights_imag",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes),
        )
        weights = weights_real + 1j * weights_imag

        x_fft = jnp.fft.rfft(x, axis=1)
        transformed = jnp.einsum(
            "bmi,iom->bmo",
            x_fft[:, :kept_modes, :],
            weights[:, :, :kept_modes],
        )

        out_fft = jnp.zeros(
            (batch_size, grid_size // 2 + 1, self.out_channels),
            dtype=jnp.complex64,
        )
        out_fft = out_fft.at[:, :kept_modes, :].set(transformed)
        return jnp.fft.irfft(out_fft, n=grid_size, axis=1)


class FNO1d(nn.Module):
    """Small 1D Fourier Neural Operator for DNO regression."""

    modes: int
    width: int
    n_blocks: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.width, name="input_proj")(x)

        for block_idx in range(self.n_blocks):
            spectral = SpectralConv1d(
                in_channels=self.width,
                out_channels=self.width,
                modes=self.modes,
                name=f"spectral_{block_idx}",
            )(x)
            residual = nn.Dense(self.width, name=f"residual_{block_idx}")(x)
            x = spectral + residual
            if block_idx < self.n_blocks - 1:
                x = nn.relu(x)

        x = nn.relu(nn.Dense(128, name="hidden_proj")(x))
        return nn.Dense(1, name="output_proj")(x)
