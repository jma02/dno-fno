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
        # x: (B, N, C_in)
        batch_size, grid_size, _ = x.shape
        kept_modes = min(self.modes, grid_size // 2 + 1)  # K = min(M, F)
        weights_real = self.param(
            "weights_real",
            nn.initializers.glorot_normal(),
            (self.in_channels, self.out_channels, self.modes),  # (C_in, C_out, M)
        )
        weights_imag = self.param(
            "weights_imag",
            nn.initializers.glorot_normal(),
            (self.in_channels, self.out_channels, self.modes),  # (C_in, C_out, M)
        )
        weights = weights_real + 1j * weights_imag  # (C_in, C_out, M)

        x_fft = jnp.fft.rfft(x, axis=1, norm="forward")  # (B, F, C_in)  where F = N//2+1
        transformed = jnp.einsum(
            "bmi,iom->bmo",
            x_fft[:, :kept_modes, :],   # (B, K, C_in)
            weights[:, :, :kept_modes],  # (C_in, C_out, K)
        )  # (B, K, C_out)

        out_fft = jnp.zeros(
            (batch_size, grid_size // 2 + 1, self.out_channels),  # (B, F, C_out)
            dtype=jnp.complex64,
        )
        out_fft = out_fft.at[:, :kept_modes, :].set(transformed)  # (B, F, C_out)
        return jnp.fft.irfft(out_fft, n=grid_size, axis=1, norm="forward")  # (B, N, C_out)


class SoftGating(nn.Module):
    """Soft-gating skip: x * sigmoid(Wx)."""
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, C)
        gate = nn.sigmoid(nn.Dense(self.channels, name="gate")(x))  # (B, N, C)
        return x * gate  # (B, N, C)


class ChannelMLP(nn.Module):
    """Pointwise MLP across channels with soft-gating skip."""
    channels: int
    expansion: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, C)
        hidden = int(self.channels * self.expansion)
        skip = SoftGating(self.channels, name="skip")(x)  # (B, N, C)
        h = nn.Dense(hidden, name="down")(x)  # (B, N, H)
        h = nn.gelu(h)  # (B, N, H)
        h = nn.Dense(self.channels, name="up")(h)  # (B, N, C)
        return h + skip  # (B, N, C)


class FNO1d(nn.Module):
    modes: int
    width: int
    n_blocks: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 2)
        x = nn.Dense(self.width, name="input_proj")(x)  # (B, N, W)

        for block_idx in range(self.n_blocks):
            spectral = SpectralConv1d(
                in_channels=self.width,
                out_channels=self.width,
                modes=self.modes,
                name=f"spectral_{block_idx}",
            )(x)  # (B, N, W)
            residual = nn.Dense(self.width, name=f"residual_{block_idx}")(x)  # (B, N, W)
            x = spectral + residual  # (B, N, W)
            if block_idx < self.n_blocks - 1:
                x = nn.gelu(x)  # (B, N, W)
            x = ChannelMLP(self.width, name=f"channel_mlp_{block_idx}")(x)  # (B, N, W)
            if block_idx < self.n_blocks - 1:
                x = nn.gelu(x)  # (B, N, W)

        x = nn.gelu(nn.Dense(128, name="hidden_proj")(x))  # (B, N, 128)
        return nn.Dense(1, name="output_proj")(x)  # (B, N, 1)
