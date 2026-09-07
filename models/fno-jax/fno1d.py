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
        # Weight-norm per mode: each mode's (Ci, Co) matrix gets unit Frobenius norm,
        # then a learned scalar gain controls magnitude. Prevents any mode from dominating.
        norm = jnp.sqrt(jnp.sum(jnp.abs(weights) ** 2, axis=(0, 1), keepdims=True) + 1e-8)
        gain = self.param("mode_gain", nn.initializers.ones, (1, 1, self.modes))
        weights = (weights / norm) * gain  # (C_in, C_out, M)

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
        h = nn.tanh(h)  # (B, N, H)
        h = nn.Dense(self.channels, name="up")(h)  # (B, N, C)
        return h + skip  # (B, N, C)


class FiLMConditioner(nn.Module):
    """Map a scalar condition (log-depth) to per-block (gamma, beta) FiLM params."""
    width: int
    n_blocks: int

    @nn.compact
    def __call__(self, c: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # c: (B, 1)
        h = nn.Dense(self.width, name="fc1")(c)        # (B, W)
        h = nn.tanh(h)
        h = nn.Dense(self.width, name="fc2")(h)        # (B, W)
        h = nn.tanh(h)
        gamma = nn.Dense(self.n_blocks * self.width, name="gamma")(h)  # (B, n_blocks*W)
        beta = nn.Dense(self.n_blocks * self.width, name="beta")(h)    # (B, n_blocks*W)
        gamma = gamma.reshape((-1, self.n_blocks, self.width))  # (B, L, W)
        beta = beta.reshape((-1, self.n_blocks, self.width))    # (B, L, W)
        return gamma, beta


class FNO1d(nn.Module):
    modes: int
    width: int
    n_blocks: int = 4
    domain_length: float = 164.0
    xi_scale: float = 1.0       # feature_absmax for xi channel
    target_scale: float = 1.0   # target_absmax for gxi
    h_clip_max: float = 5.0     # saturate log(h) at log(h_clip_max); kh_sat = 5 at k_min=1, tanh(5)≈1

    @nn.compact
    def __call__(self, x: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 2)  — channel 0 = eta, channel 1 = xi (both normalized)
        # depth: (B, 1) — log(h)
        batch_size, grid_size, _ = x.shape

        depth = jnp.minimum(depth, jnp.log(self.h_clip_max))

        # --- Linear DNO baseline: G0*xi = IFFT(k*tanh(h*k) * FFT(xi)) ---
        # Undo xi normalization, apply G0 in physical space, then normalize to target space
        xi_raw = x[:, :, 1] * self.xi_scale  # (B, N) — physical xi
        h = jnp.exp(depth)  # (B, 1) — physical depth (clipped)
        dk = 2.0 * jnp.pi / self.domain_length
        k = dk * jnp.arange(grid_size // 2 + 1)  # (F,)  rfft wavenumbers
        g0 = k[None, :] * jnp.tanh(h * k[None, :])  # (B, F)
        xi_hat = jnp.fft.rfft(xi_raw, axis=1)  # (B, F)
        baseline = jnp.fft.irfft(g0 * xi_hat, n=grid_size, axis=1) / self.target_scale  # (B, N)

        # --- Derivative features ---
        d1 = (jnp.roll(x, -1, axis=1) - jnp.roll(x, 1, axis=1)) / 2.0
        d2 = jnp.roll(x, -1, axis=1) - 2.0 * x + jnp.roll(x, 1, axis=1)
        x_in = jnp.concatenate([x, d1, d2], axis=-1)  # (B, N, 6)
        x = nn.Dense(self.width, name="input_proj")(x_in)  # (B, N, W)

        film = FiLMConditioner(self.width, self.n_blocks, name="film")
        gamma, beta = film(depth)  # each (B, n_blocks, W)

        for block_idx in range(self.n_blocks):
            spectral = SpectralConv1d(
                in_channels=self.width,
                out_channels=self.width,
                modes=self.modes,
                name=f"spectral_{block_idx}",
            )(x)  # (B, N, W)
            residual = nn.Dense(self.width, name=f"residual_{block_idx}")(x)  # (B, N, W)
            x = spectral + residual  # (B, N, W)
            g = gamma[:, block_idx, :][:, None, :]  # (B, 1, W)
            b = beta[:, block_idx, :][:, None, :]    # (B, 1, W)
            x = (1.0 + g) * x + b
            if block_idx < self.n_blocks - 1:
                x = nn.tanh(x)  # (B, N, W)
            x = ChannelMLP(self.width, name=f"channel_mlp_{block_idx}")(x)  # (B, N, W)
            if block_idx < self.n_blocks - 1:
                x = nn.tanh(x)  # (B, N, W)

        x = nn.tanh(nn.Dense(128, name="hidden_proj")(x))  # (B, N, 128)
        residual_pred = nn.Dense(1, name="output_proj")(x)  # (B, N, 1)
        return residual_pred + baseline[:, :, None]  # (B, N, 1)
