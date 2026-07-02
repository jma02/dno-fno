"""Spectrally factorized neural Dirichlet-to-Neumann operator.

The model keeps the physically important linearity in ``xi`` by constructing an
eta-conditioned linear operator and applying it to ``xi``.  The learned part is a
positive spectral sandwich,

    C(eta)^T F(eta) C(eta),

added to the flat-surface linear DNO baseline G0.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class SpectralConv1d(nn.Module):
    in_channels: int
    out_channels: int
    modes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, grid_size, _ = x.shape
        kept_modes = min(self.modes, grid_size // 2 + 1)
        weights_real = self.param(
            "weights_real",
            nn.initializers.glorot_normal(),
            (self.in_channels, self.out_channels, self.modes),
        )
        weights_imag = self.param(
            "weights_imag",
            nn.initializers.glorot_normal(),
            (self.in_channels, self.out_channels, self.modes),
        )
        weights = weights_real + 1j * weights_imag

        x_fft = jnp.fft.rfft(x, axis=1, norm="forward")
        transformed = jnp.einsum(
            "bki,iok->bko",
            x_fft[:, :kept_modes, :],
            weights[:, :, :kept_modes],
        )
        out_fft = jnp.zeros(
            (batch_size, grid_size // 2 + 1, self.out_channels),
            dtype=jnp.complex64,
        )
        out_fft = out_fft.at[:, :kept_modes, :].set(transformed)
        return jnp.fft.irfft(out_fft, n=grid_size, axis=1, norm="forward")


class SpectralDNO(nn.Module):
    modes: int
    width: int
    n_blocks: int = 4
    latent: int = 64
    domain_length: float = 6.283185307179586
    xi_scale: float = 1.0
    target_scale: float = 1.0
    h_clip_max: float = 5.0

    def _linear_baseline(self, xi_norm: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        grid_size = xi_norm.shape[1]
        xi_raw = xi_norm * self.xi_scale
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(grid_size // 2 + 1)
        g0 = k[None, :] * jnp.tanh(h * k[None, :])
        xi_fft = jnp.fft.rfft(xi_raw, axis=1)
        return jnp.fft.irfft(g0 * xi_fft, n=grid_size, axis=1) / self.target_scale

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        # inputs: (B, N, 2), normalized eta/xi. depth: (B, 1), log physical depth.
        eta = inputs[..., :1]
        xi = inputs[..., 1]
        batch_size, grid_size = xi.shape
        n_freq = grid_size // 2 + 1

        depth = jnp.minimum(depth, jnp.log(self.h_clip_max))
        depth_channel = jnp.broadcast_to(depth[:, None, :], (batch_size, grid_size, 1))
        eta_features = jnp.concatenate(
            [
                eta,
                (jnp.roll(eta, -1, axis=1) - jnp.roll(eta, 1, axis=1)) / 2.0,
                jnp.roll(eta, -1, axis=1) - 2.0 * eta + jnp.roll(eta, 1, axis=1),
                depth_channel,
            ],
            axis=-1,
        )

        h = nn.Dense(self.width, name="input_proj")(eta_features)
        h = nn.gelu(h)
        for block_idx in range(self.n_blocks):
            spectral = SpectralConv1d(
                in_channels=self.width,
                out_channels=self.width,
                modes=self.modes,
                name=f"spectral_{block_idx}",
            )(h)
            residual = nn.Dense(self.width, name=f"residual_{block_idx}")(h)
            h = nn.gelu(spectral + residual)
            h = nn.gelu(nn.Dense(self.width, name=f"channel_mlp_{block_idx}")(h))

        c = nn.Dense(self.latent, name="spatial_modulation")(h)
        pooled = jnp.mean(h, axis=1)
        filter_logits = nn.Dense(
            n_freq * self.latent,
            name="spectral_filter",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-6.0),
        )(pooled)
        spectral_filter = nn.softplus(filter_logits).reshape((batch_size, n_freq, self.latent)) + 1e-6

        xi_raw = xi * self.xi_scale
        modulated_xi = c * xi_raw[..., None]
        modulated_fft = jnp.fft.rfft(modulated_xi, axis=1, norm="forward")
        filtered = jnp.fft.irfft(
            spectral_filter * modulated_fft,
            n=grid_size,
            axis=1,
            norm="forward",
        )
        residual = jnp.sum(c * filtered, axis=-1) / self.target_scale
        baseline = self._linear_baseline(xi, depth)
        return (baseline + residual)[..., None]
