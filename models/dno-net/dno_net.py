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


class EtaBackbone(nn.Module):
    """FNO-style backbone that extracts eta-dependent spatial features."""

    modes: int
    width: int
    n_blocks: int

    @nn.compact
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        if eta.ndim == 2:
            eta = eta[..., None]
        if eta.ndim != 3 or eta.shape[-1] != 1:
            raise ValueError("eta must have shape (batch, grid) or (batch, grid, 1)")

        x = nn.Dense(self.width, name="input_proj")(eta)
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

        return x


class FactorFieldHead(nn.Module):
    """Maps backbone features to the low-rank factor field L_theta(eta)."""

    width: int
    rank: int

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        hidden = nn.relu(nn.Dense(self.width, name="hidden_proj")(features))
        return nn.Dense(self.rank, name="output_proj")(hidden)


class SpectralCorrectionHead(nn.Module):
    """Maps backbone features to a positive Fourier multiplier."""

    width: int
    spectral_floor: float

    @nn.compact
    def __call__(self, features: jnp.ndarray, num_modes: int) -> jnp.ndarray:
        pooled_mean = jnp.mean(features, axis=1)
        pooled_max = jnp.max(features, axis=1)
        pooled = jnp.concatenate((pooled_mean, pooled_max), axis=-1)

        hidden = nn.relu(nn.Dense(self.width, name="hidden_proj")(pooled))
        raw_multiplier = nn.Dense(num_modes, name="output_proj")(hidden)
        return nn.softplus(raw_multiplier) + self.spectral_floor


def apply_positive_spectral_multiplier(
    xi: jnp.ndarray,
    multiplier: jnp.ndarray,
) -> jnp.ndarray:
    """Apply F^{-1} diag(multiplier) F to a real signal xi."""
    xi_fft = jnp.fft.rfft(xi, axis=1)
    corrected_fft = xi_fft * multiplier
    return jnp.fft.irfft(corrected_fft, n=xi.shape[1], axis=1)


class DNONet(nn.Module):
    """Structure-preserving DNO model.

    The learned operator has the form
        T(eta, xi) = (L_theta(eta) L_theta(eta)^T + D_theta(eta)) xi
    where D_theta(eta) is a positive spectral multiplier.

    The architecture enforces:
    - linearity in xi
    - self-adjointness
    - positive definiteness for eps > 0
    """

    rank: int
    modes: int
    width: int
    n_blocks: int = 4
    spectral_width: int | None = None
    spectral_floor: float = 1e-3
    x: jnp.ndarray | None = None

    def _split_inputs(
        self,
        eta_or_inputs: jnp.ndarray,
        xi: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if xi is None:
            if eta_or_inputs.ndim != 3 or eta_or_inputs.shape[-1] != 2:
                raise ValueError(
                    "Expected concatenated inputs with shape (batch, grid, 2) when xi is omitted"
                )
            eta = eta_or_inputs[..., :1]
            xi = eta_or_inputs[..., 1:]
        else:
            eta = eta_or_inputs
            if eta.ndim == 2:
                eta = eta[..., None]
            if xi.ndim == 2:
                xi = xi[..., None]

        if eta.ndim != 3 or eta.shape[-1] != 1:
            raise ValueError("eta must have shape (batch, grid, 1)")
        if xi.ndim != 3 or xi.shape[-1] != 1:
            raise ValueError("xi must have shape (batch, grid, 1)")
        if eta.shape[:2] != xi.shape[:2]:
            raise ValueError("eta and xi must share the same batch and grid dimensions")
        return eta, xi

    @nn.compact
    def __call__(
        self,
        eta_or_inputs: jnp.ndarray,
        xi: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        eta, xi = self._split_inputs(eta_or_inputs, xi)
        eta_values = eta[..., 0]
        xi_values = xi[..., 0]
        grid_size = eta_values.shape[1]

        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if self.rank > grid_size:
            raise ValueError("rank must not exceed the spatial grid size")
        if self.spectral_width is not None and self.spectral_width < 1:
            raise ValueError("spectral_width must be positive when provided")

        backbone_features = EtaBackbone(
            modes=self.modes,
            width=self.width,
            n_blocks=self.n_blocks,
            name="eta_backbone",
        )(eta_values)

        low_rank_factor = FactorFieldHead(
            width=self.width,
            rank=self.rank,
            name="factor_field_head",
        )(backbone_features)
        # Keep the operator scale stable as rank changes across CARBS trials.
        low_rank_factor = low_rank_factor / jnp.sqrt(jnp.asarray(self.rank, dtype=eta_values.dtype))
        projected_xi = jnp.einsum("bnk,bn->bk", low_rank_factor, xi_values)
        low_rank_output = jnp.einsum("bnk,bk->bn", low_rank_factor, projected_xi)

        num_fourier_modes = grid_size // 2 + 1
        spectral_width = self.spectral_width or self.width
        spectral_multiplier = SpectralCorrectionHead(
            width=spectral_width,
            spectral_floor=self.spectral_floor,
            name="spectral_correction_head",
        )(backbone_features, num_modes=num_fourier_modes)
        spectral_output = apply_positive_spectral_multiplier(xi_values, spectral_multiplier)

        return (low_rank_output + spectral_output)[..., None]
