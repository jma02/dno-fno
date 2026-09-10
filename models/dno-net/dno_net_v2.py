"""Craig-Sulem neural DNO: analytic G0 + G1 plus a learned correction.

Here eta is surface elevation and xi is surface velocity potential.
Each correction branch applies M[spatial_weights(eta) * M[xi]], where M is
a depth-dependent real Fourier filter. The correction is self-adjoint,
linear in xi, and starts at order eta^2.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class DepthAwareMultiplier(nn.Module):
    """Learn a real Fourier multiplier for each wavenumber and branch.

    Depth is log(h), shaped (batch, 1); output is (batch, frequency, branch).
    """
    out_channels: int
    domain_length: float
    h_clip_max: float
    hidden: int = 32

    @nn.compact
    def __call__(self, depth: jnp.ndarray, n_freq: int) -> jnp.ndarray:
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq)
        k, h = jnp.broadcast_arrays(k[None, :], h)
        kh = h * k
        features = jnp.stack([k, h, jnp.tanh(kh), k * jnp.tanh(kh)], axis=-1)
        hidden = nn.tanh(nn.Dense(self.hidden, name="hidden")(features))
        return nn.Dense(
            self.out_channels, name="out", kernel_init=nn.initializers.lecun_normal(),
        )(hidden)


class CraigSulemBlock(nn.Module):
    """Sum parallel branches: filter xi, multiply by spatial weights, filter again."""
    n_branches: int
    domain_length: float
    h_clip_max: float
    mult_hidden: int = 32

    @nn.compact
    def __call__(
        self,
        eta_features: jnp.ndarray,
        xi_phys: jnp.ndarray,
        depth: jnp.ndarray,
    ) -> jnp.ndarray:
        grid_size = eta_features.shape[1]

        # Start with zero spatial weights, so the initial model is just G0 + G1.
        spatial_weights = nn.Dense(
            self.n_branches, name="phi_proj",
            kernel_init=nn.initializers.zeros, use_bias=False,
        )(eta_features)

        # Use the same real filter on both sides to keep the correction self-adjoint.
        multiplier = DepthAwareMultiplier(
            out_channels=self.n_branches,
            domain_length=self.domain_length,
            h_clip_max=self.h_clip_max,
            hidden=self.mult_hidden,
            name="m_shared",
        )(depth, grid_size // 2 + 1)

        xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")
        filtered_xi = jnp.fft.irfft(
            multiplier * xi_hat[..., None], n=grid_size, axis=1, norm="forward",
        )
        weighted_xi = spatial_weights * filtered_xi
        weighted_xi_hat = jnp.fft.rfft(weighted_xi, axis=1, norm="forward")
        correction_hat = jnp.sum(multiplier * weighted_xi_hat, axis=-1)
        return jnp.fft.irfft(correction_hat, n=grid_size, axis=-1, norm="forward")


class CraigSulemDNO(nn.Module):
    """Compute G0(xi) + G1(eta, xi) + learned_correction(eta, xi, depth)."""
    width: int                        # Each shared eta layer has width // 2 channels.
    n_blocks: int = 4                 # Groups of parallel branches, summed together.
    latent: int = 64                  # Branches per group.

    # Polynomial and derivative features of normalized eta.
    n_polys: int = 3                  # eta, eta^2, eta^3
    use_first_deriv: bool = True      # First spatial derivative.
    use_second_deriv: bool = True     # Second spatial derivative.
    use_half_deriv: bool = True       # Fourier multiplier sqrt(abs(k)).
    use_hilbert: bool = True          # Hilbert transform.
    mult_hidden: int = 32             # Hidden channels in each multiplier network.

    domain_length: float = 2 * jnp.pi
    # Multiply normalized inputs/outputs by these scales to get physical units.
    xi_scale: float = 1.0
    eta_scale: float = 1.0
    target_scale: float = 1.0
    h_clip_max: float = 5.0

    def _g0_apply(
        self, x_phys: jnp.ndarray, h: jnp.ndarray, grid_size: int,
    ) -> jnp.ndarray:
        """Apply the flat-surface DNO by multiplying Fourier modes by k * tanh(h*k)."""
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(grid_size // 2 + 1)
        g0 = k[None, :] * jnp.tanh(h * k[None, :])
        x_fft = jnp.fft.rfft(x_phys, axis=-1)
        return jnp.fft.irfft(g0 * x_fft, n=grid_size, axis=-1)

    def _linear_baseline(self, xi_norm: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        grid_size = xi_norm.shape[1]
        xi_phys = xi_norm * self.xi_scale
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        return (self._g0_apply(xi_phys, h, grid_size) / self.target_scale).astype(xi_norm.dtype)

    def _g1_baseline(
        self, eta_norm: jnp.ndarray, xi_norm: jnp.ndarray, depth: jnp.ndarray,
    ) -> jnp.ndarray:
        """G1 = -G0(eta * G0(xi)) - dx(eta * dx(xi)), divided by target_scale.

        Use physical inputs and float64 to reduce FFT roundoff; return the input dtype.
        """
        grid_size = xi_norm.shape[1]
        eta_phys = (eta_norm * self.eta_scale).astype(jnp.float64)
        xi_phys = (xi_norm * self.xi_scale).astype(jnp.float64)
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max))).astype(jnp.float64)

        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
            grid_size // 2 + 1, dtype=jnp.float64,
        )

        g0_xi = self._g0_apply(xi_phys, h, grid_size)
        g0_mult = k[None, :] * jnp.tanh(h * k[None, :])
        g0_term_hat = -g0_mult * jnp.fft.rfft(eta_phys * g0_xi, axis=-1)
        g0_term = jnp.fft.irfft(g0_term_hat, n=grid_size, axis=-1)

        dx_xi = jnp.fft.irfft(
            1j * k[None, :] * jnp.fft.rfft(xi_phys, axis=-1),
            n=grid_size, axis=-1,
        )
        dx_term_hat = -(1j * k)[None, :] * jnp.fft.rfft(eta_phys * dx_xi, axis=-1)
        dx_term = jnp.fft.irfft(dx_term_hat, n=grid_size, axis=-1)

        return ((g0_term + dx_term) / self.target_scale).astype(xi_norm.dtype)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        # Inputs: (batch, grid, 2) normalized [eta, xi]; depth: (batch, 1) log(h).
        batch_size, grid_size, _ = inputs.shape
        clipped_log_depth = jnp.minimum(depth, jnp.log(self.h_clip_max))

        eta_norm, xi_norm = inputs[..., 0], inputs[..., 1]
        xi_phys = xi_norm * self.xi_scale

        # Compute the analytic baseline in physical units, then normalize its output.
        baseline = (
            self._linear_baseline(xi_norm, depth)
            + self._g1_baseline(eta_norm, xi_norm, depth)
        )

        # Build polynomial and spatial derivative features from eta.
        features = [eta_norm ** p for p in range(1, self.n_polys + 1)]
        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
            grid_size // 2 + 1, dtype=eta_norm.dtype,
        )
        k_op = k_arr[None, :]
        eta_hat = jnp.fft.rfft(eta_norm, axis=-1)

        for enabled, multiplier in (
            (self.use_first_deriv, 1j * k_op),
            (self.use_second_deriv, -(k_op ** 2)),
            (self.use_half_deriv, jnp.sqrt(k_op)),
            (self.use_hilbert, -1j * jnp.sign(k_arr).astype(eta_hat.dtype)[None, :]),
        ):
            if enabled:
                features.append(jnp.fft.irfft(
                    multiplier * eta_hat, n=grid_size, axis=-1,
                ))
        eta_features = jnp.stack(features, axis=-1)

        # No biases: zero eta must produce zero features.
        for name in ("eta_feat_proj", "eta_feat_mix"):
            eta_features = nn.gelu(
                nn.Dense(self.width // 2, name=name, use_bias=False)(eta_features)
            )
        # Near zero, z * tanh(z) behaves like z^2: the correction starts at eta^2.
        eta_features = eta_features * jnp.tanh(eta_features)

        # Sum parallel correction groups, starting from zero in xi's dtype.
        correction = jnp.zeros((batch_size, grid_size), dtype=xi_phys.dtype)
        for block_idx in range(self.n_blocks):
            correction = correction + CraigSulemBlock(
                n_branches=self.latent,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                mult_hidden=self.mult_hidden,
                name=f"cs_block_{block_idx}",
            )(eta_features, xi_phys, clipped_log_depth)

        # Divide by sqrt(total branches), then convert to the target's normalized units.
        correction = correction / float(self.n_blocks * self.latent) ** 0.5
        correction = correction / self.target_scale
        return (baseline + correction)[..., None]
