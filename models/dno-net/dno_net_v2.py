"""Craig-Sulem neural DNO: analytic G0 + G1 plus a learned correction.

The correction is linear in ξ and starts at order η². Each block sums
M(D, h)[Φ(η) · M(D, h)ξ] over learned branches, using the same real Fourier
multiplier on both sides to preserve self-adjointness. Zero-initialized
η projections make the initial correction zero.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class DepthAwareMultiplier(nn.Module):
    """Real Fourier multipliers from (k, h, tanh(hk), k tanh(hk)).

    Input depth is log h with shape (B, 1); output is (B, n_freq, channels).
    """
    out_channels: int
    domain_length: float
    h_clip_max: float
    hidden: int = 32

    @nn.compact
    def __call__(self, depth: jnp.ndarray, n_freq: int) -> jnp.ndarray:
        h_phys = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))  # (B, 1)

        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq)
        batch_size = depth.shape[0]
        k_b = jnp.broadcast_to(k_arr[None, :], (batch_size, n_freq))   # (B, K)
        h_b = jnp.broadcast_to(h_phys, (batch_size, n_freq))           # (B, K)
        kh = h_b * k_b                                                  # (B, K)

        feats = jnp.stack(
            [k_b, h_b, jnp.tanh(kh), k_b * jnp.tanh(kh)], axis=-1
        )                                                           # (B, K, 4)

        hidden_layer = nn.Dense(self.hidden, name="hidden")(feats)
        hidden_layer = nn.tanh(hidden_layer)
        return nn.Dense(
            self.out_channels, name="out", kernel_init=nn.initializers.lecun_normal(),
        )(hidden_layer)


class CraigSulemBlock(nn.Module):
    """Sum branches M_i(D, h)[Φ_i(η) · M_i(D, h)ξ].

    Each Φ_i is a learned projection of the shared η features to one channel.
    Products are pointwise within each branch, without cross-branch products.
    """
    n_branches: int
    domain_length: float
    h_clip_max: float
    mult_hidden: int = 32

    @nn.compact
    def __call__(
        self,
        eta_features: jnp.ndarray,   # (B, N, C_eta)
        xi_phys: jnp.ndarray,        # (B, N)
        depth: jnp.ndarray,          # (B, 1)
    ) -> jnp.ndarray:                # (B, N)
        grid_size = eta_features.shape[1]
        n_freq = grid_size // 2 + 1

        # Zero-initialized η projections leave the analytic baseline unchanged.
        phi = nn.Dense(
            self.n_branches, name="phi_proj",
            kernel_init=nn.initializers.zeros,
            use_bias=False,
        )(eta_features)                                                # (B, N, n_branches)

        # A shared real input/output multiplier makes the block self-adjoint.
        m_shared = DepthAwareMultiplier(
            out_channels=self.n_branches,
            domain_length=self.domain_length,
            h_clip_max=self.h_clip_max,
            hidden=self.mult_hidden,
            name="m_shared",
        )(depth, n_freq)                                              # (B, K, n_branches)

        xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")        # (B, K)
        xi_branched_hat = m_shared * xi_hat[..., None]                # (B, K, n_branches)
        xi_branched = jnp.fft.irfft(
            xi_branched_hat, n=grid_size, axis=1, norm="forward",
        )                                                              # (B, N, n_branches)

        # Multiply each branch in physical space.
        prods = phi * xi_branched                                      # (B, N, n_branches)

        # Apply the output multiplier, then sum over branches.
        prods_hat = jnp.fft.rfft(prods, axis=1, norm="forward")        # (B, K, n_branches)
        out_hat = jnp.sum(m_shared * prods_hat, axis=-1)               # (B, K)
        return jnp.fft.irfft(out_hat, n=grid_size, axis=-1, norm="forward")


class CraigSulemDNO(nn.Module):
    """G0 + G1 plus an order-η² learned correction, all linear in ξ."""
    # Core architecture hyperparameters.
    width: int                       # used to size the η-feature projection
    n_blocks: int = 4                # number of Craig--Sulem blocks summed
    latent: int = 64                 # alias for n_branches per block

    # Polynomial and derivative features of normalized η.
    n_polys: int = 3                 # η, η², η³  (set to 1 to disable η²/η³)
    use_first_deriv: bool = True     # ∂_x η
    use_second_deriv: bool = True    # ∂_x² η
    use_half_deriv: bool = True      # |D|^{1/2} η  (natural in deep-water G_1)
    use_hilbert: bool = True         # ℋ η
    mult_hidden: int = 32            # hidden size of multiplier MLP

    # Physical scaling parameters.
    domain_length: float = 2 * jnp.pi
    xi_scale: float = 1.0
    # η-channel feature_absmax used to recover physical η for the analytic
    # G_1 baseline. Only meaningful under norm=scale.
    eta_scale: float = 1.0
    target_scale: float = 1.0
    h_clip_max: float = 5.0

    def _g0_apply(
        self, x_phys: jnp.ndarray, h: jnp.ndarray, grid_size: int,
    ) -> jnp.ndarray:
        """Apply the flat-surface DNO G_0(h) = |D| tanh(h|D|) in spectral space."""
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(grid_size // 2 + 1)
        g0 = k[None, :] * jnp.tanh(h * k[None, :])
        x_fft = jnp.fft.rfft(x_phys, axis=-1)
        return jnp.fft.irfft(g0 * x_fft, n=grid_size, axis=-1)

    def _linear_baseline(self, xi_norm: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        grid_size = xi_norm.shape[1]
        out_dtype = xi_norm.dtype
        xi_raw = xi_norm * self.xi_scale
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        return (self._g0_apply(xi_raw, h, grid_size) / self.target_scale).astype(out_dtype)

    def _g1_baseline(
        self, eta_norm: jnp.ndarray, xi_norm: jnp.ndarray, depth: jnp.ndarray,
    ) -> jnp.ndarray:
        """G1 = -G0(η G0 ξ) - ∂x(η ∂x ξ), divided by target_scale.

        Recover physical inputs using scale normalization. Evaluate G1 in
        float64 to reduce FFT roundoff, then cast back to the input dtype.
        """
        grid_size = xi_norm.shape[1]
        out_dtype = xi_norm.dtype
        eta_phys = (eta_norm * self.eta_scale).astype(jnp.float64)
        xi_phys = (xi_norm * self.xi_scale).astype(jnp.float64)
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max))).astype(jnp.float64)

        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
            grid_size // 2 + 1, dtype=jnp.float64,
        )

        g0_xi = self._g0_apply(xi_phys, h, grid_size)                  # (B, N)
        g0_mult = k[None, :] * jnp.tanh(h * k[None, :])
        term1_hat = -g0_mult * jnp.fft.rfft(eta_phys * g0_xi, axis=-1)
        term1 = jnp.fft.irfft(term1_hat, n=grid_size, axis=-1)         # (B, N)

        dx_xi = jnp.fft.irfft(
            1j * k[None, :] * jnp.fft.rfft(xi_phys, axis=-1),
            n=grid_size, axis=-1,
        )
        term2_hat = -(1j * k)[None, :] * jnp.fft.rfft(eta_phys * dx_xi, axis=-1)
        term2 = jnp.fft.irfft(term2_hat, n=grid_size, axis=-1)         # (B, N)

        return ((term1 + term2) / self.target_scale).astype(out_dtype)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        # inputs: (B, N, 2) — normalized [η, ξ].  depth: (B, 1) — log h.
        batch_size, grid_size, _ = inputs.shape
        n_freq = grid_size // 2 + 1
        depth_clip = jnp.minimum(depth, jnp.log(self.h_clip_max))

        eta_norm = inputs[..., 0]                                     # (B, N)
        xi_norm = inputs[..., 1]                                      # (B, N)
        xi_phys = xi_norm * self.xi_scale                             # (B, N)

        baseline = (
            self._linear_baseline(xi_norm, depth)
            + self._g1_baseline(eta_norm, xi_norm, depth)
        )

        # η spatial features (pointwise polynomials + spectral derivatives).
        feats = [eta_norm ** p for p in range(1, self.n_polys + 1)]
        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
            n_freq, dtype=eta_norm.dtype,
        )
        k_op = k_arr[None, :]
        eta_hat = jnp.fft.rfft(eta_norm, axis=-1)

        if self.use_first_deriv:
            feats.append(jnp.fft.irfft(
                1j * k_op * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_second_deriv:
            feats.append(jnp.fft.irfft(
                -(k_op ** 2) * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_half_deriv:
            feats.append(jnp.fft.irfft(
                jnp.sqrt(k_op) * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_hilbert:
            sgn = jnp.sign(k_arr).astype(eta_hat.dtype)
            feats.append(jnp.fft.irfft(
                -1j * sgn[None, :] * eta_hat, n=grid_size, axis=-1,
            ))
        eta_features = jnp.stack(feats, axis=-1)                       # (B, N, C_eta_raw)

        # The learned residual begins at O(η²), so both trunk layers and the
        # per-block projection are structurally bias-free.
        for name in ("eta_feat_proj", "eta_feat_mix"):
            eta_features = nn.Dense(
                self.width // 2, name=name, use_bias=False,
            )(eta_features)
            eta_features = nn.gelu(eta_features)                      # (B, N, width/2)
        # If η -> εη, the bias-free trunk is O(ε); this smooth lift is
        # O(ε²) and cannot overwrite the exact G_1 backbone.
        eta_features = eta_features * jnp.tanh(eta_features)

        # Keep ξ's dtype instead of letting the default zero array widen to float64.
        residual = jnp.zeros((batch_size, grid_size), dtype=xi_phys.dtype)
        for block_idx in range(self.n_blocks):
            residual = residual + CraigSulemBlock(
                n_branches=self.latent,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                mult_hidden=self.mult_hidden,
                name=f"cs_block_{block_idx}",
            )(eta_features, xi_phys, depth_clip)

        # Fan-in normalization keeps a fresh Adam update's function-space size
        # independent of the chosen block/branch count.
        residual = residual / float(self.n_blocks * self.latent) ** 0.5
        residual = residual / self.target_scale
        return (baseline + residual)[..., None]                        # (B, N, 1)
