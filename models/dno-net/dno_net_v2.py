"""Craig-Sulem-structured neural Dirichlet-to-Neumann operator.

This module implements the polynomial-in-η + depth-aware spectral filter
architecture proposed in ``notes/rollout_accuracy_report.tex``. It is a drop-in
replacement for ``SpectralDNO`` (same ``__call__`` signature and IO shape) with a
fundamentally different inductive bias.

Mathematical motivation
-----------------------
The true Dirichlet--Neumann operator admits a Craig--Sulem expansion

    G(η; h) ξ = G_0(h) ξ + G_1(η, h) ξ + G_2(η ⊗ η, h) ξ + ...

where:
  * G_0(h) = |D| tanh(h|D|)  is the flat-surface linear DNO.
  * G_n is multilinear in η^{⊗n} and linear in ξ, formed by alternating
    pointwise multiplications by powers of η with Fourier multipliers
    m(D, h) — i.e. expressions like  m_out(D, h) [ η · m_xi(D, h) ξ ].

The original ``SpectralDNO`` learns *one* effective G_1-shaped term via a single
rank-`latent` spatial × spectral sandwich. This module replaces that with a
sum of ``n_blocks`` independent Craig--Sulem-shaped blocks, each of the form

    Block(η, ξ, h)(x) = M_out(D, h) [ sum_i Φ_i(η)(x) · ( M_i(D, h) ξ )(x) ]

where:
  * Φ_i(η)(x) are pointwise spatial features of η (pointwise polynomials and
    spectral derivatives — η, η², η³, ∂_x η, ∂_x^2 η, |D|^{1/2} η, ℋ η).
  * M_i(D, h) and M_out(D, h) are learned Fourier multipliers parameterised as
    small MLPs of the *physical* features (k, h, tanh(hk), k tanh(hk)) — so
    they inherit the dispersive scaling for free and only need to learn
    perturbations on it.
  * The product is computed pointwise in x, mixed by a Dense layer, then sent
    through the output multiplier.

The whole architecture is therefore:
  * **Linear in ξ by construction** — preserves the SpectralDNO advantage on
    the linear regime (no sideband hallucination).
  * **Polynomial in η by construction** — gives the architecture explicit
    higher-order Craig--Sulem-like terms, addressing the steepness/bandwidth
    failure mode identified by the rollout forensics.
  * **Depth-aware spectral filters by construction** — multipliers receive
    (k, h, tanh(hk)) as inputs rather than learning the dispersion relation
    from scratch.

Block outputs are zero-initialised, so at step 0 the model returns the linear
baseline exactly.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class DepthAwareMultiplier(nn.Module):
    """Learned Fourier multiplier conditioned on (k, h).

    Produces a real-valued per-mode multiplier of shape ``(B, n_freq, channels)``
    from the log-depth ``depth`` and the physical wavenumber grid. The MLP
    sees (k, h, tanh(hk), k tanh(hk)) so it can reproduce the linear DNO
    dispersion scaling exactly when its weights match.
    """
    out_channels: int
    domain_length: float
    h_clip_max: float
    hidden: int = 32
    zero_init_output: bool = False

    @nn.compact
    def __call__(self, depth: jnp.ndarray, n_freq: int) -> jnp.ndarray:
        # depth: (B, 1)  log h (already clipped externally is fine; we clip again)
        depth_clip = jnp.minimum(depth, jnp.log(self.h_clip_max))
        h_phys = jnp.exp(depth_clip)                                # (B, 1)

        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq)
        batch_size = depth.shape[0]
        k_b = jnp.broadcast_to(k_arr[None, :], (batch_size, n_freq))   # (B, K)
        h_b = jnp.broadcast_to(h_phys, (batch_size, n_freq))           # (B, K)
        kh = h_b * k_b                                                  # (B, K)

        feats = jnp.stack(
            [k_b, h_b, jnp.tanh(kh), k_b * jnp.tanh(kh)], axis=-1
        )                                                               # (B, K, 4)

        hidden_layer = nn.Dense(self.hidden, name="hidden")(feats)
        hidden_layer = nn.tanh(hidden_layer)
        out_init = nn.initializers.zeros if self.zero_init_output else nn.initializers.lecun_normal()
        out = nn.Dense(self.out_channels, name="out", kernel_init=out_init)(hidden_layer)
        return out                                                      # (B, K, C)


class CraigSulemBlock(nn.Module):
    """One Craig--Sulem-shaped block.

    Computes a sum of ``n_branches`` Craig--Sulem-shaped terms,

        out(x) = sum_i M_out_i(D, h) [ Φ_i(η)(x) · ( M_xi_i(D, h) ξ )(x) ].

    Each ``Φ_i`` is one of ``n_branches`` learned projections of the shared
    η-feature stack down to one spatial channel; each branch has its own
    input multiplier ``M_xi_i`` and output multiplier ``M_out_i``. The
    product is **pointwise** in (x, branch_index) — no tensor product
    between η-channels and ξ-branches — so the memory footprint scales
    linearly in n_branches, not quadratically.

    The η→branches projection is zero-initialised so the block starts inert
    (the model initially predicts the linear DNO baseline).
    """
    n_branches: int
    domain_length: float
    h_clip_max: float
    mult_hidden: int = 32
    tie_xi_out_mult: bool = False
    phi_bias_free: bool = False
    fft_fp64: bool = False

    @nn.compact
    def __call__(
        self,
        eta_features: jnp.ndarray,   # (B, N, C_eta)
        xi_phys: jnp.ndarray,        # (B, N)
        depth: jnp.ndarray,          # (B, 1)
    ) -> jnp.ndarray:                # (B, N)
        batch_size, grid_size, _ = eta_features.shape
        n_freq = grid_size // 2 + 1
        out_dtype = xi_phys.dtype
        if self.fft_fp64:
            xi_phys = xi_phys.astype(jnp.float64)

        # 1. Project shared η-features down to one spatial channel per branch.
        #    Zero-init so the block is inert at step 0.
        phi = nn.Dense(
            self.n_branches, name="phi_proj",
            kernel_init=nn.initializers.zeros,
            use_bias=not self.phi_bias_free,
        )(eta_features)                                                # (B, N, n_branches)

        # 2. Apply n_branches depth-aware Fourier multipliers to ξ in k-space.
        if self.tie_xi_out_mult:
            # Single shared multiplier used as both M_xi and M_out. With real-
            # valued multipliers and a real pointwise φ, this makes the block
            # self-adjoint in ξ ↔ ψ (matching the true G's symmetry).
            m_shared = DepthAwareMultiplier(
                out_channels=self.n_branches,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                hidden=self.mult_hidden,
                name="m_shared",
            )(depth, n_freq)                                          # (B, K, n_branches)
            m_xi = m_shared
            m_out = m_shared
        else:
            m_xi = DepthAwareMultiplier(
                out_channels=self.n_branches,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                hidden=self.mult_hidden,
                name="m_xi",
            )(depth, n_freq)                                          # (B, K, n_branches)
            m_out = DepthAwareMultiplier(
                out_channels=self.n_branches,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                hidden=self.mult_hidden,
                name="m_out",
            )(depth, n_freq)                                          # (B, K, n_branches)

        xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")        # (B, K)
        xi_branched_hat = m_xi * xi_hat[..., None]                    # (B, K, n_branches)
        xi_branched = jnp.fft.irfft(
            xi_branched_hat, n=grid_size, axis=1, norm="forward",
        )                                                              # (B, N, n_branches)

        # 3. Pointwise product per branch — NO outer product over channels.
        prods = phi * xi_branched                                      # (B, N, n_branches)

        # 4. Per-branch output multiplier, then sum over branches.
        prods_hat = jnp.fft.rfft(prods, axis=1, norm="forward")        # (B, K, n_branches)
        out_hat = jnp.sum(m_out * prods_hat, axis=-1)                  # (B, K)
        out_phys = jnp.fft.irfft(out_hat, n=grid_size, axis=-1, norm="forward")
        return out_phys.astype(out_dtype)


class CraigSulemDNO(nn.Module):
    """Drop-in replacement for ``SpectralDNO`` with Craig--Sulem block structure.

    Output is linear in ξ by construction and polynomial in η through the
    learned Φ_i(η) features (pointwise polynomials and spectral derivatives).
    """
    # CLI hyperparams — preserved from SpectralDNO so the training entrypoint
    # can be wired up with minimal changes.
    modes: int                       # unused; kept for CLI compatibility
    width: int                       # used to size the η-feature projection
    n_blocks: int = 4                # number of Craig--Sulem blocks summed
    latent: int = 64                 # alias for n_branches per block

    # New hyperparams.
    n_polys: int = 3                 # η, η², η³  (set to 1 to disable η²/η³)
    use_first_deriv: bool = True     # ∂_x η
    use_second_deriv: bool = True    # ∂_x² η
    use_half_deriv: bool = True      # |D|^{1/2} η  (natural in deep-water G_1)
    use_hilbert: bool = True         # ℋ η
    # JCP09 Tier 2D: nested G_0(η) features. True G_2/G_3 involve G_0 acting on η;
    # exposing them analytically saves the multiplier MLPs from learning that shape.
    use_g0_eta: bool = False         # G_0(h) η  (depth-aware)
    use_g0_eta_dx: bool = False      # ∂_x G_0(h) η
    mult_hidden: int = 32            # hidden size of multiplier MLP

    # Hard-code the closed-form first-order Craig-Sulem term
    #   G_1(η, h) ξ = - G_0(η G_0 ξ) - ∂_x (η ∂_x ξ)
    # into the baseline. Forces the small-amplitude / linear-eigenmode limit
    # to be correct by construction; the learned blocks then model G_2+.
    use_g1_baseline: bool = False

    # Zero the G_1 term's output modes at k >= g1_k_cut. The double k-amplified
    # f32 FFT chain (G_0∘mult∘G_0, ∂x∘mult∘∂x) produces rounding noise that
    # exceeds the genuine G_1 signal above k≈128; the H¹ training loss then
    # weights that band by k, creating an unlearnable ~3e-3 loss floor.
    # 0 disables the cutoff.
    g1_k_cut: int = 128

    # Run every FFT chain in the model in fp64 / complex128 (G_0 baseline, ∂x,
    # η spatial features, G_1 baseline if on, and every CraigSulemBlock's
    # rfft → mult → irfft → mult → rfft → mult → irfft pipeline). Learned
    # multiplier weights, Dense layers, and optimizer state stay fp32 — only
    # the spectral compute path runs in fp64 via promotion at the boundary.
    # Requires jax_enable_x64; otherwise the cast silently downgrades to fp32.
    fft_fp64: bool = False

    # Tie M_xi = M_out inside every CraigSulemBlock so each block is
    # self-adjoint in ξ ↔ ψ (matching the true G(η)'s symmetry).
    tie_xi_out_mult: bool = False

    # Drop all bias terms on the η-feature trunk and the per-block φ
    # projections. Every η-feature vanishes at η = 0 and gelu(0) = 0, so with
    # no biases every block output is identically zero at η = 0 and the
    # prediction collapses to the G_0 baseline FOR ALL trained weights — the
    # flat-surface limit is exact by construction, not just at init. Directly
    # targets the linear-regime sideband hallucination.
    phi_bias_free: bool = False

    # Carried over from SpectralDNO.
    domain_length: float = 6.283185307179586
    xi_scale: float = 1.0
    # η-channel feature_absmax: used to recover physical η from the normalized
    # input channel for the analytic G_1 baseline. Default 1.0 keeps existing
    # behavior when use_g1_baseline=False. Only meaningful under norm=scale.
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

    def _dx_apply(self, x_phys: jnp.ndarray, grid_size: int) -> jnp.ndarray:
        """Apply ∂_x in spectral space."""
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(grid_size // 2 + 1)
        x_fft = jnp.fft.rfft(x_phys, axis=-1)
        return jnp.fft.irfft(1j * k[None, :] * x_fft, n=grid_size, axis=-1)

    def _linear_baseline(self, xi_norm: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        grid_size = xi_norm.shape[1]
        out_dtype = xi_norm.dtype
        if self.fft_fp64:
            xi_raw = (xi_norm * self.xi_scale).astype(jnp.float64)
            h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max))).astype(jnp.float64)
        else:
            xi_raw = xi_norm * self.xi_scale
            h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        return (self._g0_apply(xi_raw, h, grid_size) / self.target_scale).astype(out_dtype)

    def _g1_baseline(
        self, eta_norm: jnp.ndarray, xi_norm: jnp.ndarray, depth: jnp.ndarray,
    ) -> jnp.ndarray:
        """Closed-form first-order Craig-Sulem term, in normalized output units.

        Physical:  G_1(η_phys, h) ξ_phys = - G_0(η_phys · G_0 ξ_phys) - ∂_x(η_phys · ∂_x ξ_phys)
        Returned:  G_1(...) / target_scale  to match the normalized residual convention.

        Assumes norm=scale so that η_phys = eta_norm * eta_scale and
        ξ_phys = xi_norm * xi_scale.

        When fft_fp64 is set, the entire chain (G_0∘mult∘G_0 and ∂x∘mult∘∂x)
        runs in fp64/complex128 to defeat the f32 FFT noise floor. The cast
        back to the caller's dtype happens at the return so the rest of the
        model stays in its native precision.
        """
        grid_size = xi_norm.shape[1]
        out_dtype = xi_norm.dtype
        if self.fft_fp64:
            eta_phys = (eta_norm * self.eta_scale).astype(jnp.float64)
            xi_phys = (xi_norm * self.xi_scale).astype(jnp.float64)
            h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max))).astype(jnp.float64)
            k_dtype = jnp.float64
        else:
            eta_phys = eta_norm * self.eta_scale
            xi_phys = xi_norm * self.xi_scale
            h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
            k_dtype = xi_phys.dtype

        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(grid_size // 2 + 1, dtype=k_dtype)
        # Mask the *outer* multipliers above g1_k_cut (see field comment). The
        # in-band (k < cut) noise contribution of the inner products is
        # negligible, so only the final transforms need masking.
        mask = jnp.ones_like(k) if self.g1_k_cut <= 0 else (
            jnp.arange(grid_size // 2 + 1) < self.g1_k_cut
        ).astype(k.dtype)

        g0_xi = self._g0_apply(xi_phys, h, grid_size)                  # (B, N)
        g0_mult = mask[None, :] * k[None, :] * jnp.tanh(h * k[None, :])
        term1_hat = -g0_mult * jnp.fft.rfft(eta_phys * g0_xi, axis=-1)
        term1 = jnp.fft.irfft(term1_hat, n=grid_size, axis=-1)         # (B, N)

        dx_xi = self._dx_apply(xi_phys, grid_size)                     # (B, N)
        term2_hat = -(mask * 1j * k)[None, :] * jnp.fft.rfft(eta_phys * dx_xi, axis=-1)
        term2 = jnp.fft.irfft(term2_hat, n=grid_size, axis=-1)         # (B, N)

        return ((term1 + term2) / self.target_scale).astype(out_dtype)

    def _eta_spatial_features(
        self, eta: jnp.ndarray, n_freq: int, grid_size: int,
        depth: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build pointwise η-features: polynomials + spectral derivatives.

        Returns shape (B, N, C_eta). If use_g0_eta or use_g0_eta_dx is set, depth
        (B,1) log-h must be provided so G_0(h)·η can be evaluated per-sample.
        """
        feats = []
        for p in range(1, self.n_polys + 1):
            feats.append(eta ** p)

        # FFT chain runs in fp64 when requested; cast back to eta.dtype before
        # appending so the stacked features stay in the model's working precision.
        out_dtype = eta.dtype
        if self.fft_fp64:
            eta_fft_input = eta.astype(jnp.float64)
            k_dtype = jnp.float64
        else:
            eta_fft_input = eta
            k_dtype = eta.dtype
        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq, dtype=k_dtype)
        eta_hat = jnp.fft.rfft(eta_fft_input, axis=-1)

        if self.use_first_deriv:
            feats.append(jnp.fft.irfft(
                1j * k_arr[None, :] * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_second_deriv:
            feats.append(jnp.fft.irfft(
                -(k_arr ** 2)[None, :] * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_half_deriv:
            feats.append(jnp.fft.irfft(
                jnp.sqrt(k_arr)[None, :] * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_hilbert:
            sgn = jnp.sign(k_arr).astype(eta_hat.dtype)
            feats.append(jnp.fft.irfft(
                -1j * sgn[None, :] * eta_hat, n=grid_size, axis=-1,
            ))
        if self.use_g0_eta or self.use_g0_eta_dx:
            # G_0(h) = |k| tanh(h|k|); broadcast h over batch. Depth arrives as log(h),
            # clipped to h_clip_max so the symbol matches the linear baseline.
            h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))   # (B, 1)
            g0_sym = k_arr[None, :] * jnp.tanh(h * k_arr[None, :])      # (B, n_freq)
            g0_eta_hat = g0_sym * eta_hat                                # (B, n_freq)
            if self.use_g0_eta:
                feats.append(jnp.fft.irfft(g0_eta_hat, n=grid_size, axis=-1))
            if self.use_g0_eta_dx:
                feats.append(jnp.fft.irfft(1j * k_arr[None, :] * g0_eta_hat,
                                           n=grid_size, axis=-1))
        return jnp.stack(feats, axis=-1).astype(out_dtype)            # (B, N, C_eta_raw)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        # inputs: (B, N, 2) — normalized [η, ξ].  depth: (B, 1) — log h.
        batch_size, grid_size, _ = inputs.shape
        n_freq = grid_size // 2 + 1
        depth_clip = jnp.minimum(depth, jnp.log(self.h_clip_max))

        eta_norm = inputs[..., 0]                                     # (B, N)
        xi_norm = inputs[..., 1]                                      # (B, N)
        xi_phys = xi_norm * self.xi_scale                             # (B, N)

        baseline = self._linear_baseline(xi_norm, depth)              # (B, N)
        if self.use_g1_baseline:
            baseline = baseline + self._g1_baseline(eta_norm, xi_norm, depth)

        # η spatial features (pointwise polynomials + spectral derivatives).
        raw_eta_feats = self._eta_spatial_features(eta_norm, n_freq, grid_size,
                                                   depth=depth_clip)
        # Project to a richer pointwise feature space. With phi_bias_free, both
        # layers carry no bias so features (and hence all blocks) vanish
        # identically at η = 0.
        use_bias = not self.phi_bias_free
        eta_features = nn.Dense(self.width // 2, name="eta_feat_proj", use_bias=use_bias)(raw_eta_feats)
        eta_features = nn.gelu(eta_features)
        eta_features = nn.Dense(self.width // 2, name="eta_feat_mix", use_bias=use_bias)(eta_features)
        eta_features = nn.gelu(eta_features)                          # (B, N, width/2)

        # Sum of n_blocks Craig--Sulem blocks. Explicit dtype: jnp.zeros without
        # one defaults to fp64 when x64 is enabled, which would silently widen
        # the whole residual+baseline sum to fp64 even under mixed precision.
        residual = jnp.zeros((batch_size, grid_size), dtype=xi_phys.dtype)
        for block_idx in range(self.n_blocks):
            block_out = CraigSulemBlock(
                n_branches=self.latent,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                mult_hidden=self.mult_hidden,
                tie_xi_out_mult=self.tie_xi_out_mult,
                phi_bias_free=self.phi_bias_free,
                fft_fp64=self.fft_fp64,
                name=f"cs_block_{block_idx}",
            )(eta_features, xi_phys, depth_clip)
            residual = residual + block_out                           # (B, N)

        residual = residual / self.target_scale
        return (baseline + residual)[..., None]                       # (B, N, 1)
