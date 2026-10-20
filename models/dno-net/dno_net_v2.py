"""Craig-Sulem neural DNO with a selectable analytic baseline.

Here eta is surface elevation and xi is surface velocity potential.
Ordinary branches apply M[spatial_weights(eta) * M[xi]]; optional nested
branches apply gamma M[eta N[eta M[xi]]], with real depth-dependent filters.
The tied-filter correction is self-adjoint,
linear in xi. It starts at order eta^3 with G0+G1+G2, eta^2 with G0+G1,
eta with G0 alone, or uses a constant feature to learn the full DNO.
"""
from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from flax import linen as nn

from solver.solvers.dno_series_jax import dno_series_eval


def surface_parity_products(features: jnp.ndarray) -> jnp.ndarray:
    """Replace the two odd surface channels by their three even quadratic products."""
    eta, eta2, eta3, dx, dxx, half, hilbert = jnp.moveaxis(features, -1, 0)
    return jnp.stack((eta, eta2, eta3, dx**2, dxx, half, hilbert**2, dx * hilbert), axis=-1)


def surface_amplitude(eta_norm: jnp.ndarray, reference_rms: float) -> jnp.ndarray:
    """Smooth per-surface RMS amplitude relative to the fixed training RMS."""
    return jnp.sqrt(jnp.mean((eta_norm / reference_rms) ** 2, axis=-1, keepdims=True) + 1e-6)


def phase_rank_one_hat(
    eta_hat: jnp.ndarray, xi_hat: jnp.ndarray, log_depth: jnp.ndarray,
    coefficients: jnp.ndarray, domain_length: float,
) -> jnp.ndarray:
    """Physical, unnormalized even-grid rFFT of beta(h) u mean(u xi), u = |D|^.5 eta_x."""
    grid_size = 2 * (eta_hat.shape[-1] - 1)
    k = (2.0 * jnp.pi / domain_length) * jnp.arange(
        eta_hat.shape[-1], dtype=eta_hat.real.dtype,
    )
    u_hat = (1j * k * jnp.sqrt(k) * eta_hat).at[..., -1].set(0)
    weights = jnp.full(k.shape, 2, dtype=k.dtype).at[0].set(1).at[-1].set(1)
    moment = jnp.sum(
        weights * jnp.real(jnp.conj(u_hat) * xi_hat), axis=-1, keepdims=True,
    ) / grid_size**2
    beta = coefficients[0] + coefficients[1] * (
        log_depth + jnp.log(2.0 * jnp.pi / domain_length)
    )
    return beta * moment * u_hat


def phase_wavenumber_hat(
    eta_hat: jnp.ndarray, xi_hat: jnp.ndarray, multiplier: jnp.ndarray,
    domain_length: float,
) -> jnp.ndarray:
    """Unnormalized rFFT of .5[u mean(v xi) + v mean(u xi)], v = m(kh)u."""
    grid_size = 2 * (eta_hat.shape[-1] - 1)
    k = (2.0 * jnp.pi / domain_length) * jnp.arange(
        eta_hat.shape[-1], dtype=eta_hat.real.dtype,
    )
    u_hat = (1j * k * jnp.sqrt(k) * eta_hat).at[..., -1].set(0)
    v_hat = multiplier * u_hat
    weights = jnp.full(k.shape, 2, dtype=k.dtype).at[0].set(1).at[-1].set(1)
    moments = jnp.sum(
        weights * jnp.real(jnp.conj(jnp.stack((u_hat, v_hat), axis=-2))
                           * xi_hat[..., None, :]), axis=-1, keepdims=True,
    ) / grid_size**2
    return 0.5 * (u_hat * moments[..., 1, :] + v_hat * moments[..., 0, :])


class PhaseWavenumberMultiplier(nn.Module):
    """Signed scalar multiplier conditioned on physical kh at the actual depth."""
    hidden: int
    domain_length: float

    @nn.compact
    def __call__(self, log_depth: jnp.ndarray, n_freq: int) -> jnp.ndarray:
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq, dtype=log_depth.dtype)
        kh = jnp.exp(log_depth) * k[None, :]
        features = jnp.stack((jnp.log1p(kh), jnp.tanh(kh)), axis=-1)
        hidden = nn.tanh(nn.Dense(self.hidden, name="hidden")(features))
        return nn.Dense(
            1, name="out", kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
        )(hidden)[..., 0]


class DepthAwareMultiplier(nn.Module):
    """Learn a real Fourier multiplier for each wavenumber and branch.

    Depth is log(h), shaped (batch, 1); output is (batch, frequency, branch).
    """
    out_channels: int
    domain_length: float
    h_clip_max: float
    hidden: int = 32
    multiplier_skip: bool = False
    log_multiplier_inputs: bool = False

    @nn.compact
    def __call__(self, depth: jnp.ndarray, n_freq: int) -> jnp.ndarray:
        h = jnp.exp(jnp.minimum(depth, jnp.log(self.h_clip_max)))
        k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(n_freq)
        k, h = jnp.broadcast_arrays(k[None, :], h)
        kh = h * k
        features = jnp.stack([k, h, jnp.tanh(kh), k * jnp.tanh(kh)], axis=-1)
        if self.log_multiplier_inputs:
            features = jnp.stack([jnp.log1p(k), jnp.log(h), jnp.tanh(kh),
                                  jnp.log1p(k * jnp.tanh(kh))], axis=-1)
        hidden = nn.tanh(nn.Dense(self.hidden, name="hidden")(features))
        multiplier = nn.Dense(
            self.out_channels, name="out", kernel_init=nn.initializers.lecun_normal(),
        )(hidden)
        if self.multiplier_skip:
            k_ref = 2.0 * jnp.pi * 128 / self.domain_length
            symbols = jnp.stack((jnp.ones_like(k), k * jnp.tanh(kh) / k_ref,
                                 (k / k_ref) ** 2), axis=-1)
            multiplier = multiplier + nn.Dense(
                self.out_channels, name="symbol_skip", use_bias=False,
                kernel_init=nn.initializers.zeros,
            )(symbols)
        return multiplier


class CraigSulemBlock(nn.Module):
    """Sum tied-filter branches, optionally nesting a filter between two eta factors."""
    n_branches: int
    domain_length: float
    h_clip_max: float
    mult_hidden: int = 32
    nested: bool = False
    multiplier_skip: bool = False
    log_multiplier_inputs: bool = False

    @nn.compact
    def __call__(
        self,
        eta_features: jnp.ndarray,
        xi_phys: jnp.ndarray,
        depth: jnp.ndarray,
        *,
        eta_norm: jnp.ndarray | None = None,
        nested_features: jnp.ndarray | None = None,
        correction_hat_addition: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        grid_size = eta_features.shape[1]

        # Initialize the learned contribution to zero.
        spatial_weights = None
        if not self.nested:
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
            multiplier_skip=self.multiplier_skip,
            log_multiplier_inputs=self.log_multiplier_inputs,
            name="m_shared",
        )(depth, grid_size // 2 + 1)

        xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")
        filtered_xi = jnp.fft.irfft(
            multiplier * xi_hat[..., None], n=grid_size, axis=1, norm="forward",
        )
        if self.nested:
            inner_multiplier = DepthAwareMultiplier(
                self.n_branches, self.domain_length, self.h_clip_max,
                self.mult_hidden, multiplier_skip=self.multiplier_skip,
                log_multiplier_inputs=self.log_multiplier_inputs, name="m_inner",
            )(depth, grid_size // 2 + 1)
            surface = cast(jnp.ndarray, eta_norm)[..., None]
            if nested_features is not None:
                surface_mix = self.param(
                    "surface_mix", nn.initializers.zeros, (2, self.n_branches), jnp.float32,
                )
                surface = surface + nested_features @ surface_mix
            inner_hat = jnp.fft.rfft(surface * filtered_xi, axis=1, norm="forward")
            inner_field = jnp.fft.irfft(
                inner_multiplier * inner_hat, n=grid_size, axis=1, norm="forward",
            )
            gamma = self.param("gamma", nn.initializers.zeros, (self.n_branches,), jnp.float32)
            weighted_xi = gamma * surface * inner_field
        else:
            weighted_xi = cast(jnp.ndarray, spatial_weights) * filtered_xi
        weighted_xi_hat = jnp.fft.rfft(weighted_xi, axis=1, norm="forward")
        correction_hat = jnp.sum(multiplier * weighted_xi_hat, axis=-1)
        if correction_hat_addition is not None:
            correction_hat = correction_hat + correction_hat_addition
        return jnp.fft.irfft(correction_hat, n=grid_size, axis=-1, norm="forward")


class CraigSulemDNO(nn.Module):
    """Learn the remainder above the selected analytic Craig-Sulem order."""
    width: int                        # Each shared eta layer has width // 2 channels.
    n_blocks: int = 4                 # Groups of parallel branches, summed together.
    latent: int = 64                  # Branches per group.
    nested_blocks: int = 0            # Replace this many final groups with M eta N eta M.
    nested_surface_modulation: bool = False  # Bounded eta/h modulation of both nested factors.
    nested_surface_features: bool = False  # Mix even linear surface features per nested branch.
    phase_rank_one: bool = False      # Signed quadratic correction along |D|^.5 eta_x.
    phase_wavenumber_hidden: int = 0  # Positive width enables a symmetric kh-aware phase head.

    # Polynomial and derivative features of normalized eta.
    n_polys: int = 3                  # eta, eta^2, eta^3
    use_first_deriv: bool = True      # First spatial derivative.
    use_second_deriv: bool = True     # Second spatial derivative.
    curvature_cutoff: float = 0.0     # Physical wavenumber; zero keeps exact curvature.
    use_half_deriv: bool = True       # Fourier multiplier sqrt(abs(k)).
    use_hilbert: bool = True          # Hilbert transform.
    use_relative_elevation: bool = False  # Append physical eta / actual depth.
    use_learned_eta_filters: bool = False  # Modulate the two even surface filters.
    surface_feature_scales: tuple[float, ...] | None = None  # Fixed training-feature RMS.
    surface_amplitude_reference: float = 0.0  # Positive raw-eta TRAIN RMS enables per-surface conditioning.
    surface_feature_transform: tuple[tuple[float, ...], ...] | None = None  # Fixed, zero-preserving whitening.
    surface_context_width: int = 0    # Optional narrow nonlocal surface encoder.
    surface_context_gate: bool = False  # Reuse context as a bounded multiplicative gate.
    surface_polynomial_skip: bool = False  # Direct quadratic/cubic features at the head.
    surface_parity_products: bool = False  # Even dx/Hilbert products in the trunk and head.
    surface_depth_gate: bool = False  # Condition lifted features on depth alone.
    mult_hidden: int = 32             # Hidden channels in each multiplier network.
    multiplier_skip: bool = False    # Learn constant, G0, and k^2 symbol coefficients.
    log_multiplier_inputs: bool = False  # Compress multiplier coordinates before tanh.
    baseline_order: int = 1           # -1: none; otherwise CS sum through this order.
    baseline_true_depth: bool = False  # Analytic terms only; learned filters still clip h.

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
        h = jnp.exp(depth if self.baseline_true_depth else jnp.minimum(depth, jnp.log(self.h_clip_max)))
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
        h = jnp.exp(depth if self.baseline_true_depth else jnp.minimum(depth, jnp.log(self.h_clip_max))).astype(jnp.float64)

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

    def _encode_surface(
        self, eta_norm: jnp.ndarray, depth: jnp.ndarray, *, amplitude: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
        """Shared geometry trunk before the baseline-dependent Taylor gate."""
        grid_size = eta_norm.shape[1]
        clipped_log_depth = jnp.minimum(depth, jnp.log(self.h_clip_max))
        # Build polynomial and spatial derivative features from eta.
        features = [eta_norm ** p for p in range(1, self.n_polys + 1)]
        if self.baseline_order == -1:
            features.insert(0, jnp.ones_like(eta_norm))
        k_arr = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
            grid_size // 2 + 1, dtype=eta_norm.dtype,
        )
        k_op = k_arr[None, :]
        eta_hat = jnp.fft.rfft(eta_norm, axis=-1)
        second_symbol, half_symbol = -(k_op ** 2), jnp.sqrt(k_op)
        if self.curvature_cutoff > 0:
            second_symbol = second_symbol / jnp.sqrt(1.0 + (k_op / self.curvature_cutoff) ** 2)
        if self.use_learned_eta_filters:
            surface_k, surface_log_h = jnp.broadcast_arrays(k_op, clipped_log_depth)
            surface_conditioning = jnp.stack((
                jnp.log1p(surface_k), surface_log_h,
                jnp.tanh(jnp.exp(surface_log_h) * surface_k),
            ), axis=-1)
            surface_hidden = nn.tanh(nn.Dense(
                self.mult_hidden, name="eta_filter_hidden",
            )(surface_conditioning))
            surface_modulation = 1.0 + 0.5 * nn.tanh(nn.Dense(
                2, name="eta_filter_out", kernel_init=nn.initializers.zeros,
            )(surface_hidden))
            second_symbol = second_symbol * surface_modulation[..., 0]
            half_symbol = half_symbol * surface_modulation[..., 1]

        for enabled, multiplier in (
            (self.use_first_deriv, 1j * k_op),
            (self.use_second_deriv, second_symbol),
            (self.use_half_deriv, half_symbol),
            (self.use_hilbert, -1j * jnp.sign(k_arr).astype(eta_hat.dtype)[None, :]),
        ):
            if enabled:
                features.append(jnp.fft.irfft(
                    multiplier * eta_hat, n=grid_size, axis=-1,
                ))
        if self.use_relative_elevation:
            features.append(eta_norm * self.eta_scale / jnp.exp(depth))
        eta_features = jnp.stack(features, axis=-1)
        if self.surface_parity_products:
            eta_features = surface_parity_products(eta_features)
        if amplitude is not None:
            powers = jnp.asarray(
                list(range(1, self.n_polys + 1)) + [1] * (len(features) - self.n_polys),
                dtype=eta_features.dtype,
            )
            eta_features = eta_features / amplitude[..., None] ** powers
        if self.surface_feature_scales is not None:
            eta_features = eta_features / jnp.asarray(
                self.surface_feature_scales, dtype=eta_features.dtype,
            )
        nested_features = None
        if self.nested_surface_features:
            curvature_index = self.n_polys + int(self.baseline_order == -1) + int(self.use_first_deriv)
            nested_features = jnp.stack((
                eta_features[..., curvature_index + 1], eta_features[..., curvature_index],
            ), axis=-1)
        skips = ((eta_features[..., 1:3],) if self.surface_polynomial_skip else ())
        if self.surface_parity_products:
            skips += (eta_features[..., jnp.asarray((3, 6, 7))],)
        polynomial_features = jnp.concatenate(skips, axis=-1) if skips else None
        if self.surface_feature_transform is not None:
            eta_features = eta_features @ jnp.asarray(self.surface_feature_transform, dtype=eta_features.dtype)

        # With no constant input, zero eta gives zero features.
        context = None
        for name in ("eta_feat_proj", "eta_feat_mix"):
            if name == "eta_feat_mix" and amplitude is not None:
                gain = self.param("eta_amplitude_gain", nn.initializers.zeros, (self.width // 2,), jnp.float32)
                eta_features = eta_features * (1.0 + gain * jnp.log(amplitude[..., None]))
            eta_features = nn.gelu(
                nn.Dense(self.width // 2, name=name, use_bias=False)(eta_features)
            )
            if name == "eta_feat_proj" and self.surface_context_width:
                context = nn.Dense(
                    self.surface_context_width, name="eta_context_in", use_bias=False,
                )(eta_features)
                context_k, context_log_h = jnp.broadcast_arrays(k_op, clipped_log_depth)
                conditioning = jnp.stack((
                    jnp.log1p(context_k), context_log_h,
                    jnp.tanh(jnp.exp(context_log_h) * context_k),
                ), axis=-1)
                context_hidden = nn.tanh(nn.Dense(
                    self.surface_context_width, name="eta_context_hidden",
                )(conditioning))
                context_multiplier = 0.1 * nn.tanh(nn.Dense(
                    self.surface_context_width, name="eta_context_out",
                )(context_hidden))
                context = jnp.fft.irfft(
                    context_multiplier * jnp.fft.rfft(context, axis=1),
                    n=grid_size, axis=1,
                )
                context = nn.Dense(
                    self.width // 2, name="eta_context_expand", use_bias=False,
                )(context)
                eta_features = eta_features + context
        return eta_features, nested_features, polynomial_features, context

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, depth: jnp.ndarray, *, return_baseline: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Inputs: (batch, grid, 2) normalized [eta, xi]; depth: (batch, 1) log(h).
        batch_size, grid_size, _ = inputs.shape
        clipped_log_depth = jnp.minimum(depth, jnp.log(self.h_clip_max))

        eta_norm, xi_norm = inputs[..., 0], inputs[..., 1]
        xi_phys = xi_norm * self.xi_scale

        # Compute the analytic baseline in physical units, then normalize its output.
        baseline = jnp.zeros_like(xi_norm)
        if self.baseline_order >= 0:
            baseline = self._linear_baseline(xi_norm, depth)
        if self.baseline_order >= 1:
            baseline = baseline + self._g1_baseline(eta_norm, xi_norm, depth)
        m1_baseline = baseline
        if self.baseline_order == 2:
            # Retain M1 separately for training objectives that use fixed sample weights.
            k64 = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
                grid_size // 2 + 1, dtype=jnp.float64,
            )
            baseline = dno_series_eval(
                (eta_norm * self.eta_scale).astype(jnp.float64),
                xi_phys.astype(jnp.float64), k64,
                jnp.exp(depth if self.baseline_true_depth else clipped_log_depth).astype(jnp.float64), order=2,
            ) / self.target_scale

        amplitude = (surface_amplitude(eta_norm, self.surface_amplitude_reference)
                     if self.surface_amplitude_reference > 0 else None)
        eta_features, nested_features, polynomial_features, context = self._encode_surface(
            eta_norm, depth, amplitude=amplitude,
        )
        # G0-only must allow O(eta) corrections; its bias-free GELU trunk already does.
        # Preserve the analytic Taylor terms; a constant input removes this constraint.
        if self.baseline_order == 2:
            eta_features = eta_features * jnp.tanh(eta_features) ** 2
        elif self.baseline_order != 0:
            eta_features = eta_features * jnp.tanh(eta_features)
        if self.surface_context_gate and context is not None:
            eta_features = eta_features * (1.0 + jnp.tanh(context))
        if polynomial_features is not None:
            eta_features = jnp.concatenate((eta_features, polynomial_features), axis=-1)
        if self.surface_depth_gate:
            t = clipped_log_depth / 4.0
            gains = 1.0 + 0.5 * jnp.tanh(nn.Dense(
                eta_features.shape[-1], use_bias=False,
                kernel_init=nn.initializers.zeros, name="eta_depth_gate",
            )(jnp.concatenate((t, t**2), axis=-1)))
            eta_features = eta_features * gains[:, None, :]
        if amplitude is not None:
            eta_features = eta_features * amplitude[..., None] ** 2

        # Sum parallel correction groups, starting from zero in xi's dtype.
        correction = jnp.zeros((batch_size, grid_size), dtype=xi_phys.dtype)
        # Nested eta factors retain raw training-RMS scaling when the trunk is normalized.
        nested_surface = eta_norm
        if self.surface_amplitude_reference > 0:
            nested_surface = nested_surface / self.surface_amplitude_reference
        elif self.surface_feature_scales is not None:
            nested_surface = nested_surface / self.surface_feature_scales[0]
        if self.nested_surface_modulation:
            gain = self.param("nested_surface_gain", nn.initializers.zeros, ())
            nested_surface = nested_surface * (
                1.0 + jnp.tanh(gain) * jnp.tanh(eta_norm * self.eta_scale / jnp.exp(depth))
            )
        phase_hat = None
        if self.phase_rank_one:
            coefficients = self.param(
                "phase_rank_one_coefficients", nn.initializers.zeros, (2,), jnp.float32,
            )
            phase_hat = phase_rank_one_hat(
                jnp.fft.rfft(eta_norm, axis=-1) * self.eta_scale,
                jnp.fft.rfft(xi_phys, axis=-1), depth, coefficients, self.domain_length,
            )
        elif self.phase_wavenumber_hidden:
            multiplier = PhaseWavenumberMultiplier(
                self.phase_wavenumber_hidden, self.domain_length, name="phase_wavenumber",
            )(depth, grid_size // 2 + 1)
            phase_hat = phase_wavenumber_hat(
                jnp.fft.rfft(eta_norm, axis=-1) * self.eta_scale,
                jnp.fft.rfft(xi_phys, axis=-1), multiplier, self.domain_length,
            )
        if phase_hat is not None:
            # Inject before the existing inverse FFT, compensating its forward norm
            # and the outer branch normalization; this head is not another tied branch.
            phase_hat = phase_hat * (float(self.n_blocks * self.latent) ** 0.5 / grid_size)
        for block_idx in range(self.n_blocks):
            correction = correction + CraigSulemBlock(
                n_branches=self.latent,
                domain_length=self.domain_length,
                h_clip_max=self.h_clip_max,
                mult_hidden=self.mult_hidden,
                nested=block_idx >= self.n_blocks - self.nested_blocks,
                multiplier_skip=self.multiplier_skip,
                log_multiplier_inputs=self.log_multiplier_inputs,
                name=f"cs_block_{block_idx}",
            )(eta_features, xi_phys, clipped_log_depth,
              eta_norm=nested_surface, nested_features=nested_features,
              correction_hat_addition=phase_hat if block_idx == self.n_blocks - 1 else None)

        # Divide by sqrt(total branches), then convert to the target's normalized units.
        correction = correction / float(self.n_blocks * self.latent) ** 0.5
        correction = correction / self.target_scale
        prediction = (baseline + correction)[..., None]
        if return_baseline:
            return prediction, baseline[..., None], m1_baseline[..., None], correction[..., None]
        return prediction


class CraigSulemTermsResidual(CraigSulemDNO):
    """A CS baseline plus a geometry-conditioned linear decoder of its term fields.

    The residual starts one surface order beyond the baseline and has zero spatial
    mean. Unlike the tied-filter model, its spatially varying coefficients do not
    guarantee self-adjointness. Split encoding/decoding permits CPU term production.
    """
    baseline_order: int = 3
    baseline_true_depth: bool = True

    @nn.compact
    def encode_geometry(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        """Return (batch, grid, order+1) coefficients independent of xi."""
        eta_norm = inputs[..., 0].astype(jnp.float32)
        features, _, _, _ = self._encode_surface(eta_norm, depth.astype(jnp.float32))
        # Column i starts at order baseline_order+1-i, without forcing nodal zeros.
        gate = jnp.tanh(features)
        lifted = features[..., None] * jnp.stack(
            [gate ** power for power in range(self.baseline_order, -1, -1)], axis=-1,
        )
        return nn.Einsum(
            shape=(features.shape[-1], self.baseline_order + 1), einsum_str="...fc,fc->...c",
            use_bias=False, kernel_init=nn.initializers.zeros, name="terms_head",
        )(lifted)

    def decode_terms(
        self, coefficients: jnp.ndarray, terms_phys: jnp.ndarray, *, return_baseline: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Join encoded geometry with physical terms (batch, order+1, grid)."""
        terms_phys = terms_phys.astype(jnp.float64)
        baseline = jnp.sum(terms_phys, axis=-2) / self.target_scale
        m1_baseline = jnp.sum(terms_phys[..., :2, :], axis=-2) / self.target_scale
        correction = jnp.sum(coefficients * jnp.swapaxes(terms_phys, -2, -1), axis=-1)
        correction = (correction - jnp.mean(correction, axis=-1, keepdims=True)) / self.target_scale
        prediction = (baseline + correction)[..., None]
        if return_baseline:
            return prediction, baseline[..., None], m1_baseline[..., None], correction[..., None]
        return prediction

    def __call__(
        self, inputs: jnp.ndarray, depth: jnp.ndarray, *, terms_phys: jnp.ndarray | None = None,
        return_baseline: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        coefficients = self.encode_geometry(inputs, depth)
        if terms_phys is None:
            k = (2.0 * jnp.pi / self.domain_length) * jnp.arange(
                inputs.shape[1] // 2 + 1, dtype=jnp.float64,
            )
            terms_phys = dno_series_eval(
                inputs[..., 0].astype(jnp.float64) * self.eta_scale,
                inputs[..., 1].astype(jnp.float64) * self.xi_scale,
                k, jnp.exp(depth.astype(jnp.float64)), order=self.baseline_order,
                pad_factor=2, return_terms=True,
            )
        return self.decode_terms(coefficients, terms_phys, return_baseline=return_baseline)
