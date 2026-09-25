"""Craig-Sulem neural DNO: analytic G0 + G1 plus a learned correction.

Here eta is surface elevation and xi is surface velocity potential.
The default correction branches apply M[spatial_weights(eta) * M[xi]], where M is
a depth-dependent real Fourier filter. The correction is self-adjoint,
linear in xi, and starts at order eta^2.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def fourier_resample_even(field: jnp.ndarray, size: int, *, restriction_adjoint: bool = False) -> jnp.ndarray:
    """Resample axis 1; optional upsampling is the mean-inner-product restriction adjoint."""
    source_size = field.shape[1]
    if source_size == size:
        return field
    if size < 2 or size % 2 or source_size % 2:
        raise ValueError("Fourier resampling requires positive even grid sizes")
    spectrum = jnp.fft.rfft(field, axis=1, norm="forward")
    if size < source_size:
        spectrum = spectrum[:, :size // 2 + 1]
        spectrum = spectrum.at[:, -1].set(2 * spectrum[:, -1].real)
    elif not restriction_adjoint:
        spectrum = spectrum.at[:, -1].multiply(0.5)
    return jnp.fft.irfft(spectrum, n=size, axis=1, norm="forward")


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
    fuse_fft: bool = False

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
        branched_spectrum = multiplier * xi_hat[..., None]
        if self.fuse_fft and branched_spectrum.dtype == jnp.complex64:
            from fused_fft import fused_roundtrip

            weighted_xi_hat = fused_roundtrip(branched_spectrum, spatial_weights)
        else:
            # Keep the float64 path for finite-secant physics regularization.
            filtered_xi = jnp.fft.irfft(
                branched_spectrum, n=grid_size, axis=1, norm="forward",
            )
            weighted_xi_hat = jnp.fft.rfft(
                spatial_weights * filtered_xi, axis=1, norm="forward",
            )
        correction_hat = jnp.sum(multiplier * weighted_xi_hat, axis=-1)
        return jnp.fft.irfft(correction_hat, n=grid_size, axis=-1, norm="forward")


class CompactCorrection(nn.Module):
    """Experimental P.T C(eta, depth) P xi using a fixed real Fourier basis.

    C is symmetric and O(eta^2). The rank limit and lack of structural
    translation equivariance are intentional restrictions of this option.
    """
    rank: int = 64
    hidden: int = 128

    @nn.compact
    def __call__(self, eta: jnp.ndarray, xi: jnp.ndarray,
                 depth: jnp.ndarray) -> jnp.ndarray:
        size = eta.shape[1]
        if self.rank < 2 or self.rank % 2 or self.rank >= size or self.hidden < 1:
            raise ValueError("Compact correction needs positive hidden width and an even rank in [2, grid size)")
        angle = (2 * jnp.pi / size) * jnp.arange(size, dtype=eta.dtype)[:, None]
        angle = angle * jnp.arange(1, self.rank // 2 + 1, dtype=eta.dtype)[None, :]
        basis = jnp.concatenate((jnp.cos(angle), jnp.sin(angle)), axis=1) * (2 / size) ** 0.5

        # Bias-free eta maps vanish at eta=0. Depth only modulates their values.
        hidden = nn.gelu(nn.Dense(self.hidden, use_bias=False, name="eta_encoder")(eta))
        hidden *= 1 + nn.tanh(nn.Dense(self.hidden, name="depth_gate")(depth))
        left = nn.Dense(self.rank**2, use_bias=False, name="matrix_left")(hidden)
        right = nn.Dense(
            self.rank**2, use_bias=False, name="matrix_right",
            kernel_init=nn.initializers.zeros,
        )(hidden)
        # Zero only one factor: initial correction is zero, but its gradient is not.
        matrix = (left * right).reshape(eta.shape[0], self.rank, self.rank)
        matrix = (matrix + matrix.swapaxes(-1, -2)) / (2 * self.rank**0.5)
        coefficients = xi @ basis
        corrected = jnp.einsum("bij,bj->bi", matrix, coefficients)
        return corrected @ basis.T


class SpectralMLPCorrection(nn.Module):
    """FFT -> global dense MLP -> inverse FFT -> pointwise decoder.

    Uses every independent real Fourier coefficient on the learned grid.
    Unlike the branch correction, this does not enforce linearity in xi,
    self-adjointness, translation equivariance or quadratic order in eta.
    """
    hidden: int = 256
    layers: int = 4
    channels: int = 16
    decoder_hidden: int = 64

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        batch, size, _ = inputs.shape
        if size % 2 or min(self.hidden, self.layers, self.channels, self.decoder_hidden) < 1:
            raise ValueError("Spectral MLP needs an even grid and positive layer dimensions")
        spectrum = jnp.fft.rfft(inputs, axis=1, norm="ortho")
        # DC and Nyquist have no imaginary degrees of freedom on an even real grid.
        hidden = jnp.concatenate((
            spectrum.real.reshape(batch, -1),
            spectrum[:, 1:-1].imag.reshape(batch, -1), depth,
        ), axis=-1)
        for index in range(self.layers):
            hidden = nn.gelu(nn.Dense(self.hidden, name=f"hidden_{index}")(hidden))
        coefficients = nn.Dense(size * self.channels, name="spectrum_out")(hidden)
        coefficients = coefficients.reshape(batch, size, self.channels)
        n_freq = size // 2 + 1
        real = coefficients[:, :n_freq]
        imaginary = jnp.pad(coefficients[:, n_freq:], ((0, 0), (1, 1), (0, 0)))
        spatial = jnp.fft.irfft(real + 1j * imaginary, n=size, axis=1, norm="ortho")
        decoded = nn.gelu(nn.Dense(self.decoder_hidden, name="decoder_hidden")(spatial))
        correction = nn.Dense(
            1, use_bias=False, kernel_init=nn.initializers.zeros, name="decoder_out",
        )(decoded)[..., 0]
        return correction - correction.mean(axis=1, keepdims=True)


class CanonicalFNOCorrection(nn.Module):
    """Benchmark candidate: four width-32 Fourier blocks retaining every mode."""
    fold_spatial: bool = True
    spectral_gemm: str = "packed"
    transform: str = "fft_backward"
    blocks: int = 4
    output_channels: int = 1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        batch, size, _ = inputs.shape
        condition = jnp.broadcast_to(depth[:, None, :], (batch, size, 1))
        hidden = nn.Dense(32, name="encoder")(jnp.concatenate((inputs, condition), axis=-1))
        channels_first = self.transform == "fft_channels_first"
        if channels_first:
            hidden = hidden.swapaxes(1, 2)
        fft_axis = 2 if channels_first else 1
        channel_axis = 1 if channels_first else 2
        equation = "bik,kio->bok" if channels_first else "bki,kio->bko"
        fused = self.transform == "fused" and hidden.dtype == jnp.float32 and size == 256
        if (self.transform == "fused" or channels_first) and not self.fold_spatial:
            raise ValueError("Fused/channel-first FFT requires folding the spatial branch")
        norm = "backward" if self.transform in ("fft_backward", "fft_channels_first", "fused") else "ortho"
        if self.transform == "dft":
            angle = 2 * np.pi * np.arange(size)[:, None] * np.arange(size // 2 + 1)[None, :] / size
            basis = np.concatenate((np.cos(angle), -np.sin(angle[:, 1:-1])), axis=1) / size**0.5
            multiplicity = np.full(size, 2.0)
            multiplicity[[0, size // 2]] = 1
            inverse = (basis * multiplicity[None, :]).T
            # Single-pass TF32 loses too much accuracy over repeated transforms.
            precision = "TF32_TF32_F32_X3" if hidden.dtype == jnp.float32 and jax.default_backend() == "gpu" else None
        for index in range(self.blocks):
            if self.transform == "dft":
                coefficients = jnp.einsum(
                    "bnc,nk->bkc", hidden, jnp.asarray(basis, dtype=hidden.dtype), precision=precision,
                )
                spectrum = coefficients[:, :size // 2 + 1] + 1j * jnp.pad(
                    coefficients[:, size // 2 + 1:], ((0, 0), (1, 1), (0, 0)),
                )
            elif not fused or index == 0:
                spectrum = jnp.fft.rfft(hidden, axis=fft_axis, norm=norm)
            shape = (size // 2 + 1, 32, 32)
            # Global x64 is enabled for the analytic baseline and physics losses.
            # Learned parameters should match Dense's float32 default explicitly.
            real = self.param(f"spectral_real_{index}", nn.initializers.normal(1 / 32**0.5), shape, jnp.float32)
            imaginary = self.param(f"spectral_imag_{index}", nn.initializers.normal(1 / 32**0.5), shape, jnp.float32)
            kernel = self.param(f"spatial_kernel_{index}", nn.initializers.lecun_normal(), (32, 32), jnp.float32)
            bias = self.param(f"spatial_bias_{index}", nn.initializers.zeros, (32,), jnp.float32)
            if self.fold_spatial:
                # A constant channel matrix commutes with the spatial FFT.
                real = real.astype(spectrum.real.dtype) + kernel
            if self.spectral_gemm == "packed":
                packed = jnp.concatenate((spectrum.real, spectrum.imag), axis=channel_axis)
                weights = jnp.concatenate((
                    jnp.concatenate((real, imaginary), axis=-1),
                    jnp.concatenate((-imaginary, real), axis=-1),
                ), axis=-2)
                product = jnp.einsum(equation, packed, weights)
                product_real, product_imag = jnp.split(product, 2, axis=channel_axis)
                mixed = product_real + 1j * product_imag
            elif self.spectral_gemm == "split":
                real_part = jnp.einsum(equation, spectrum.real, real)
                real_part -= jnp.einsum(equation, spectrum.imag, imaginary)
                imaginary_part = jnp.einsum(equation, spectrum.real, imaginary)
                imaginary_part += jnp.einsum(equation, spectrum.imag, real)
                mixed = real_part + 1j * imaginary_part
            elif self.spectral_gemm == "complex":
                mixed = jnp.einsum(equation, spectrum, real + 1j * imaginary)
            else:
                raise ValueError(f"Unknown FNO spectral GEMM: {self.spectral_gemm!r}")
            if fused and index < self.blocks - 1:
                from fused_fft import fused_gelu_roundtrip

                spectrum = fused_gelu_roundtrip(mixed, bias, interpret=jax.default_backend() == "cpu")
                continue
            if self.transform == "dft":
                coefficients = jnp.concatenate((mixed.real, mixed[:, 1:-1].imag), axis=1)
                spatial = jnp.einsum(
                    "bkc,kn->bnc", coefficients, jnp.asarray(inverse, dtype=coefficients.dtype), precision=precision,
                )
            else:
                spatial = jnp.fft.irfft(mixed, n=size, axis=fft_axis, norm=norm)
            if not self.fold_spatial:
                spatial = spatial + hidden @ kernel
            hidden = nn.gelu(spatial + (bias[None, :, None] if channels_first else bias))
        if channels_first:
            hidden = hidden.swapaxes(1, 2)
        hidden = nn.gelu(nn.Dense(64, name="decoder_hidden")(hidden))
        correction = nn.Dense(
            self.output_channels, use_bias=False, kernel_init=nn.initializers.zeros, name="decoder_out",
        )(hidden)
        if self.output_channels == 1:
            correction = correction[..., 0]
            return correction - correction.mean(axis=1, keepdims=True)
        return correction


class SelfAdjointFNOCorrection(nn.Module):
    """Surface-only FNO gates a linear, self-adjoint potential path; no eta-order constraint."""
    blocks: int = 4

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        size = inputs.shape[1]
        weights = CanonicalFNOCorrection(
            blocks=self.blocks, output_channels=32, name="conditioner",
        )(inputs[..., :1], depth)
        multiplier = self.param(
            "filter", nn.initializers.normal(1.0), (size // 2 + 1, 32), jnp.float32,
        ).astype(inputs.dtype)
        # A zero DC filter preserves the existing mean-free correction on both sides.
        multiplier = multiplier.at[0, :].set(0)
        spectrum = jnp.fft.rfft(inputs[..., 1], axis=1, norm="backward")
        filtered = jnp.fft.irfft(spectrum[..., None] * multiplier, n=size, axis=1, norm="backward")
        weighted = jnp.fft.rfft(weights * filtered, axis=1, norm="backward")
        return jnp.fft.irfft(jnp.sum(multiplier * weighted, axis=-1), n=size, axis=1, norm="backward") / 32**0.5


class FullSpectrumMLPCorrection(nn.Module):
    """Learn full-spectrum compression; a surface MLP gates a tied linear xi map.

    The xi operator has rank at most hidden, with a learned Fourier subspace
    spanning all input frequencies. No hard low-pass or eta-order constraint.
    """
    hidden: int = 256
    layers: int = 4

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        size = inputs.shape[1]
        if size % 2 or self.hidden < 1 or self.layers < 1:
            raise ValueError("Full-spectrum MLP requires an even grid and positive dimensions")
        # Single-pass TF32 rounds xi before projection, measurably degrading
        # linearity. Three-pass TF32 retains tensor cores with near-float32 accuracy.
        precision = "TF32_TF32_F32_X3" if inputs.dtype == jnp.float32 and jax.default_backend() == "gpu" else None
        spectrum = jnp.fft.rfft(inputs, axis=1, norm="ortho")
        # Orthonormal real coordinates make a tied matrix transpose the true
        # adjoint, including DC and Nyquist (which have no imaginary component).
        real = spectrum.real.at[:, 1:-1, :].multiply(2**0.5)
        coefficients = jnp.concatenate((real, spectrum[:, 1:-1].imag * 2**0.5), axis=1)
        hidden = jnp.concatenate((coefficients[..., 0], depth), axis=-1)
        for index in range(self.layers):
            hidden = nn.gelu(nn.Dense(self.hidden, precision=precision, name=f"hidden_{index}")(hidden))
        weights = nn.Dense(
            self.hidden, use_bias=False, kernel_init=nn.initializers.zeros, name="decoder_out",
            precision=precision,
        )(hidden)
        projection = self.param(
            "xi_projection", nn.initializers.lecun_normal(), (size - 1, self.hidden), jnp.float32,
        ).astype(inputs.dtype)
        # Exclude DC on both sides: constants are annihilated and output is mean-free.
        latent = jnp.matmul(coefficients[:, 1:, 1], projection, precision=precision)
        output = jnp.matmul(latent * weights, projection.T, precision=precision)
        output = jnp.pad(output, ((0, 0), (1, 0)))
        n_freq = size // 2 + 1
        real = output[:, :n_freq].at[:, 1:-1].divide(2**0.5)
        imaginary = jnp.pad(output[:, n_freq:] / 2**0.5, ((0, 0), (1, 1)))
        return jnp.fft.irfft(real + 1j * imaginary, n=size, axis=1, norm="ortho")


class AttentionResidual(nn.Module):
    heads: int = 4

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        channels = inputs.shape[-1]
        precision = "TF32_TF32_F32_X3" if inputs.dtype == jnp.float32 and jax.default_backend() == "gpu" else None
        normalized = nn.LayerNorm(name="attention_norm")(inputs)
        hidden = inputs + nn.MultiHeadDotProductAttention(
            num_heads=self.heads, qkv_features=channels, out_features=channels,
            precision=precision, name="attention",
        )(normalized, deterministic=True)
        normalized = nn.LayerNorm(name="mlp_norm")(hidden)
        mixed = nn.gelu(nn.Dense(2 * channels, precision=precision, name="mlp_in")(normalized))
        return hidden + nn.Dense(channels, precision=precision, name="mlp_out")(mixed)


class SpatialSpectralAttentionCorrection(nn.Module):
    """Full-grid surface attention generates weights for a tied linear xi path."""
    channels: int = 128
    heads: int = 4
    window: int = 64
    branches: int = 32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        batch, size, _ = inputs.shape
        if (size % 2 or min(self.channels, self.heads, self.window, self.branches) < 1
                or size % self.window or self.channels % self.heads):
            raise ValueError("Attention requires an even grid divisible by window, and channels divisible by heads")
        precision = "TF32_TF32_F32_X3" if inputs.dtype == jnp.float32 and jax.default_backend() == "gpu" else None
        condition = jnp.broadcast_to(depth[:, None, :], (batch, size, 1))
        hidden = jnp.concatenate((inputs[..., :1], condition), axis=-1)
        for index in range(2):
            hidden = nn.gelu(nn.Dense(self.channels, precision=precision, name=f"encoder_{index}")(hidden))
        position = self.param("window_position", nn.initializers.normal(0.01),
                              (self.window, self.channels), jnp.float32)
        windows = hidden.reshape(batch * (size // self.window), self.window, self.channels)
        windows = AttentionResidual(heads=self.heads, name="spatial")(
            windows + position.astype(inputs.dtype),
        )
        hidden = windows.reshape(batch, size, self.channels)
        spectrum = jnp.fft.rfft(hidden, axis=1, norm="ortho")
        tokens = jnp.concatenate((spectrum.real, spectrum.imag), axis=-1)
        frequency = self.param("frequency_position", nn.initializers.normal(0.01),
                               (size // 2 + 1, 2 * self.channels), jnp.float32)
        tokens = AttentionResidual(heads=self.heads, name="frequency")(
            tokens + frequency.astype(inputs.dtype),
        )
        real, imaginary = jnp.split(tokens, 2, axis=-1)
        imaginary = imaginary.at[:, 0].set(0).at[:, -1].set(0)
        spatial = jnp.fft.irfft(real + 1j * imaginary, n=size, axis=1, norm="ortho")
        hidden = nn.gelu(nn.Dense(self.channels, precision=precision, name="decoder_hidden")(spatial))
        weights = nn.Dense(
            self.branches, use_bias=False, kernel_init=nn.initializers.zeros,
            precision=precision, name="decoder_out",
        )(hidden)
        multiplier = self.param("filter", nn.initializers.normal(1.0),
                                (size // 2 + 1, self.branches), jnp.float32).astype(inputs.dtype)
        multiplier = multiplier.at[0].set(0)
        potential = jnp.fft.rfft(inputs[..., 1], axis=1, norm="backward")
        filtered = jnp.fft.irfft(potential[..., None] * multiplier, n=size, axis=1, norm="backward")
        weighted = jnp.fft.rfft(weights * filtered, axis=1, norm="backward")
        return jnp.fft.irfft(jnp.sum(multiplier * weighted, axis=-1), n=size, axis=1, norm="backward") / self.branches**0.5


class CraigSulemDNO(nn.Module):
    """Compute G0(xi) + G1(eta, xi) + learned_correction(eta, xi, depth)."""
    width: int                        # Each shared eta layer has width // 2 channels.
    n_blocks: int = 4                 # Groups of parallel branches, summed together.
    latent: int = 64                  # Branches per group.
    learned_grid: int | None = None   # Analytic baseline remains on the input grid.
    fuse_fft: bool = False            # Experimental 256-point float32 kernel.
    correction_kind: str = "branches"  # "compact" opts into a fixed-rank matrix.
    compact_rank: int = 64
    compact_hidden: int = 128
    spectral_hidden: int = 256
    spectral_layers: int = 4
    spectral_channels: int = 16
    spectral_decoder_hidden: int = 64
    attention_channels: int = 128
    attention_heads: int = 4
    attention_window: int = 64
    fno_fold_spatial: bool = True
    fno_spectral_gemm: str = "packed"
    fno_transform: str = "fft_backward"

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
        if self.correction_kind not in ("branches", "compact", "spectral_mlp", "canonical_fno", "symmetric_fno", "full_spectrum_mlp", "spatial_spectral_attention"):
            raise ValueError(f"Unknown correction kind: {self.correction_kind!r}")
        if self.correction_kind != "branches" and self.fuse_fft:
            raise ValueError("FFT fusion applies only to the branches correction")
        clipped_log_depth = jnp.minimum(depth, jnp.log(self.h_clip_max))

        eta_norm, xi_norm = inputs[..., 0], inputs[..., 1]
        xi_phys = xi_norm * self.xi_scale

        # Compute the analytic baseline in physical units, then normalize its output.
        baseline = (
            self._linear_baseline(xi_norm, depth)
            + self._g1_baseline(eta_norm, xi_norm, depth)
        )
        if self.correction_kind == "spatial_spectral_attention":
            correction = SpatialSpectralAttentionCorrection(
                channels=self.attention_channels, heads=self.attention_heads,
                window=self.attention_window, branches=self.latent, name="spatial_spectral_attention",
            )(inputs, clipped_log_depth)
            return (baseline + correction)[..., None]
        if self.correction_kind == "full_spectrum_mlp":
            correction = FullSpectrumMLPCorrection(
                hidden=self.spectral_hidden, layers=self.spectral_layers, name="full_spectrum_mlp",
            )(inputs, clipped_log_depth)
            return (baseline + correction)[..., None]
        output_grid = grid_size
        if self.learned_grid is not None and self.learned_grid != grid_size:
            inputs = fourier_resample_even(inputs, self.learned_grid)
            grid_size = inputs.shape[1]
            eta_norm, xi_norm = inputs[..., 0], inputs[..., 1]
            xi_phys = xi_norm * self.xi_scale
        if self.correction_kind in ("compact", "spectral_mlp", "canonical_fno", "symmetric_fno"):
            if self.correction_kind == "compact":
                correction = CompactCorrection(
                    rank=self.compact_rank, hidden=self.compact_hidden, name="compact",
                )(eta_norm, xi_phys, clipped_log_depth) / self.target_scale
            elif self.correction_kind == "symmetric_fno":
                correction = SelfAdjointFNOCorrection(blocks=self.n_blocks, name="symmetric_fno")(
                    inputs, clipped_log_depth,
                )
            elif self.correction_kind == "canonical_fno":
                correction = CanonicalFNOCorrection(
                    fold_spatial=self.fno_fold_spatial, spectral_gemm=self.fno_spectral_gemm,
                    transform=self.fno_transform, name="canonical_fno",
                )(inputs, clipped_log_depth)
            else:
                correction = SpectralMLPCorrection(
                    hidden=self.spectral_hidden, layers=self.spectral_layers,
                    channels=self.spectral_channels, decoder_hidden=self.spectral_decoder_hidden,
                    name="spectral_mlp",
                )(inputs, clipped_log_depth)
            if grid_size != output_grid:
                correction = fourier_resample_even(
                    correction[..., None], output_grid,
                    restriction_adjoint=self.correction_kind == "symmetric_fno",
                )[..., 0]
            return (baseline + correction)[..., None]
        if self.fuse_fft and (grid_size != 256 or self.latent % 8):
            raise ValueError("Fused FFT requires a 256-point learned grid and latent divisible by 8")

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
                fuse_fft=self.fuse_fft,
                name=f"cs_block_{block_idx}",
            )(eta_features, xi_phys, clipped_log_depth)

        # Divide by sqrt(total branches), then convert to the target's normalized units.
        correction = correction / float(self.n_blocks * self.latent) ** 0.5
        correction = correction / self.target_scale
        if grid_size != output_grid:
            correction = fourier_resample_even(correction[..., None], output_grid)[..., 0]
        return (baseline + correction)[..., None]
