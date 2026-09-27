"""Small spectral transformer with physical surface features and constrained gates."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model import AttentionBlock, baseline
from real_fft import irfft, rfft


def surface_features(eta: Tensor, length: float) -> Tensor:
    """Physical eta, powers, first/second/half derivatives and Hilbert transform."""
    k = torch.arange(eta.shape[1] // 2 + 1, device=eta.device, dtype=eta.dtype) * (2 * math.pi / length)
    multipliers = torch.stack((1j * k, -k.square(), k.sqrt(), -1j * k.sign()), -1)
    derivatives = torch.fft.irfft(torch.fft.rfft(eta, dim=1)[..., None] * multipliers,
                                 n=eta.shape[1], dim=1)
    return torch.cat((torch.stack((eta, eta.square(), eta**3), -1), derivatives), -1)


class SpectralDNO(nn.Module):
    def __init__(self, n: int = 1024, width: int = 64, branches: int = 32,
                 heads: int = 4, length: float = 2 * math.pi, depth: int = 2,
                 bf16: bool = False, feature_scales: Tensor | None = None,
                 max_mode: int | None = None) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if max_mode is not None and not 0 <= max_mode <= n // 2:
            raise ValueError("max_mode must be between zero and the Nyquist mode")
        self.n, self.length, self.depth, self.bf16 = n, length, depth, bf16
        self.max_mode = max_mode
        self.register_buffer("feature_scales", torch.ones(7) if feature_scales is None else feature_scales)
        self.register_buffer("k", torch.arange(n // 2 + 1) * (2 * math.pi / length))
        self.encoder = nn.Sequential(nn.Linear(7, width, bias=False), nn.GELU(),
                                     nn.Linear(width, width, bias=False), nn.GELU())
        self.frequency_in = nn.Linear(2 * width + 4, width)
        self.frequency = nn.Sequential(*(AttentionBlock(width, heads) for _ in range(depth)))
        self.frequency_out = nn.Linear(width, 2 * width)
        self.decoder = nn.Linear(width, branches, bias=False)
        nn.init.zeros_(self.decoder.weight)
        self.depth_filters = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, branches))

    def correction(self, eta: Tensor, xi: Tensor, depth: Tensor) -> Tensor:
        features = surface_features(eta, self.length) / self.feature_scales
        k, h = torch.broadcast_tensors(self.k[None, :], depth.reshape(-1, 1).clamp(max=5))
        tanh_kh = torch.tanh(k * h)
        # Same physical frequency/depth features as the production multiplier.
        filter_features = torch.stack((k, h, tanh_kh, k * tanh_kh), -1)
        token_condition = torch.stack((k / self.k[-1], h / 5, tanh_kh,
                                       k / self.k[-1] * tanh_kh), -1)
        with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            local = self.encoder(features)
        # Keep each FFT's spatial axis contiguous; GEMMs still use channels last.
        fft_input = local.transpose(1, 2).float() if self.bf16 else local.transpose(1, 2)
        spectrum = rfft(fft_input.contiguous(), norm="ortho").transpose(1, 2)
        width = local.shape[-1]
        # Interleave complex activation pairs; permute the small weights instead
        # of constructing separate real/imaginary activation gradients.
        packed = torch.view_as_real(spectrum).flatten(-2)
        weight = self.frequency_in.weight
        weight = torch.cat((weight[:, :2 * width].reshape(width, 2, width)
                            .transpose(1, 2).reshape(width, 2 * width), weight[:, 2 * width:]), -1)
        with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            tokens = F.linear(torch.cat((packed, token_condition), -1), weight, self.frequency_in.bias)
            tokens = self.frequency(tokens)
            weight_out = self.frequency_out.weight.reshape(2, width, width).transpose(0, 1).reshape(2 * width, width)
            bias_out = self.frequency_out.bias.reshape(2, width).T.reshape(2 * width)
            coefficients = F.linear(tokens, weight_out, bias_out)
        coefficients = coefficients.float() if self.bf16 else coefficients
        packed = coefficients.reshape(*coefficients.shape[:-1], width, 2).transpose(1, 2).contiguous()
        context = irfft(torch.view_as_complex(packed), n=self.n, norm="ortho").transpose(1, 2)
        with torch.autocast(eta.device.type, enabled=False):
            # The local path is O(eta^2); bounded attention modulation cannot remove it.
            local = local.float() if self.bf16 else local
            anchored = local * local.tanh()
            weights = self.decoder(anchored * (1 + context.tanh()))
            filters = (self.depth_filters(filter_features) * (self.k != 0)[None, :, None]).transpose(1, 2).contiguous()
            # Real views let Inductor fuse multiplier/reduction kernels instead of
            # materializing separate complex multiplies and conjugate gradients.
            spectrum_xi = torch.view_as_real(torch.fft.rfft(xi))[:, None]
            filtered = irfft(torch.view_as_complex(spectrum_xi * filters[..., None]), n=self.n)
            weighted = rfft(weights.transpose(1, 2) * filtered)
            output = (torch.view_as_real(weighted) * filters[..., None]).sum(1)
            return irfft(torch.view_as_complex(output), n=self.n) / math.sqrt(filters.shape[1])

    def project(self, value: Tensor) -> Tensor:
        if self.max_mode is None:
            return value
        mask = torch.arange(self.n // 2 + 1, device=value.device) <= self.max_mode
        return irfft(rfft(value) * mask, n=self.n)

    def forward(self, eta: Tensor, xi: Tensor, depth: Tensor) -> Tensor:
        # P G(eta) P preserves linearity and self-adjointness in xi.
        xi = self.project(xi)
        return self.project(baseline(eta, xi, depth, self.length) + self.correction(eta, xi, depth))
