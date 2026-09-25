"""Small spectral transformer with physical surface features and constrained gates."""

import math

import torch
from torch import Tensor, nn

from model import AttentionBlock, baseline


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
                 bf16: bool = False, feature_scales: Tensor | None = None) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.n, self.length, self.depth, self.bf16 = n, length, depth, bf16
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
        local = local.float() if self.bf16 else local
        spectrum = torch.fft.rfft(local, dim=1, norm="ortho")
        with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            tokens = self.frequency_in(torch.cat((spectrum.real, spectrum.imag, token_condition), -1))
            tokens = self.frequency(tokens)
            coefficients = self.frequency_out(tokens)
        coefficients = coefficients.float() if self.bf16 else coefficients
        real, imag = coefficients.chunk(2, -1)
        context = torch.fft.irfft(torch.complex(real, imag), n=self.n, dim=1, norm="ortho")
        with torch.autocast(eta.device.type, enabled=False):
            # The local path is O(eta^2); bounded attention modulation cannot remove it.
            anchored = local * local.tanh()
            weights = self.decoder(anchored * (1 + context.tanh()))
            filters = self.depth_filters(filter_features) * (self.k != 0)[None, :, None]
            filtered = torch.fft.irfft(torch.fft.rfft(xi, dim=1)[..., None] * filters, n=self.n, dim=1)
            weighted = torch.fft.rfft(weights * filtered, dim=1)
            return torch.fft.irfft((weighted * filters).sum(-1), n=self.n, dim=1) / math.sqrt(filters.shape[-1])

    def forward(self, eta: Tensor, xi: Tensor, depth: Tensor) -> Tensor:
        return baseline(eta, xi, depth, self.length) + self.correction(eta, xi, depth)
