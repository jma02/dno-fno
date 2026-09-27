"""Full spatial attention -> FFT -> frequency attention -> IFFT, conditioned on eta."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def baseline(eta: Tensor, xi: Tensor, depth: Tensor, length: float) -> Tensor:
    """Physical G0(xi) + G1(eta, xi); MPS uses float32, CPU/CUDA use float64."""
    dtype = xi.dtype
    work_dtype = torch.float32 if xi.device.type == "mps" else torch.float64
    eta, xi, depth = (v.to(work_dtype) for v in (eta, xi, depth))
    n = xi.shape[-1]
    k = torch.arange(n // 2 + 1, device=xi.device, dtype=work_dtype) * (2 * math.pi / length)
    symbol = k * torch.tanh(depth.reshape(-1, 1).clamp(max=5) * k)
    spectrum = torch.fft.rfft(xi)
    g0 = torch.fft.irfft(symbol * spectrum, n=n)
    dx = torch.fft.irfft(1j * k * spectrum, n=n)
    g1 = torch.fft.irfft(-symbol * torch.fft.rfft(eta * g0) - 1j * k * torch.fft.rfft(eta * dx), n=n)
    return (g0 + g1).to(dtype)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        # Explicit projections avoid MultiheadAttention's inference-only fast path.
        batch, positions, width = x.shape
        heads = self.attention.num_heads
        qkv = F.linear(normalized, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q, k, v = (t.reshape(batch, positions, heads, width // heads).transpose(1, 2)
                   for t in qkv.chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, positions, width)
        x = x + self.attention.out_proj(attended)
        return x + self.mlp(self.norm2(x))


class DNO(nn.Module):
    def __init__(self, n: int = 1024, width: int = 128, branches: int = 32,
                 heads: int = 4, length: float = 2 * math.pi,
                 depth: int = 1, bf16: bool = False) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.n, self.length = n, length
        self.depth, self.bf16 = depth, bf16
        self.encoder = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.position = nn.Parameter(.01 * torch.randn(n, width))
        self.spatial = AttentionBlock(width, heads) if depth == 1 else nn.Sequential(
            *(AttentionBlock(width, heads) for _ in range(depth)))
        self.frequency_position = nn.Parameter(.01 * torch.randn(n // 2 + 1, 2 * width))
        self.frequency = AttentionBlock(2 * width, heads) if depth == 1 else nn.Sequential(
            *(AttentionBlock(2 * width, heads) for _ in range(depth)))
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, branches, bias=False))
        nn.init.zeros_(self.decoder[-1].weight)
        self.filters = nn.Parameter(torch.randn(n // 2 + 1, branches))
        mask = torch.ones(n // 2 + 1, 1)
        mask[0] = 0
        self.register_buffer("nonzero_modes", mask)

    def correction(self, eta: Tensor, xi: Tensor, depth: Tensor) -> Tensor:
        """Tied real filters make the correction linear in xi and self-adjoint."""
        log_depth = depth.reshape(-1, 1).clamp(max=5).log().expand_as(eta)
        with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            features = self.encoder(torch.stack((eta, log_depth), -1))
            features = self.spatial(features + self.position)  # All 1024 positions.
        features = features.float() if self.bf16 else features
        spectrum = torch.fft.rfft(features, dim=1, norm="ortho")
        tokens = torch.cat((spectrum.real, spectrum.imag), dim=-1)
        with torch.autocast(eta.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            tokens = self.frequency(tokens + self.frequency_position)
        tokens = tokens.float() if self.bf16 else tokens
        real, imag = tokens.chunk(2, dim=-1)
        # irfft ignores the imaginary DC/Nyquist components.
        features = torch.fft.irfft(torch.complex(real, imag), n=self.n, dim=1, norm="ortho")
        with torch.autocast(eta.device.type, enabled=False):
            weights = self.decoder(features)
        filters = self.filters * self.nonzero_modes
        filtered = torch.fft.irfft(torch.fft.rfft(xi, dim=1)[..., None] * filters, n=self.n, dim=1)
        weighted = torch.fft.rfft(weights * filtered, dim=1)
        return torch.fft.irfft((weighted * filters).sum(-1), n=self.n, dim=1) / math.sqrt(filters.shape[-1])

    def forward(self, eta: Tensor, xi: Tensor, depth: Tensor) -> Tensor:
        # Inputs: physical eta, xi [B,N], positive depth [B] or [B,1]. Output [B,N].
        return baseline(eta, xi, depth, self.length) + self.correction(eta, xi, depth)
