"""Real FFT adjoints without a full complex inverse transform in backward."""

from typing import Any

import torch
from torch import Tensor


class _RFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, n: int, norm: str) -> Tensor:
        ctx.n, ctx.norm = n, norm
        return torch.fft.rfft(x, n=n, norm=norm)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None, None]:
        k = torch.arange(ctx.n // 2 + 1, device=gradient.device)
        endpoint = (k == 0) | ((ctx.n % 2 == 0) & (k == ctx.n // 2))
        scale = torch.where(endpoint, 1., .5).to(gradient.real.dtype)
        packed = torch.view_as_real(gradient) * scale[:, None]
        norm = {"backward": "forward", "forward": "backward", "ortho": "ortho"}[ctx.norm]
        return torch.fft.irfft(torch.view_as_complex(packed), n=ctx.n, norm=norm), None, None


class _IRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, n: int, norm: str) -> Tensor:
        ctx.n, ctx.norm = n, norm
        return torch.fft.irfft(x, n=n, norm=norm)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None, None]:
        norm = {"backward": "forward", "forward": "backward", "ortho": "ortho"}[ctx.norm]
        transformed = torch.fft.rfft(gradient.contiguous(), n=ctx.n, norm=norm)
        k = torch.arange(ctx.n // 2 + 1, device=gradient.device)
        endpoint = (k == 0) | ((ctx.n % 2 == 0) & (k == ctx.n // 2))
        scale = torch.where(endpoint, 1., 2.).to(gradient.dtype)
        return torch.view_as_complex(torch.view_as_real(transformed) * scale[:, None]), None, None


def rfft(x: Tensor, n: int | None = None, norm: str = "backward") -> Tensor:
    """Transform the last axis; explicit n must equal its input length."""
    return _RFFT.apply(x, x.shape[-1] if n is None else n, norm)


def irfft(x: Tensor, n: int, norm: str = "backward") -> Tensor:
    """Transform an n//2+1-bin spectrum along its last axis."""
    return _IRFFT.apply(x, n, norm)
