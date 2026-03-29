from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer that only keeps a fixed number of low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def complex_multiply(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bix,iox->box", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, grid_size = x.shape
        x_fft = torch.fft.rfft(x)
        kept_modes = min(self.modes, x_fft.shape[-1])

        out_fft = torch.zeros(
            batch_size,
            self.out_channels,
            grid_size // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_fft[:, :, :kept_modes] = self.complex_multiply(
            x_fft[:, :, :kept_modes],
            self.weights[:, :, :kept_modes],
        )
        return torch.fft.irfft(out_fft, n=grid_size)


class FNO1d(nn.Module):
    """Small 1D Fourier Neural Operator for DNO regression."""

    def __init__(self, modes: int, width: int, n_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_blocks)]
        )
        self.residual_layers = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_blocks)]
        )
        self.hidden_proj = nn.Linear(width, 128)
        self.output_proj = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as (batch, grid, channels).
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)

        for block_idx, (spectral_layer, residual_layer) in enumerate(
            zip(self.spectral_layers, self.residual_layers)
        ):
            x = spectral_layer(x) + residual_layer(x)
            if block_idx < len(self.spectral_layers) - 1:
                x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = F.relu(self.hidden_proj(x))
        return self.output_proj(x)
