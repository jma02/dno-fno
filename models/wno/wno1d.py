import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from pytorch_wavelets import DWT1DForward, DWT1DInverse

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class WaveEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 2
        self.proj1 = nn.Conv1d(in_channels + 2, mid, kernel_size=1)
        self.proj2 = nn.Conv1d(mid, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, N = x.shape
        # Compute global amplitude statistics across all dims
        flat = x.reshape(B, C * N)
        std_val = flat.std(dim=-1)  # (B,)
        amp_val = (flat.max(dim=-1)[0] - flat.min(dim=-1)[0]) / 2  # (B,)
        # Broadcast stats to spatial dim
        log_std = torch.log1p(std_val).view(B, 1, 1).expand(B, 1, N)
        log_amp = torch.log1p(amp_val).view(B, 1, 1).expand(B, 1, N)
        x_aug = torch.cat([x, log_std, log_amp], dim=1)  # (B, C+2, N)
        x_out = F.gelu(self.proj1(x_aug))
        x_out = self.proj2(x_out)
        # Return features and conditioning signal (B, 2)
        cond = torch.stack([std_val, amp_val], dim=1)  # (B, 2)
        return x_out, cond


class WaveletConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, J: int = 4, wave: str = "sym4", mode: str = "periodization"):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wave, mode=mode)
        self.idwt = DWT1DInverse(wave=wave, mode=mode)
        self.mix_low = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.mix_high = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1) for _ in range(J)
        ])
        self.dw_low = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels)
        self.dw_high = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels) for _ in range(J)
        ])
        self.norm_low = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.norm_high = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels) for _ in range(J)
        ])
        self.film_low = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2 * out_channels)
        )
        self.film_high = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim, 64),
                nn.GELU(),
                nn.Linear(64, 2 * out_channels)
            ) for _ in range(J)
        ])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        yl, yh = self.dwt(x)
        yl = self.mix_low(yl)
        yl = self.dw_low(yl)
        yl = self.norm_low(yl)
        film_low = self.film_low(cond).unsqueeze(-1)
        gamma_low, beta_low = film_low.chunk(2, dim=1)
        yl = F.gelu(yl) * (1 + gamma_low) + beta_low

        yh_out: List[torch.Tensor] = []
        for j, conv in enumerate(self.mix_high):
            hj = conv(yh[j])
            hj = self.dw_high[j](hj)
            hj = self.norm_high[j](hj)
            film_h = self.film_high[j](cond).unsqueeze(-1)
            gamma_h, beta_h = film_h.chunk(2, dim=1)
            hj = F.gelu(hj) * (1 + gamma_h) + beta_h
            yh_out.append(hj)
        return self.idwt((yl, yh_out))


class WNO1d(nn.Module):
    def __init__(
        self,
        width: int = 64,
        depth: int = 4,
        J_blocks: int = 4,
        wave: str = "db4",
        mode: str = "periodization",
    ):
        super().__init__()
        self.width = width
        self.padding = 2
        self.encoder = WaveEncoder(2, self.width)

        self.conv_layers = nn.ModuleList([
            WaveletConv1d(self.width, self.width, cond_dim=2, J=J_blocks, wave=wave, mode=mode) for _ in range(depth)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(depth)
        ])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x, cond = self.encoder(x)  # cond: (B, 2) amplitude conditioning

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x, cond)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
