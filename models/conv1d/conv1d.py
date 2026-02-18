import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(Cin, 64, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 64, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )
    def forward(self, x):  # (B,N,C)
        x = x.permute(0,2,1)
        y = self.net(x)
        return y.permute(0,2,1)
