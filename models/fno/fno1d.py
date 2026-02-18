import torch
from torch import nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Ensure we don't try to take more modes than available from rfft
        n_modes = min(self.modes1, x_ft.size(-1))
        out_ft[:, :, :n_modes] = self.compl_mul1d(x_ft[:, :, :n_modes], self.weights1[:, :, :n_modes])
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, n_blocks=4):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.n_blocks = n_blocks
        self.fc0 = nn.Linear(2, self.width)
        
        self.spectral_layers = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes1) for _ in range(self.n_blocks)])
        self.skip_layers = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_blocks)])
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        for i in range(self.n_blocks):
            x1 = self.spectral_layers[i](x)
            x2 = self.skip_layers[i](x)
            x = x1 + x2
            if i < self.n_blocks - 1:
                x = F.relu(x)
                
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
