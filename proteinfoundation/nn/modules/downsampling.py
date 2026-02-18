import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool1D(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        # Anti-aliasing kernel [1, 2, 1]
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        # Reshape to [1, 1, 3] for depthwise conv
        self.register_buffer('kernel', kernel[None, None, :])
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        # x: [b, c, n]
        # Depthwise convolution with padding to preserve size before striding
        return F.conv1d(x, self.kernel.expand(self.channels, -1, -1), 
                        stride=self.stride, padding=1, groups=self.channels)

class DownsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.pool = BlurPool1D(dim, stride=2)

    def forward(self, x):
        # x: [b, n, d] -> permute -> [b, d, n]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        # Back to [b, n', d]
        return x.transpose(1, 2)

class UpsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, target_length):
        # x: [b, n', d]
        x = x.transpose(1, 2)
        # Linear interpolation to original length
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        x = self.conv(x)
        x = self.act(x)
        return x.transpose(1, 2)