import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool1D(nn.Module):
    """
    Shift-invariant downsampling using a fixed Gaussian kernel.
    Prevents aliasing when downsampling.
    """
    def __init__(self, channels, stride=2):
        super().__init__()
        # Anti-aliasing kernel [1, 2, 1]
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :])
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        # x: [b, c, n]
        return F.conv1d(x, self.kernel.expand(self.channels, -1, -1), 
            stride=self.stride, padding=1, groups=self.channels)

class DownsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Project to hidden dimension (optional, but good for mixing)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim) # Added Norm for stability
        self.act = nn.SiLU()
        # 2. BlurPool for downsampling
        self.pool = BlurPool1D(dim, stride=2)

    def forward(self, x):
        # x: [b, n, d]
        
        # Conv branch
        residual = x
        x_in = x.transpose(1, 2) # [b, d, n]
        x_in = self.conv(x_in)
        x_in = x_in.transpose(1, 2) # [b, n, d]
        
        # Norm + Act
        x_in = self.norm(x_in)
        x_in = self.act(x_in)
        
        # Pool
        x_out = self.pool(x_in.transpose(1, 2)).transpose(1, 2)
        return x_out

class UpsampleBlock(nn.Module):
    """
    SOTA 'Resize-Blur-Convolution' approach.
    Uses Nearest Neighbor upsampling, an anti-aliasing blur, 
    followed by a Convolution to maintain shift-equivariance.
    """
    def __init__(self, dim):
        super().__init__()
        # 1. Added the Blur filter
        self.blur = BlurPool1D(dim)
        
        # 2. Refinement Convolution
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, target_length):
        # x: [b, n', d]
        
        # 1. Nearest Neighbor Upsampling (Learnable "Resize")
        x = x.transpose(1, 2) # [b, d, n']
        x = F.interpolate(x, size=target_length, mode='nearest')
        
        # 2. Anti-aliasing Blur (Smooths the NN "step edges")
        x = self.blur(x)
        
        # 3. Convolution (The "Refinement" step)
        x = self.conv(x)
        x = x.transpose(1, 2) # Back to [b, n, d]
        
        # 4. Norm + Act
        x = self.norm(x)
        x = self.act(x)
        
        return x