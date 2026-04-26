import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool1D(nn.Module):
    """
    Shift-invariant downsampling using a fixed Gaussian-ish [1,2,1]/4 kernel.
    When a mask is provided the kernel is renormalised per-position so boundary
    residues next to padding are not attenuated. Default behavior (no mask)
    is identical to plain depthwise conv1d with the [1,2,1]/4 kernel.
    """
    def __init__(self, channels, stride=2):
        super().__init__()
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :])
        self.stride = stride
        self.channels = channels

    def forward(self, x, mask=None):
        # x: [b, c, n]; mask: [b, n] bool or None
        out = F.conv1d(x, self.kernel.expand(self.channels, -1, -1),
            stride=self.stride, padding=1, groups=self.channels)
        if mask is not None:
            # Weight-sum at each output position = how much of the [1,2,1]/4 kernel
            # fell on valid residues. Dividing renormalises so a real residue next
            # to a padded slot doesn't lose 25-50% of its magnitude to the blur.
            mask_f = mask.to(dtype=x.dtype).unsqueeze(1)  # [b, 1, n]
            weight_sum = F.conv1d(mask_f, self.kernel,
                stride=self.stride, padding=1)            # [b, 1, n_out]
            out = out / weight_sum.clamp(min=1e-6)
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.pool = BlurPool1D(dim, stride=2)
        # Zero-init output projection: at init this block is a pure stride-2 downsample.
        # Gradients will grow the projection's contribution over time.
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, mask=None):
        # x: [b, n, d]; mask: [b, n] bool or None for boundary-aware blur
        residual = self.pool(x.transpose(1, 2), mask=mask).transpose(1, 2)
        x_in = self.conv(x.transpose(1, 2))
        x_in = self.act(self.norm(x_in.transpose(1, 2)))
        x_out = self.pool(x_in.transpose(1, 2), mask=mask).transpose(1, 2)
        return residual + self.out_proj(x_out)

class UpsampleBlock(nn.Module):
    """
    Resize-Blur-Convolution upsampler with zero-init output and parity injection.
    NN upsample -> + parity_emb[i % 2] -> [1,2,1] blur -> conv -> LN -> SiLU
    -> zero-init linear.

    The parity embedding breaks the NN-interp symmetry between adjacent upsampled
    positions: without it, positions 2k and 2k+1 carry identical values and the
    conv has to learn an asymmetric response to distinguish them. With it, the
    conv sees a parity-distinguishable input from the start. parity_emb is
    zero-init so it doesn't disturb the at-init identity of the down-trunk-up
    pipeline (zero-init out_proj already gives that).
    """
    def __init__(self, dim):
        super().__init__()
        self.blur = BlurPool1D(dim, stride=1)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Zero-init parity embedding: model grows it iff the parity signal helps.
        self.parity_emb = nn.Parameter(torch.zeros(2, dim))

    def forward(self, x, target_length, mask=None):
        # x: [b, n', d]; mask: [b, target_length] bool or None
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_length, mode='nearest')  # [b, d, n]
        parity_idx = torch.arange(target_length, device=x.device) % 2
        parity = self.parity_emb[parity_idx].t().unsqueeze(0)     # [1, d, n]
        x = x + parity
        x = self.blur(x, mask=mask)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = self.out_proj(x)
        return x